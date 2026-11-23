"""Matched LOC alignment with fixed or predicted proposal priors, entirely offline."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import math
import os
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from hybrid_drive.assets.files import check_components
from hybrid_drive.neural.grounded_loc_probe import (
    GroundedLocProbe,
    fixed_quarter_boxes,
    grounded_presence_scores,
    supervised_grounded_loc_loss,
)
from hybrid_drive.neural.losses import Reason2DriveLoss
from hybrid_drive.neural.pretrained_alignment import load_grounded_alignment_backbones
from hybrid_drive.neural.spatial_data import (
    SpatialAlignmentDataset,
    collate_spatial_alignment,
    verify_spatial_splits,
)
from .bounded_io import decode_json, read_bounded, require_hex
from .io import write_json
from .neural_checkpoint import load_neural_weights, module_state_hash, save_neural_weights
from .proposal_alignment_report import write_proposal_report
from .proposal_prior_source import load_spatial_prior_source
from .protocol import content_hash
from .spatial_experiment import SPLIT_FRAMES
from .spatial_metrics import (
    SpatialPredictions,
    calibrate_spatial_threshold,
    shuffled_by_recording,
    spatial_metrics,
)


SCOPE = "openscene_four_band_proposal_alignment_ablation_v1"
CONDITIONS = ("fixed_quarters", "spatial_proposals")
LEARNING_RATE = 0.002


@dataclass(frozen=True)
class ProposalRecipe:
    epochs: int = 3
    batch_size: int = 4
    seeds: tuple[int, ...] = (0, 1)

    def __post_init__(self):
        if type(self.epochs) is not int or not 1 <= self.epochs <= 6:
            raise ValueError("epochs must be an integer from 1 to 6")
        if type(self.batch_size) is not int or not 1 <= self.batch_size <= 8:
            raise ValueError("batch size must be an integer from 1 to 8")
        seeds = tuple(self.seeds)
        if (
            not 1 <= len(seeds) <= 3
            or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds)
            or len(set(seeds)) != len(seeds)
        ):
            raise ValueError("Expected 1..3 distinct bounded seeds")
        object.__setattr__(self, "seeds", seeds)


@dataclass(frozen=True)
class _Encoded:
    features: Tensor
    proposals: Tensor
    presence: Tensor
    target_boxes: Tensor
    recordings: tuple[str, ...]

    def __len__(self):
        return len(self.recordings)

    def batch(self, indices, condition, proposal_order=None):
        features = self.features[indices]
        if condition == "fixed_quarters":
            boxes = fixed_quarter_boxes(features)
        elif condition == "spatial_proposals":
            boxes = self.proposals[indices if proposal_order is None else proposal_order[indices]]
        else:
            raise ValueError("Unknown proposal condition")
        return features, boxes, self.presence[indices], self.target_boxes[indices]


def run_proposal_alignment(
    data_root,
    model_root,
    prior_run_root,
    results_root,
    *,
    revision,
    expected_prior_result_sha256,
    recipe=ProposalRecipe(),
):
    require_hex(revision, 40, "revision")
    require_hex(expected_prior_result_sha256, 64, "expected prior result SHA256")
    if not isinstance(recipe, ProposalRecipe):
        raise ValueError("Expected ProposalRecipe")
    root = _new_result_directory(results_root, (data_root, model_root, prior_run_root))
    manifest = {
        "schema": "hybrid-drive-proposal-alignment-v1",
        "scope": SCOPE,
        "revision": revision,
        "recipe": asdict(recipe),
        "planned_frames": SPLIT_FRAMES.copy(),
        "expected_prior_result_sha256": expected_prior_result_sha256,
        "conditions": list(CONDITIONS),
        "roi_count_per_frame": 4,
        "holdout": "prior_used_reference_only_open_labels_after_all_weights_and_thresholds_frozen",
        "shuffled_control": "proposal_boxes_only_within_recording_global_rgb_and_features_unchanged",
        "full_stage1_claim": False,
        "driving_claim": False,
        "optimizer": {"name": "AdamW", "learning_rate": LEARNING_RATE, "weight_decay": 0},
        "torch_version": torch.__version__,
        "torch_threads": torch.get_num_threads(),
    }
    result = {"status": "running", "scope": SCOPE, "revision": revision, "runs": []}
    trace = []
    write_json(root / "manifest.json", manifest)
    write_json(root / "result.json", result)
    try:
        with torch.random.fork_rng(devices=[]), torch.device("cpu"):
            _run(root, data_root, model_root, prior_run_root, recipe, manifest, result, trace)
        result["status"] = "passed"
    except KeyboardInterrupt:
        result.update(status="interrupted", error="Interrupted by operator")
    except Exception as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    write_json(root / "manifest.json", manifest)
    result["manifest_hash"] = content_hash(manifest)
    result["optimizer_steps_completed"] = len(trace)
    write_json(root / "trace.json", trace)
    try:
        write_proposal_report(root, manifest, result)
    except Exception as exc:
        result["report_error"] = f"{type(exc).__name__}: {exc}"
        if result["status"] == "passed":
            result.update(status="failed", error="Aggregate report could not be saved")
    write_json(root / "result.json", result)
    return root, result


def _new_result_directory(output, inputs):
    base = Path(os.path.abspath(output))
    ancestors = (base, *base.parents)
    for path in inputs:
        source = Path(os.path.abspath(path))
        if source in ancestors or (
            source.exists()
            and any(candidate.exists() and candidate.samefile(source) for candidate in ancestors)
        ):
            raise ValueError("Outputs must be outside inputs")
    check_components(base)
    root = base / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    root.mkdir(parents=True, exist_ok=False)
    return root


def _run(root, data_root, model_root, prior_run_root, recipe, manifest, result, trace):
    datasets = {role: SpatialAlignmentDataset(data_root, role) for role in ("train", "dev")}
    verify_spatial_splits(datasets, SPLIT_FRAMES)
    dataset_hash = datasets["train"].dataset_hash
    vision, language, provenance = load_grounded_alignment_backbones(model_root)
    prior = load_spatial_prior_source(
        prior_run_root,
        expected_result_sha256=manifest["expected_prior_result_sha256"],
        dataset_hash=dataset_hash,
        vision_provenance=provenance["vision"],
    )
    frozen = _frozen_backbone_states(vision, language)
    encoded = {role: _encode(dataset, vision, prior, recipe.batch_size) for role, dataset in datasets.items()}
    manifest.update(dataset_hash=dataset_hash, backbones=provenance, prior_receipt=prior.receipt)
    write_json(root / "manifest.json", manifest)
    criterion = Reason2DriveLoss()
    training = encoded["train"]
    for seed in recipe.seeds:
        generator = torch.Generator().manual_seed(seed)
        orders = [torch.randperm(len(training), generator=generator) for _ in range(recipe.epochs)]
        order_hash = content_hash([order.tolist() for order in orders])
        comparison_hash = None
        for condition in CONDITIONS:
            torch.manual_seed(seed)
            model = GroundedLocProbe(language, visual_dim=vision.visual_dim)
            initial = module_state_hash(model)
            if comparison_hash is not None and comparison_hash != initial:
                raise ValueError("Paired arms must begin at identical weights")
            comparison_hash = initial
            print(f"Seed {seed}, {condition}: {recipe.epochs} epochs", flush=True)
            _train(
                root, model, language, criterion, training, condition, orders, recipe.batch_size, seed, trace
            )
            final = module_state_hash(model)
            if final == initial:
                raise ValueError("The proposal-alignment weights did not update")
            metrics, thresholds = _calibrate(
                _predict(model, language, encoded["dev"], condition, recipe.batch_size)
            )
            metadata = {
                "scope": SCOPE,
                "dataset_hash": dataset_hash,
                "revision": manifest["revision"],
                "backbone_hash": content_hash(provenance),
                "prior_receipt_hash": content_hash(prior.receipt),
                "condition": condition,
                "seed": seed,
                "train_order_hash": order_hash,
            }
            filename = f"{condition}-seed-{seed}.safetensors"
            checkpoint = save_neural_weights(root / filename, model.weight_modules(), metadata=metadata)
            result["runs"].append(
                {
                    "seed": seed,
                    "condition": condition,
                    "initial_weights_hash": initial,
                    "final_weights_hash": final,
                    "train_order_hash": order_hash,
                    "dev_thresholds": thresholds,
                    "dev": metrics,
                    "checkpoint": dict(checkpoint, file=filename),
                    "checkpoint_metadata": metadata,
                }
            )
            del model

    thresholds = _freeze_before_holdout(root, manifest, result, encoded["dev"].recordings)
    datasets["holdout"] = SpatialAlignmentDataset(data_root, "holdout")
    verify_spatial_splits(datasets, SPLIT_FRAMES)
    holdout = _encode(datasets["holdout"], vision, prior, recipe.batch_size)
    manifest["recordings"] = {role: sorted(set(dataset.recording_ids)) for role, dataset in datasets.items()}
    _evaluate_frozen(root, result, vision.visual_dim, language, holdout, recipe.batch_size, thresholds)
    result["frozen_vision_and_language_unchanged"] = _frozen_backbone_states(vision, language) == frozen
    if not result["frozen_vision_and_language_unchanged"]:
        raise ValueError("Frozen backbones changed")
    result["paired_holdout_deltas"] = _paired_deltas(result["runs"], recipe.seeds)


def _frozen_backbone_states(vision, language):
    for module in (vision, language):
        if module.training or any(p.requires_grad or p.grad is not None for p in module.parameters()):
            raise ValueError("Vision and language backbones must remain frozen")
    return {"vision": module_state_hash(vision), "language": module_state_hash(language)}


@torch.no_grad()
def _encode(dataset, vision, prior, batch_size):
    features, proposals, present, targets, seen, recordings = [], [], [], [], [], []
    for batch in DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_spatial_alignment
    ):
        patches = vision(batch.images)
        if (
            patches.ndim != 5
            or patches.shape[:3] != (len(batch.sample_ids), 1, vision.visual_dim)
            or not torch.isfinite(patches).all()
        ):
            raise ValueError("Expected finite patch grids with unchanged sample order")
        predicted = prior.predict(patches)
        if (
            predicted.shape != batch.boxes_2d.shape
            or predicted.requires_grad
            or not torch.isfinite(predicted).all()
        ):
            raise ValueError("Prior source must provide one finite box per ordered band without gradients")
        features.append(patches)
        proposals.append(predicted)
        present.append(batch.presence)
        targets.append(batch.boxes_2d)
        seen.extend(batch.sample_ids)
        recordings.extend(batch.recording_ids)
    if tuple(seen) != dataset.sample_ids or tuple(recordings) != dataset.recording_ids:
        raise ValueError("Data loader changed the prepared sample or recording order")
    return _Encoded(
        torch.cat(features), torch.cat(proposals), torch.cat(present), torch.cat(targets), tuple(recordings)
    )


def _train(root, model, language, criterion, train, condition, orders, batch_size, seed, trace):
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0)
    model.train()
    for epoch, order in enumerate(orders, start=1):
        for batch_index, indices in enumerate(order.split(batch_size)):
            features, rois, presence, targets = train.batch(indices, condition)
            optimizer.zero_grad(set_to_none=True)
            predicted = model(features, rois, language)
            loss = supervised_grounded_loc_loss(criterion, predicted, language, presence, targets)
            text_bridge = _ce_through_bridge(model, loss.text) if epoch == 1 and batch_index == 0 else None
            loss.total.backward()
            grads = {name: _gradient_norm(module) for name, module in model.weight_modules().items()}
            optimizer.step()
            if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
                raise ValueError("Nonfinite proposal-alignment weights after optimizer step")
            trace.append(
                {
                    "condition": condition,
                    "seed": seed,
                    "epoch": epoch,
                    "step": len(trace) + 1,
                    "total": float(loss.total.detach()),
                    "text": float(loss.text.detach()),
                    "presence": float(loss.classification.detach()),
                    "giou_2d": float(loss.giou_2d.detach()),
                    "events": loss.supervised_events,
                    "positive_boxes": loss.supervised_image_boxes,
                    "grad_norms": grads,
                    "ce_only_bridge_grad_norms": text_bridge,
                }
            )
        write_json(root / "trace.json", trace)
        print(f"Seed {seed}, {condition}, epoch {epoch}/{len(orders)}", flush=True)


def _ce_through_bridge(model, text_loss):
    modules = model.weight_modules()
    params = {name: tuple(modules[name].parameters()) for name in ("tokenizer", "projector")}
    return {
        name: _finite_nonzero_norm(
            torch.autograd.grad(text_loss, values, retain_graph=True, allow_unused=True)
        )
        for name, values in params.items()
    }


def _gradient_norm(module):
    return _finite_nonzero_norm(p.grad for p in module.parameters())


def _finite_nonzero_norm(values):
    grads = [g.detach().double().square().sum() for g in values if g is not None]
    norm = float(torch.stack(grads).sum().sqrt()) if grads else 0.0
    if norm <= 0 or not math.isfinite(norm):
        raise ValueError("Expected a finite nonzero gradient for every trainable group")
    return norm


@torch.no_grad()
def _predict(model, language, encoded, condition, batch_size, proposal_order=None):
    model.eval()
    decoder_scores, language_scores, boxes = [], [], []
    for indices in torch.arange(len(encoded)).split(batch_size):
        features, rois, _, _ = encoded.batch(indices, condition, proposal_order)
        validity, margin, decoded = grounded_presence_scores(model(features, rois, language), language)
        decoder_scores.append(validity)
        language_scores.append(margin)
        boxes.append(decoded)
    present, targets, boxes = encoded.presence.numpy(), encoded.target_boxes.numpy(), torch.cat(boxes).numpy()
    return {
        "decoder": SpatialPredictions(
            torch.cat(decoder_scores).numpy(), present, boxes, targets, encoded.recordings
        ),
        "text": SpatialPredictions(
            torch.cat(language_scores).numpy(), present, boxes, targets, encoded.recordings
        ),
    }


def _metrics(predictions, thresholds):
    output = {
        key: spatial_metrics(values, thresholds[key]["threshold"]) for key, values in predictions.items()
    }
    for metrics in (output["text"], *output["text"]["per_recording"].values()):
        for key in tuple(metrics):
            if "iou" in key or "l2_image_fraction" in key:
                del metrics[key]
    return output


def _calibrate(predictions):
    thresholds = {key: calibrate_spatial_threshold(value) for key, value in predictions.items()}
    return _metrics(predictions, thresholds), thresholds


def _freeze_before_holdout(root, manifest, result, recordings):
    freeze = {
        "schema": "hybrid-drive-proposal-alignment-calibration-v1",
        "dataset_hash": manifest["dataset_hash"],
        "proposal_receipt_hash": content_hash(manifest["prior_receipt"]),
        "dev_recordings": sorted(set(recordings)),
        "runs": [
            {
                "condition": row["condition"],
                "seed": row["seed"],
                "dev_thresholds": row["dev_thresholds"],
                "checkpoint_sha256": row["checkpoint"]["sha256"],
                "train_order_hash": row["train_order_hash"],
            }
            for row in result["runs"]
        ],
    }
    write_json(root / "calibration.json", freeze)
    result["holdout_gate_hash"] = content_hash(freeze)
    write_json(root / "result.json", result)
    persisted = decode_json(read_bounded(root / "calibration.json", 128 * 1024))
    if content_hash(persisted) != result["holdout_gate_hash"]:
        raise ValueError("The persisted dev calibration differs from the holdout gate")
    return {(row["seed"], row["condition"]): row["dev_thresholds"] for row in persisted["runs"]}


def _evaluate_frozen(root, result, visual_dim, language, holdout, batch_size, thresholds):
    order = torch.as_tensor(shuffled_by_recording(holdout.recordings), dtype=torch.long)
    result["mismatched_proposal_fraction"] = float(
        (holdout.proposals != holdout.proposals[order]).any(dim=-1).float().mean()
    )
    for row in result["runs"]:
        restored = GroundedLocProbe(language, visual_dim=visual_dim).eval()
        checkpoint = row["checkpoint"]
        load_neural_weights(
            root / checkpoint["file"],
            restored.weight_modules(),
            expected_sha256=checkpoint["sha256"],
            expected_metadata=row["checkpoint_metadata"],
        )
        if module_state_hash(restored) != row["final_weights_hash"]:
            raise ValueError("Reloaded alignment checkpoint differs from the frozen state")
        row["checkpoint_state_bitwise_equal"] = True
        condition = row["condition"]
        cutoff = thresholds[(row["seed"], condition)]
        predictions = _predict(restored, language, holdout, condition, batch_size)
        row["holdout"] = _metrics(predictions, cutoff)
        if condition == "spatial_proposals":
            wrong_boxes = _predict(restored, language, holdout, condition, batch_size, order)
            row["holdout_shuffled_rois_same_frame"] = _metrics(wrong_boxes, cutoff)
            row["mean_absolute_score_change_on_proposal_shuffle"] = {
                key: float(np.abs(values.scores - wrong_boxes[key].scores).mean())
                for key, values in predictions.items()
            }
        write_json(root / "result.json", result)


def _paired_deltas(rows, seeds):
    def compare(left, right):
        return {
            section: {
                key: (
                    left[section][key] - right[section][key]
                    if left[section][key] is not None and right[section][key] is not None
                    else None
                )
                for key in keys
            }
            for section, keys in (
                ("decoder", ("macro_balanced_accuracy", "macro_roc_auc", "macro_positive_iou")),
                ("text", ("macro_balanced_accuracy", "macro_roc_auc")),
            )
        }

    output = []
    for seed in seeds:
        fixed, proposals = [
            next(row for row in rows if row["seed"] == seed and row["condition"] == condition)
            for condition in CONDITIONS
        ]
        output.append(
            {
                "seed": seed,
                "proposals_minus_fixed": compare(proposals["holdout"], fixed["holdout"]),
                "proposals_minus_shuffled_rois": compare(
                    proposals["holdout"], proposals["holdout_shuffled_rois_same_frame"]
                ),
            }
        )
    return output
