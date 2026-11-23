"""Matched LOC alignment with fixed or predicted proposal priors, entirely offline."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import math
import os
from pathlib import Path

import torch

from hybrid_drive.assets.files import check_components
from hybrid_drive.neural.grounded_loc_probe import (
    GroundedLocProbe,
    supervised_grounded_loc_loss,
)
from hybrid_drive.neural.losses import Reason2DriveLoss
from hybrid_drive.neural.pretrained_alignment import load_grounded_alignment_backbones
from hybrid_drive.neural.spatial_data import (
    SpatialAlignmentDataset,
    verify_spatial_splits,
)
from .bounded_io import decode_json, read_bounded, require_hex
from .fresh_spatial_holdout import read_fresh_spatial_holdout
from .io import write_json
from .localization_gallery import write_localization_gallery
from .neural_checkpoint import module_state_hash, save_neural_weights
from .proposal_alignment_report import write_proposal_report
from .proposal_evaluation import (
    CONDITIONS,
    calibrate,
    encode,
    evaluate_direct_spatial,
    evaluate_frozen,
    paired_deltas,
    predict,
)
from .proposal_prior_source import load_spatial_prior_source
from .protocol import content_hash
from .spatial_experiment import SPLIT_FRAMES


SCOPE = "openscene_four_band_proposal_alignment_ablation_v1"
FRESH_SCOPE = "openscene_four_band_roi_memory_fresh_v1"
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


def run_proposal_alignment(
    data_root,
    model_root,
    prior_run_root,
    results_root,
    *,
    revision,
    expected_prior_result_sha256,
    recipe=ProposalRecipe(),
    fresh_holdout_root=None,
):
    require_hex(revision, 40, "revision")
    require_hex(expected_prior_result_sha256, 64, "expected prior result SHA256")
    if not isinstance(recipe, ProposalRecipe):
        raise ValueError("Expected ProposalRecipe")
    inputs = (data_root, model_root, prior_run_root)
    fresh = fresh_holdout_root is not None
    if fresh:
        inputs += (fresh_holdout_root,)
    root = _new_result_directory(results_root, inputs)
    scope = FRESH_SCOPE if fresh else SCOPE
    manifest = {
        "schema": "hybrid-drive-proposal-alignment-v1",
        "scope": scope,
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
    if fresh:
        manifest.update(
            memory_mode="roi_tokens",
            holdout="pinned_fresh_recordings_open_labels_after_all_weights_and_dev_thresholds_frozen",
            gallery_seed=recipe.seeds[0],
        )
    result = {"status": "running", "scope": scope, "revision": revision, "runs": []}
    trace = []
    write_json(root / "manifest.json", manifest)
    write_json(root / "result.json", result)
    try:
        with torch.random.fork_rng(devices=[]), torch.device("cpu"):
            _run(root, data_root, model_root, prior_run_root, fresh_holdout_root, recipe, manifest, result, trace)
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


def _run(root, data_root, model_root, prior_run_root, fresh_root, recipe, manifest, result, trace):
    fresh = read_fresh_spatial_holdout(data_root, fresh_root) if fresh_root is not None else None
    memory_mode = manifest.get("memory_mode", "patches_and_priors")
    if fresh is not None:
        manifest["fresh_holdout_manifest_hash"] = fresh["manifest_hash"]
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
    encoded = {role: encode(dataset, vision, prior, recipe.batch_size) for role, dataset in datasets.items()}
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
            model = GroundedLocProbe(language, visual_dim=vision.visual_dim, memory_mode=memory_mode)
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
            metrics, thresholds = calibrate(
                predict(model, language, encoded["dev"], condition, recipe.batch_size)
            )
            metadata = {
                "scope": manifest["scope"],
                "dataset_hash": dataset_hash,
                "revision": manifest["revision"],
                "backbone_hash": content_hash(provenance),
                "prior_receipt_hash": content_hash(prior.receipt),
                "condition": condition,
                "seed": seed,
                "train_order_hash": order_hash,
            }
            if fresh is not None:
                metadata.update(memory_mode=memory_mode, fresh_holdout_manifest_hash=fresh["manifest_hash"])
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
    if fresh is None:
        datasets["holdout"] = SpatialAlignmentDataset(data_root, "holdout")
        verify_spatial_splits(datasets, SPLIT_FRAMES)
    else:
        datasets["holdout"] = SpatialAlignmentDataset(
            fresh_root, "holdout", expected_manifest_hash=fresh["manifest_hash"]
        )
    holdout = encode(datasets["holdout"], vision, prior, recipe.batch_size, capture_direct=fresh is not None)
    manifest["recordings"] = {role: sorted(set(dataset.recording_ids)) for role, dataset in datasets.items()}
    gallery = evaluate_frozen(
        root, result, vision.visual_dim, language, holdout, recipe.batch_size, thresholds,
        memory_mode=memory_mode, gallery_seed=manifest.get("gallery_seed"),
    )
    if fresh is not None:
        cutoff = prior.receipt["calibration"]["threshold"]
        direct, evidence = evaluate_direct_spatial(holdout, cutoff)
        result["direct_spatial"] = direct
        target = root / "gallery.html"
        write_localization_gallery(
            target, fresh_root, fresh, {
                "Прямая spatial-голова": evidence,
                "Четыре полные четверти": gallery["fixed_quarters"],
                "Четыре предсказанные RoI": gallery["spatial_proposals"],
            },
        )
        result["gallery"] = target.name
    result["frozen_vision_and_language_unchanged"] = _frozen_backbone_states(vision, language) == frozen
    if not result["frozen_vision_and_language_unchanged"]:
        raise ValueError("Frozen backbones changed")
    result["paired_holdout_deltas"] = paired_deltas(result["runs"], recipe.seeds)


def _frozen_backbone_states(vision, language):
    for module in (vision, language):
        if module.training or any(p.requires_grad or p.grad is not None for p in module.parameters()):
            raise ValueError("Vision and language backbones must remain frozen")
    return {"vision": module_state_hash(vision), "language": module_state_hash(language)}


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
    if "fresh_holdout_manifest_hash" in manifest:
        freeze["fresh_holdout_manifest_hash"] = manifest["fresh_holdout_manifest_hash"]
        freeze["memory_mode"] = manifest["memory_mode"]
    write_json(root / "calibration.json", freeze)
    result["holdout_gate_hash"] = content_hash(freeze)
    write_json(root / "result.json", result)
    persisted = decode_json(read_bounded(root / "calibration.json", 128 * 1024))
    if content_hash(persisted) != result["holdout_gate_hash"]:
        raise ValueError("The persisted dev calibration differs from the holdout gate")
    return {(row["seed"], row["condition"]): row["dev_thresholds"] for row in persisted["runs"]}
