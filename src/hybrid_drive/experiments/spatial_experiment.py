"""Train a spatial probe and a matched zero-feature control before opening final labels."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import time

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from hybrid_drive.assets.files import check_components
from hybrid_drive.neural.pretrained_alignment import load_alignment_vision
from hybrid_drive.neural.spatial_data import (
    SpatialAlignmentDataset,
    collate_spatial_alignment,
    verify_spatial_splits,
)
from hybrid_drive.neural.spatial_loc import SpatialLocLoss, SpatialLocProbe
from .bounded_io import require_hex
from .io import write_json
from .neural_checkpoint import load_neural_weights, module_state_hash, save_neural_weights
from .protocol import content_hash
from .spatial_metrics import (
    SpatialPredictions,
    calibrate_spatial_threshold,
    shuffled_by_recording,
    spatial_metrics,
)
from .spatial_report import write_spatial_report


SCOPE = "openscene_spatial_loc_four_band_v2"
SPLIT_FRAMES = {"train": 96, "dev": 48, "holdout": 64}
CONDITIONS = ("image_features", "constant_zero_features")


@dataclass(frozen=True)
class SpatialRecipe:
    epochs: int = 8
    batch_size: int = 8
    seeds: tuple[int, ...] = (0, 1)

    def __post_init__(self):
        if type(self.epochs) is not int or not 1 <= self.epochs <= 16:
            raise ValueError("epochs must be an integer from 1 to 16")
        if type(self.batch_size) is not int or not 1 <= self.batch_size <= 16:
            raise ValueError("batch size must be an integer from 1 to 16")
        seeds = tuple(self.seeds)
        if (
            not 1 <= len(seeds) <= 4
            or any(type(s) is not int or not 0 <= s < 2**32 for s in seeds)
            or len(set(seeds)) != len(seeds)
        ):
            raise ValueError("Expected 1..4 distinct bounded seeds")
        object.__setattr__(self, "seeds", seeds)


@dataclass(frozen=True)
class _Encoded:
    features: Tensor
    present: Tensor
    target_boxes: Tensor
    recording_ids: tuple[str, ...]
    feature_wall_s: float

    def __len__(self):
        return len(self.recording_ids)

    def batch(self, idx, condition, feature_order=None):
        source = idx if feature_order is None else feature_order[idx]
        selected = self.features[source]
        if condition == "constant_zero_features":
            selected = torch.zeros_like(selected)
        return selected, self.present[idx], self.target_boxes[idx]


def run_spatial_experiment(data_root, model_root, results_root, *, revision, recipe=SpatialRecipe()):
    require_hex(revision, 40, "revision")
    if not isinstance(recipe, SpatialRecipe):
        raise ValueError("Expected SpatialRecipe")
    results_root = Path(os.path.abspath(results_root))
    for source in (Path(os.path.abspath(data_root)), Path(os.path.abspath(model_root))):
        ancestors = (results_root, *results_root.parents)
        if source in ancestors or (
            source.exists()
            and any(candidate.exists() and candidate.samefile(source) for candidate in ancestors)
        ):
            raise ValueError("Outputs must be outside inputs")
    check_components(results_root)
    root = results_root / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    root.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema": "hybrid-drive-spatial-experiment-v1",
        "scope": SCOPE,
        "revision": revision,
        "recipe": asdict(recipe),
        "planned_frames": SPLIT_FRAMES.copy(),
        "optimizer": {"name": "AdamW", "learning_rate": 0.002, "weight_decay": 0},
        "conditions": list(CONDITIONS),
        "training_input": "current_frame_features_only",
        "calibration": "dev_classwise_macro_balanced_accuracy_by_recording",
        "holdout": "open_only_after_all_weights_and_dev_thresholds_frozen",
        "shuffled_diagnostic": "rotate_features_by_half_recording_never_cross_recordings",
        "llm_used": False,
        "safety_or_driving_claim": False,
        "full_stage1_claim": False,
        "torch_version": torch.__version__,
        "torch_threads": torch.get_num_threads(),
    }
    result = {"status": "running", "scope": SCOPE, "revision": revision, "runs": []}
    trace = []
    write_json(root / "manifest.json", manifest)
    write_json(root / "result.json", result)
    try:
        with torch.random.fork_rng(devices=[]), torch.device("cpu"):
            _run(root, data_root, model_root, recipe, manifest, result, trace)
        result["status"] = "passed"
    except KeyboardInterrupt:
        result.update(status="interrupted", error="Interrupted by operator")
    except Exception as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    write_json(root / "manifest.json", manifest)
    result["manifest_hash"] = content_hash(manifest)
    result["optimizer_steps_completed"] = len(trace)
    write_json(root / "trace.json", trace)
    write_json(root / "result.json", result)
    write_spatial_report(root, manifest, result)
    return root, result


def _run(root, data_root, model_root, recipe, manifest, result, trace):
    datasets = {role: SpatialAlignmentDataset(data_root, role) for role in ("train", "dev")}
    verify_spatial_splits(datasets, SPLIT_FRAMES)
    manifest["dataset_hash"] = datasets["train"].dataset_hash
    vision, provenance = load_alignment_vision(model_root)
    if vision.training or any(p.requires_grad or p.grad is not None for p in vision.parameters()):
        raise ValueError("Vision must be frozen before spatial training")
    frozen = module_state_hash(vision)
    manifest["vision"] = provenance
    encoded = {role: _encode(dataset, vision, recipe.batch_size) for role, dataset in datasets.items()}

    training = encoded["train"]
    positives = int(training.present.sum())
    negatives = training.present.numel() - positives
    if not positives or not negatives:
        raise ValueError("Both classes are needed to weight training")
    weight = negatives / positives
    manifest["train_presence"] = {"positive": positives, "negative": negatives, "positive_weight": weight}
    write_json(root / "manifest.json", manifest)
    criterion = SpatialLocLoss(weight)
    prepared = []

    for seed in recipe.seeds:
        generator = torch.Generator().manual_seed(seed)
        orders = [torch.randperm(len(training), generator=generator) for _ in range(recipe.epochs)]
        order_hash = content_hash([order.tolist() for order in orders])
        initial_hash = None
        for condition in CONDITIONS:
            torch.manual_seed(seed)
            probe = SpatialLocProbe(vision.visual_dim)
            initial = module_state_hash(probe)
            if initial_hash is not None and initial != initial_hash:
                raise ValueError("Matched arms must initialize identically")
            initial_hash = initial
            print(f"Seed {seed}, {condition}: {recipe.epochs} epochs", flush=True)
            _train(probe, criterion, training, condition, orders, seed, recipe.batch_size, trace, root)
            final_hash = module_state_hash(probe)
            if final_hash == initial:
                raise ValueError("Spatial model weights did not update")
            dev_predictions = _predict(probe, encoded["dev"], condition, recipe.batch_size)
            calibration = calibrate_spatial_threshold(dev_predictions)
            metadata = {
                "scope": SCOPE,
                "dataset_hash": manifest["dataset_hash"],
                "revision": manifest["revision"],
                "vision_hash": content_hash(provenance),
                "seed": seed,
                "condition": condition,
                "recipe": manifest["recipe"],
                "train_order_hash": order_hash,
            }
            filename = f"{condition}-seed-{seed}.safetensors"
            checkpoint = save_neural_weights(root / filename, {"spatial": probe}, metadata=metadata)
            run = {
                "seed": seed,
                "condition": condition,
                "initial_weights_hash": initial,
                "final_weights_hash": final_hash,
                "train_order_hash": order_hash,
                "dev_threshold": calibration,
                "dev": spatial_metrics(dev_predictions, calibration["threshold"]),
                "checkpoint": dict(checkpoint, file=filename),
                "checkpoint_metadata": metadata,
            }
            prepared.append(run)
            del probe

    # The manifest reader binds all role paths, but train/dev readers never open
    # final targets. The test reader is not even constructed until this commit point.
    freeze = {
        "schema": "hybrid-drive-spatial-calibration-v1",
        "dataset_hash": manifest["dataset_hash"],
        "calibration_recordings": sorted(set(encoded["dev"].recording_ids)),
        "runs": [
            {
                "seed": run["seed"],
                "condition": run["condition"],
                "threshold": run["dev_threshold"]["threshold"],
                "checkpoint_sha256": run["checkpoint"]["sha256"],
                "train_order_hash": run["train_order_hash"],
            }
            for run in prepared
        ],
    }
    write_json(root / "calibration.json", freeze)
    result["calibration_hash"] = content_hash(freeze)
    result["runs"] = prepared
    write_json(root / "result.json", result)

    datasets["holdout"] = SpatialAlignmentDataset(data_root, "holdout")
    verify_spatial_splits(datasets, SPLIT_FRAMES)
    held = _encode(datasets["holdout"], vision, recipe.batch_size)
    encoded["holdout"] = held
    manifest["split_recordings"] = {role: sorted(set(ds.recording_ids)) for role, ds in datasets.items()}
    feature_order = torch.as_tensor(shuffled_by_recording(held.recording_ids), dtype=torch.long)
    for run in prepared:
        restored = SpatialLocProbe(vision.visual_dim).eval()
        artifact = run["checkpoint"]
        load_neural_weights(
            root / artifact["file"],
            {"spatial": restored},
            expected_sha256=artifact["sha256"],
            expected_metadata=run["checkpoint_metadata"],
        )
        if module_state_hash(restored) != run["final_weights_hash"]:
            raise ValueError("Loaded spatial checkpoint differs from the frozen trained state")
        run["checkpoint_state_bitwise_equal"] = True
        condition, threshold = run["condition"], run["dev_threshold"]["threshold"]
        run["holdout"] = spatial_metrics(_predict(restored, held, condition, recipe.batch_size), threshold)
        if condition == "image_features":
            run["holdout_shuffled_same_recording"] = spatial_metrics(
                _predict(restored, held, condition, recipe.batch_size, feature_order), threshold
            )
        write_json(root / "result.json", result)

    result["paired_holdout_deltas"] = _paired_deltas(prepared, recipe.seeds)
    result["frozen_vision_unchanged"] = module_state_hash(vision) == frozen
    if not result["frozen_vision_unchanged"] or any(p.grad is not None for p in vision.parameters()):
        raise ValueError("Frozen vision state or gradients changed")
    result["feature_encode_wall_s"] = {role: value.feature_wall_s for role, value in encoded.items()}


def _paired_deltas(runs, seeds):
    def differences(left, right):
        return {
            key: left[key] - right[key] if left[key] is not None and right[key] is not None else None
            for key in ("macro_balanced_accuracy", "macro_roc_auc", "macro_positive_iou")
        }

    output = []
    for seed in seeds:
        image, blind = [
            next(run for run in runs if run["seed"] == seed and run["condition"] == condition)
            for condition in CONDITIONS
        ]
        output.append(
            {
                "seed": seed,
                "image_minus_zero": differences(image["holdout"], blind["holdout"]),
                "image_minus_shuffled": differences(
                    image["holdout"], image["holdout_shuffled_same_recording"]
                ),
            }
        )
    return output


@torch.no_grad()
def _encode(dataset, vision, batch_size):
    inputs, labels, boxes, ids, recordings = [], [], [], [], []
    start = time.monotonic()
    for batch in DataLoader(
        dataset, batch_size=batch_size, num_workers=0, shuffle=False, collate_fn=collate_spatial_alignment
    ):
        features = vision(batch.images)
        if (
            features.ndim != 5
            or features.shape[:3] != (len(batch.sample_ids), 1, vision.visual_dim)
            or not torch.isfinite(features).all()
        ):
            raise ValueError("Frozen vision did not return ordered, finite patch grids")
        inputs.append(features[:, 0])
        labels.append(batch.presence)
        boxes.append(batch.boxes_2d)
        ids.extend(batch.sample_ids)
        recordings.extend(batch.recording_ids)
    if tuple(ids) != dataset.sample_ids or tuple(recordings) != dataset.recording_ids:
        raise ValueError("DataLoader did not preserve recording/sample correspondence")
    return _Encoded(
        torch.cat(inputs), torch.cat(labels), torch.cat(boxes), tuple(recordings), time.monotonic() - start
    )


def _train(probe, criterion, encoded, condition, orders, seed, batch_size, trace, root):
    optimiser = torch.optim.AdamW(probe.parameters(), lr=0.002, weight_decay=0)
    probe.train()
    for epoch, order in enumerate(orders, start=1):
        for indices in order.split(batch_size):
            data, label, target = encoded.batch(indices, condition)
            optimiser.zero_grad(set_to_none=True)
            loss = criterion(probe(data), label, target)
            loss.total.backward()
            norms = [
                p.grad.detach().double().square().sum() for p in probe.parameters() if p.grad is not None
            ]
            if not norms:
                raise ValueError("The spatial model did not receive gradients")
            norm = float(torch.stack(norms).sum().sqrt())
            if not math.isfinite(norm) or norm <= 0:
                raise ValueError("Expected a finite nonzero spatial gradient")
            optimiser.step()
            if any(not torch.isfinite(p).all() for p in probe.parameters()):
                raise ValueError("Optimizer produced nonfinite spatial state")
            trace.append(
                {
                    "seed": seed,
                    "condition": condition,
                    "epoch": epoch,
                    "optimizer_step": len(trace) + 1,
                    "presence_count": loss.supervised_presence_count,
                    "positive_count": loss.positive_count,
                    "loss": float(loss.total.detach()),
                    "presence_loss": float(loss.presence.detach()),
                    "cell_loss": float(loss.cell.detach()),
                    "box_loss": float(loss.box.detach()),
                    "giou": float(loss.giou.detach()),
                    "grad_norm": norm,
                }
            )
        write_json(root / "trace.json", trace)


@torch.no_grad()
def _predict(probe, encoded, condition, batch_size, feature_order=None):
    probe.eval()
    scores, boxes = [], []
    for idx in torch.arange(len(encoded)).split(batch_size):
        features, _, _ = encoded.batch(idx, condition, feature_order)
        prediction = probe(features)
        scores.append(prediction.presence_logits)
        boxes.append(prediction.boxes_xyxy)
    return SpatialPredictions(
        torch.cat(scores).numpy(),
        encoded.present.numpy(),
        torch.cat(boxes).numpy(),
        encoded.target_boxes.numpy(),
        encoded.recording_ids,
    )
