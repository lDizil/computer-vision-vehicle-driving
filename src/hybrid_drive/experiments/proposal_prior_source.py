"""Admit the frozen spatial seed-0 image arm as a source of unconditional boxes."""

from copy import deepcopy
from dataclasses import dataclass
import hashlib
from pathlib import Path

import torch
from torch import Tensor

from hybrid_drive.assets.files import check_components
from hybrid_drive.contracts._validation import finite_real
from hybrid_drive.neural.spatial_loc import SpatialLocProbe
from .bounded_io import decode_json, read_bounded, require_hex, require_keys
from .neural_checkpoint import SCHEMA as NEURAL_WEIGHTS_SCHEMA, load_neural_weights, module_state_hash
from .protocol import content_hash


_SCOPE = "openscene_spatial_loc_four_band_v2"
_MANIFEST_SCHEMA = "hybrid-drive-spatial-experiment-v1"
_CALIBRATION_SCHEMA = "hybrid-drive-spatial-calibration-v1"
_TRAINING_INPUT = "current_frame_features_only"
_CALIBRATION_METHOD = "dev_classwise_macro_balanced_accuracy_by_recording"
_HOLDOUT_POLICY = "open_only_after_all_weights_and_dev_thresholds_frozen"
_SEED = 0
_CONDITION = "image_features"
_CHECKPOINT_FILE = f"{_CONDITION}-seed-{_SEED}.safetensors"
_VISUAL_DIM = 384
_HIDDEN_DIM = 64
_BANDS = 4
_MAX_JSON_BYTES = 256 * 1024
_MAX_CHECKPOINT_BYTES = 1024 * 1024


@dataclass(frozen=True)
class FrozenSpatialPriorSource:
    _probe: SpatialLocProbe
    _receipt: dict

    @property
    def receipt(self) -> dict:
        return deepcopy(self._receipt)

    @torch.no_grad()
    def predict(self, feature_maps: Tensor) -> Tensor:
        """Return [B,4,4] normalized xyxy boxes in batch and left-to-right band order."""
        if (
            not isinstance(feature_maps, Tensor)
            or feature_maps.ndim != 5
            or feature_maps.shape[0] == 0
            or feature_maps.shape[1:3] != (1, _VISUAL_DIM)
            or feature_maps.dtype != torch.float32
            or feature_maps.device.type != "cpu"
        ):
            raise ValueError("Expected CPU float32 feature maps [B,1,384,H,W]")

        boxes = self._probe(feature_maps[:, 0]).boxes_xyxy
        if (
            boxes.shape != (len(feature_maps), _BANDS, 4)
            or boxes.dtype != feature_maps.dtype
            or boxes.device != feature_maps.device
            or not torch.isfinite(boxes).all()
            or ((boxes < 0) | (boxes > 1)).any()
            or (boxes[..., 2:] <= boxes[..., :2]).any()
        ):
            raise ValueError("Prior must return four finite positive-area normalized xyxy boxes per frame")
        centers_x = (boxes[..., 0] + boxes[..., 2]) * 0.5
        lower = torch.arange(_BANDS, device=boxes.device, dtype=boxes.dtype) / _BANDS
        if ((centers_x < lower) | (centers_x >= lower + 1 / _BANDS)).any():
            raise ValueError("Prior box centers must preserve left-to-right quarter order")
        return boxes


def load_spatial_prior_source(
    run_root: str | Path, *, expected_result_sha256: str, dataset_hash: str, vision_provenance: dict
) -> FrozenSpatialPriorSource:
    """Verify the local artifact chain against the caller-supplied result byte SHA-256."""
    require_hex(expected_result_sha256, 64, "expected_result_sha256")
    require_hex(dataset_hash, 64, "dataset_hash")
    root = Path(run_root).absolute()
    check_components(root)
    if not root.is_dir():
        raise ValueError("Spatial prior run root must be a local directory")

    result_bytes = read_bounded(root / "result.json", _MAX_JSON_BYTES)
    if hashlib.sha256(result_bytes).hexdigest() != expected_result_sha256:
        raise ValueError("Spatial prior result differs from expected SHA256")
    result = _json_object(result_bytes, "result")
    if result.get("status") != "passed" or result.get("frozen_vision_unchanged") is not True:
        raise ValueError("Spatial prior result must record a passed run with unchanged frozen vision")
    manifest = _bound_document(root / "manifest.json", result.get("manifest_hash"))
    calibration = _bound_document(root / "calibration.json", result.get("calibration_hash"))
    vision_hash = _validate_manifest(manifest, result, dataset_hash, vision_provenance)
    run = _pinned_run(result, "result")
    checkpoint, metadata = _validate_checkpoint_record(run, manifest, vision_hash)
    threshold = _validate_calibration(calibration, manifest, run, checkpoint)

    path = root / _CHECKPOINT_FILE
    check_components(path)
    if not path.is_file() or path.stat().st_size != checkpoint["size_bytes"]:
        raise ValueError("Spatial prior checkpoint file or size differs")
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        probe = SpatialLocProbe(_VISUAL_DIM, hidden_dim=_HIDDEN_DIM).float()
    load_neural_weights(
        path,
        {"spatial": probe},
        expected_sha256=checkpoint["sha256"],
        expected_metadata=metadata,
    )
    if module_state_hash(probe) != run["final_weights_hash"]:
        raise ValueError("Spatial prior checkpoint differs from the recorded trained state")
    probe.requires_grad_(False).eval()

    return FrozenSpatialPriorSource(
        probe,
        {
            "schema": "hybrid-drive-frozen-spatial-prior-source-v1",
            "scope": _SCOPE,
            "source_revision": manifest["revision"],
            "dataset_hash": dataset_hash,
            "vision_hash": vision_hash,
            "torch_version": manifest["torch_version"],
            "recipe": manifest["recipe"],
            "training_input": _TRAINING_INPUT,
            "seed": _SEED,
            "condition": _CONDITION,
            "architecture": {
                "name": "SpatialLocProbe",
                "visual_dim": _VISUAL_DIM,
                "hidden_dim": _HIDDEN_DIM,
                "bands": _BANDS,
            },
            "result_sha256": expected_result_sha256,
            "manifest_hash": result["manifest_hash"],
            "calibration_hash": result["calibration_hash"],
            "calibration": {
                "method": _CALIBRATION_METHOD,
                "holdout_policy": _HOLDOUT_POLICY,
                "recordings": calibration["calibration_recordings"],
                "threshold": threshold,
                "applied_to_proposals": False,
            },
            "checkpoint": checkpoint,
            "train_order_hash": run["train_order_hash"],
            "final_weights_hash": run["final_weights_hash"],
        },
    )


def _json_object(data, name):
    value = decode_json(data)
    if not isinstance(value, dict):
        raise ValueError(f"Spatial prior {name} must be a JSON object")
    return value


def _bound_document(path, expected_hash):
    name = path.stem
    require_hex(expected_hash, 64, name + "_hash")
    document = _json_object(read_bounded(path, _MAX_JSON_BYTES), name)
    if content_hash(document) != expected_hash:
        raise ValueError(f"Spatial prior {name} differs from the recorded content hash")
    return document


def _validate_manifest(manifest, result, dataset_hash, vision_provenance):
    if (
        manifest.get("schema") != _MANIFEST_SCHEMA
        or manifest.get("scope") != _SCOPE
        or result.get("scope") != _SCOPE
    ):
        raise ValueError("Unsupported spatial prior manifest schema or scope")
    revision = manifest.get("revision")
    require_hex(revision, 40, "prior source revision")
    if result.get("revision") != revision:
        raise ValueError("Spatial prior result/manifest source revision differs")
    if manifest.get("dataset_hash") != dataset_hash:
        raise ValueError("Spatial prior dataset hash differs")
    if (
        manifest.get("training_input") != _TRAINING_INPUT
        or manifest.get("calibration") != _CALIBRATION_METHOD
        or manifest.get("holdout") != _HOLDOUT_POLICY
    ):
        raise ValueError(
            "Spatial prior must record current-frame training and dev calibration frozen before holdout"
        )

    vision = manifest.get("vision")
    if not isinstance(vision, dict) or not isinstance(vision_provenance, dict):
        raise ValueError("Expected spatial prior and active vision provenance objects")
    vision_hash = content_hash(vision_provenance)
    if content_hash(vision) != vision_hash:
        raise ValueError("Spatial prior vision provenance differs")
    runtime = vision.get("runtime")
    if (
        not isinstance(runtime, dict)
        or runtime.get("torch") != manifest.get("torch_version")
        or runtime.get("torch") != str(torch.__version__)
        or runtime.get("device") != "cpu"
        or runtime.get("dtype") != "float32"
        or runtime.get("attention") != "eager"
    ):
        raise ValueError("Spatial prior manifest and active vision runtime differ")

    recipe = manifest.get("recipe")
    require_keys(recipe, {"epochs", "batch_size", "seeds"}, "spatial prior recipe")
    seeds = recipe["seeds"]
    if (
        any(type(recipe[key]) is not int or not 1 <= recipe[key] <= 16 for key in ("epochs", "batch_size"))
        or not isinstance(seeds, list)
        or not 1 <= len(seeds) <= 4
        or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds)
        or len(set(seeds)) != len(seeds)
        or _SEED not in seeds
        or manifest.get("conditions") != [_CONDITION, "constant_zero_features"]
    ):
        raise ValueError("Invalid spatial prior recipe for the pinned image arm and seed")
    return vision_hash


def _pinned_run(document, name):
    runs = document.get("runs")
    if not isinstance(runs, list) or any(not isinstance(run, dict) for run in runs):
        raise ValueError(f"Spatial prior {name} must contain run records")
    pinned = [
        run
        for run in runs
        if type(run.get("seed")) is int and run["seed"] == _SEED and run.get("condition") == _CONDITION
    ]
    if len(pinned) != 1:
        raise ValueError(f"Spatial prior {name} must contain one {_CONDITION} seed {_SEED} record")
    return pinned[0]


def _validate_checkpoint_record(run, manifest, vision_hash):
    for key in ("train_order_hash", "initial_weights_hash", "final_weights_hash"):
        require_hex(run.get(key), 64, key)
    if (
        run.get("checkpoint_state_bitwise_equal") is not True
        or run["initial_weights_hash"] == run["final_weights_hash"]
    ):
        raise ValueError("Spatial prior result must record a round-trip of the updated checkpoint state")
    artifact = run.get("checkpoint")
    require_keys(
        artifact, {"file", "schema", "sha256", "size_bytes", "metadata_sha256"}, "spatial prior checkpoint"
    )
    if artifact["file"] != _CHECKPOINT_FILE:
        raise ValueError("Spatial prior checkpoint path differs from the pinned filename")
    if (
        artifact["schema"] != NEURAL_WEIGHTS_SCHEMA
        or type(artifact["size_bytes"]) is not int
        or not 0 < artifact["size_bytes"] <= _MAX_CHECKPOINT_BYTES
    ):
        raise ValueError("Invalid spatial prior checkpoint schema or size")
    require_hex(artifact["sha256"], 64, "checkpoint sha256")
    metadata = {
        "scope": _SCOPE,
        "dataset_hash": manifest["dataset_hash"],
        "revision": manifest["revision"],
        "vision_hash": vision_hash,
        "seed": _SEED,
        "condition": _CONDITION,
        "recipe": manifest["recipe"],
        "train_order_hash": run["train_order_hash"],
    }
    metadata_hash = content_hash(metadata)
    if (
        content_hash(run.get("checkpoint_metadata")) != metadata_hash
        or artifact["metadata_sha256"] != metadata_hash
    ):
        raise ValueError("Spatial prior checkpoint metadata differs from the validated manifest/run")
    return artifact, metadata


def _validate_calibration(calibration, manifest, run, checkpoint):
    if (
        calibration.get("schema") != _CALIBRATION_SCHEMA
        or calibration.get("dataset_hash") != manifest["dataset_hash"]
    ):
        raise ValueError("Spatial prior calibration schema or dataset differs")
    recordings = manifest.get("split_recordings")
    require_keys(recordings, {"train", "dev", "holdout"}, "spatial prior split recordings")
    seen = set()
    for names in recordings.values():
        if (
            not isinstance(names, list)
            or not names
            or any(not isinstance(name, str) or not name for name in names)
            or len(set(names)) != len(names)
            or seen.intersection(names)
        ):
            raise ValueError("Spatial prior split recordings must be distinct")
        seen.update(names)
    if calibration.get("calibration_recordings") != recordings["dev"]:
        raise ValueError("Spatial prior calibration must bind to its dev recordings")

    freeze = _pinned_run(calibration, "calibration")
    calibrated = run.get("dev_threshold")
    if not isinstance(calibrated, dict) or calibrated.get("weighting") != "classwise_macro_by_recording":
        raise ValueError("Spatial prior result must record the expected dev threshold weighting")
    threshold = finite_real(freeze.get("threshold"), "spatial prior calibration threshold")
    if (
        threshold != finite_real(calibrated.get("threshold"), "spatial prior result threshold")
        or freeze.get("checkpoint_sha256") != checkpoint["sha256"]
        or freeze.get("train_order_hash") != run["train_order_hash"]
    ):
        raise ValueError("Spatial prior calibration differs from the pinned result run")
    return threshold
