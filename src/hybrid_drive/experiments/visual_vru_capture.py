"""Capture current six-camera DINOv2 maps without exposing any labels."""

from hashlib import sha256

from safetensors.torch import load, save_file
import torch

from hybrid_drive.experiments.bounded_io import decode_json, read_bounded, require_hex
from hybrid_drive.experiments.box_motion_inputs import load_motion_snapshot
from hybrid_drive.experiments.fast_capture import artifact_hash
from hybrid_drive.experiments.io import write_json
from hybrid_drive.experiments.visual_vru_data import load_vru_inputs
from hybrid_drive.experiments.visual_vru_geometry import GRID_COLS, GRID_ROWS, GRID_STEP_M, GRID_X_MIN, GRID_Y_MIN
from hybrid_drive.experiments.visual_vru_model import GRID_SHAPE, VISUAL_DIM
from hybrid_drive.fast.metric_bev import BevGrid, lift_features
from hybrid_drive.neural.pretrained_alignment import load_alignment_vision
from hybrid_drive.neural.surround_features import SharedSurroundEncoder


CAPTURE_SCHEMA = "hybrid-drive-visual-vru-feature-cache-v1"
GRID = BevGrid(GRID_X_MIN, GRID_Y_MIN, GRID_STEP_M, GRID_ROWS, GRID_COLS, (.40, .9, 1.4))


@torch.no_grad()
def capture_vru(input_root, model_root, output, *, revision):
    require_hex(revision, 40, "capture revision")
    require_new_output(output)
    index = load_vru_inputs(input_root)
    vision, source = load_alignment_vision(model_root)
    identity = source["source"]
    encoder = SharedSurroundEncoder(vision, encoder_id=f'{identity["repository"]}@{identity["revision"]}')
    maps, coverage, captured = [], [], []
    for anchor in index["anchors"]:
        now = anchor["sample_token"]
        snapshot = load_motion_snapshot(input_root, now, index["frames"][now])
        feature = encoder.encode(snapshot)
        lifted, support = lift_features(feature, snapshot, GRID)
        maps.append(lifted.to(torch.float16))
        coverage.append(support[0].clone())
        captured.append({"sample_token": now, "recording": anchor["recording"], "snapshot_sha256": snapshot.content_sha256})
        if len(maps) % 32 == 0:
            print(f"Captured {index['split']}: {len(maps)}/{len(index['anchors'])}", flush=True)
    output.mkdir(parents=True)
    features_path = output / "camera-bev.safetensors"
    save_file({"appearance": torch.stack(maps), "coverage": torch.stack(coverage)}, features_path)
    metadata = {"schema": CAPTURE_SCHEMA, "revision": revision, "split": index["split"],
                "input_sha256": artifact_hash(input_root / "index.json"), "features_sha256": artifact_hash(features_path),
                "vision": identity, "annotations_mounted": False, "rows": captured}
    write_json(output / "capture.json", metadata)
    return metadata


def load_vru_capture(input_root, root, revision):
    index = load_vru_inputs(input_root)
    receipt = decode_json(read_bounded(root / "capture.json", 256 * 1024))
    if (receipt["schema"] != CAPTURE_SCHEMA or receipt["revision"] != revision or receipt["split"] != index["split"]
            or receipt["annotations_mounted"] is not False or receipt["input_sha256"] != artifact_hash(input_root / "index.json")):
        raise ValueError("Capture must bind to the exact observation pack, source split and revision")
    for row, original in zip(receipt["rows"], index["anchors"], strict=True):
        if (row["sample_token"], row["recording"]) != (original["sample_token"], original["recording"]):
            raise ValueError("Feature order differs from the observation index")
        require_hex(row["snapshot_sha256"], 64, "captured snapshot identity")
    raw = read_bounded(root / "camera-bev.safetensors", 80 * 1024**2)
    if sha256(raw).hexdigest() != receipt["features_sha256"]:
        raise ValueError("Feature bytes no longer match the label-free capture")
    tensors = load(raw)
    maps, coverage = tensors["appearance"], tensors["coverage"]
    n = len(index["anchors"])
    if (set(tensors) != {"appearance", "coverage"} or maps.shape != (n, VISUAL_DIM, *GRID_SHAPE)
            or maps.dtype != torch.float16 or coverage.shape != (n, *GRID_SHAPE) or coverage.dtype != torch.float32
            or not torch.isfinite(maps).all() or not torch.isfinite(coverage).all()
            or (coverage < 0).any() or (coverage > 1).any()):
        raise ValueError("Feature tensor shape, dtype or finite-domain validation failed")
    return index, receipt, maps, coverage


def require_new_output(output):
    if output.exists() or output.is_symlink():
        raise ValueError("Expected a new experiment output directory")
