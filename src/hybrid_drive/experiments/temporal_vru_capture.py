"""Lift past/current shared camera features into the current ego frame."""

from hashlib import sha256

from safetensors.torch import load, save_file
import torch

from hybrid_drive.experiments.bounded_io import decode_json, read_bounded, require_hex
from hybrid_drive.experiments.box_motion_inputs import load_motion_snapshot
from hybrid_drive.experiments.fast_capture import artifact_hash
from hybrid_drive.experiments.io import write_json
from hybrid_drive.experiments.temporal_vru_data import load_temporal_vru_inputs
from hybrid_drive.experiments.temporal_vru_model import MAX_HISTORY_S, MIN_HISTORY_S
from hybrid_drive.experiments.visual_vru_capture import GRID, require_new_output
from hybrid_drive.experiments.visual_vru_model import GRID_SHAPE, VISUAL_DIM
from hybrid_drive.fast.metric_bev import lift_features
from hybrid_drive.neural.pretrained_alignment import load_alignment_vision
from hybrid_drive.neural.surround_features import SharedSurroundEncoder


CAPTURE_SCHEMA = "hybrid-drive-temporal-vru-capture-v1"
_KEYS = {"current", "past_in_current", "current_coverage", "past_coverage", "history_s"}


@torch.no_grad()
def capture_temporal_vru(input_root, model_root, output, *, revision):
    require_hex(revision, 40, "capture revision")
    require_new_output(output)
    index = load_temporal_vru_inputs(input_root)
    vision, source = load_alignment_vision(model_root)
    identity = source["source"]
    encoder = SharedSurroundEncoder(vision, encoder_id=f'{identity["repository"]}@{identity["revision"]}')
    values = {key: [] for key in sorted(_KEYS)}
    receipts = []
    for anchor in index["anchors"]:
        now, previous = anchor["sample_token"], anchor["previous_token"]
        current_snapshot = load_motion_snapshot(input_root, now, index["frames"][now])
        past_snapshot = load_motion_snapshot(input_root, previous, index["frames"][previous])
        dt = (current_snapshot.ref_time_us - past_snapshot.ref_time_us) / 1e6
        if not MIN_HISTORY_S <= dt <= MAX_HISTORY_S:
            raise ValueError("Past snapshot is not within the observed lookback window")
        current = encoder.encode(current_snapshot)
        past = encoder.encode(past_snapshot)
        now_grid, now_support = lift_features(current, current_snapshot, GRID)
        past_grid, past_support = lift_features(past, current_snapshot, GRID)
        for key, item in zip(("current", "past_in_current", "current_coverage", "past_coverage", "history_s"),
                             (now_grid.half(), past_grid.half(), now_support[0], past_support[0], torch.tensor(dt)),
                             strict=True):
            values[key].append(item)
        receipts.append({**anchor, "current_sha256": current_snapshot.content_sha256,
                         "past_sha256": past_snapshot.content_sha256})
        if len(receipts) % 24 == 0:
            print(f"Aligned capture {index['split']}: {len(receipts)}/{len(index['anchors'])}", flush=True)
    output.mkdir(parents=True)
    path = output / "temporal-bev.safetensors"
    save_file({key: torch.stack(seq) for key, seq in values.items()}, path)
    metadata = {"schema": CAPTURE_SCHEMA, "revision": revision, "split": index["split"],
                "input_sha256": artifact_hash(input_root / "index.json"), "features_sha256": artifact_hash(path),
                "vision": identity, "annotations_mounted": False, "past_alignment": "current_ego", "rows": receipts}
    write_json(output / "capture.json", metadata)
    return metadata


def load_temporal_capture(input_root, root, revision):
    index = load_temporal_vru_inputs(input_root)
    receipt = decode_json(read_bounded(root / "capture.json", 256 * 1024))
    if (receipt["schema"] != CAPTURE_SCHEMA or receipt["revision"] != revision or receipt["split"] != index["split"]
            or receipt["annotations_mounted"] is not False or receipt["past_alignment"] != "current_ego"
            or receipt["input_sha256"] != artifact_hash(input_root / "index.json")):
        raise ValueError("Temporal feature cache must match the observation and alignment provenance")
    for recorded, expected in zip(receipt["rows"], index["anchors"], strict=True):
        if any(recorded[key] != expected[key] for key in ("sample_token", "previous_token", "recording")):
            raise ValueError("Camera feature order does not match the temporal windows")
        require_hex(recorded["current_sha256"], 64, "current camera observation")
        require_hex(recorded["past_sha256"], 64, "past camera observation")
    raw = read_bounded(root / "temporal-bev.safetensors", 165 * 1024**2)
    if sha256(raw).hexdigest() != receipt["features_sha256"]:
        raise ValueError("Temporal feature bytes differ from the capture receipt")
    values = load(raw)
    n = len(index["anchors"])
    if set(values) != _KEYS:
        raise ValueError("Wrong temporal feature fields")
    for key, shape, dtype in (
        ("current", (n, VISUAL_DIM, *GRID_SHAPE), torch.float16),
        ("past_in_current", (n, VISUAL_DIM, *GRID_SHAPE), torch.float16),
        ("current_coverage", (n, *GRID_SHAPE), torch.float32),
        ("past_coverage", (n, *GRID_SHAPE), torch.float32),
        ("history_s", (n,), torch.float32),
    ):
        if values[key].shape != shape or values[key].dtype != dtype or not torch.isfinite(values[key]).all():
            raise ValueError(f"Invalid {key} shape, dtype or domain")
    expected_dt = torch.tensor([(index["frames"][a["sample_token"]]["ref_time_us"] -
                                 index["frames"][a["previous_token"]]["ref_time_us"]) / 1e6 for a in index["anchors"]])
    if not torch.equal(values["history_s"], expected_dt):
        raise ValueError("Stored history interval differs from recorded associations")
    for key in ("current_coverage", "past_coverage"):
        if (values[key] < 0).any() or (values[key] > 1).any():
            raise ValueError("Temporal coverage outside [0,1]")
    return index, receipt, values
