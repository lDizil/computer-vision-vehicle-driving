"""Capture paired ground-road and original object projections without labels."""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
from math import atan2
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np

from hybrid_drive.adapters.nuscenes_surround import load_nuscenes_surround
from hybrid_drive.experiments.bounded_io import decode_json, read_bounded, require_hex
from hybrid_drive.experiments.box_motion_inputs import load_motion_inputs, load_motion_snapshot
from hybrid_drive.experiments.fast_capture import artifact_hash
from hybrid_drive.experiments.io import write_json
from hybrid_drive.fast.ground_road_projection import GROUND_HEIGHT_M, ground_temporal_bev
from hybrid_drive.fast.metric_bev import BevGrid, past_forward_speed, temporal_bev
from hybrid_drive.neural.pretrained_alignment import load_alignment_vision
from hybrid_drive.neural.surround_features import SharedSurroundEncoder


SCHEMA = "hybrid-drive-ground-road-capture-v1"
MINI_CAPTURE_REVISION = "d93ae54ba71d7b8873c6519e6e204a45c72ddfa8"
_CHANNELS = 773
_COUNTS = {"train": 56, "dev": 7, "test": 40}
_MAX_OBSERVATIONS_BYTES = 2 * 1024 * 1024


def capture_ground_road(
    mini_input_root: Path, mini_capture: Path, external_input_root: Path,
    model_root: Path, output: Path, *, revision: str,
) -> None:
    """Use frozen Mini train/dev observations and the independent OpenScene anchors."""
    require_hex(revision, 40, "revision")
    if output.exists() or output.is_symlink():
        raise ValueError("Expected a new capture directory")
    mini_meta, mini_sha, grid = _load_mini_receipt(mini_input_root, mini_capture)
    mini_rows = {
        role: _mini_observations(mini_capture, mini_meta, role)
        for role in ("train", "dev")
    }
    test_index = load_motion_inputs(external_input_root)
    if len(test_index["anchors"]) != _COUNTS["test"]:
        raise ValueError("Expected the 40 frozen OpenScene anchors")
    test_index_sha = artifact_hash(external_input_root / "index.json")

    vision, provenance = load_alignment_vision(model_root)
    if provenance != mini_meta["encoder"]:
        raise ValueError("DINO and the preprocessing/runtime must match the original capture")
    source = provenance["source"]
    encoder = SharedSurroundEncoder(vision, encoder_id=f'{source["repository"]}@{source["revision"]}')
    started = perf_counter()
    output.parent.mkdir(parents=True, exist_ok=True)

    def mini_snapshot(token):
        return load_nuscenes_surround(mini_input_root, token, max_skew_us=50_000)

    def test_snapshot(token):
        return load_motion_snapshot(external_input_root, token, test_index["frames"][token])

    with TemporaryDirectory(prefix=".ground-road-", dir=output.parent) as temp:
        stage = Path(temp) / "capture"
        stage.mkdir()
        results = {}
        for role, (rows, original) in mini_rows.items():
            results[role] = _capture_role(
                stage / role, role, rows, encoder, mini_snapshot, grid, original_observations=original,
            )
        test_rows = [{**row, "previous_token": row["history_tokens"][-1]} for row in test_index["anchors"]]
        results["test"] = _capture_role(stage / "test", "test", test_rows, encoder, test_snapshot, grid)
        write_json(stage / "capture.json", {
            "schema": SCHEMA,
            "revision": revision,
            "grid": mini_meta["grid"],
            "ground_plane_m": GROUND_HEIGHT_M,
            "source_mini_capture_sha256": mini_sha,
            "source_test_index_sha256": test_index_sha,
            "test_targets_opened": False,
            "encoder": provenance,
            "wall_s": perf_counter() - started,
            "roles": results,
        })
        if output.exists() or output.is_symlink():
            raise ValueError("Capture output appeared while processing inputs")
        stage.rename(output)


def _load_mini_receipt(input_root, capture_root):
    raw = read_bounded(capture_root / "capture.json", _MAX_OBSERVATIONS_BYTES)
    meta = decode_json(raw)
    if (meta["schema"] != "hybrid-drive-fast-capture-v1" or meta["revision"] != MINI_CAPTURE_REVISION
            or meta["targets_opened"] is not False
            or meta["observation_index_sha256"] != artifact_hash(input_root / "index.json")):
        raise ValueError("Expected the original target-free Mini capture and its observation index")
    fields = dict(meta["grid"])
    fields["heights_m"] = tuple(fields["heights_m"])
    grid = BevGrid(**fields)
    if grid != BevGrid():
        raise ValueError("Expected the frozen 2 m BEV lattice")
    return meta, sha256(raw).hexdigest(), grid


def _mini_observations(root, meta, role):
    part = meta["splits"][role]
    raw = read_bounded(root / role / "observations.json", _MAX_OBSERVATIONS_BYTES)
    if sha256(raw).hexdigest() != part["artifacts"]["observations.json"]:
        raise ValueError("Original Mini observations changed")
    rows = decode_json(raw)
    if not isinstance(rows, list) or len(rows) != part["count"] or len(rows) != _COUNTS[role]:
        raise ValueError("Unexpected Mini observation count")
    if len({row["sample_token"] for row in rows}) != len(rows):
        raise ValueError("Duplicate Mini anchors")
    return rows, raw


def _capture_role(root, role, rows, encoder, load_snapshot, grid, *, original_observations=None):
    root.mkdir()
    external = role == "test"
    shape = (len(rows), _CHANNELS, grid.rows, grid.cols)
    names = ["features_ground.npy", "current_coverage.npy", "history_coverage.npy", "observations.json"]
    ground = np.lib.format.open_memmap(root / names[0], mode="w+", dtype=np.float16, shape=shape)
    object_maps = None
    if external:
        names.extend(("features_object.npy", "object_coverage.npy"))
        object_maps = np.lib.format.open_memmap(root / names[4], mode="w+", dtype=np.float16, shape=shape)
    current_seen = np.empty((len(rows), grid.rows, grid.cols), dtype=np.bool_)
    previous_seen = np.empty_like(current_seen)
    object_seen = np.empty_like(current_seen) if external else None
    encoded = _OncePerSnapshot(encoder, load_snapshot, rows)
    external_observations = []
    for i, row in enumerate(rows):
        now, previous = (encoded.take(row[k]) for k in ("sample_token", "previous_token"))
        if not external:
            _verify_input_binding(now.snapshot, previous.snapshot, row)
        values, now_mask, prior_mask = ground_temporal_bev(now, previous, grid)
        _write_feature(ground, i, values)
        current_seen[i] = now_mask.cpu().numpy()
        previous_seen[i] = prior_mask.cpu().numpy()
        if external:
            original, old_mask = temporal_bev(now, previous, grid)
            _write_feature(object_maps, i, original)
            object_seen[i] = old_mask.cpu().numpy()
            external_observations.append(_external_observation(row, now.snapshot, previous.snapshot))
        print(f"{role}: captured {i+1}/{len(rows)}, {encoded.encodes} surround encodes", flush=True)
    ground.flush()
    np.save(root / names[1], current_seen, allow_pickle=False)
    np.save(root / names[2], previous_seen, allow_pickle=False)
    if external:
        object_maps.flush()
        np.save(root / names[5], object_seen, allow_pickle=False)
        write_json(root / names[3], external_observations)
    else:
        # Existing road targets bind to this exact file, including JSON bytes.
        (root / names[3]).write_bytes(original_observations)
    return {
        "count": len(rows), "shape": list(shape), "encodes": encoded.encodes,
        "artifacts": {name: artifact_hash(root / name) for name in names},
    }


class _OncePerSnapshot:
    def __init__(self, encoder, load_snapshot, rows):
        self.remaining = Counter(row[k] for row in rows for k in ("sample_token", "previous_token"))
        self.encoder, self.load_snapshot = encoder, load_snapshot
        self.cache = {}
        self.encodes = 0

    def take(self, token):
        if token not in self.cache:
            sample = self.load_snapshot(token)
            if sample.snapshot_id != token:
                raise ValueError("Requested token does not match the loaded snapshot")
            value = self.encoder.encode(sample)
            self.encodes += 1
        else:
            value = self.cache.pop(token)
        self.remaining[token] -= 1
        if self.remaining[token]:
            if len(self.cache) >= 4:
                raise ValueError("Too many live shared snapshots; check capture chronology")
            self.cache[token] = value
        return value


def _verify_input_binding(now, previous, row):
    if (now.content_sha256 != row["snapshot_sha256"]
            or previous.content_sha256 != row["previous_sha256"]
            or now.ref_time_us != row["ref_time_us"]
            or previous.ref_time_us != row["earlier_ref_time_us"]):
        raise ValueError("Mini cameras or reference pose changed since the label-bound capture")


def _external_observation(row, now, previous):
    pose = now.ego_to_world
    return {
        "sample_token": row["sample_token"],
        "previous_token": row["previous_token"],
        "recording": row["recording"],
        "snapshot_sha256": now.content_sha256,
        "previous_sha256": previous.content_sha256,
        "ref_time_us": now.ref_time_us,
        "earlier_ref_time_us": previous.ref_time_us,
        "timing_basis": now.timing.value,
        "ego_world_xy_yaw": [float(pose[0, 3]), float(pose[1, 3]), atan2(pose[1, 0], pose[0, 0])],
        "forward_speed_mps_from_past": past_forward_speed(now, previous),
    }


def _write_feature(values, index, tensor):
    if tuple(tensor.shape) != values.shape[1:]:
        raise ValueError("Unexpected BEV channel layout")
    values[index] = tensor.cpu().numpy()
    if not np.isfinite(values[index]).all():
        raise ValueError("Encoded BEV cannot be stored in float16")
