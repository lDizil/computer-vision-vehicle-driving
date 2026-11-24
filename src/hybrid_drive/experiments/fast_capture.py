"""Capture frozen DINO features with access to inputs only, not targets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from math import atan2
from pathlib import Path
from time import perf_counter

import numpy as np

from hybrid_drive.adapters.nuscenes_surround import load_nuscenes_surround
from hybrid_drive.experiments.bounded_io import decode_json, read_bounded, require_hex
from hybrid_drive.experiments.io import write_json
from hybrid_drive.fast.metric_bev import BevGrid, past_forward_speed, temporal_bev
from hybrid_drive.neural.pretrained_alignment import load_alignment_vision
from hybrid_drive.neural.surround_features import SharedSurroundEncoder


SPLITS = ("train", "dev", "test")
MAX_CAPTURE_BYTES = 512 * 1024 * 1024


def load_json(path: Path):
    return decode_json(read_bounded(path, 8 * 1024 * 1024))


def artifact_hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def capture_fast(input_root: Path, model_root: Path, output: Path, *, revision: str) -> None:
    require_hex(revision, 40, "revision")
    if output.exists():
        raise ValueError("Output directory must be new")
    index = load_json(input_root / "index.json")
    _check_splits(index)
    vision, provenance = load_alignment_vision(model_root)
    model = provenance["source"]
    encoder = SharedSurroundEncoder(vision, encoder_id=f'{model["repository"]}@{model["revision"]}')
    grid = BevGrid()
    output.mkdir(parents=True)
    checks = {}
    started = perf_counter()
    for split in SPLITS:
        checks[split] = _capture_split(input_root, output, split, index["splits"][split], encoder, grid)
    write_json(output / "capture.json", {
        "schema": "hybrid-drive-fast-capture-v1",
        "revision": revision,
        "grid": asdict(grid),
        "raw_source_sha256": index["source_sha256"],
        "observation_index_sha256": artifact_hash(input_root / "index.json"),
        "encoder": provenance,
        "targets_opened": False,
        "wall_s": perf_counter() - started,
        "splits": checks,
    })


def _check_splits(index):
    if index.get("schema") != "hybrid-drive-fast-mini-inputs-v1" or set(index["splits"]) != set(SPLITS):
        raise ValueError("Unexpected observation index")
    require_hex(index.get("source_sha256"), 64, "raw source SHA256")
    recordings, observations = set(), set()
    for name in SPLITS:
        rows = index["splits"][name]
        if not 1 <= len(rows) <= 100:
            raise ValueError("Unexpected number of observations")
        ids = [row["sample_token"] for row in rows]
        logs = {row["recording"] for row in rows}
        seen = set(ids) | {row["previous_token"] for row in rows}
        if len(set(ids)) != len(ids) or logs & recordings or seen & observations:
            raise ValueError("Captures cross split or contain duplicate anchors")
        recordings |= logs
        observations |= seen


def _capture_split(root, output, split, rows, encoder, grid):
    out = output / split
    out.mkdir()
    mem = None
    coverage = np.zeros((len(rows), grid.rows, grid.cols), np.bool_)
    observations = []
    cached = None
    n_encoded = 0
    for i, row in enumerate(rows):
        current = encoder.encode(load_nuscenes_surround(root, row["sample_token"], max_skew_us=50_000))
        n_encoded += 1
        if cached is not None and cached.snapshot.snapshot_id == row["previous_token"]:
            previous = cached
        else:
            previous = encoder.encode(load_nuscenes_surround(root, row["previous_token"], max_skew_us=50_000))
            n_encoded += 1
        values, mask = temporal_bev(current, previous, grid)
        if mem is None:
            mem = np.lib.format.open_memmap(out / "features.npy", mode="w+", dtype=np.float16,
                                            shape=(len(rows), *values.shape))
        mem[i] = values.cpu().numpy()
        if not np.isfinite(mem[i]).all():
            raise ValueError("Encoded BEV cannot be stored in float16")
        coverage[i] = mask.cpu().numpy()
        speed = past_forward_speed(current.snapshot, previous.snapshot)
        pose = current.snapshot.ego_to_world
        observations.append({
            **row,
            "snapshot_sha256": current.snapshot.content_sha256,
            "previous_sha256": previous.snapshot.content_sha256,
            "ref_time_us": current.snapshot.ref_time_us,
            "earlier_ref_time_us": previous.snapshot.ref_time_us,
            "last_input_capture_us": max(
                current.snapshot.ref_time_us,
                *(v.capture_us for v in current.snapshot.views if v.capture_us is not None),
            ),
            "observed_skew_us": current.snapshot.measured_skew_us,
            "forward_speed_mps_from_past": speed,
            "ego_world_xy_yaw": [float(pose[0, 3]), float(pose[1, 3]), atan2(pose[1, 0], pose[0, 0])],
        })
        cached = current
        print(f"{split}: captured {i+1}/{len(rows)}, {n_encoded} surround encodes", flush=True)
    mem.flush()
    np.save(out / "coverage.npy", coverage, allow_pickle=False)
    write_json(out / "observations.json", observations)
    return {
        "count": len(rows),
        "encodes": n_encoded,
        "shape": list(mem.shape),
        "artifacts": {name: artifact_hash(out / name) for name in ("features.npy", "coverage.npy", "observations.json")},
    }


@dataclass(frozen=True)
class CaptureSplit:
    observations: list[dict]
    values: np.ndarray
    coverage: np.ndarray
    grid: BevGrid


def load_capture(root: Path, split: str) -> CaptureSplit:
    if split not in SPLITS:
        raise ValueError("Unknown split")
    meta = load_json(root / "capture.json")
    if meta["schema"] != "hybrid-drive-fast-capture-v1" or meta["targets_opened"] is not False:
        raise ValueError("Expected a completed observation-only capture")
    g = dict(meta["grid"])
    g["heights_m"] = tuple(g["heights_m"])
    grid = BevGrid(**g)
    section = meta["splits"][split]
    for name in ("features.npy", "coverage.npy", "observations.json"):
        path = root / split / name
        if path.is_symlink() or not 0 < path.stat().st_size < MAX_CAPTURE_BYTES:
            raise ValueError("Invalid capture file")
        if artifact_hash(path) != section["artifacts"][name]:
            raise ValueError("Capture file changed")
    values = np.load(root / split / "features.npy", mmap_mode="r", allow_pickle=False)
    mask = np.load(root / split / "coverage.npy", allow_pickle=False)
    observations = load_json(root / split / "observations.json")
    if (list(values.shape) != section["shape"] or values.dtype != np.float16 or values.ndim != 4
            or tuple(values.shape[2:]) != (grid.rows, grid.cols)
            or mask.shape != (values.shape[0], grid.rows, grid.cols) or mask.dtype != np.bool_
            or len(observations) != section["count"] or len(observations) != values.shape[0]):
        raise ValueError("Wrong capture layout")
    return CaptureSplit(observations, values, mask, grid)
