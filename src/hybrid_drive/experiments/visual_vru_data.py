"""Prepare observation-only camera packs with physically separate VRU labels."""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
from pathlib import Path
import pickle
import sys
from tempfile import TemporaryDirectory

import numpy as np

from hybrid_drive.assets.files import PreparationError, check_components
from hybrid_drive.contracts.sensors import timestamp_us
from hybrid_drive.experiments.bounded_io import canonical_json, decode_json, read_bounded, require_hex, require_keys
from hybrid_drive.experiments.box_motion_data import (
    CAMERAS,
    _METADATA_LIMIT,
    _check_recording_groups,
    _chronological,
    _observation,
    _read_selected,
)
from hybrid_drive.experiments.frozen_transfer_data import _timeline_sample, _validate_frame
from hybrid_drive.experiments.io import write_json
from hybrid_drive.experiments.visual_vru_geometry import GRID_COLS, GRID_ROWS, probe_pair, visible_vru_cells
from hybrid_drive.geometry.rigid import matrix


SOURCE_SHA256 = "f2322a9ef2a96ddc1aaff8b72b3bcd699e17c3e7f4c87257909d678122576183"
INPUT_SCHEMA = "hybrid-drive-visual-vru-observations-v1"
TARGET_SCHEMA = "hybrid-drive-visual-vru-targets-v1"
INDEX_LIMIT = 8 * 1024**2
LABEL_LIMIT = 2 * 1024**2
_SOURCE_SCHEMA = "hybrid-drive-visual-vru-sources-v1"
_SHARDS = {4: "train", 19: "train", 23: "development", 24: "heldout", 26: "heldout"}
_RECORDINGS = {
    "train": frozenset(("2021.06.09.17.23.18_veh-38", "2021.08.24.13.12.55_veh-45",
                        "2021.06.23.16.54.19_veh-35", "2021.08.09.17.55.59_veh-28")),
    "development": frozenset(("2021.06.07.18.53.26_veh-26", "2021.06.23.17.31.36_veh-16")),
    "heldout": frozenset(("2021.06.03.12.02.06_veh-35", "2021.06.07.12.54.00_veh-35",
                          "2021.06.28.15.02.02_veh-38", "2021.08.17.16.57.11_veh-08")),
}
_PAIR_CAP, _POSITIVE_CAP, _EMPTY_CAP = 12, 20, 8
_MIN_GAP_US = 2_500_000
_IMAGE_LIMIT = 2 * 1024**2
_ANNOTATION_KEYS = {"gt_names", "gt_boxes", "track_tokens"}


def prepare_vru_data(metadata_archive: Path, camera_root: Path, sources_path: Path, output: Path) -> dict:
    """Select fixed offline probes and publish separate inputs and labels atomically."""
    if sys.platform != "linux":
        raise PreparationError("OpenScene VRU preparation requires the isolated Linux image")
    source = _load_source(sources_path)
    output = Path(output)
    check_components(output)
    if output.exists():
        raise PreparationError("Visual VRU output must not already exist")
    output.parent.mkdir(parents=True, exist_ok=True)
    catalog = {split: {} for split in _RECORDINGS}
    wanted_images = {shard: {} for shard in source["camera_archives"]}
    recordings, seen_tokens = {}, set()
    for spec, frames in _read_recordings(metadata_archive, source):
        shard, split, recording = str(spec["camera_shard"]), spec["split"], spec["recording"]
        selected, recordings[recording] = _select_samples(frames)
        for index, cells, pair in selected:
            frame = frames[index]
            token = frame["token"]
            if token in seen_tokens:
                raise PreparationError("Selected recordings share a visual VRU sample token")
            seen_tokens.add(token)
            poses, blocked, future_blocked = pair if pair is not None else (None, None, None)
            observation = _observation(frame, recording, spec["log_name"],
                                       source["camera_archives"][shard]["archive_prefix"], wanted_images[shard])
            catalog[split][token] = (observation, poses, cells, blocked, future_blocked)

    with TemporaryDirectory(dir=output.parent, prefix=".visual-vru-") as temp:
        stage = Path(temp) / "data"
        ownership = {recording: split for split, groups in _RECORDINGS.items() for recording in groups}
        _extract_images(Path(camera_root), source["camera_archives"], wanted_images, stage, ownership)
        splits = {split: _write_split(stage / split, split, rows) for split, rows in catalog.items()}
        check_components(output)
        if output.exists():
            raise PreparationError("Visual VRU output appeared during extraction")
        stage.rename(output)
    return {"anchors": len(seen_tokens), "camera_jpegs": sum(map(len, wanted_images.values())),
            "splits": splits, "recordings": recordings}


def load_vru_inputs(root: Path) -> dict:
    """Load and validate only the observation index; JPEGs need not be mounted."""
    return _read_index(root)[1]


def load_vru_labels(input_root: Path, labels_path: Path) -> dict:
    """Bind separately mounted labels to one exact observation index and order."""
    raw, index = _read_index(input_root)
    targets = decode_json(read_bounded(Path(labels_path), LABEL_LIMIT))
    _validate_targets(targets, index, sha256(raw).hexdigest())
    return targets


def _load_source(path):
    source = decode_json(read_bounded(Path(path), 16 * 1024))
    if (not isinstance(source, dict) or source.get("schema") != _SOURCE_SCHEMA
            or sha256(canonical_json(source)).hexdigest() != SOURCE_SHA256):
        raise PreparationError("Visual VRU sources must match the checked-in digest")
    logs = source["logs"]
    _check_recording_groups({"logs": logs, "previous_recordings": []})
    if ({str(shard) for shard in _SHARDS} != source["camera_archives"].keys()
            or Counter(row["camera_shard"] for row in logs) != {shard: 2 for shard in _SHARDS}
            or any(row["split"] != _SHARDS[row["camera_shard"]]
                   or row["recording"] not in _RECORDINGS[row["split"]] for row in logs)):
        raise PreparationError("Expected the fixed disjoint Visual VRU recording split")
    return source


def _read_recordings(path, source):
    specs = {row["member"]: row for row in source["logs"]}
    seen = set()
    for member, data in _read_selected(path, source["metadata_archive"], set(specs), _METADATA_LIMIT):
        spec = specs[member]
        # Check the exact member before invoking the pickle decoder in isolated Linux only.
        if len(data) != spec["size_bytes"] or sha256(data).hexdigest() != spec["sha256"]:
            raise PreparationError("Visual VRU metadata member differs from its pinned SHA256 or size")
        frames = pickle.loads(data)
        if (not isinstance(frames, list) or not 2 <= len(frames) <= 4096
                or len(frames) != spec.get("frame_count", len(frames))
                or any(not isinstance(f, dict) or f.get("log_name") != spec["log_name"] for f in frames)):
            raise PreparationError("Visual VRU metadata recording does not match its pinned log")
        frames = _chronological(frames)
        physical = frames[0]["log_token"]
        if physical in seen:
            raise PreparationError("Selected Visual VRU recordings share a physical log")
        seen.add(physical)
        yield spec, frames


def _select_samples(frames):
    stamps = [timestamp_us(frame["timestamp"], "OpenScene association stamp") for frame in frames]
    pairs, positives, empty, cells_by_index = {}, [], [], {}
    for i, frame in enumerate(frames):
        anns = frame.get("anns")
        if (not isinstance(anns, dict) or not _ANNOTATION_KEYS <= anns.keys()
                or any(not frame.get("cams", {}).get(cam, {}).get("data_path") for cam in CAMERAS)):
            continue
        cells = visible_vru_cells(frame)
        cells_by_index[i] = cells
        (positives if cells else empty).append(i)
        if cells and i and (pair := probe_pair(frames, i)) is not None:
            pairs[i] = pair
    selected_pairs = _select_pairs(pairs, stamps)
    extra_positives = _sample_spaced(positives, stamps, selected_pairs, _POSITIVE_CAP)
    extra_empty = _sample_spaced(empty, stamps, selected_pairs + extra_positives, _EMPTY_CAP)
    if not selected_pairs and not extra_positives and not extra_empty:
        raise PreparationError("No annotated six-camera Visual VRU anchors in a pinned recording")
    chosen = sorted((*selected_pairs, *extra_positives, *extra_empty))
    stats = {"eligible": len(cells_by_index), "available_positive": len(positives),
             "available_empty": len(empty), **_pair_counts(pairs.values(), "available_"),
             "selected": len(chosen), "extra_positive": len(extra_positives), "extra_empty": len(extra_empty),
             **_pair_counts([pairs[i] for i in selected_pairs], "selected_")}
    paired = set(selected_pairs)
    return [(i, cells_by_index[i], pairs[i] if i in paired else None) for i in chosen], stats


def _select_pairs(pairs, stamps):
    selected, direction_counts = [], [0, 0]
    for _ in range(_PAIR_CAP):
        viable = [i for i in pairs if _separated(i, selected, stamps)]
        if not viable:
            break

        def priority(i):
            _, blocked, future_blocked = pairs[i]
            spread = min((abs(stamps[i] - stamps[j]) for j in selected), default=0)
            return direction_counts[blocked], future_blocked != blocked, -spread, stamps[i]

        chosen = min(viable, key=priority)
        selected.append(chosen)
        direction_counts[pairs[chosen][1]] += 1
    return selected


def _sample_spaced(candidates, stamps, reserved, cap):
    pool = []
    for i in candidates:
        if ((not pool or stamps[i] - stamps[pool[-1]] >= _MIN_GAP_US)
                and _separated(i, reserved, stamps)):
            pool.append(i)
    return _timeline_sample(pool, stamps, min(cap, len(pool)))


def _separated(index, others, stamps):
    return all(abs(stamps[index] - stamps[other]) >= _MIN_GAP_US for other in others)


def _pair_counts(pairs, prefix):
    pairs = tuple(pairs)
    return {prefix + "probes": len(pairs), prefix + "blocked_left": sum(p[1] == 0 for p in pairs),
            prefix + "blocked_right": sum(p[1] == 1 for p in pairs),
            prefix + "future_probes": sum(p[2] is not None for p in pairs),
            prefix + "future_agrees": sum(p[1] == p[2] for p in pairs)}


def _extract_images(camera_root, archives, wanted, output, ownership):
    seen = {}
    for shard, spec in archives.items():
        images = wanted[shard]
        for member, payload in _read_selected(camera_root / spec["filename"], spec, set(images), _IMAGE_LIMIT):
            rel, view, recording = images[member]
            if not payload.startswith(b"\xff\xd8\xff"):
                raise PreparationError("Expected a JPEG in a Visual VRU camera archive")
            digest = sha256(payload).hexdigest()
            if seen.setdefault(digest, recording) != recording:
                raise PreparationError("Visual VRU recordings share camera JPEG bytes")
            view["sha256"] = digest
            target = output / ownership[recording] / "inputs" / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)


def _write_split(root, split, catalog):
    anchors, frames, rows = [], {}, []
    ordered = sorted(catalog, key=lambda token: (catalog[token][0]["recording"], catalog[token][0]["ref_time_us"]))
    for token in ordered:
        frame, poses, cells, blocked, future_blocked = catalog[token]
        anchors.append({"sample_token": token, "recording": frame["recording"], "probe_poses": poses})
        frames[token] = frame
        rows.append({"sample_token": token, "visible_cells": [list(cell) for cell in cells],
                     "blocked_index": blocked, "future_blocked_index": future_blocked})
    index = {"schema": INPUT_SCHEMA, "source_sha256": SOURCE_SHA256, "split": split,
             "anchors": anchors, "frames": frames}
    _validate_index(index)
    index_file = root / "inputs" / "index.json"
    write_json(index_file, index)
    digest = sha256(read_bounded(index_file, INDEX_LIMIT)).hexdigest()
    labels = {"schema": TARGET_SCHEMA, "input_sha256": digest, "rows": rows}
    _validate_targets(labels, index, digest)
    label_file = root / "labels" / "labels.json"
    label_file.parent.mkdir(parents=True)
    write_json(label_file, labels)
    pair_counts = _pair_counts([(None, r["blocked_index"], r["future_blocked_index"]) for r in rows
                                if r["blocked_index"] is not None], "")
    return {"anchors": len(rows), "positive": sum(bool(row["visible_cells"]) for row in rows),
            "empty": sum(not row["visible_cells"] for row in rows), **pair_counts, "input_sha256": digest,
            "labels_sha256": sha256(read_bounded(label_file, LABEL_LIMIT)).hexdigest()}


def _read_index(root):
    raw = read_bounded(Path(root) / "index.json", INDEX_LIMIT)
    index = decode_json(raw)
    _validate_index(index)
    return raw, index


def _validate_index(index):
    require_keys(index, {"schema", "source_sha256", "split", "anchors", "frames"}, "Visual VRU observations")
    anchors, frames, split = index["anchors"], index["frames"], index["split"]
    if (index["schema"] != INPUT_SCHEMA or index["source_sha256"] != SOURCE_SHA256
            or not isinstance(split, str) or split not in _RECORDINGS
            or not isinstance(anchors, list) or not isinstance(frames, dict) or len(anchors) != len(frames)
            or not 1 <= len(anchors) <= len(_RECORDINGS[split]) * (_PAIR_CAP + _POSITIVE_CAP + _EMPTY_CAP)):
        raise ValueError("Unexpected Visual VRU perception split or observation count")
    seen, counts, probe_counts, jpgs = set(), Counter(), Counter(), {}
    prior = None
    for anchor in anchors:
        require_keys(anchor, {"sample_token", "recording", "probe_poses"}, "Visual VRU anchor")
        token, recording = anchor["sample_token"], anchor["recording"]
        require_hex(token, 16, "Visual VRU anchor token")
        if token in seen or token not in frames or not isinstance(recording, str) or recording not in _RECORDINGS[split]:
            raise ValueError("Mixed split, missing or duplicated Visual VRU frame")
        seen.add(token)
        counts[recording] += 1
        stamp = _validate_frame(frames[token], token, recording, jpgs)
        if prior is not None and (recording < prior[0] or (recording == prior[0] and stamp - prior[1] < _MIN_GAP_US)):
            raise ValueError("Visual VRU anchors must be sorted and spaced")
        prior = recording, stamp
        if anchor["probe_poses"] is not None:
            _validate_probe(anchor["probe_poses"])
            probe_counts[recording] += 1
    if (counts.keys() != _RECORDINGS[split]
            or any(n > _PAIR_CAP + _POSITIVE_CAP + _EMPTY_CAP for n in counts.values())
            or any(n > _PAIR_CAP for n in probe_counts.values())):
        raise ValueError("Unexpected Visual VRU per-recording sample or probe counts")


def _validate_probe(value):
    poses = matrix(value, (2, 5, 3), "Visual VRU probe poses")
    if (np.any(np.abs(poses) > 30) or np.any(poses[0, :, 0] <= 0) or np.any(np.diff(poses[0, :, 0]) <= 0)
            or np.any(poses[0, :, 1] <= 0) or not np.allclose(poses[0, :, 0], poses[1, :, 0], atol=1e-8, rtol=0)
            or not np.allclose(poses[0, :, 1:], -poses[1, :, 1:], atol=1e-8, rtol=0)):
        raise ValueError("Visual VRU probes must share forward progress and mirror lateral poses")


def _validate_targets(targets, index, digest):
    require_keys(targets, {"schema", "input_sha256", "rows"}, "Visual VRU labels")
    rows = targets["rows"]
    if (targets["schema"] != TARGET_SCHEMA or targets["input_sha256"] != digest
            or not isinstance(rows, list) or len(rows) != len(index["anchors"])):
        raise ValueError("Visual VRU labels do not match the exact observation index")
    for row, anchor in zip(rows, index["anchors"]):
        require_keys(row, {"sample_token", "visible_cells", "blocked_index", "future_blocked_index"}, "Visual VRU label row")
        cells = row["visible_cells"]
        if (row["sample_token"] != anchor["sample_token"] or not isinstance(cells, list)
                or any(not isinstance(cell, list) or len(cell) != 2 or any(type(x) is not int for x in cell)
                       or not (0 <= cell[0] < GRID_ROWS and 0 <= cell[1] < GRID_COLS) for cell in cells)
                or cells != [list(cell) for cell in sorted({tuple(cell) for cell in cells})]):
            raise ValueError("Visual VRU label order or visible cell coordinates are invalid")
        blocked, future = row["blocked_index"], row["future_blocked_index"]
        for value in (blocked, future):
            if value is not None and (type(value) is not int or value not in (0, 1)):
                raise ValueError("Visual VRU probe index must be an integer side or null")
        if ((blocked is None) != (anchor["probe_poses"] is None) or (blocked is None and future is not None)
                or (blocked is not None and not cells)):
            raise ValueError("Visual VRU label does not describe the associated diagnostic probe")
