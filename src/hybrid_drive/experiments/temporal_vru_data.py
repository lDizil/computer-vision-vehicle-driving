"""Build pinned past/current camera inputs and separate short-horizon VRU labels."""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
from math import hypot
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

from hybrid_drive.assets.files import PreparationError, check_components
from hybrid_drive.contracts.sensors import timestamp_us
from hybrid_drive.experiments.bounded_io import canonical_json, decode_json, read_bounded, require_hex, require_keys
from hybrid_drive.experiments.box_motion_data import CAMERAS, _check_recording_groups, _observation, _pose
from hybrid_drive.experiments.frozen_transfer_contacts import _eligible_vulnerable, _tracked_boxes
from hybrid_drive.experiments.frozen_transfer_data import VULNERABLE_CLASSES, _timeline_sample, _validate_frame
from hybrid_drive.experiments.io import write_json
from hybrid_drive.experiments.visual_vru_data import _extract_images, _read_recordings
from hybrid_drive.experiments.visual_vru_geometry import GRID_COLS, GRID_ROWS, GRID_STEP_M, GRID_X_MIN, GRID_Y_MIN
from hybrid_drive.geometry.rigid import invert_pose, matrix


SOURCE_SHA256 = "5308acdf7fd9c0617faedce9cd819e4bb8bd223e1f8d5642f2b00e31d6ddb1ef"
INPUT_SCHEMA = "hybrid-drive-temporal-vru-inputs-v1"
TARGET_SCHEMA = "hybrid-drive-temporal-vru-targets-v1"
INDEX_LIMIT = 16 * 1024**2
LABEL_LIMIT = 4 * 1024**2
_SOURCE_SCHEMA = "hybrid-drive-temporal-vru-sources-v1"
_SHARDS = {4: "train", 19: "train", 23: "development", 1: "heldout", 24: "heldout", 26: "heldout"}
_RECORDINGS = {
    "train": frozenset(("2021.06.09.17.23.18_veh-38", "2021.08.24.13.12.55_veh-45",
                        "2021.06.23.16.54.19_veh-35", "2021.08.09.17.55.59_veh-28")),
    "development": frozenset(("2021.06.07.18.53.26_veh-26", "2021.06.23.17.31.36_veh-16")),
    "heldout": frozenset(("2021.06.03.12.02.06_veh-35", "2021.06.07.12.54.00_veh-35",
                          "2021.06.28.15.02.02_veh-38", "2021.08.17.16.57.11_veh-08",
                          "2021.10.05.07.10.04_veh-52", "2021.10.01.19.16.42_veh-28")),
}
_MOVING_CAP, _QUIET_CAP, _EMPTY_CAP = 12, 8, 4
_RECORDING_CAP = _MOVING_CAP + _QUIET_CAP + _EMPTY_CAP
_MIN_GAP_US = 3_000_000
_PAST_WINDOW_US = (400_000, 600_000)
_FUTURE_WINDOWS_US = ((400_000, 600_000), (900_000, 1_100_000))
_FUTURE_STEP_US = (300_000, 700_000)
_ANNOTATION_KEYS = {"gt_names", "gt_boxes", "track_tokens"}
_MOVING_M, _QUIET_M = .75, .25


def prepare_temporal_vru(metadata_archive: Path, camera_root: Path, sources_path: Path, output: Path) -> dict:
    """Select curated windows before model scoring and publish data atomically."""
    if sys.platform != "linux":
        raise PreparationError("Temporal VRU preparation requires the isolated Linux image")
    source = _load_source(sources_path)
    output = Path(output)
    check_components(output)
    if output.exists():
        raise PreparationError("Temporal VRU output must not already exist")
    output.parent.mkdir(parents=True, exist_ok=True)
    frames_by_split = {split: {} for split in _RECORDINGS}
    samples_by_split = {split: [] for split in _RECORDINGS}
    images = {shard: {} for shard in source["camera_archives"]}
    recordings, input_tokens, future_tokens = {}, set(), set()
    for spec, frames in _read_recordings(metadata_archive, source):
        recording, split, shard = spec["recording"], spec["split"], str(spec["camera_shard"])
        selected, recordings[recording] = _select_samples(frames)
        for i, category, tracks, target_dt_s in selected:
            history, current = frames[i - 1:i + 1]
            future_tokens.update(frame["token"] for frame in frames[i + 1:i + 3])
            for frame in (history, current):
                token = frame["token"]
                if token in input_tokens:
                    raise PreparationError("Temporal VRU windows share an observed token")
                input_tokens.add(token)
                frames_by_split[split][token] = _observation(
                    frame, recording, spec["log_name"], source["camera_archives"][shard]["archive_prefix"], images[shard],
                )
            samples_by_split[split].append((current["token"], history["token"], recording, category, tracks, target_dt_s))
    if input_tokens & future_tokens:
        raise PreparationError("Temporal VRU observations contain a following-frame token")

    with TemporaryDirectory(dir=output.parent, prefix=".temporal-vru-") as temp:
        stage = Path(temp) / "data"
        ownership = {recording: split for split, groups in _RECORDINGS.items() for recording in groups}
        _extract_images(Path(camera_root), source["camera_archives"], images, stage, ownership)
        splits = {split: _write_split(stage / split, split, observed, samples_by_split[split])
                  for split, observed in frames_by_split.items()}
        check_components(output)
        if output.exists():
            raise PreparationError("Temporal VRU output appeared during preparation")
        stage.rename(output)
    return {"anchors": sum(len(rows) for rows in samples_by_split.values()),
            "observed_frames": len(input_tokens), "camera_jpegs": sum(map(len, images.values())),
            "splits": splits, "recordings": recordings}


def load_temporal_vru_inputs(root: Path) -> dict:
    """Load only the observation index. JPEGs and labels need not be mounted."""
    return _read_index(root)[1]


def load_temporal_vru_labels(root: Path, labels_path: Path) -> dict:
    """Verify that separately mounted labels bind the exact observation bytes."""
    raw, index = _read_index(root)
    targets = decode_json(read_bounded(Path(labels_path), LABEL_LIMIT))
    _validate_targets(targets, index, sha256(raw).hexdigest())
    return targets


def _load_source(path):
    source = decode_json(read_bounded(Path(path), 16 * 1024))
    if (not isinstance(source, dict) or source.get("schema") != _SOURCE_SCHEMA
            or sha256(canonical_json(source)).hexdigest() != SOURCE_SHA256):
        raise PreparationError("Temporal VRU sources must match the checked-in digest")
    logs = source["logs"]
    _check_recording_groups({"logs": logs, "previous_recordings": []})
    if ({str(shard) for shard in _SHARDS} != source["camera_archives"].keys()
            or Counter(row["camera_shard"] for row in logs) != {shard: 2 for shard in _SHARDS}
            or any(row["split"] != _SHARDS[row["camera_shard"]]
                   or row["recording"] not in _RECORDINGS[row["split"]] for row in logs)):
        raise PreparationError("Expected the fixed disjoint Temporal VRU recording split")
    return source


def _select_samples(frames):
    stamps = [timestamp_us(frame["timestamp"], "OpenScene association stamp") for frame in frames]
    pools = {name: [] for name in ("moving", "quiet", "other", "empty")}
    by_index = {}
    for i in range(1, len(frames) - 2):
        sample = _label_window(frames[i - 1:i + 3], stamps[i - 1:i + 3])
        if sample is None:
            continue
        category, tracks, dt_s = sample
        pools[category].append(i)
        by_index[i] = (category, tracks, dt_s)
    selected = _choose_spaced(pools["moving"], stamps, (), _MOVING_CAP)
    quiet = _choose_spaced(pools["quiet"], stamps, selected, _QUIET_CAP)
    selected.extend(quiet)
    selected.extend(_choose_spaced(pools["other"], stamps, selected, _QUIET_CAP - len(quiet)))
    selected.extend(_choose_spaced(pools["empty"], stamps, selected, _EMPTY_CAP))
    if not selected:
        raise PreparationError("No complete Temporal VRU samples in a pinned recording")
    samples = [(i, *by_index[i]) for i in sorted(selected)]
    stats = {"available": {key: len(items) for key, items in pools.items()},
             "selected": _sample_stats(samples)}
    return samples, stats


def _label_window(frames, stamps):
    history, current, next_frame, last_frame = frames
    previous, now, soon, later = stamps
    if (len({(f["log_token"], f["log_name"]) for f in frames}) != 1
            or not _PAST_WINDOW_US[0] <= now - previous <= _PAST_WINDOW_US[1]
            or not _future_offsets_fit(soon - now, later - now)
            or any(not _has_cameras(f) for f in (history, current))
            or any(not _has_annotations(f) for f in (current, next_frame, last_frame))):
        return None
    observed = _tracked_boxes(current["anns"])
    in_grid = {token: (name, box) for token, (name, box) in observed.items()
               if name in VULNERABLE_CLASSES and _in_grid(box[:2])}
    eligible = _eligible_vulnerable(current, in_grid)
    anchor_from_world = invert_pose(_pose(current))
    next_boxes, last_boxes = (_vru_centers(frame, anchor_from_world) for frame in (next_frame, last_frame))
    if not eligible and (in_grid or next_boxes or last_boxes):
        # No supported anchor track is not evidence that the window was empty.
        return None
    tracks = []
    for token, kind in sorted(eligible.items()):
        if any(token not in boxes or boxes[token][0] != kind for boxes in (next_boxes, last_boxes)):
            # Includes a matched actor that leaves the supervised grid.
            return None
        tracks.append({"centers_xy_m": [in_grid[token][1][:2].tolist(), next_boxes[token][1], last_boxes[token][1]]})
    return _category(tracks), tracks, [(soon - now) / 1e6, (later - now) / 1e6]


def _vru_centers(frame, anchor_from_world):
    relative = anchor_from_world @ _pose(frame)
    centers = {}
    for token, (kind, box) in _tracked_boxes(frame["anns"]).items():
        if kind not in VULNERABLE_CLASSES:
            continue
        xy = (relative[:3, :3] @ box[:3] + relative[:3, 3])[:2]
        if _in_grid(xy):
            centers[token] = (kind, xy.tolist())
    return centers


def _in_grid(xy):
    x, y = xy
    return (GRID_X_MIN <= x < GRID_X_MIN + GRID_ROWS * GRID_STEP_M
            and GRID_Y_MIN <= y < GRID_Y_MIN + GRID_COLS * GRID_STEP_M)


def _has_cameras(frame):
    return all(frame.get("cams", {}).get(camera, {}).get("data_path") for camera in CAMERAS)


def _has_annotations(frame):
    annotations = frame.get("anns")
    return isinstance(annotations, dict) and _ANNOTATION_KEYS <= annotations.keys()


def _future_offsets_fit(near, far):
    (near_lo, near_hi), (far_lo, far_hi) = _FUTURE_WINDOWS_US
    return (near_lo <= near <= near_hi and far_lo <= far <= far_hi
            and _FUTURE_STEP_US[0] <= far - near <= _FUTURE_STEP_US[1])


def _movement(track):
    start, _, finish = track["centers_xy_m"]
    return hypot(finish[0] - start[0], finish[1] - start[1])


def _category(tracks):
    largest = max(map(_movement, tracks), default=None)
    if largest is None:
        return "empty"
    if largest >= _MOVING_M:
        return "moving"
    return "quiet" if largest <= _QUIET_M else "other"


def _choose_spaced(candidates, stamps, reserved, cap):
    pool = []
    for i in candidates:
        if ((not pool or stamps[i] - stamps[pool[-1]] >= _MIN_GAP_US)
                and all(abs(stamps[i] - stamps[j]) >= _MIN_GAP_US for j in reserved)):
            pool.append(i)
    return _timeline_sample(pool, stamps, min(cap, len(pool)))


def _sample_stats(samples):
    categories = Counter(category for _, category, _, _ in samples)
    tracks = [_movement(track) for _, _, records, _ in samples for track in records]
    return {"anchors": len(samples), "tracks": len(tracks), "moving_tracks": sum(x >= _MOVING_M for x in tracks),
            "quiet_tracks": sum(x <= _QUIET_M for x in tracks),
            **{key: categories[key] for key in ("moving", "quiet", "other", "empty")}}


def _write_split(root, split, observed, samples):
    ordered = sorted(samples, key=lambda item: (item[2], observed[item[0]]["ref_time_us"]))
    anchors = [{"sample_token": current, "previous_token": history, "recording": recording}
               for current, history, recording, _, _, _ in ordered]
    rows = [{"sample_token": current, "tracks": tracks, "target_dt_s": dt}
            for current, _, _, _, tracks, dt in ordered]
    index = {"schema": INPUT_SCHEMA, "source_sha256": SOURCE_SHA256, "split": split,
             "anchors": anchors, "frames": observed}
    _validate_index(index)
    index_file = root / "inputs" / "index.json"
    write_json(index_file, index)
    digest = sha256(read_bounded(index_file, INDEX_LIMIT)).hexdigest()
    labels = {"schema": TARGET_SCHEMA, "input_sha256": digest, "rows": rows}
    _validate_targets(labels, index, digest)
    label_file = root / "labels" / "labels.json"
    label_file.parent.mkdir(parents=True)
    write_json(label_file, labels)
    summaries = [(a["sample_token"], cat, tracks, dt) for a, (_, _, _, cat, tracks, dt) in zip(anchors, ordered, strict=True)]
    return {**_sample_stats(summaries), "recordings": len({a["recording"] for a in anchors}),
            "observed_frames": len(observed),
            "moving_recordings": len({recording for _, _, recording, cat, _, _ in ordered if cat == "moving"}),
            "input_sha256": digest, "labels_sha256": sha256(read_bounded(label_file, LABEL_LIMIT)).hexdigest()}


def _read_index(root):
    raw = read_bounded(Path(root) / "index.json", INDEX_LIMIT)
    index = decode_json(raw)
    _validate_index(index)
    return raw, index


def _validate_index(index):
    require_keys(index, {"schema", "source_sha256", "split", "anchors", "frames"}, "Temporal VRU input")
    frames, anchors, split = index["frames"], index["anchors"], index["split"]
    if (index["schema"] != INPUT_SCHEMA or index["source_sha256"] != SOURCE_SHA256
            or not isinstance(split, str) or split not in _RECORDINGS
            or not isinstance(anchors, list) or not 1 <= len(anchors) <= len(_RECORDINGS[split]) * _RECORDING_CAP
            or not isinstance(frames, dict) or len(frames) != 2 * len(anchors)):
        raise ValueError("Unexpected Temporal VRU split or observation count")
    used, counts, images = set(), Counter(), {}
    last = None
    for anchor in anchors:
        require_keys(anchor, {"sample_token", "previous_token", "recording"}, "Temporal VRU anchor")
        current, history, recording = anchor["sample_token"], anchor["previous_token"], anchor["recording"]
        if not isinstance(recording, str) or recording not in _RECORDINGS[split]:
            raise ValueError("Temporal VRU anchor belongs to another split")
        stamps = []
        for token in (history, current):
            require_hex(token, 16, "Temporal VRU observation token")
            if token in used or token not in frames:
                raise ValueError("Temporal VRU windows share or miss an observation")
            used.add(token)
            stamps.append(_validate_frame(frames[token], token, recording, images))
        previous, now = stamps
        if (not _PAST_WINDOW_US[0] <= now - previous <= _PAST_WINDOW_US[1]
                or last is not None and (recording < last[0] or recording == last[0] and now - last[1] < _MIN_GAP_US)):
            raise ValueError("Temporal VRU cadence, chronological order, or window spacing is invalid")
        counts[recording] += 1
        last = recording, now
    if counts.keys() != _RECORDINGS[split] or any(n > _RECORDING_CAP for n in counts.values()):
        raise ValueError("Unexpected Temporal VRU recording counts")


def _validate_targets(targets, index, digest):
    require_keys(targets, {"schema", "input_sha256", "rows"}, "Temporal VRU labels")
    rows = targets["rows"]
    if (targets["schema"] != TARGET_SCHEMA or targets["input_sha256"] != digest
            or not isinstance(rows, list) or len(rows) != len(index["anchors"])):
        raise ValueError("Temporal VRU labels do not match the exact observation index")
    for row, anchor in zip(rows, index["anchors"], strict=True):
        require_keys(row, {"sample_token", "tracks", "target_dt_s"}, "Temporal VRU target row")
        tracks = row["tracks"]
        if row["sample_token"] != anchor["sample_token"] or not isinstance(tracks, list) or len(tracks) > 256:
            raise ValueError("Temporal VRU target row order or actor count is invalid")
        times = matrix(row["target_dt_s"], (2,), "Temporal VRU target times")
        if not _future_offsets_fit(*(float(value) * 1e6 for value in times)):
            raise ValueError("Temporal VRU target cadence is invalid")
        for track in tracks:
            require_keys(track, {"centers_xy_m"}, "Temporal VRU target track")
            positions = matrix(track["centers_xy_m"], (3, 2), "Temporal VRU track positions")
            if not all(_in_grid(xy) for xy in positions):
                raise ValueError("Temporal VRU track goes outside the labelled grid")
