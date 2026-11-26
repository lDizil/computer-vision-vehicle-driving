"""Prepare fixed observation-only windows for the VRU fusion and chain trials."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

from hybrid_drive.assets.files import PreparationError, check_components
from hybrid_drive.contracts.sensors import timestamp_us
from hybrid_drive.experiments.bounded_io import canonical_json, decode_json, read_bounded, require_hex, require_keys
from hybrid_drive.experiments.box_motion_data import CAMERAS, _check_recording_groups, _observation, _read_selected
from hybrid_drive.experiments.frozen_transfer_data import _validate_frame
from hybrid_drive.experiments.io import write_json
from hybrid_drive.experiments.temporal_vru_data import (
    INDEX_LIMIT,
    LABEL_LIMIT,
    TARGET_SCHEMA as TEMPORAL_TARGET_SCHEMA,
    _RECORDINGS as TEMPORAL_RECORDINGS,
    _choose_spaced,
    _label_window,
    _validate_targets,
)
from hybrid_drive.experiments.triple_vru_inputs import _third_status
from hybrid_drive.experiments.visual_vru_data import _read_recordings


SOURCE_SHA256 = "33fc31b846496be039c4fd3030e27ee9382589dc81ec12f1402b200148e82b08"
CHAIN_CONTROL_SOURCE_SHA256 = "20a000f8edf22c5aaf8e5eb4b4e2d04ce6961abfd8bfa9a2f4241c28e3099773"
INPUT_SCHEMA = "hybrid-drive-vru-fusion-holdout-inputs-v1"
TARGET_SCHEMA = "hybrid-drive-vru-fusion-holdout-targets-v1"
_SOURCE_SCHEMA = "hybrid-drive-vru-fusion-holdout-sources-v1"
_SHARDS = frozenset((3, 5, 22))
_RECORDINGS = frozenset((
    "2021.05.12.23.36.44_veh-35", "2021.06.09.14.58.55_veh-35", "2021.06.14.16.32.09_veh-35",
    "2021.06.14.19.22.11_veh-38", "2021.07.16.20.45.29_veh-35", "2021.10.11.07.12.18_veh-50",
))
_CONTROL_SHARDS = frozenset((11, 16))
_CONTROL_RECORDINGS = frozenset((
    "2021.06.09.14.03.17_veh-12", "2021.07.24.23.50.16_veh-17",
    "2021.06.14.18.42.45_veh-12", "2021.10.06.17.43.07_veh-28",
))
_CAPS = {"moving": 8, "quiet": 4, "empty": 2}
_RECORDING_CAP = sum(_CAPS.values())
_MIN_GAP_US = 3_000_000
_STEP_US = (400_000, 600_000)
_IMAGE_LIMIT = 2 * 1024**2


@dataclass(frozen=True)
class _SourceProfile:
    shards: frozenset[int]
    recordings: frozenset[str]
    previous_recordings: frozenset[str]


def prepare_fusion_holdout(metadata_archive: Path, camera_root: Path, source_path: Path, output: Path) -> dict:
    """Extract only selected observed JPEGs and publish labels on a separate path."""
    if sys.platform != "linux":
        raise PreparationError("Fusion holdout preparation requires the isolated Linux image")
    source = _load_source(source_path)
    output = Path(output)
    check_components(output)
    if output.exists():
        raise PreparationError("Fusion holdout output must not already exist")
    output.parent.mkdir(parents=True, exist_ok=True)

    wanted = {shard: {} for shard in source["camera_archives"]}
    index, rows, recordings = _collect_observations(metadata_archive, source, wanted)
    with TemporaryDirectory(dir=output.parent, prefix=".fusion-holdout-") as tmp:
        stage = Path(tmp) / "holdout"
        inputs = stage / "inputs"
        _extract_cameras(camera_root, source["camera_archives"], wanted, inputs)
        _validate_index(index)
        index_path = inputs / "index.json"
        write_json(index_path, index)
        digest = sha256(read_bounded(index_path, INDEX_LIMIT)).hexdigest()
        labels = {"schema": TARGET_SCHEMA, "input_sha256": digest, "rows": rows}
        _validate_labels(labels, index, digest)
        label_path = stage / "labels" / "labels.json"
        label_path.parent.mkdir()
        write_json(label_path, labels)
        check_components(output)
        if output.exists():
            raise PreparationError("Fusion holdout output appeared during preparation")
        stage.rename(output)
    return {"anchors": len(rows), "observed_frames": len(index["frames"]),
            "camera_jpegs": len(index["frames"]) * len(CAMERAS),
            "older_observations": sum(row["older_token"] is not None for row in index["anchors"]),
            "input_sha256": digest, "recordings": recordings}


def load_fusion_inputs(root: Path) -> dict:
    """Read the observation index without needing JPEGs or labels mounted."""
    return _read_index(root)[1]


def load_fusion_labels(input_root: Path, labels_path: Path) -> dict:
    """Read separately mounted labels bound to the exact observation index bytes."""
    raw, index = _read_index(input_root)
    labels = decode_json(read_bounded(Path(labels_path), LABEL_LIMIT))
    _validate_labels(labels, index, sha256(raw).hexdigest())
    return labels


def _load_source(source_path):
    source = decode_json(read_bounded(Path(source_path), 16 * 1024))
    if not isinstance(source, dict) or source.get("schema") != _SOURCE_SCHEMA:
        raise PreparationError("Fusion holdout sources must match the checked-in digest")
    try:
        profile = _source_profile(sha256(canonical_json(source)).hexdigest())
    except ValueError as exc:
        raise PreparationError("Fusion holdout sources must match the checked-in digest") from exc
    if set(source["previous_recordings"]) != profile.previous_recordings:
        raise PreparationError("Fusion holdout must exclude every prior VRU recording in its profile")
    _check_recording_groups(source)
    logs = source["logs"]
    if (source["camera_archives"].keys() != {str(shard) for shard in profile.shards}
            or Counter(row["camera_shard"] for row in logs) != {shard: 2 for shard in profile.shards}
            or {row["recording"] for row in logs} != profile.recordings):
        raise PreparationError("Unexpected pinned Fusion VRU recordings or camera shards")
    return source


def _source_profile(digest):
    prior = frozenset().union(*TEMPORAL_RECORDINGS.values())
    if digest == SOURCE_SHA256:
        return _SourceProfile(_SHARDS, _RECORDINGS, prior)
    if digest == CHAIN_CONTROL_SOURCE_SHA256:
        return _SourceProfile(_CONTROL_SHARDS, _CONTROL_RECORDINGS, prior | _RECORDINGS)
    raise ValueError("Expected a pinned Fusion VRU source digest")


def _collect_observations(metadata_archive, source, wanted):
    digest = sha256(canonical_json(source)).hexdigest()
    profile = _source_profile(digest)
    observed, recordings, samples, future_tokens = {}, {}, [], set()
    for spec, frames in _read_recordings(metadata_archive, source):
        recording, shard = spec["recording"], str(spec["camera_shard"])
        if recording in recordings:
            raise PreparationError("Duplicate Fusion VRU recording")
        selected, summary = _select_samples(frames)
        older_status = Counter()
        for i, tracks, dt in selected:
            history, current = frames[i - 1:i + 1]
            older = frames[i - 2] if i >= 2 else None
            status = _third_status(older, history)
            older_status[status] += 1
            if status != "ready":
                older = None
            for frame in (history, current) if older is None else (older, history, current):
                token = frame["token"]
                if token in observed:
                    raise PreparationError("Fusion VRU episodes share an observed token")
                observed[token] = _observation(
                    frame, recording, spec["log_name"], source["camera_archives"][shard]["archive_prefix"], wanted[shard],
                )
            future_tokens.update(frame["token"] for frame in frames[i + 1:i + 3])
            anchor = {"sample_token": current["token"], "previous_token": history["token"],
                      "older_token": None if older is None else older["token"], "recording": recording}
            samples.append((anchor, {"sample_token": current["token"], "tracks": tracks, "target_dt_s": dt}))
        recordings[recording] = {**summary, "older_statuses": dict(older_status)}
    if recordings.keys() != profile.recordings or observed.keys() & future_tokens:
        raise PreparationError("Fusion VRU is missing a pinned recording or contains a future observation")
    samples.sort(key=lambda pair: (pair[0]["recording"], observed[pair[0]["sample_token"]]["ref_time_us"]))
    anchors, rows = zip(*samples, strict=True)
    index = {"schema": INPUT_SCHEMA, "source_sha256": digest, "anchors": list(anchors), "frames": observed}
    return index, list(rows), recordings


def _select_samples(frames):
    stamps = [timestamp_us(frame["timestamp"], "OpenScene association stamp") for frame in frames]
    pools = {category: [] for category in _CAPS}
    targets = {}
    for i in range(1, len(frames) - 2):
        record = _label_window(frames[i - 1:i + 3], stamps[i - 1:i + 3])
        if record is not None and record[0] in pools:
            category, tracks, dt = record
            pools[category].append(i)
            targets[i] = (category, tracks, dt)
    chosen = []
    for category, cap in _CAPS.items():
        chosen.extend(_choose_spaced(pools[category], stamps, chosen, cap))
    if not chosen:
        raise PreparationError("No complete Fusion VRU windows in a pinned recording")
    counts = Counter(targets[i][0] for i in chosen)
    return [(i, *targets[i][1:]) for i in sorted(chosen)], {
        "available": {key: len(values) for key, values in pools.items()},
        "selected": {key: counts[key] for key in _CAPS},
    }


def _extract_cameras(camera_root, archives, wanted, inputs):
    images = set()
    for shard, spec in archives.items():
        selected = wanted[shard]
        for member, payload in _read_selected(Path(camera_root) / spec["filename"], spec, set(selected), _IMAGE_LIMIT):
            relative, view, _ = selected[member]
            digest = sha256(payload).hexdigest()
            if not payload.startswith(b"\xff\xd8\xff") or digest in images:
                raise PreparationError("Expected unique JPEG bytes for Fusion VRU observations")
            images.add(digest)
            view["sha256"] = digest
            path = inputs / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)


def _read_index(root):
    raw = read_bounded(Path(root) / "index.json", INDEX_LIMIT)
    index = decode_json(raw)
    _validate_index(index)
    return raw, index


def _validate_index(index):
    require_keys(index, {"schema", "source_sha256", "anchors", "frames"}, "Fusion VRU input")
    profile = _source_profile(index["source_sha256"])
    anchors, frames = index["anchors"], index["frames"]
    if (index["schema"] != INPUT_SCHEMA
            or not isinstance(anchors, list) or not 1 <= len(anchors) <= len(profile.recordings) * _RECORDING_CAP
            or not isinstance(frames, dict) or not 2 * len(anchors) <= len(frames) <= 3 * len(anchors)):
        raise ValueError("Unexpected Fusion VRU observation index shape or source")
    used, images, counts = set(), {}, Counter()
    prior = None
    for anchor in anchors:
        require_keys(anchor, {"sample_token", "previous_token", "older_token", "recording"}, "Fusion VRU anchor")
        older, previous, now, recording = (anchor[k] for k in ("older_token", "previous_token", "sample_token", "recording"))
        if not isinstance(recording, str) or recording not in profile.recordings:
            raise ValueError("Unexpected Fusion VRU recording")
        stamps = []
        for token in (previous, now) if older is None else (older, previous, now):
            require_hex(token, 16, "Fusion VRU observation token")
            if token in used or token not in frames:
                raise ValueError("Fusion VRU windows share or miss an observation")
            used.add(token)
            stamps.append(_validate_frame(frames[token], token, recording, images))
        stamp = stamps[-1]
        if (any(not _STEP_US[0] <= right - left <= _STEP_US[1] for left, right in zip(stamps, stamps[1:]))
                or prior is not None and (recording < prior[0] or recording == prior[0] and stamp - prior[1] < _MIN_GAP_US)):
            raise ValueError("Fusion VRU observation cadence, chronology or window spacing is invalid")
        counts[recording] += 1
        prior = recording, stamp
    if used != frames.keys() or len(images) != len(frames) * len(CAMERAS):
        raise ValueError("Fusion VRU contains unreferenced observations or duplicate JPEG bytes")
    if counts.keys() != profile.recordings or any(count > _RECORDING_CAP for count in counts.values()):
        raise ValueError("Unexpected Fusion VRU counts by recording")


def _validate_labels(labels, index, digest):
    require_keys(labels, {"schema", "input_sha256", "rows"}, "Fusion VRU labels")
    if labels["schema"] != TARGET_SCHEMA:
        raise ValueError("Unexpected Fusion VRU labels schema")
    _validate_targets({**labels, "schema": TEMPORAL_TARGET_SCHEMA}, index, digest)
