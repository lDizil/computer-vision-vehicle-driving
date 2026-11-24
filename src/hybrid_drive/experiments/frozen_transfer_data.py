"""Prepare six-camera inputs and an evaluator-only selection ledger from pinned logs."""

from __future__ import annotations

from bisect import bisect_left
from hashlib import sha256
from pathlib import Path
import pickle
import sys
from tempfile import TemporaryDirectory

from hybrid_drive.assets.files import PreparationError, check_components
from hybrid_drive.contracts.sensors import timestamp_us
from hybrid_drive.experiments.bounded_io import canonical_json, decode_json, read_bounded, require_hex, require_keys
from hybrid_drive.experiments.box_motion_data import (
    CAMERAS,
    _LOG_PATTERN,
    _METADATA_LIMIT,
    _check_recording_groups,
    _chronological,
    _observation,
    _read_selected,
)
from hybrid_drive.experiments.io import write_json
from hybrid_drive.geometry.rigid import matrix, rigid_transform


SOURCES_SHA256 = "c5c1c8fb0f0fd00064611e14301b8106ae6f2cb3ff5d04cef7a359952d028d77"
INPUT_SCHEMA = "hybrid-drive-frozen-transfer-observations-v1"
SELECTION_SCHEMA = "hybrid-drive-frozen-transfer-selection-v1"
VULNERABLE_CLASSES = frozenset({"pedestrian", "bicycle"})
INDEX_LIMIT = 8 * 1024**2
_SOURCE_SCHEMA = "hybrid-drive-frozen-transfer-sources-v1"
_LOG_COUNT = 4
_UNIFORM_COUNT = 24
_VULNERABLE_CAP = 12
_MIN_ANCHOR_GAP_US = 4_000_000
_STEP_US = (250_000, 750_000)
_IMAGE_LIMIT = 2 * 1024**2
_CALIBRATION_SHAPES = {
    "intrinsic": (3, 3), "camera_to_ego_rotation": (3, 3),
    "camera_to_ego_translation": (3,), "distortion": (5,),
}


def load_transfer_source(source_path: Path) -> dict:
    source = decode_json(read_bounded(Path(source_path), 16 * 1024))
    _verify_source(source)
    _check_recording_groups(source)
    if len(source["logs"]) != _LOG_COUNT:
        raise PreparationError("Unexpected transfer recording count")
    if {str(log["camera_shard"]) for log in source["logs"]} != source["camera_archives"].keys():
        raise PreparationError("No pinned camera archive for a selected recording")
    return source


def read_transfer_recordings(metadata_archive: Path, source: dict):
    """Yield each recording's raw, chronologically sorted frames in isolated Linux."""
    if sys.platform != "linux":
        raise PreparationError("OpenScene pickle preparation requires the isolated Linux image")
    _verify_source(source)
    member_specs = {row["member"]: row for row in source["logs"]}
    seen_physical_ids = set()
    for member, data in _read_selected(metadata_archive, source["metadata_archive"], set(member_specs), _METADATA_LIMIT):
        spec = member_specs[member]
        # The member digest is checked again after decompression, before deserialization.
        if len(data) != spec["size_bytes"] or sha256(data).hexdigest() != spec["sha256"]:
            raise PreparationError("Transfer metadata member differs from its pinned SHA256 or size")
        frames = pickle.loads(data)
        if (not isinstance(frames, list) or not 2 <= len(frames) <= 4096
                or any(not isinstance(frame, dict) or frame.get("log_name") != spec["log_name"] for frame in frames)):
            raise PreparationError("Transfer metadata recording does not match its pinned log")
        frames = _chronological(frames)
        physical_id = frames[0]["log_token"]
        if physical_id in seen_physical_ids:
            raise PreparationError("One physical recording appears more than once in the transfer cohort")
        seen_physical_ids.add(physical_id)
        yield spec, frames


def prepare_transfer(metadata_archive: Path, camera_root: Path, source_path: Path, output: Path) -> dict:
    """Publish the observation pack and separate selection atomically; no outcome file."""
    source = load_transfer_source(source_path)
    camera_root, output = Path(camera_root), Path(output)
    check_components(output)
    if output.exists():
        raise PreparationError("Transfer output must not already exist")
    output.parent.mkdir(parents=True, exist_ok=True)

    observations, anchors, ledger, stats = {}, [], [], {}
    images = {shard: {} for shard in source["camera_archives"]}
    for spec, frames in read_transfer_recordings(metadata_archive, source):
        chosen, summary = _select_windows(frames)
        recording = spec["recording"]
        stats[recording] = summary
        shard = str(spec["camera_shard"])
        camera = source["camera_archives"][shard]
        for index, population in chosen:
            previous, current = frames[index - 1 : index + 1]
            anchors.append({"sample_token": current["token"], "recording": recording,
                            "history_tokens": [previous["token"]]})
            ledger.append({"sample_token": current["token"], "recording": recording,
                           "ref_time_us": int(current["timestamp"]), "frame_index": index, "population": population})
            for frame in (previous, current):
                token = frame["token"]
                if token in observations:
                    raise PreparationError("Selected transfer windows share an observed frame")
                observations[token] = _observation(
                    frame, recording, spec["log_name"], camera["archive_prefix"], images[shard],
                )
    ledger.sort(key=lambda row: (row["recording"], row["ref_time_us"]))
    anchors.sort(key=lambda row: (row["recording"], observations[row["sample_token"]]["ref_time_us"]))

    with TemporaryDirectory(dir=output.parent, prefix=".frozen-transfer-") as temp:
        staging = Path(temp) / "transfer"
        root = staging / "inputs"
        root.mkdir(parents=True)
        _extract_images(camera_root, source["camera_archives"], images, root)
        index = {"schema": INPUT_SCHEMA, "source_sha256": SOURCES_SHA256,
                 "anchors": anchors, "frames": observations}
        _validate_inputs(index)
        write_json(root / "index.json", index)
        input_sha = sha256(read_bounded(root / "index.json", INDEX_LIMIT)).hexdigest()
        evaluation = staging / "evaluation"
        evaluation.mkdir()
        write_json(evaluation / "selection.json", {"schema": SELECTION_SCHEMA, "source_sha256": SOURCES_SHA256,
                                                   "input_sha256": input_sha, "rows": ledger})
        check_components(output)
        if output.exists():
            raise PreparationError("Transfer output appeared during extraction")
        staging.rename(output)
    return {"anchors": len(anchors), "observed_frames": len(observations),
            "camera_jpegs": sum(map(len, images.values())), "input_sha256": input_sha, "recordings": stats}


def load_transfer_inputs(root: Path) -> dict:
    """Validate the observation index before inference or scoring, without opening RGB."""
    index = decode_json(read_bounded(Path(root) / "index.json", INDEX_LIMIT))
    _validate_inputs(index)
    return index


def _verify_source(source):
    if (not isinstance(source, dict) or source.get("schema") != _SOURCE_SCHEMA
            or sha256(canonical_json(source)).hexdigest() != SOURCES_SHA256):
        raise PreparationError("Transfer sources must match the checked-in digest")


def _select_windows(frames):
    stamps = [timestamp_us(frame["timestamp"], "OpenScene association stamp") for frame in frames]
    eligible = [i for i in range(1, len(frames)) if _valid_history(frames[i - 1], frames[i], stamps[i] - stamps[i - 1])]
    timeline = _spaced(eligible, stamps)
    if len(timeline) < _UNIFORM_COUNT:
        raise PreparationError(f"Need {_UNIFORM_COUNT} spaced observation windows; got {len(timeline)}")
    uniform = _timeline_sample(timeline, stamps, _UNIFORM_COUNT)
    uniform_stamps = [stamps[i] for i in uniform]

    challenge = []
    for i in eligible:
        t = stamps[i]
        slot = bisect_left(uniform_stamps, t)
        if any(abs(t - other) < _MIN_ANCHOR_GAP_US for other in uniform_stamps[max(0, slot - 1) : slot + 1]):
            continue
        if _current_vulnerable_in_corridor(frames[i]):
            challenge.append(i)
    spaced_challenge = _spaced(challenge, stamps)
    vulnerable = _timeline_sample(spaced_challenge, stamps, min(_VULNERABLE_CAP, len(spaced_challenge)))
    selected = sorted([(i, "uniform") for i in uniform] + [(i, "vulnerable") for i in vulnerable])
    return selected, {"eligible_windows": len(eligible), "uniform": len(uniform),
                      "eligible_vulnerable_windows_after_uniform": len(challenge), "vulnerable": len(vulnerable)}


def _valid_history(previous, current, gap):
    return (previous["log_token"] == current["log_token"] and _STEP_US[0] <= gap <= _STEP_US[1]
            and all(all(f.get("cams", {}).get(name, {}).get("data_path") for name in CAMERAS)
                    for f in (previous, current)))


def _spaced(indices, stamps):
    result = []
    for i in indices:
        if not result or stamps[i] - stamps[result[-1]] >= _MIN_ANCHOR_GAP_US:
            result.append(i)
    return result


def _timeline_sample(indices, stamps, count):
    if not count:
        return []
    times = [stamps[i] for i in indices]
    first, span = times[0], times[-1] - times[0]
    result, lo = [], 0
    for rank in range(count):
        target = first + (2 * rank + 1) * span // (2 * count)
        hi = len(times) - (count - rank - 1)
        slot = bisect_left(times, target, lo, hi)
        j = min(range(max(lo, slot - 1), min(hi, slot + 1)), key=lambda k: abs(times[k] - target))
        result.append(indices[j])
        lo = j + 1
    return result


def _current_vulnerable_in_corridor(frame):
    annotation = frame.get("anns") or {}
    names, boxes = annotation.get("gt_names", ()), annotation.get("gt_boxes", ())
    if len(names) != len(boxes):
        raise PreparationError("Current-frame class names and boxes are not aligned")
    for name, box in zip(names, boxes):
        if name in VULNERABLE_CLASSES:
            x, y = matrix(box, (7,), "current vulnerable box")[:2]
            if 4 <= x <= 32 and abs(y) <= 6:
                return True
    return False


def _extract_images(camera_root, archives, images, root):
    seen = {}
    for shard, spec in archives.items():
        selected = images[shard]
        for member, payload in _read_selected(camera_root / spec["filename"], spec, set(selected), _IMAGE_LIMIT):
            rel, view, recording = selected[member]
            if not payload.startswith(b"\xff\xd8\xff"):
                raise PreparationError("Expected a JPEG in the transfer camera archive")
            digest = sha256(payload).hexdigest()
            if seen.setdefault(digest, recording) != recording:
                raise PreparationError("JPEG bytes are shared by supposedly disjoint recordings")
            view["sha256"] = digest
            destination = root / rel
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)


def _validate_inputs(index):
    require_keys(index, {"schema", "source_sha256", "anchors", "frames"}, "Transfer observations")
    frames, anchors = index["frames"], index["anchors"]
    if (index["schema"] != INPUT_SCHEMA or index["source_sha256"] != SOURCES_SHA256
            or not isinstance(frames, dict) or not isinstance(anchors, list)
            or not _LOG_COUNT * _UNIFORM_COUNT <= len(anchors) <= _LOG_COUNT * (_UNIFORM_COUNT + _VULNERABLE_CAP)
            or len(frames) != 2 * len(anchors)):
        raise ValueError("Unexpected transfer observation identity or cohort size")
    referenced, last, counts, groups, images = set(), {}, {}, {}, {}
    for anchor in anchors:
        require_keys(anchor, {"sample_token", "recording", "history_tokens"}, "Transfer anchor")
        recording, token, history = anchor["recording"], anchor["sample_token"], anchor["history_tokens"]
        match = _LOG_PATTERN.fullmatch(f"{recording}_00000_00000")
        if match is None or groups.setdefault(match.groups(), recording) != recording:
            raise ValueError("Transfer recordings must be disjoint day/vehicle groups")
        if not isinstance(history, list) or len(history) != 1:
            raise ValueError("Transfer anchor must bind one earlier observation")
        prev = history[0]
        times = []
        for frame in (prev, token):
            require_hex(frame, 16, "Transfer observation token")
            if frame not in frames or frame in referenced:
                raise ValueError("Transfer windows cannot refer to missing or shared frames")
            referenced.add(frame)
            times.append(_validate_frame(frames[frame], frame, recording, images))
        if (not _STEP_US[0] <= times[1] - times[0] <= _STEP_US[1]
                or (recording in last and times[1] - last[recording] < _MIN_ANCHOR_GAP_US)):
            raise ValueError("Transfer observation cadence or anchor separation is invalid")
        last[recording] = times[1]
        counts[recording] = counts.get(recording, 0) + 1
    if (referenced != frames.keys() or len(counts) != _LOG_COUNT
            or any(not _UNIFORM_COUNT <= n <= _UNIFORM_COUNT + _VULNERABLE_CAP for n in counts.values())):
        raise ValueError("Unexpected per-recording sizes or unreferenced observations")


def _validate_frame(frame, token, recording, images):
    require_keys(frame, {"recording", "ref_time_us", "ego_to_world", "views"}, "Transfer observation")
    if frame["recording"] != recording:
        raise ValueError("Transfer observation belongs to another physical recording")
    rigid_transform(frame["ego_to_world"], "Transfer ego pose")
    views = frame["views"]
    if not isinstance(views, list) or len(views) != len(CAMERAS):
        raise ValueError("Transfer observation must carry six camera records")
    for camera, view in zip(CAMERAS, views):
        require_keys(view, {"camera", "path", "sha256", *_CALIBRATION_SHAPES}, "Transfer camera metadata")
        if view["camera"] != camera or view["path"] != f"rgb/{token}/{camera}.jpg":
            raise ValueError("Transfer JPEG path and camera order must bind the frame")
        digest = view["sha256"]
        require_hex(digest, 64, "Transfer JPEG SHA256")
        if images.setdefault(digest, recording) != recording:
            raise ValueError("Transfer observation index shares JPEGs across recordings")
        for key, shape in _CALIBRATION_SHAPES.items():
            matrix(view[key], shape, key)
    return timestamp_us(frame["ref_time_us"], "Transfer observation stamp")
