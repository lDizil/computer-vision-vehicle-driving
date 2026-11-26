"""Recover past annotations for the exact existing current/future targets."""

from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
from pathlib import Path
import sys

from hybrid_drive.assets.files import AtomicFile, PreparationError, check_components
from hybrid_drive.contracts.sensors import timestamp_us
from hybrid_drive.experiments.bounded_io import canonical_json, decode_json, read_bounded, require_keys
from hybrid_drive.experiments.box_motion_data import _pose
from hybrid_drive.experiments.frozen_transfer_contacts import _eligible_vulnerable, _tracked_boxes
from hybrid_drive.experiments.frozen_transfer_data import VULNERABLE_CLASSES
from hybrid_drive.experiments.temporal_vru_data import LABEL_LIMIT, _has_annotations, _in_grid, _label_window
from hybrid_drive.experiments.triple_vru_inputs import _check_base_metadata, _third_status
from hybrid_drive.experiments.visual_vru_data import _read_recordings
from hybrid_drive.experiments.vru_fusion_holdout_data import _load_source, _read_index, _validate_labels
from hybrid_drive.geometry.rigid import invert_pose, matrix


TRUTH_SCHEMA = "hybrid-drive-vru-chain-history-truth-v1"


def prepare_vru_chain_truth(metadata_archive: Path, sources_path: Path, input_root: Path,
                            labels_path: Path, output_path: Path) -> dict:
    """Publish a separate, diagnostic-only truth file without reading JPEGs."""
    if sys.platform != "linux":
        raise PreparationError("VRU history truth requires the isolated Linux image")
    output_path = Path(output_path)
    check_components(output_path)
    if output_path.exists():
        raise PreparationError("VRU history truth output must not already exist")
    index, labels, binding = _load_bound_labels(input_root, labels_path)
    source = _load_source(sources_path)
    if sha256(canonical_json(source)).hexdigest() != index["source_sha256"]:
        raise PreparationError("VRU history truth sources do not match the observation index")
    rows = _recover_rows(metadata_archive, source, index, labels)
    truth = {"schema": TRUTH_SCHEMA, **binding, "rows": rows}
    _validate_truth(truth, index, labels, binding)
    payload = canonical_json(truth) + b"\n"
    if len(payload) > LABEL_LIMIT:
        raise PreparationError("VRU history truth exceeds the artifact byte budget")
    with AtomicFile(output_path) as output:
        output.out.write(payload)
        output.commit()
    return {"windows": len(rows), "tracks": sum(len(row["tracks"]) for row in rows),
            "previous_centers": _count_centers(rows, "previous_xy_m"),
            "older_centers": _count_centers(rows, "older_xy_m"), "truth_sha256": sha256(payload).hexdigest(),
            **binding}


def load_vru_chain_truth(input_root: Path, labels_path: Path, truth_path: Path) -> dict:
    """Verify history truth is paired with both exact observation and label bytes."""
    index, labels, binding = _load_bound_labels(input_root, labels_path)
    truth = decode_json(read_bounded(Path(truth_path), LABEL_LIMIT))
    _validate_truth(truth, index, labels, binding)
    return truth


def _load_bound_labels(input_root, labels_path):
    raw, index = _read_index(input_root)
    label_bytes = read_bounded(Path(labels_path), LABEL_LIMIT)
    binding = {"input_sha256": sha256(raw).hexdigest(), "labels_sha256": sha256(label_bytes).hexdigest()}
    labels = decode_json(label_bytes)
    _validate_labels(labels, index, binding["input_sha256"])
    return index, labels["rows"], binding


def _recover_rows(metadata_archive, source, index, labels):
    by_recording = defaultdict(list)
    for i, anchor in enumerate(index["anchors"]):
        by_recording[anchor["recording"]].append((i, anchor))
    rows = [None] * len(labels)
    for spec, frames in _read_recordings(metadata_archive, source):
        recording = spec["recording"]
        wanted = by_recording.pop(recording, None)
        if wanted is None:
            raise PreparationError("Pinned metadata recording is not in the exact observation index")
        positions = {frame["token"]: i for i, frame in enumerate(frames)}
        prefix = source["camera_archives"][str(spec["camera_shard"])]["archive_prefix"]
        for position, anchor in wanted:
            i = positions.get(anchor["sample_token"])
            if i is None or not 1 <= i < len(frames) - 2:
                raise PreparationError("VRU history truth anchor is not a complete pinned window")
            rows[position] = _recover_window(anchor, labels[position], index, frames, i, spec["log_name"], prefix)
    if by_recording:
        raise PreparationError("Missing pinned metadata recordings for VRU history truth")
    return rows


def _recover_window(anchor, label, index, frames, i, log_name, prefix):
    window = frames[i - 1:i + 3]
    previous, current = window[:2]
    older = frames[i - 2] if i >= 2 else None
    if _third_status(older, previous) != "ready":
        older = None
    older_token = older["token"] if older is not None else None
    if (previous["token"], older_token) != (anchor["previous_token"], anchor["older_token"]):
        raise PreparationError("VRU history truth does not match the exact historical frames")
    for frame in (previous, current) if older is None else (older, previous, current):
        _check_base_metadata(frame, index["frames"][frame["token"]], anchor["recording"], log_name, prefix)
    stamps = [timestamp_us(frame["timestamp"], "OpenScene history truth stamp") for frame in window]
    recovered = _label_window(window, stamps)
    if recovered is None or recovered[1:] != (label["tracks"], label["target_dt_s"]):
        raise PreparationError("VRU history truth does not reproduce the frozen current and future targets")

    ordered = _ordered_reference_tracks(current, label["tracks"])
    anchor_from_world = invert_pose(_pose(current))
    previous_status, previous_centers = _history_centers(previous, anchor_from_world)
    older_status, older_centers = _history_centers(older, anchor_from_world)
    tracks = [{"previous_xy_m": _center(previous_centers, token, kind),
               "older_xy_m": _center(older_centers, token, kind)} for token, kind in ordered]
    return {"sample_token": anchor["sample_token"], "previous_token": previous["token"], "older_token": older_token,
            "previous_status": previous_status, "older_status": older_status, "tracks": tracks}


def _ordered_reference_tracks(current, labels):
    boxes = {token: (kind, box) for token, (kind, box) in _tracked_boxes(current["anns"]).items()
             if kind in VULNERABLE_CLASSES and _in_grid(box[:2])}
    tracks = sorted(_eligible_vulnerable(current, boxes).items())
    if (len(tracks) != len(labels) or
            any(boxes[token][1][:2].tolist() != label["centers_xy_m"][0]
                for (token, _), label in zip(tracks, labels, strict=True))):
        raise PreparationError("VRU history truth cannot recover the ordered label tracks")
    return tracks


def _history_centers(frame, anchor_from_world):
    if frame is None:
        return "missing_frame", {}
    if not _has_annotations(frame):
        return "missing_annotations", {}
    relative = anchor_from_world @ _pose(frame)
    centers = {token: (kind, (relative[:3, :3] @ box[:3] + relative[:3, 3])[:2].tolist())
               for token, (kind, box) in _tracked_boxes(frame["anns"]).items() if kind in VULNERABLE_CLASSES}
    return "available", centers


def _center(centers, token, kind):
    entry = centers.get(token)
    return entry[1] if entry is not None and entry[0] == kind else None


def _validate_truth(truth, index, labels, binding):
    require_keys(truth, {"schema", "input_sha256", "labels_sha256", "rows"}, "VRU history truth")
    rows = truth["rows"]
    if (truth["schema"] != TRUTH_SCHEMA or any(truth[key] != val for key, val in binding.items())
            or not isinstance(rows, list) or len(rows) != len(labels)):
        raise ValueError("VRU history truth does not match the exact observation index and labels")
    for row, anchor, targets in zip(rows, index["anchors"], labels, strict=True):
        require_keys(row, {"sample_token", "previous_token", "older_token", "tracks", "previous_status",
                           "older_status"}, "VRU history truth row")
        if (any(row[key] != anchor[key] for key in ("sample_token", "previous_token", "older_token"))
                or not isinstance(row["tracks"], list) or len(row["tracks"]) != len(targets["tracks"])):
            raise ValueError("VRU history truth row does not match the exact annotation order or frame tokens")
        for side in ("previous", "older"):
            status = row[side + "_status"]
            valid = ("missing_frame",) if row[side + "_token"] is None else ("available", "missing_annotations")
            if status not in valid:
                raise ValueError("Invalid VRU history truth annotation status")
        for track in row["tracks"]:
            require_keys(track, {"previous_xy_m", "older_xy_m"}, "VRU history truth track")
            for side in ("previous", "older"):
                xy = track[side + "_xy_m"]
                if xy is not None:
                    if row[side + "_status"] != "available":
                        raise ValueError("VRU history truth cannot have a center without annotations")
                    matrix(xy, (2,), "VRU history truth center")


def _count_centers(rows, field):
    return sum(track[field] is not None for row in rows for track in row["tracks"])
