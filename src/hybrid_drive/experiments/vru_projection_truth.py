"""Project the pinned current VRU annotations into the detector camera rasters."""

from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
from itertools import product
from math import cos, sin
from pathlib import Path
import sys

import numpy as np

from hybrid_drive.adapters.real_camera import _openscene_calibrations
from hybrid_drive.assets.files import AtomicFile, PreparationError, check_components
from hybrid_drive.contracts.sensors import timestamp_us
from hybrid_drive.experiments.bounded_io import canonical_json, decode_json, read_bounded, require_keys
from hybrid_drive.experiments.frozen_transfer_contacts import _tracked_boxes
from hybrid_drive.experiments.temporal_vru_data import LABEL_LIMIT, _label_window
from hybrid_drive.experiments.triple_vru_inputs import _check_base_metadata
from hybrid_drive.experiments.visual_vru_data import _read_recordings
from hybrid_drive.experiments.vru_chain_truth import _load_bound_labels, _ordered_reference_tracks
from hybrid_drive.experiments.vru_fusion_holdout_data import _load_source
from hybrid_drive.geometry.rigid import matrix


TRUTH_SCHEMA = "hybrid-drive-vru-projection-truth-v1"
RECTIFIED_WH = (768, 432)
_NATIVE_WH = (1920, 1080)
_CLASS_MAP = {"pedestrian": "person", "bicycle": "bicycle"}
_CORNERS = np.array(tuple(product((-0.5, 0.5), repeat=3)))
_EDGES = tuple((i, i ^ bit) for i in range(8) for bit in (1, 2, 4) if i < (i ^ bit))


def prepare_projection_truth(metadata_archive: Path, sources_path: Path, input_root: Path,
                             labels_path: Path, output_path: Path) -> dict:
    """Recover oriented boxes and approximate 2D envelopes, not RGB visibility."""
    if sys.platform != "linux":
        raise PreparationError("VRU projection truth requires the isolated Linux image")
    output_path = Path(output_path)
    check_components(output_path)
    if output_path.exists():
        raise PreparationError("VRU projection truth output must not already exist")
    index, labels, binding = _load_bound_labels(input_root, labels_path)
    source = _load_source(sources_path)
    if sha256(canonical_json(source)).hexdigest() != index["source_sha256"]:
        raise PreparationError("VRU projection truth source does not match the observation index")
    rows = _recover_rows(metadata_archive, source, index, labels)
    truth = {"schema": TRUTH_SCHEMA, **binding, "rows": rows}
    _validate_truth(truth, index, labels, binding)
    payload = canonical_json(truth) + b"\n"
    if len(payload) > LABEL_LIMIT:
        raise PreparationError("VRU projection truth exceeds the artifact byte budget")
    with AtomicFile(output_path) as output:
        output.out.write(payload)
        output.commit()
    targets = [target for row in rows for target in row["targets"]]
    return {"windows": len(rows), "targets": len(targets),
            "camera_envelopes": sum(len(target["views"]) for target in targets),
            "truth_sha256": sha256(payload).hexdigest(), **binding}


def load_projection_truth(input_root: Path, labels_path: Path, path: Path) -> dict:
    """Load the proxy artifact bound to these exact index and label bytes."""
    index, labels, binding = _load_bound_labels(input_root, labels_path)
    truth = decode_json(read_bounded(Path(path), LABEL_LIMIT))
    _validate_truth(truth, index, labels, binding)
    return truth


def _recover_rows(metadata_archive, source, index, labels):
    anchors_by_recording = defaultdict(list)
    for position, anchor in enumerate(index["anchors"]):
        anchors_by_recording[anchor["recording"]].append((position, anchor))
    rows = [None] * len(labels)
    for spec, frames in _read_recordings(metadata_archive, source):
        tasks = anchors_by_recording.pop(spec["recording"], None)
        if tasks is None:
            raise PreparationError("Pinned metadata recording is not in the exact observation index")
        positions = {frame["token"]: i for i, frame in enumerate(frames)}
        prefix = source["camera_archives"][str(spec["camera_shard"])]["archive_prefix"]
        for position, anchor in tasks:
            token = anchor["sample_token"]
            i = positions.get(token)
            if i is None or not 1 <= i < len(frames) - 2 or frames[i - 1]["token"] != anchor["previous_token"]:
                raise PreparationError("VRU projection truth anchor is not a complete pinned window")
            rows[position] = _recover_window(anchor, labels[position], index["frames"][token], frames[i - 1:i + 3],
                                             spec["log_name"], prefix)
    if anchors_by_recording:
        raise PreparationError("Missing pinned metadata recordings for VRU projection truth")
    return rows


def _recover_window(anchor, label, entry, window, log_name, prefix):
    current = window[1]
    _check_base_metadata(current, entry, anchor["recording"], log_name, prefix)
    stamps = [timestamp_us(frame["timestamp"], "OpenScene projection truth stamp") for frame in window]
    recovered = _label_window(window, stamps)
    if recovered is None or recovered[1:] != (label["tracks"], label["target_dt_s"]):
        raise PreparationError("VRU projection truth does not reproduce the frozen current and future targets")
    ordered = _ordered_reference_tracks(current, label["tracks"])
    boxes = _tracked_boxes(current["anns"])
    calibrations = tuple(_calibration(view) for view in entry["views"])
    targets = [_project_target(_CLASS_MAP.get(kind, "unsupported"), boxes[token][1], calibrations)
               for token, kind in ordered]
    return {"sample_token": anchor["sample_token"], "targets": targets}


def _calibration(view):
    _, resized, _ = _openscene_calibrations(
        view["camera"], *_NATIVE_WH, view["intrinsic"], view["camera_to_ego_rotation"],
        view["camera_to_ego_translation"], view["distortion"], *RECTIFIED_WH,
    )
    return resized


def _project_target(kind, box, calibrations):
    c, s = cos(box[6]), sin(box[6])
    rotation = np.array(((c, -s, 0.), (s, c, 0.), (0., 0., 1.)))
    corners = (_CORNERS * box[3:6]) @ rotation.T + box[:3]
    views = []
    for camera in calibrations:
        envelope = _project_envelope(corners, camera)
        if envelope is not None:
            views.append({"camera_name": camera.name, "xyxy": envelope})
    return {"kind": kind, "box_ego": box.tolist(), "views": views}


def _project_envelope(corners, camera):
    transform = camera.ego_to_camera
    camera_points = corners @ transform[:3, :3].T + transform[:3, 3]
    near = camera.near_m
    front = camera_points[:, 2] >= near
    clipped = list(camera_points[front])
    for i, j in _EDGES:
        if front[i] == front[j]:
            continue
        a, b = camera_points[i], camera_points[j]
        alpha = (near - a[2]) / (b[2] - a[2])
        crossing = a + alpha * (b - a)
        crossing[2] = near
        clipped.append(crossing)
    if len(clipped) < 3:
        return None
    homogeneous = np.stack(clipped) @ camera.intrinsic.T
    pixels = homogeneous[:, :2] / homogeneous[:, 2, None]
    if not np.isfinite(pixels).all() or np.max(np.abs(pixels)) > 1e12:
        raise ValueError("Cuboid projection exceeds the finite pixel range")

    # Clip the projected silhouette before its AABB: clamping the AABB alone
    # overstates coverage when only an empty corner touches the viewport.
    polygon = _convex_hull(pixels)
    for axis, boundary, direction in ((0, 0, 1), (0, camera.width - 1, -1),
                                      (1, 0, 1), (1, camera.height - 1, -1)):
        polygon = _clip_polygon(polygon, axis, boundary, direction)
        if len(polygon) < 3:
            return None
    low_x, low_y = np.min(polygon, axis=0)
    high_x, high_y = np.max(polygon, axis=0)
    if low_x >= high_x or low_y >= high_y:
        return None
    return [float(v) for v in (low_x, low_y, high_x, high_y)]


def _convex_hull(points):
    points = sorted({tuple(row) for row in points})

    def half(sequence):
        result = []
        for point in sequence:
            while len(result) >= 2:
                a, b = result[-2:]
                turn = (b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0])
                if turn > 0:
                    break
                result.pop()
            result.append(point)
        return result

    return half(points)[:-1] + half(reversed(points))[:-1]


def _clip_polygon(polygon, axis, boundary, direction):
    result = []
    for a, b in zip(polygon[-1:] + polygon[:-1], polygon, strict=True):
        inside_a = direction * (a[axis] - boundary) >= 0
        inside_b = direction * (b[axis] - boundary) >= 0
        if inside_a != inside_b:
            other = 1 - axis
            alpha = (boundary - a[axis]) / (b[axis] - a[axis])
            across = a[other] + alpha * (b[other] - a[other])
            result.append((boundary, across) if axis == 0 else (across, boundary))
        if inside_b:
            result.append(b)
    return result


def _validate_truth(truth, index, labels, binding):
    require_keys(truth, {"schema", "input_sha256", "labels_sha256", "rows"}, "VRU projection truth")
    rows = truth["rows"]
    if (truth["schema"] != TRUTH_SCHEMA or any(truth[k] != value for k, value in binding.items())
            or not isinstance(rows, list) or len(rows) != len(labels)):
        raise ValueError("VRU projection truth does not match the exact observation index and labels")
    for row, anchor, label in zip(rows, index["anchors"], labels, strict=True):
        require_keys(row, {"sample_token", "targets"}, "VRU projection truth row")
        token = anchor["sample_token"]
        if (row["sample_token"] != token or not isinstance(row["targets"], list)
                or len(row["targets"]) != len(label["tracks"])):
            raise ValueError("VRU projection truth row does not match the exact annotation order or frame")
        latest = index["frames"][token]
        ranks = {view["camera"]: i for i, view in enumerate(latest["views"])}
        for target, track in zip(row["targets"], label["tracks"], strict=True):
            _validate_target(target, track, ranks)


def _validate_target(target, track, ranks):
    require_keys(target, {"kind", "box_ego", "views"}, "VRU projection truth target")
    box = matrix(target["box_ego"], (7,), "VRU projection truth box")
    if np.any(box[3:6] <= 0.) or box[:2].tolist() != track["centers_xy_m"][0]:
        raise ValueError("VRU projection truth box must match the ordered center and have positive dimensions")
    if target["kind"] not in ("person", "bicycle", "unsupported") or not isinstance(target["views"], list):
        raise ValueError("Invalid VRU projection truth target kind or views")
    last_rank = -1
    width, height = RECTIFIED_WH
    for view in target["views"]:
        require_keys(view, {"camera_name", "xyxy"}, "VRU projection truth view")
        name = view["camera_name"]
        rank = ranks.get(name) if isinstance(name, str) else None
        if rank is None or rank <= last_rank:
            raise ValueError("VRU projection truth views must be unique and ordered by the six-camera rig")
        last_rank = rank
        left, top, right, bottom = matrix(view["xyxy"], (4,), "VRU projection truth envelope")
        if not (0 <= left < right <= width - 1 and 0 <= top < bottom <= height - 1):
            raise ValueError("VRU projection truth envelope must be inside the rectified viewport")
