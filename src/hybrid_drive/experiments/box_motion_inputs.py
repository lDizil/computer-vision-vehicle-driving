"""Read only the observation pack for the independent OpenScene motion test."""

from __future__ import annotations

from hashlib import sha256
from io import BytesIO
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image

from hybrid_drive.adapters.real_camera import _rectify_native_openscene_rgb
from hybrid_drive.contracts.sensors import CameraSnapshot, SURROUND_ORDER, SurroundView, timestamp_us
from hybrid_drive.experiments.bounded_io import require_hex, read_bounded
from hybrid_drive.experiments.fast_capture import load_json


CAMERAS = ("CAM_F0", "CAM_L0", "CAM_R0", "CAM_L2", "CAM_R2", "CAM_B0")


def load_motion_inputs(root: Path) -> dict:
    index = load_json(root / "index.json")
    if index.get("schema") != "hybrid-drive-box-motion-inputs-v1" or not isinstance(index.get("source"), dict):
        raise ValueError("Unexpected observation pack")
    frames = index["frames"]
    anchors = index["anchors"]
    if not 1 <= len(anchors) <= 80 or not len(anchors) <= len(frames) <= 320:
        raise ValueError("Unexpected holdout size")
    referenced = set()
    for anchor in anchors:
        history = anchor["history_tokens"]
        current = anchor["sample_token"]
        span = [*history, current]
        if len(span) != 4 or len(set(span)) != 4 or any(k not in frames for k in span):
            raise ValueError("Expected three earlier observations and a current frame")
        if any(frames[token]["recording"] != anchor["recording"] for token in span):
            raise ValueError("Cannot mix recording groups in one observation window")
        times = [timestamp_us(frames[token]["ref_time_us"], "ref_time_us") for token in span]
        if any(not 250_000 <= right-left <= 1_000_000 for left,right in zip(times,times[1:])):
            raise ValueError("Observation windows must have four ordered, nearby frames")
        referenced.update(span)
    if referenced != set(frames) or len({a["sample_token"] for a in anchors}) != len(anchors):
        raise ValueError("Duplicate anchors or unreferenced snapshots")
    return index


def load_motion_snapshot(root: Path, token: str, entry: dict, *, rectified_wh=(384, 216)) -> CameraSnapshot:
    if (not isinstance(rectified_wh, tuple) or len(rectified_wh) != 2
            or any(type(n) is not int or not 4 <= n <= native for n, native in zip(rectified_wh, (1920, 1080)))):
        raise ValueError("Rectified width/height must fit the native OpenScene camera")
    views = entry["views"]
    if tuple(view["camera"] for view in views) != CAMERAS:
        raise ValueError("OpenScene six-camera order changed")
    result = []
    for direction, view in zip(SURROUND_ORDER, views, strict=True):
        path = _safe_image(root, view["path"])
        digest = view["sha256"]
        require_hex(digest, 64, "JPEG SHA256")
        body = read_bounded(path, 2 * 1024 * 1024)
        if sha256(body).hexdigest() != digest:
            raise ValueError("Holdout JPEG changed after preparation")
        with Image.open(BytesIO(body), formats=("JPEG",)) as file:
            if file.mode != "RGB" or file.size != (1920,1080):
                raise ValueError("Unexpected native OpenScene JPEG")
            rgb = np.asarray(file)
        pixels, calib = _rectify_native_openscene_rgb(
            view["camera"], rgb, view["intrinsic"], view["camera_to_ego_rotation"],
            view["camera_to_ego_translation"], view["distortion"], *rectified_wh,
        )
        result.append(SurroundView(direction, calib, pixels))
    return CameraSnapshot(token, entry["ref_time_us"], entry["ego_to_world"], None, result,
                          rig_id="openscene-v1.1-mini-six-selected-v1")


def _safe_image(root, value):
    rel = PurePosixPath(value)
    if (rel.is_absolute() or len(rel.parts) < 2 or rel.as_posix() != value
            or ".." in rel.parts or "\\" in value or rel.suffix != ".jpg"):
        raise ValueError("Expected a confined JPEG path")
    return root.joinpath(*rel.parts)
