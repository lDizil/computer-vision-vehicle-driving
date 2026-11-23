"""Read one six-camera keyframe from the original nuScenes v1.0-mini layout."""

from __future__ import annotations

import io
import json
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image

from hybrid_drive.contracts._validation import identifier
from hybrid_drive.contracts.motion import EgoMotion
from hybrid_drive.contracts.sensors import CameraSnapshot, SurroundView, ViewDirection, timestamp_us
from hybrid_drive.geometry.camera import CameraCalibration
from hybrid_drive.geometry.rigid import invert_pose, pose_from_quaternion


_VERSION = "v1.0-mini"
_RIG_ID = "nuscenes-v1.0-mini"
_REFERENCE_CHANNEL = "LIDAR_TOP"
_DIRECTIONS = {
    "CAM_FRONT": ViewDirection.FRONT,
    "CAM_FRONT_LEFT": ViewDirection.FRONT_LEFT,
    "CAM_FRONT_RIGHT": ViewDirection.FRONT_RIGHT,
    "CAM_BACK_LEFT": ViewDirection.REAR_LEFT,
    "CAM_BACK_RIGHT": ViewDirection.REAR_RIGHT,
    "CAM_BACK": ViewDirection.REAR,
}
_MAX_TABLE_BYTES = 64 * 1024 * 1024
_MAX_JPEG_BYTES = 16 * 1024 * 1024
_MAX_IMAGE_PIXELS = 16_000_000


def load_nuscenes_surround(
    root: str | Path,
    sample_token: str,
    *,
    max_skew_us: int,
    motion: EgoMotion | None = None,
) -> CameraSnapshot:
    """Read six rectified JPEGs and put every calibration in one ego frame.

    The LIDAR_TOP keyframe pose, not its sensor extrinsic or point cloud, is
    the geometric reference. The exposure timestamps still describe separate
    instants; this does not time-warp dynamic objects or claim decision time.
    Neither annotations nor neighboring frames are used. EgoPose has no speed
    or acceleration; callers can supply separately measured motion at reference.
    """
    sample_token = identifier(sample_token, "sample_token")
    max_skew_us = timestamp_us(max_skew_us, "max_skew_us")
    if motion is not None and not isinstance(motion, EgoMotion):
        raise ValueError("motion must be measured EgoMotion or None")
    root = Path(root).resolve(strict=True)
    sample = _record(_index(root, "sample"), sample_token, "sample")
    timestamp_us(sample.get("timestamp"), "sample timestamp")  # Not a camera/LiDAR clock replacement.
    selected = _keyframes(root, sample_token)
    poses = _index(root, "ego_pose")
    reference_time, world_from_reference = _capture_pose(selected[_REFERENCE_CHANNEL][0], poses)
    captures = {
        channel: _capture_pose(selected[channel][0], poses)
        for channel in _DIRECTIONS
    }
    views = []
    for channel, direction in _DIRECTIONS.items():
        row, calibration = selected[channel]
        capture_time, world_from_ego_at_capture = captures[channel]
        ego_at_capture_from_camera = pose_from_quaternion(
            calibration.get("translation"), calibration.get("rotation"),
        )
        world_from_camera = world_from_ego_at_capture @ ego_at_capture_from_camera
        ego_to_camera = invert_pose(world_from_camera) @ world_from_reference
        rgb = _read_jpeg(root, row, channel)
        height, width = rgb.shape[:2]
        pinhole = CameraCalibration(
            channel, width, height, calibration.get("camera_intrinsic"), ego_to_camera,
        )
        views.append(SurroundView(direction, pinhole, rgb, capture_us=capture_time))
    return CameraSnapshot(
        sample_token, reference_time, world_from_reference, motion, views,
        rig_id=_RIG_ID, max_skew_us=max_skew_us,
    )


def _keyframes(root, sample_token):
    sensors = _index(root, "sensor")
    calibrations = _index(root, "calibrated_sensor")
    selected = {}
    expected = {*_DIRECTIONS, _REFERENCE_CHANNEL}
    for row in _index(root, "sample_data").values():
        if row.get("sample_token") != sample_token or row.get("is_key_frame") is False:
            continue
        if row.get("is_key_frame") is not True:
            raise ValueError("sample_data is_key_frame must be a boolean")
        calibration = _record(calibrations, row.get("calibrated_sensor_token"), "calibrated_sensor")
        sensor = _record(sensors, calibration.get("sensor_token"), "sensor")
        channel = identifier(sensor.get("channel"), "sensor channel")
        modality = sensor.get("modality")
        if channel not in expected:
            if modality == "camera":
                raise ValueError(f"Unexpected nuScenes camera channel {channel}")
            continue
        if modality != ("lidar" if channel == _REFERENCE_CHANNEL else "camera"):
            raise ValueError(f"Unexpected sensor modality for {channel}")
        if channel in selected:
            raise ValueError(f"Duplicate keyframe channel {channel}")
        selected[channel] = row, calibration
    if missing := expected - selected.keys():
        raise ValueError(f"Missing keyframe channels: {', '.join(sorted(missing))}")
    return selected


def _capture_pose(row, poses):
    capture_time = timestamp_us(row.get("timestamp"), "sample_data timestamp")
    pose = _record(poses, row.get("ego_pose_token"), "ego_pose")
    if timestamp_us(pose.get("timestamp"), "ego_pose timestamp") != capture_time:
        raise ValueError("ego_pose timestamp differs from its sample_data timestamp")
    return capture_time, pose_from_quaternion(pose.get("translation"), pose.get("rotation"))


def _read_jpeg(root, row, channel):
    width, height = row.get("width"), row.get("height")
    if (type(width) is not int or type(height) is not int
            or width <= 0 or height <= 0 or width * height > _MAX_IMAGE_PIXELS):
        raise ValueError(f"Invalid image dimensions for {channel}")
    if row.get("fileformat") != "jpg":
        raise ValueError(f"{channel} requires a nuScenes JPEG")
    filename = identifier(row.get("filename"), "image filename")
    relative = PurePosixPath(filename)
    if (len(relative.parts) != 3 or relative.parts[:2] != ("samples", channel)
            or relative.suffix != ".jpg" or relative.as_posix() != filename or "\\" in filename):
        raise ValueError(f"{channel} requires its own samples/<channel>/*.jpg path")
    data = _read_bounded(root, relative, _MAX_JPEG_BYTES)
    try:
        with Image.open(io.BytesIO(data), formats=("JPEG",)) as image:
            if image.mode != "RGB" or image.size != (width, height):
                raise ValueError(f"JPEG pixels do not match RGB metadata for {channel}")
            return np.array(image)
    except OSError as exc:
        raise ValueError(f"Invalid JPEG for {channel}") from exc


def _record(records, token, name):
    token = identifier(token, f"{name} token")
    try:
        return records[token]
    except KeyError as exc:
        raise ValueError(f"No {name} record for token {token!r}") from exc


def _index(root, name):
    raw = _read_bounded(root, Path(_VERSION, f"{name}.json"), _MAX_TABLE_BYTES)
    try:
        rows = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"Invalid nuScenes {name} JSON") from exc
    if not isinstance(rows, list):
        raise ValueError(f"nuScenes {name} must be a list of records")
    result = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"nuScenes {name} must contain only records")
        key = identifier(row.get("token"), f"{name} token")
        if key in result:
            raise ValueError(f"Duplicate {name} token {key!r}")
        result[key] = row
    return result


def _read_bounded(root, relative, size_limit):
    path = root / relative
    try:
        if path.resolve(strict=True) != path or not path.is_file():
            raise ValueError(f"Expected a regular file inside the extracted dataset: {relative}")
        with path.open("rb") as handle:
            data = handle.read(size_limit + 1)
    except OSError as exc:
        raise ValueError(f"Cannot read nuScenes file: {relative}") from exc
    if len(data) > size_limit:
        raise ValueError(f"nuScenes file exceeds the size limit: {relative}")
    return data
