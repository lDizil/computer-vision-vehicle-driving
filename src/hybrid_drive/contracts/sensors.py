"""A six-view input in the shared metric reference frame, independent of models."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from hashlib import sha256
import json
from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike, NDArray

from hybrid_drive.geometry.camera import CameraCalibration
from hybrid_drive.geometry.rigid import rigid_transform

from ._validation import identifier, tuple_snapshot
from .motion import EgoMotion


class ViewDirection(str, Enum):
    FRONT = "front"
    FRONT_LEFT = "front_left"
    FRONT_RIGHT = "front_right"
    REAR_LEFT = "rear_left"
    REAR_RIGHT = "rear_right"
    REAR = "rear"


SURROUND_ORDER = tuple(ViewDirection)


class TimingBasis(str, Enum):
    CAMERA_CAPTURE = "sensor_capture"
    SOURCE_ASSOCIATION = "source_association"


@dataclass(frozen=True, eq=False, init=False)
class SurroundView:
    """Rectified RGB whose calibration maps reference ego points to this camera."""

    direction: ViewDirection
    calibration: CameraCalibration
    capture_us: int | None
    pixels_sha256: str
    _pixels: bytes = field(repr=False)

    def __init__(self, direction: ViewDirection, calibration: CameraCalibration, rgb: NDArray[np.uint8],
                 *, capture_us: int | None = None):
        if not isinstance(direction, ViewDirection) or not isinstance(calibration, CameraCalibration):
            raise ValueError("SurroundView requires a ViewDirection and CameraCalibration")
        height, width = calibration.height, calibration.width
        if width * height > 16_000_000:
            raise ValueError("Surround camera image exceeds 16 megapixels")
        if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.shape != (height, width, 3):
            raise ValueError("Rectified RGB must be uint8 with shape matching calibration")
        pixels = rgb.tobytes(order="C")
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "calibration", calibration)
        object.__setattr__(self, "capture_us", None if capture_us is None else timestamp_us(capture_us, "capture_us"))
        object.__setattr__(self, "pixels_sha256", sha256(pixels).hexdigest())
        object.__setattr__(self, "_pixels", pixels)

    @property
    def rgb(self) -> NDArray[np.uint8]:
        return np.frombuffer(self._pixels, np.uint8).reshape(self.calibration.height, self.calibration.width, 3)


@dataclass(frozen=True, eq=False, init=False)
class CameraSnapshot:
    """One canonical six-view snapshot, not proof of simultaneous exposures.

    The validated skew includes the reference time. An association-only source
    has no per-view clocks and cannot pass a measured skew requirement.
    """

    snapshot_id: str
    rig_id: str
    ref_time_us: int
    motion: EgoMotion | None
    views: tuple[SurroundView, ...]
    timing: TimingBasis
    measured_skew_us: int | None
    max_skew_us: int | None
    content_sha256: str
    _ego_to_world: bytes = field(repr=False)

    def __init__(self, snapshot_id: str, ref_time_us: int, ego_to_world: ArrayLike,
                 motion: EgoMotion | None, views, *, rig_id: str, max_skew_us: int | None = None):
        snapshot_id = identifier(snapshot_id, "snapshot_id")
        rig_id = identifier(rig_id, "rig_id")
        reference = timestamp_us(ref_time_us, "ref_time_us")
        pose = rigid_transform(ego_to_world, "ego_to_world")
        if motion is not None and not isinstance(motion, EgoMotion):
            raise ValueError("motion must be EgoMotion or unknown (None)")
        ordered = _ordered_views(views)
        timing, measured, permitted = _timing(reference, ordered, max_skew_us)
        content = _snapshot_hash(snapshot_id, rig_id, reference, pose, motion, ordered)
        for name, value in (
            ("snapshot_id", snapshot_id), ("rig_id", rig_id), ("ref_time_us", reference),
            ("motion", motion), ("views", ordered), ("timing", timing),
            ("measured_skew_us", measured), ("max_skew_us", permitted),
            ("content_sha256", content), ("_ego_to_world", pose.tobytes()),
        ):
            object.__setattr__(self, name, value)

    @property
    def ego_to_world(self) -> NDArray[np.float64]:
        return np.frombuffer(self._ego_to_world, np.float64).reshape(4, 4)


def _ordered_views(views):
    views = tuple_snapshot(views, "views")
    if len(views) != len(SURROUND_ORDER) or any(not isinstance(v, SurroundView) for v in views):
        raise ValueError("Expected exactly six calibrated surround views")
    by_direction = {v.direction: v for v in views}
    if set(by_direction) != set(SURROUND_ORDER) or len({v.calibration.name for v in views}) != len(views):
        raise ValueError("Surround direction and native camera name must each be unique")
    return tuple(by_direction[direction] for direction in SURROUND_ORDER)


def _timing(reference, views, limit):
    times = [view.capture_us for view in views if view.capture_us is not None]
    if not times:
        if limit is not None:
            raise ValueError("Cannot require capture skew with association-only timestamps")
        return TimingBasis.SOURCE_ASSOCIATION, None, None
    if len(times) != len(views):
        raise ValueError("Cannot mix measured and missing camera capture times")
    if limit is None:
        raise ValueError("Measured capture times require an explicit max_skew_us")
    limit = timestamp_us(limit, "max_skew_us")
    span = max(reference, *times) - min(reference, *times)
    if span > limit:
        raise ValueError(f"Camera/reference skew {span} us exceeds {limit} us")
    return TimingBasis.CAMERA_CAPTURE, span, limit


def _snapshot_hash(snapshot_id, rig_id, reference, pose, motion, views):
    payload = {
        "schema": "surround-snapshot-v1",
        "snapshot_id": snapshot_id,
        "rig_id": rig_id,
        "ref_time_us": reference,
        "ego_to_world": pose.tolist(),
        "motion": None if motion is None else vars(motion),
        "views": [{
            "direction": view.direction.value,
            "name": view.calibration.name,
            "size": [view.calibration.width, view.calibration.height],
            "intrinsic": view.calibration.intrinsic.tolist(),
            "ego_to_camera": view.calibration.ego_to_camera.tolist(),
            "near_m": view.calibration.near_m,
            "capture_us": view.capture_us,
            "pixels_sha256": view.pixels_sha256,
        } for view in views],
    }
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def timestamp_us(value, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or not 0 <= value < 2**63:
        raise ValueError(f"{name} must be a nonnegative integer in microseconds")
    return int(value)
