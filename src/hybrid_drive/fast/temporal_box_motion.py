"""Conservative offline velocity baseline from three frames of predicted boxes."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

from hybrid_drive.contracts._validation import identifier, tuple_snapshot
from hybrid_drive.contracts.sensors import timestamp_us
from hybrid_drive.fast.forecast_proposals import ForecastObject
from hybrid_drive.geometry.rigid import rigid_transform


_MIN_GAP_US = 250_000
_MAX_GAP_US = 1_000_000
_MAX_BOXES = 128
_MAX_SPEED_MPS = 28.0
_CENTER_SLACK_M = 1.0
_MAX_HEIGHT_DELTA_M = 1.25
_MAX_LOG_SIZE_RATIO = math.log(1.85)
_MIN_BOX_AXIS_COS = 0.5  # Axial, so a 180-degree heading flip does not break a match.
_UNIQUENESS_GAP_M = 0.65
_QUIET_SPEED_MPS = 1.0
_MAX_SPEED_DISAGREEMENT_MPS = 2.5  # Euclidean difference of the two velocity vectors.
_MIN_VELOCITY_COS = math.cos(math.radians(35.0))
_ZERO = (0.0, 0.0)


@dataclass(frozen=True, eq=False, init=False)
class MotionFrame:
    """Predicted boxes in a recording's local ego axes at one geometric anchor."""

    recording: str
    ref_time_us: int
    boxes: tuple[ForecastObject, ...]
    _pose_data: bytes = field(repr=False)

    def __init__(self, recording: str, ref_time_us: int, ego_to_world: ArrayLike,
                 boxes: Sequence[ForecastObject]):
        recording = identifier(recording, "recording")
        ref_time_us = timestamp_us(ref_time_us, "ref_time_us")
        pose = rigid_transform(ego_to_world, "ego_to_world")
        boxes = tuple_snapshot(boxes, "boxes")
        if len(boxes) > _MAX_BOXES or any(not isinstance(box, ForecastObject) for box in boxes):
            raise ValueError("boxes must contain at most 128 ForecastObject values per frame")
        if len({box.object_id for box in boxes}) != len(boxes):
            raise ValueError("MotionFrame requires unique box IDs within each frame")
        object.__setattr__(self, "recording", recording)
        object.__setattr__(self, "ref_time_us", ref_time_us)
        object.__setattr__(self, "boxes", boxes)
        object.__setattr__(self, "_pose_data", pose.tobytes())

    @property
    def ego_to_world(self) -> NDArray[np.float64]:
        return np.frombuffer(self._pose_data, dtype=np.float64).reshape(4, 4)


@dataclass(frozen=True)
class MotionDiagnostic:
    object_id: str
    status: str
    previous_object_id: str | None = None
    earlier_object_id: str | None = None
    recent_velocity_xy: tuple[float, float] | None = None
    prior_velocity_xy: tuple[float, float] | None = None


@dataclass(frozen=True)
class MotionEstimate:
    objects: tuple[ForecastObject, ...]
    diagnostics: tuple[MotionDiagnostic, ...]


def associate_box_motion(current: MotionFrame, previous: MotionFrame, earlier: MotionFrame) -> MotionEstimate:
    """Replace learned speeds using only two unique adjacent matches, else zero.

    Historic centers and heading axes are transformed into current ego axes
    before matching. Box IDs are within-frame identifiers, not track IDs. The
    stationary result is an uncertain-motion baseline, not a safety verdict.
    """
    if not all(isinstance(frame, MotionFrame) for frame in (current, previous, earlier)):
        raise ValueError("Expected current, previous and earlier MotionFrame values")
    if not current.recording == previous.recording == earlier.recording:
        raise ValueError("Cannot associate boxes across different recordings")
    recent_s = _gap_seconds(current, previous)
    prior_s = _gap_seconds(previous, earlier)
    if not current.boxes:
        return MotionEstimate((), ())

    now, recent, prior = (_align(frame, current.ego_to_world)
                          for frame in (current, previous, earlier))
    recent_matches = _mutual_matches(now, recent, recent_s)
    prior_matches = _mutual_matches(recent, prior, prior_s)

    output, diagnostics = [], []
    for i, box in enumerate(current.boxes):
        j, problem = recent_matches[i]
        if j is None:
            speed = _ZERO
            diagnostic = MotionDiagnostic(box.object_id, problem + "_previous")
        else:
            previous_id = previous.boxes[j].object_id
            k, problem = prior_matches[j]
            if k is None:
                speed = _ZERO
                diagnostic = MotionDiagnostic(box.object_id, problem + "_earlier", previous_id)
            else:
                v_recent = (now.centers[i, :2] - recent.centers[j, :2]) / recent_s
                v_prior = (recent.centers[j, :2] - prior.centers[k, :2]) / prior_s
                speed, status = _combine(v_recent, v_prior, recent_s, prior_s)
                diagnostic = MotionDiagnostic(
                    box.object_id, status, previous_id, earlier.boxes[k].object_id,
                    _xy_tuple(v_recent), _xy_tuple(v_prior),
                )
        output.append(replace(box, velocity_xy=speed))
        diagnostics.append(diagnostic)
    return MotionEstimate(tuple(output), tuple(diagnostics))


def _gap_seconds(later: MotionFrame, earlier: MotionFrame) -> float:
    delta = later.ref_time_us - earlier.ref_time_us
    if not _MIN_GAP_US <= delta <= _MAX_GAP_US:
        raise ValueError("Adjacent history frame timestamps must increase with gaps of 0.25–1.0 seconds")
    return delta / 1_000_000


@dataclass(frozen=True)
class _Aligned:
    centers: NDArray[np.float64]
    axes_xy: NDArray[np.float64]
    valid_axes: NDArray[np.bool_]
    log_sizes: NDArray[np.float64]


def _align(frame: MotionFrame, reference_pose: NDArray[np.float64]) -> _Aligned:
    pose = frame.ego_to_world
    inverse_rotation = reference_pose[:3, :3].T
    rotation = inverse_rotation @ pose[:3, :3]
    centers = np.array([box.center_xyz for box in frame.boxes], dtype=np.float64).reshape(-1, 3)
    with np.errstate(over="ignore", invalid="ignore"):
        translation = inverse_rotation @ (pose[:3, 3] - reference_pose[:3, 3])
        centers = centers @ rotation.T + translation
    if not np.isfinite(centers).all():
        raise ValueError("Nonfinite ego-aligned box centers")
    yaw = np.array([box.yaw for box in frame.boxes])
    axes = np.column_stack((np.cos(yaw), np.sin(yaw), np.zeros_like(yaw)))
    axes_xy = (axes @ rotation.T)[:, :2]
    norms = np.linalg.norm(axes_xy, axis=1)
    valid_axes = norms > 1e-6
    axes_xy /= np.where(valid_axes, norms, 1.0)[:, None]
    log_sizes = np.log(np.array([box.size_wlh for box in frame.boxes], dtype=np.float64).reshape(-1, 3))
    return _Aligned(centers, axes_xy, valid_axes, log_sizes)


def _mutual_matches(later: _Aligned, earlier: _Aligned, dt_s: float):
    with np.errstate(over="ignore", invalid="ignore"):
        delta = later.centers[:, None, :] - earlier.centers[None, :, :]
        distance = np.hypot(delta[..., 0], delta[..., 1])
        height_delta = np.abs(delta[..., 2])
        shape_delta = np.abs(later.log_sizes[:, None, :] - earlier.log_sizes[None, :, :])
        axis_cos = np.clip(np.abs(later.axes_xy @ earlier.axes_xy.T), 0.0, 1.0)
        allowed = (
            (distance <= _MAX_SPEED_MPS * dt_s + _CENTER_SLACK_M)
            & (height_delta <= _MAX_HEIGHT_DELTA_M)
            & (np.max(shape_delta, axis=2) <= _MAX_LOG_SIZE_RATIO)
            & (axis_cos >= _MIN_BOX_AXIS_COS)
            & later.valid_axes[:, None] & earlier.valid_axes[None, :]
        )
        costs = np.where(allowed, distance + 0.5 * height_delta
                         + 0.75 * shape_delta.sum(axis=2) + 0.5 * (1.0 - axis_cos), np.inf)
    forward = [_unique_best(row) for row in costs]
    backward = [_unique_best(row) for row in costs.T]
    result = []
    for i, (j, status) in enumerate(forward):
        if j is None:
            result.append((None, status))
        elif backward[j][0] != i:
            result.append((None, "ambiguous" if backward[j][0] is None else "nonmutual"))
        else:
            result.append((j, "matched"))
    return result


def _unique_best(row: NDArray[np.float64]) -> tuple[int | None, str]:
    if not row.size:
        return None, "missing"
    index = int(np.argmin(row))
    best = row[index]
    if not np.isfinite(best):
        return None, "missing"
    # A tied minimum is always rejected; assignment does not depend on input box order.
    if np.count_nonzero(row < best + _UNIQUENESS_GAP_M) != 1:
        return None, "ambiguous"
    return index, "matched"


def _combine(recent, prior, recent_s, prior_s) -> tuple[tuple[float, float], str]:
    a = float(np.linalg.norm(recent))
    b = float(np.linalg.norm(prior))
    if max(a, b) > _MAX_SPEED_MPS:
        return _ZERO, "implausible_speed"
    if max(a, b) <= _QUIET_SPEED_MPS:
        return _ZERO, "quiet"
    if (min(a, b) <= _QUIET_SPEED_MPS or np.linalg.norm(recent - prior) > _MAX_SPEED_DISAGREEMENT_MPS
            or float(np.dot(recent, prior)) < _MIN_VELOCITY_COS * a * b):
        return _ZERO, "inconsistent_motion"
    return _xy_tuple((recent * recent_s + prior * prior_s) / (recent_s + prior_s)), "estimated"


def _xy_tuple(value) -> tuple[float, float]:
    return float(value[0]), float(value[1])
