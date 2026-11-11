from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike, NDArray

from hybrid_drive.contracts._validation import finite_real, identifier
from hybrid_drive.contracts.trajectory import Trajectory


Pixel = tuple[float, float]
PixelSegment = tuple[Pixel, Pixel]


@dataclass(frozen=True, eq=False, init=False)
class CameraCalibration:
    """Pinhole calibration from ego (forward, left, up) to camera (right, down, forward)."""

    name: str
    width: int
    height: int
    _intrinsic_data: bytes = field(repr=False)
    _ego_to_camera_data: bytes = field(repr=False)
    near_m: float

    def __init__(self, name: str, width: int, height: int, intrinsic: ArrayLike,
                 ego_to_camera: ArrayLike, near_m: float = 0.1):
        name = identifier(name, "name")
        width, height = _dimension(width, "width"), _dimension(height, "height")
        intrinsic = _matrix(intrinsic, (3, 3), "intrinsic")
        ego_to_camera = _matrix(ego_to_camera, (4, 4), "ego_to_camera")
        near_m = finite_real(near_m, "near_m")
        if near_m <= 0:
            raise ValueError("near_m must be positive")
        _validate_intrinsic(intrinsic)
        _validate_rigid_transform(ego_to_camera)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "width", width)
        object.__setattr__(self, "height", height)
        object.__setattr__(self, "_intrinsic_data", intrinsic.tobytes())
        object.__setattr__(self, "_ego_to_camera_data", ego_to_camera.tobytes())
        object.__setattr__(self, "near_m", near_m)

    @property
    def intrinsic(self) -> NDArray[np.float64]:
        return np.frombuffer(self._intrinsic_data, dtype=np.float64).reshape(3, 3)

    @property
    def ego_to_camera(self) -> NDArray[np.float64]:
        return np.frombuffer(self._ego_to_camera_data, dtype=np.float64).reshape(4, 4)

    def resized(self, width: int, height: int) -> CameraCalibration:
        """Match PIL pixel centres: u' = (u + 0.5) * sx - 0.5, likewise v."""
        width, height = _dimension(width, "width"), _dimension(height, "height")
        sx, sy = width / self.width, height / self.height
        intrinsic = self.intrinsic.copy()
        intrinsic[0] *= sx
        intrinsic[1] *= sy
        intrinsic[0, 2] += (sx - 1) / 2
        intrinsic[1, 2] += (sy - 1) / 2
        return CameraCalibration(self.name, width, height, intrinsic, self.ego_to_camera, self.near_m)


def _dimension(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or not 1 <= value <= 65536:
        raise ValueError(f"{name} must be an integer in [1, 65536]")
    return int(value)


def _matrix(value: ArrayLike, shape: tuple[int, int], name: str) -> NDArray[np.float64]:
    result = _numeric_array(value, name)
    if result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    return result


def _numeric_array(value: ArrayLike, name: str) -> NDArray[np.float64]:
    try:
        array = np.asarray(value)
        # NumPy otherwise silently coerces mixed Python booleans to integers.
        contains_bool = any(isinstance(item, (bool, np.bool_)) for item in np.asarray(value, dtype=object).flat)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain finite real numeric values") from exc
    if array.dtype.kind not in "fiu" or contains_bool:
        raise ValueError(f"{name} must contain finite real numeric values, not booleans")
    with np.errstate(over="ignore", invalid="ignore"):
        result = array.astype(np.float64, copy=True)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain finite real numeric values")
    return result


def _validate_intrinsic(intrinsic: NDArray[np.float64]) -> None:
    if not np.array_equal(intrinsic[2], [0, 0, 1]) or intrinsic[1, 0] != 0:
        raise ValueError("intrinsic must be an upper-triangular pinhole matrix with bottom row [0, 0, 1]")
    if intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0:
        raise ValueError("intrinsic focal lengths must be positive")


def _validate_rigid_transform(transform: NDArray[np.float64]) -> None:
    if not np.array_equal(transform[3], [0, 0, 0, 1]):
        raise ValueError("ego_to_camera must have homogeneous bottom row [0, 0, 0, 1]")
    rotation = transform[:3, :3]
    with np.errstate(over="ignore", invalid="ignore"):
        orthonormal = np.allclose(rotation @ rotation.T, np.eye(3), rtol=0, atol=1e-9)
        determinant = np.linalg.det(rotation)
    if not orthonormal or not np.isclose(determinant, 1, rtol=0, atol=1e-9):
        raise ValueError("ego_to_camera rotation must be proper orthonormal (determinant +1)")


def project_segments(points_xyz: ArrayLike, calibration: CameraCalibration) -> tuple[PixelSegment, ...]:
    """Project consecutive ego-frame polyline edges, clipping near plane before image bounds.

    Invisible edges are omitted independently; separated visible pieces are never joined.
    Empty and one-point [N, 3] polylines have no edges. Zero-length image edges are omitted.
    Ill-conditioned clipping is rejected: projected coordinates must be within 1e12 pixels;
    near-crossing endpoint depths must be within 1e12 times the near distance.
    """
    if not isinstance(calibration, CameraCalibration):
        raise ValueError("calibration must be CameraCalibration")
    points = _numeric_array(points_xyz, "points_xyz")
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_xyz must have shape [N, 3]")
    transform = calibration.ego_to_camera
    with np.errstate(over="ignore", invalid="ignore"):
        camera_points = points @ transform[:3, :3].T + transform[:3, 3]
    if not np.isfinite(camera_points).all():
        raise ValueError("Transformed points exceed the finite coordinate range")
    result = []
    for start, end in zip(camera_points[:-1], camera_points[1:]):
        clipped = _clip_near(start, end, calibration.near_m)
        if clipped is None:
            continue
        pixels = _project(clipped, calibration.intrinsic)
        segment = _clip_viewport(pixels, calibration.width, calibration.height)
        if segment is not None and segment[0] != segment[1]:
            result.append(segment)
    return tuple(result)


def _clip_near(start: NDArray[np.float64], end: NDArray[np.float64], near: float):
    start_visible, end_visible = start[2] >= near, end[2] >= near
    if not start_visible and not end_visible:
        return None
    if start_visible and end_visible:
        return np.stack((start, end))
    if max(abs(float(start[2])), abs(float(end[2]))) > near * 1e12:
        raise ValueError("Near-plane intersection exceeds the 1e12 depth-to-near clipping limit")
    alpha = _crossing_fraction(float(start[2]), float(end[2]), near)
    intersection = (1 - alpha) * start + alpha * end
    intersection[2] = near
    return np.stack((start, intersection) if start_visible else (intersection, end))


def _crossing_fraction(start: float, end: float, boundary: float) -> float:
    scale = max(abs(start), abs(end), abs(boundary))
    return (boundary / scale - start / scale) / (end / scale - start / scale)


def _project(points: NDArray[np.float64], intrinsic: NDArray[np.float64]) -> NDArray[np.float64]:
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        normalized = points[:, :2] / points[:, 2, None]
        pixels = normalized @ intrinsic[:2, :2].T + intrinsic[:2, 2]
    if not np.isfinite(pixels).all():
        raise ValueError("Projected points exceed the finite pixel coordinate range")
    return pixels


def _clip_viewport(pixels: NDArray[np.float64], width: int, height: int) -> PixelSegment | None:
    # Distinct viewport crossings can otherwise round to the same float parameter.
    if np.max(np.abs(pixels)) > 1e12:
        raise ValueError("Projected coordinates exceed the 1e12-pixel clipping limit")
    start, end = pixels
    enter, leave = 0.0, 1.0
    enter_boundary = leave_boundary = None
    for axis, maximum in enumerate((width - 1, height - 1)):
        first, last = float(start[axis]), float(end[axis])
        if (first < 0 and last < 0) or (first > maximum and last > maximum):
            return None
        if first == last:
            continue
        low, high = (0.0, float(maximum)) if first < last else (float(maximum), 0.0)
        if min(first, last) < 0 or max(first, last) > maximum:
            entering = _crossing_fraction(first, last, low)
            leaving = _crossing_fraction(first, last, high)
            if entering > enter:
                enter, enter_boundary = entering, (axis, low)
            if leaving < leave:
                leave, leave_boundary = leaving, (axis, high)
        if enter > leave:
            return None
    clipped_start = (1 - enter) * start + enter * end
    clipped_end = (1 - leave) * start + leave * end
    if enter_boundary is not None:
        clipped_start[enter_boundary[0]] = enter_boundary[1]
    if leave_boundary is not None:
        clipped_end[leave_boundary[0]] = leave_boundary[1]
    maximum = np.array([width - 1, height - 1])
    clipped = np.clip(np.stack((clipped_start, clipped_end)), 0, maximum)
    return tuple(tuple(float(value) for value in point) for point in clipped)


def project_trajectory(trajectory: Trajectory, calibration: CameraCalibration) -> tuple[PixelSegment, ...]:
    """Render future ego-frame XY poses on z=0, including the edge from the ego origin."""
    if not isinstance(trajectory, Trajectory):
        raise ValueError("trajectory must be Trajectory")
    points = np.zeros((trajectory.num_poses + 1, 3), dtype=np.float64)
    points[1:, :2] = trajectory.poses[:, :2]
    return project_segments(points, calibration)
