from dataclasses import dataclass, field
import math
from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import finite_real


@dataclass(frozen=True)
class Pose2D:
    """SE(2) reference pose: metres and radians, heading in [-pi, pi)."""

    x: float
    y: float
    heading: float

    def __post_init__(self):
        object.__setattr__(self, "x", finite_real(self.x, "x"))
        object.__setattr__(self, "y", finite_real(self.y, "y"))
        heading = finite_real(self.heading, "heading")
        object.__setattr__(self, "heading", float(_wrap(heading)))


@dataclass(frozen=True, eq=False, init=False)
class Trajectory:
    """Immutable future ego poses at dt, 2*dt, ...; t=0 is (0, 0, 0).

    Columns are forward x, left y (metres), and heading (radians).
    Float64 storage keeps world-to-ego conversion ahead of float32 adapters.
    Shape/finite checks do not establish dynamic feasibility or road safety.
    """

    _data: bytes = field(repr=False)
    _num_poses: int
    interval_s: float

    def __init__(self, poses: ArrayLike, interval_s: float):
        poses = _validated_poses(poses)
        interval_s = _sampling(interval_s, len(poses))
        poses[:, 2] = _wrap(poses[:, 2])
        object.__setattr__(self, "_data", poses.tobytes())
        object.__setattr__(self, "_num_poses", len(poses))
        object.__setattr__(self, "interval_s", interval_s)

    @property
    def poses(self) -> NDArray[np.float64]:
        # Fresh metadata prevents external shape/dtype changes from altering this value.
        return np.frombuffer(self._data, dtype=np.float64).reshape(self.num_poses, 3)

    @property
    def num_poses(self) -> int:
        return self._num_poses

    @property
    def horizon_s(self) -> float:
        return self.num_poses * self.interval_s

    @property
    def times_s(self) -> NDArray[np.float64]:
        return _readonly(np.arange(1, self.num_poses + 1, dtype=np.float64) * self.interval_s)

    @classmethod
    def from_world_poses(cls, poses: ArrayLike, reference_pose: Pose2D, interval_s: float) -> "Trajectory":
        world = _validated_poses(poses)
        rotation = _rotation(reference_pose)
        with np.errstate(over="ignore", invalid="ignore"):
            world[:, :2] = (world[:, :2] - [reference_pose.x, reference_pose.y]) @ rotation
        world[:, 2] = _wrap(world[:, 2]) - reference_pose.heading
        return cls(world, interval_s)

    def to_world_poses(self, reference_pose: Pose2D) -> NDArray[np.float64]:
        rotation = _rotation(reference_pose)
        world = self.poses.copy()
        with np.errstate(over="ignore", invalid="ignore"):
            world[:, :2] = world[:, :2] @ rotation.T + [reference_pose.x, reference_pose.y]
        world[:, 2] = _wrap(world[:, 2] + reference_pose.heading)
        if not np.isfinite(world).all():
            raise ValueError("World poses exceed the finite coordinate range")
        return _readonly(world)

    def resample(self, interval_s: float, num_poses: int) -> "Trajectory":
        """Interpolate within this horizon, taking the shortest angular arc per segment.

        Rotation exceeding pi between samples is ambiguous and is not reconstructed.
        """
        interval_s = _sampling(interval_s, num_poses)
        target_horizon = num_poses * interval_s
        if target_horizon > math.nextafter(self.horizon_s, math.inf):
            raise ValueError("Resampling cannot extrapolate beyond the source horizon")
        source_times = np.concatenate(([0.0], self.times_s))
        target_times = np.minimum(
            np.arange(1, num_poses + 1, dtype=np.float64) * interval_s,
            self.horizon_s,
        )
        source = np.vstack((np.zeros((1, 3)), self.poses))
        source[:, 2] = np.unwrap(source[:, 2])
        upper = np.clip(np.searchsorted(source_times, target_times, side="right"), 1, len(source_times) - 1)
        lower = upper - 1
        alpha = ((target_times - source_times[lower]) / (source_times[upper] - source_times[lower]))[:, None]
        # Avoid overflowing a coordinate delta or slope when the result is finite.
        poses = (1 - alpha) * source[lower] + alpha * source[upper]
        return Trajectory(poses, interval_s)


def _validated_poses(value: ArrayLike) -> NDArray[np.float64]:
    try:
        poses = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Poses must be a numeric array with shape [N, 3]") from exc
    if poses.ndim != 2 or poses.shape[0] == 0 or poses.shape[1] != 3:
        raise ValueError("Poses must have non-empty shape [N, 3]")
    if poses.dtype.kind not in "fiu":
        raise ValueError("Poses must contain real numeric values")
    with np.errstate(over="ignore"):
        poses = poses.astype(np.float64, copy=True)
    if not np.isfinite(poses).all():
        raise ValueError("Poses must contain only finite values")
    return poses


def _sampling(interval_s: float, num_poses: int) -> float:
    interval_s = finite_real(interval_s, "interval_s")
    if interval_s <= 0:
        raise ValueError("interval_s must be positive")
    if isinstance(num_poses, (bool, np.bool_)) or not isinstance(num_poses, Integral) or num_poses <= 0:
        raise ValueError("num_poses must be a positive integer")
    try:
        horizon = int(num_poses) * interval_s
    except OverflowError as exc:
        raise ValueError("Trajectory horizon must be finite") from exc
    if not math.isfinite(horizon):
        raise ValueError("Trajectory horizon must be finite")
    return interval_s


def _wrap(value):
    values = np.asarray(value)
    wrapped = np.where(
        (values >= -math.pi) & (values < math.pi),
        values,
        (values + math.pi) % (2 * math.pi) - math.pi,
    )
    return np.where(wrapped >= math.pi, -math.pi, wrapped)


def _readonly(values: NDArray[np.float64]) -> NDArray[np.float64]:
    # A bytes-backed array cannot be made writable again via setflags.
    return np.frombuffer(values.tobytes(), dtype=np.float64).reshape(values.shape)


def _rotation(reference_pose: Pose2D) -> NDArray[np.float64]:
    if not isinstance(reference_pose, Pose2D):
        raise ValueError("reference_pose must be Pose2D")
    c, s = math.cos(reference_pose.heading), math.sin(reference_pose.heading)
    return np.array([[c, -s], [s, c]], dtype=np.float64)
