from dataclasses import dataclass
from enum import Enum
import math

import numpy as np

from hybrid_drive.contracts._validation import finite_real
from hybrid_drive.contracts.motion import EgoMotion
from hybrid_drive.contracts.trajectory import Trajectory


@dataclass(frozen=True)
class KinematicLimits:
    interval_s: float
    max_speed_mps: float
    max_acceleration_mps2: float
    max_jerk_mps3: float

    def __post_init__(self):
        for name in ("interval_s", "max_speed_mps", "max_acceleration_mps2", "max_jerk_mps3"):
            value = finite_real(getattr(self, name), name)
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)


class KinematicViolation(str, Enum):
    TIME_GRID = "time_grid"
    INSUFFICIENT_POSES = "insufficient_poses"
    NUMERIC_RANGE = "numeric_range"
    SPEED = "speed"
    ACCELERATION = "acceleration"
    JERK = "jerk"


@dataclass(frozen=True)
class KinematicReport:
    violations: tuple[KinematicViolation, ...]
    max_speed_mps: float | None
    max_acceleration_mps2: float | None
    max_jerk_mps3: float | None

    @property
    def within_limits(self) -> bool:
        return not self.violations


def check_kinematics(trajectory: Trajectory, motion: EgoMotion, limits: KinematicLimits) -> KinematicReport:
    """Check sampled 2D estimates, not continuous dynamics or road safety.

    Speed uses interval means, acceleration their differences, and jerk the
    next differences. Integral residuals bind the first interval to measured
    velocity/acceleration without inventing an instantaneous stop at t=0.
    """
    if not isinstance(trajectory, Trajectory) or not isinstance(motion, EgoMotion):
        raise ValueError("Expected Trajectory and EgoMotion")
    if not isinstance(limits, KinematicLimits):
        raise ValueError("limits must be KinematicLimits")
    violations = []
    if not math.isclose(trajectory.interval_s, limits.interval_s, rel_tol=1e-12, abs_tol=0):
        violations.append(KinematicViolation.TIME_GRID)
    if trajectory.num_poses < 3:
        violations.append(KinematicViolation.INSUFFICIENT_POSES)
    if violations:
        return KinematicReport(tuple(violations), None, None, None)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            peaks = _measure_peaks(trajectory, motion)
    except FloatingPointError:
        return KinematicReport((KinematicViolation.NUMERIC_RANGE,), None, None, None)
    if not all(math.isfinite(value) for value in peaks):
        return KinematicReport((KinematicViolation.NUMERIC_RANGE,), None, None, None)
    thresholds = (limits.max_speed_mps, limits.max_acceleration_mps2, limits.max_jerk_mps3)
    reasons = (KinematicViolation.SPEED, KinematicViolation.ACCELERATION, KinematicViolation.JERK)
    violations = tuple(reason for peak, bound, reason in zip(peaks, thresholds, reasons) if _exceeds(peak, bound))
    return KinematicReport(violations, *peaks)


def _measure_peaks(trajectory: Trajectory, motion: EgoMotion) -> tuple[float, float, float]:
    h = trajectory.interval_s
    positions = np.vstack((np.zeros((1, 2)), trajectory.poses[:, :2]))
    velocities = np.diff(positions, axis=0) / h
    accelerations = np.diff(velocities, axis=0) / h
    jerks = np.diff(accelerations, axis=0) / h
    initial_velocity = np.array([motion.vx_mps, motion.vy_mps])
    initial_acceleration = np.array([motion.ax_mps2, motion.ay_mps2])
    # Integral residuals are exact at the start for constant acceleration/jerk.
    start_acceleration = (velocities[0] - initial_velocity) / h * 2
    start_jerk = (velocities[0] - initial_velocity - 0.5 * initial_acceleration * h) / h / h * 6
    return (
        max(_max_norm(velocities), math.hypot(*initial_velocity)),
        max(_max_norm(accelerations), math.hypot(*initial_acceleration), math.hypot(*start_acceleration)),
        max(_max_norm(jerks), math.hypot(*start_jerk)),
    )


def _max_norm(values) -> float:
    return float(np.max(np.hypot(values[:, 0], values[:, 1])))


def _exceeds(value: float, bound: float) -> bool:
    return value > bound and not math.isclose(value, bound, rel_tol=1e-9, abs_tol=1e-9)
