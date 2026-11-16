from dataclasses import dataclass
from enum import Enum
import math
from numbers import Integral
from typing import TYPE_CHECKING

import numpy as np

from hybrid_drive.contracts._validation import finite_real
from hybrid_drive.contracts.motion import EgoMotion
from hybrid_drive.contracts.trajectory import Trajectory

from ._braking_profile import build_profile, sample_distances
from ._limits import exceeds_limit
from .kinematics import KinematicLimits, KinematicReport, check_kinematics

if TYPE_CHECKING:
    from .curved_braking import CurvedBrakingReport


@dataclass(frozen=True)
class BrakingConfig:
    deceleration_mps2: float
    jerk_mps3: float

    def __post_init__(self):
        for name in ("deceleration_mps2", "jerk_mps3"):
            value = finite_real(getattr(self, name), name)
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)

    def validate_for(self, limits: KinematicLimits) -> None:
        if not isinstance(limits, KinematicLimits):
            raise ValueError("limits must be KinematicLimits")
        if self.deceleration_mps2 > limits.max_acceleration_mps2 or self.jerk_mps3 > limits.max_jerk_mps3:
            raise ValueError("Braking caps must not exceed the kinematic limits")


class FallbackKind(str, Enum):
    BRAKING = "braking"
    CURVED_BRAKING = "curved_braking"
    EMERGENCY_HOLD = "emergency_hold"


class FallbackFailure(str, Enum):
    UNSUPPORTED_MOTION = "unsupported_motion"
    INITIAL_ACCELERATION = "initial_acceleration"
    INFEASIBLE_STOP = "infeasible_stop"
    NUMERIC_RANGE = "numeric_range"
    KINEMATIC_LIMITS = "kinematic_limits"
    MISSING_HEADING_RATE = "missing_heading_rate"
    UNSUPPORTED_HEADING = "unsupported_heading"
    OBSERVATION_MISMATCH = "observation_mismatch"
    NO_CERTIFIED_PROFILE = "no_certified_profile"


@dataclass(frozen=True)
class FallbackResult:
    trajectory: Trajectory
    kind: FallbackKind
    failure_reason: FallbackFailure | None
    kinematics: KinematicReport
    stop_time_s: float | None
    stop_distance_m: float | None
    peak_speed_mps: float | None
    model_report: "CurvedBrakingReport | None" = None

    @property
    def succeeded(self) -> bool:
        return self.failure_reason is None


def braking_fallback(motion: EgoMotion, limits: KinematicLimits, config: BrakingConfig, num_poses: int) -> FallbackResult:
    """Forward straight-line jerk-limited stop, or an explicit offline failure.

    A failed emergency hold only satisfies the evaluator's output shape. It is
    not a braking command or a safe substitute for unsupported lateral motion.
    """
    if not isinstance(motion, EgoMotion) or not isinstance(config, BrakingConfig):
        raise ValueError("Expected EgoMotion and BrakingConfig")
    config.validate_for(limits)
    empty = _empty_trajectory(num_poses, limits.interval_s)
    if motion.vx_mps < 0 or motion.vy_mps != 0 or motion.ay_mps2 != 0:
        return _emergency_hold(empty, motion, limits, FallbackFailure.UNSUPPORTED_MOTION)
    if abs(motion.ax_mps2) > config.deceleration_mps2:
        return _emergency_hold(empty, motion, limits, FallbackFailure.INITIAL_ACCELERATION)
    v, a, jerk = motion.vx_mps, motion.ax_mps2, config.jerk_mps3
    if v == 0 and a == 0:
        report = check_kinematics(empty, motion, limits)
        failure = None if report.within_limits else FallbackFailure.KINEMATIC_LIMITS
        return FallbackResult(empty, FallbackKind.BRAKING, failure, report, 0.0, 0.0, 0.0)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            minimum_speed = a * (a / jerk) / 2
            if not math.isfinite(minimum_speed):
                raise FloatingPointError("Unrepresentable stopping state")
            if a < 0 and v < minimum_speed:
                return _emergency_hold(empty, motion, limits, FallbackFailure.INFEASIBLE_STOP)
            profile = build_profile(v, a, config.deceleration_mps2, jerk, minimum_speed)
            poses = np.zeros((len(empty.times_s), 3))
            poses[:, 0] = sample_distances(profile, v, a, jerk, empty.times_s)
            if not np.isfinite(poses).all() or np.any(np.diff(np.r_[0.0, poses[:, 0]]) < 0):
                raise FloatingPointError("Unrepresentable forward trajectory")
    except (FloatingPointError, OverflowError):
        return _emergency_hold(empty, motion, limits, FallbackFailure.NUMERIC_RANGE)
    trajectory = Trajectory(poses, limits.interval_s)
    report = check_kinematics(trajectory, motion, limits)
    failure = None
    if not report.within_limits or exceeds_limit(profile.peak_speed_mps, limits.max_speed_mps):
        failure = FallbackFailure.KINEMATIC_LIMITS
    return FallbackResult(
        trajectory, FallbackKind.BRAKING, failure, report,
        profile.stop_time_s, profile.stop_distance_m, profile.peak_speed_mps,
    )


def unproven_fallback(
    motion: EgoMotion, limits: KinematicLimits, num_poses: int, reason: FallbackFailure,
) -> FallbackResult:
    """Finite offline placeholder with explicit failure, never a braking command."""
    if not isinstance(motion, EgoMotion) or not isinstance(limits, KinematicLimits):
        raise ValueError("Expected EgoMotion and KinematicLimits")
    if not isinstance(reason, FallbackFailure):
        raise ValueError("reason must be FallbackFailure")
    return _emergency_hold(_empty_trajectory(num_poses, limits.interval_s), motion, limits, reason)


def _empty_trajectory(num_poses: int, interval_s: float) -> Trajectory:
    if isinstance(num_poses, (bool, np.bool_)) or not isinstance(num_poses, Integral) or num_poses <= 0:
        raise ValueError("num_poses must be a positive integer")
    try:
        horizon = int(num_poses) * interval_s
    except OverflowError as exc:
        raise ValueError("Trajectory horizon must be finite") from exc
    if not math.isfinite(horizon):
        raise ValueError("Trajectory horizon must be finite")
    return Trajectory(np.zeros((num_poses, 3)), interval_s)


def _emergency_hold(trajectory, motion, limits, reason) -> FallbackResult:
    return FallbackResult(
        trajectory, FallbackKind.EMERGENCY_HOLD, reason,
        check_kinematics(trajectory, motion, limits), None, None, None,
    )
