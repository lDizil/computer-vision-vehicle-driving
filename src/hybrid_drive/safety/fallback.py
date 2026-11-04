from dataclasses import dataclass
from enum import Enum
import math
from numbers import Integral

import numpy as np

from hybrid_drive.contracts._validation import finite_real
from hybrid_drive.contracts.motion import EgoMotion
from hybrid_drive.contracts.trajectory import Trajectory

from ._limits import exceeds_limit
from .kinematics import KinematicLimits, KinematicReport, check_kinematics


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
    EMERGENCY_HOLD = "emergency_hold"


class FallbackFailure(str, Enum):
    UNSUPPORTED_MOTION = "unsupported_motion"
    INITIAL_ACCELERATION = "initial_acceleration"
    INFEASIBLE_STOP = "infeasible_stop"
    NUMERIC_RANGE = "numeric_range"
    KINEMATIC_LIMITS = "kinematic_limits"


@dataclass(frozen=True)
class FallbackResult:
    trajectory: Trajectory
    kind: FallbackKind
    failure_reason: FallbackFailure | None
    kinematics: KinematicReport
    stop_time_s: float | None
    stop_distance_m: float | None
    peak_speed_mps: float | None

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
            profile = _build_profile(v, a, config, minimum_speed)
            poses = _sample_profile(profile, v, a, jerk, empty.times_s)
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


@dataclass(frozen=True)
class _Profile:
    down_s: float
    hold_s: float
    up_s: float
    down_distance_m: float
    stop_distance_m: float
    plateau_start_speed_mps: float
    peak_deceleration_mps2: float
    peak_speed_mps: float

    @property
    def stop_time_s(self) -> float:
        return self.down_s + self.hold_s + self.up_s


def _build_profile(v: float, a: float, config: BrakingConfig, minimum_speed: float) -> _Profile:
    cap, jerk = config.deceleration_mps2, config.jerk_mps3
    equivalent_speed = v + minimum_speed
    saturation_speed = cap * (cap / jerk)
    if equivalent_speed >= saturation_speed:
        peak = cap
        hold = (equivalent_speed - saturation_speed) / peak
        down = (a + peak) / jerk
    else:
        peak = -a if a < 0 and v == minimum_speed else math.sqrt(jerk) * math.sqrt(equivalent_speed)
        hold = 0.0
        down = (v - minimum_speed) / (peak - a) if a < 0 else (a + peak) / jerk
    if not math.isfinite(peak) or peak <= 0:
        raise FloatingPointError("Unrepresentable braking peak")
    up = peak / jerk
    release_speed = peak * up / 2
    plateau_speed = release_speed + peak * hold
    down_distance = down * (plateau_speed + down * (peak / 3 - a / 6))
    release_distance = release_speed * up / 3
    stop_distance = down_distance + hold * (release_speed + peak * hold / 2) + release_distance
    peak_speed = v + minimum_speed if a > 0 else v
    values = (equivalent_speed, down, hold, up, release_speed, plateau_speed, down_distance, stop_distance, peak_speed)
    if not all(math.isfinite(value) and value >= 0 for value in values):
        raise FloatingPointError("Unrepresentable stopping profile")
    release_start = down + hold
    stop_time = release_start + up
    if up <= 0 or not math.isfinite(stop_time) or stop_time <= release_start or stop_distance <= 0:
        raise FloatingPointError("Unrepresentable stopping endpoint")
    if hold > 0 and release_start <= down:
        raise FloatingPointError("Unrepresentable plateau duration")
    return _Profile(down, hold, up, down_distance, stop_distance, plateau_speed, peak, peak_speed)


def _sample_profile(profile: _Profile, v: float, a: float, jerk: float, times) -> np.ndarray:
    poses = np.zeros((len(times), 3))
    poses[:, 0] = profile.stop_distance_m
    down = times < profile.down_s
    t = times[down]
    poses[down, 0] = (v + (a / 2 - jerk * t / 6) * t) * t
    release_start = profile.down_s + profile.hold_s
    hold = (times >= profile.down_s) & (times < release_start)
    t = times[hold] - profile.down_s
    poses[hold, 0] = profile.down_distance_m + (profile.plateau_start_speed_mps - profile.peak_deceleration_mps2 * t / 2) * t
    up = (times >= release_start) & (times < profile.stop_time_s)
    ratio = (profile.stop_time_s - times[up]) / profile.up_s
    release_distance = profile.peak_deceleration_mps2 * profile.up_s / 2 * profile.up_s / 3
    # Evaluate backward from rest, rather than clipping a rounded negative velocity.
    poses[up, 0] = profile.stop_distance_m - release_distance * ratio**3
    return poses
