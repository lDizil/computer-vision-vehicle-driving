from dataclasses import dataclass, replace
from enum import Enum
from fractions import Fraction
import math

import numpy as np

from hybrid_drive.contracts._validation import finite_real
from hybrid_drive.contracts.motion import EgoMotion
from hybrid_drive.contracts.trajectory import Trajectory

from ._braking_profile import BrakingProfile, build_profile, sample_distances
from .fallback import BrakingConfig, FallbackFailure, FallbackKind, FallbackResult, unproven_fallback
from .kinematics import KinematicLimits, KinematicReport, check_kinematics
from .planar_motion import (
    MODEL_ASSUMPTIONS,
    PlanarMotionBounds,
    PlanarMotionError,
    PlanarMotionFailure,
    PlanarMotionHypothesis,
)


_BUDGET_SCALES = tuple(2.0 ** -index for index in range(13))


class CurvedBrakingOutcome(str, Enum):
    CERTIFIED = "certified"
    SCALAR_BUDGET = "scalar_budget"
    SCALAR_INFEASIBLE = "scalar_infeasible"
    NUMERIC_RANGE = "numeric_range"
    CONTINUOUS_LIMITS = "continuous_limits"
    SIDESLIP = "sideslip"
    HEADING_ALIAS = "heading_alias"
    SAMPLED_LIMITS = "sampled_limits"


@dataclass(frozen=True)
class CurvedBrakingAttempt:
    scale: float
    outcome: CurvedBrakingOutcome


@dataclass(frozen=True)
class CurvedBrakingReport:
    """Model-relative evidence, not a vehicle/actuator or synchronized-state certificate."""

    model_id: str
    heading_rate_radps: float
    path_curvature_per_m: float | None
    heading_curvature_per_m: float | None
    initial_sideslip_rad: float | None
    selected_scale: float | None
    continuous_bounds: PlanarMotionBounds | None
    sideslip_range_rad: tuple[float, float] | None
    max_heading_step_rad: float | None
    attempts: tuple[CurvedBrakingAttempt, ...]
    assumptions: tuple[str, ...] = MODEL_ASSUMPTIONS


@dataclass(frozen=True)
class _Candidate:
    trajectory: Trajectory
    profile: BrakingProfile
    bounds: PlanarMotionBounds
    sideslip_range_rad: tuple[float, float]
    max_heading_step_rad: float
    kinematics: KinematicReport


def curved_braking_fallback(
    motion: EgoMotion, heading_rate_radps: float | None, limits: KinematicLimits,
    config: BrakingConfig, num_poses: int,
) -> FallbackResult:
    """Search a predeclared finite set of variable-slip point-mass braking profiles.

    Original config caps bound full acceleration/jerk, not scalar budgets.
    Failure to certify these candidates does not prove global infeasibility.
    The caller owns observation association and the frozen-snapshot assumption.
    """
    if not isinstance(motion, EgoMotion) or not isinstance(config, BrakingConfig):
        raise ValueError("Expected EgoMotion and BrakingConfig")
    config.validate_for(limits)
    placeholder = unproven_fallback(motion, limits, num_poses, FallbackFailure.NO_CERTIFIED_PROFILE)
    if heading_rate_radps is None:
        return replace(placeholder, failure_reason=FallbackFailure.MISSING_HEADING_RATE)
    try:
        rate = finite_real(heading_rate_radps, "heading_rate_radps")
    except ValueError:
        return replace(placeholder, failure_reason=FallbackFailure.UNSUPPORTED_HEADING)
    if motion.vx_mps == motion.vy_mps == motion.ax_mps2 == motion.ay_mps2 == 0:
        if rate != 0:
            return replace(placeholder, failure_reason=FallbackFailure.UNSUPPORTED_HEADING)
        return _stationary_result(placeholder)
    try:
        model = PlanarMotionHypothesis.from_motion(motion, rate)
    except PlanarMotionError as exc:
        failure = (FallbackFailure.NUMERIC_RANGE if exc.reason is PlanarMotionFailure.NUMERIC_RANGE
                   else FallbackFailure.UNSUPPORTED_MOTION)
        return replace(placeholder, failure_reason=failure)
    if math.hypot(motion.ax_mps2, motion.ay_mps2) > config.deceleration_mps2:
        return replace(placeholder, failure_reason=FallbackFailure.INITIAL_ACCELERATION)
    attempts = []
    for scale in _BUDGET_SCALES:
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                candidate = _try_candidate(model, limits, config, placeholder.trajectory.times_s, scale)
        except (FloatingPointError, OverflowError, PlanarMotionError):
            candidate = CurvedBrakingOutcome.NUMERIC_RANGE
        if isinstance(candidate, CurvedBrakingOutcome):
            attempts.append(CurvedBrakingAttempt(scale, candidate))
            continue
        attempts.append(CurvedBrakingAttempt(scale, CurvedBrakingOutcome.CERTIFIED))
        report = _model_report(model, tuple(attempts), candidate)
        return FallbackResult(
            candidate.trajectory, FallbackKind.CURVED_BRAKING, None, candidate.kinematics,
            candidate.profile.stop_time_s, candidate.profile.stop_distance_m, candidate.profile.peak_speed_mps, report,
        )
    return replace(placeholder, model_report=_model_report(model, tuple(attempts)))


def _stationary_result(placeholder: FallbackResult) -> FallbackResult:
    assumptions = ("exact_zero_velocity_acceleration_and_heading_rate",)
    bounds = PlanarMotionBounds(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, assumptions)
    report = CurvedBrakingReport(
        "exact-rest-v1", 0.0, None, None, None, None, bounds, None, 0.0, (),
        assumptions,
    )
    failure = None if placeholder.kinematics.within_limits else FallbackFailure.KINEMATIC_LIMITS
    return FallbackResult(
        placeholder.trajectory, FallbackKind.CURVED_BRAKING, failure, placeholder.kinematics,
        0.0, 0.0, 0.0, report,
    )


def _try_candidate(model, limits, config, times, scale) -> _Candidate | CurvedBrakingOutcome:
    cap, jerk = config.deceleration_mps2 * scale, config.jerk_mps3 * scale
    if cap <= 0 or jerk <= 0:
        raise FloatingPointError("Unrepresentable scalar budget")
    v, a = model.speed_mps, model.tangential_acceleration_mps2
    if abs(a) > cap:
        return CurvedBrakingOutcome.SCALAR_BUDGET
    rise = Fraction(a) ** 2 / (2 * Fraction(jerk))
    if a < 0 and Fraction(v) < rise:
        return CurvedBrakingOutcome.SCALAR_INFEASIBLE
    minimum_speed = a * (a / jerk) / 2
    if not math.isfinite(minimum_speed) or (a != 0 and minimum_speed == 0):
        raise FloatingPointError("Unrepresentable stopping state")
    profile = build_profile(v, a, cap, jerk, minimum_speed)
    if profile.peak_deceleration_mps2 > cap:
        raise FloatingPointError("Rounded scalar peak exceeds its budget")
    peak_speed_bound = max(profile.peak_speed_mps, _upper_float(Fraction(v) + (rise if a > 0 else 0)))
    peak_acceleration_bound = _peak_acceleration_bound(v, a, jerk, cap, profile)
    bounds = model.continuous_bounds(peak_speed_bound, peak_acceleration_bound, jerk)
    if (bounds.speed_mps > limits.max_speed_mps
            or bounds.acceleration_mps2 > min(config.deceleration_mps2, limits.max_acceleration_mps2)
            or bounds.jerk_mps3 > min(config.jerk_mps3, limits.max_jerk_mps3)):
        return CurvedBrakingOutcome.CONTINUOUS_LIMITS
    sideslip = model.sideslip_range(profile.stop_distance_m)
    if sideslip[0] <= -math.pi / 2 or sideslip[1] >= math.pi / 2:
        return CurvedBrakingOutcome.SIDESLIP
    heading_step = _upper_float(Fraction(bounds.heading_rate_radps) * Fraction(limits.interval_s))
    if heading_step >= math.pi:
        return CurvedBrakingOutcome.HEADING_ALIAS
    distances = sample_distances(profile, v, a, jerk, times)
    if not np.isfinite(distances).all() or np.any(np.diff(np.r_[0.0, distances]) < 0):
        raise FloatingPointError("Unrepresentable nondecreasing arc lengths")
    trajectory = Trajectory(model.poses_at_distances(distances), limits.interval_s)
    report = check_kinematics(trajectory, model.motion, limits)
    if (not report.within_limits or report.max_acceleration_mps2 > config.deceleration_mps2
            or report.max_jerk_mps3 > config.jerk_mps3):
        return CurvedBrakingOutcome.SAMPLED_LIMITS
    return _Candidate(trajectory, profile, bounds, sideslip, heading_step, report)


def _upper_float(exact: Fraction) -> float:
    value = float(exact)
    if Fraction(value) < exact:
        value = math.nextafter(value, math.inf)
    if not math.isfinite(value):
        raise FloatingPointError("Unrepresentable profile bound")
    return value


def _peak_acceleration_bound(v, a, jerk, cap, profile):
    squared = Fraction(jerk) * Fraction(v) + Fraction(a) ** 2 / 2
    if squared >= Fraction(cap) ** 2:
        return cap
    peak = math.nextafter(math.sqrt(_upper_float(squared)), math.inf)
    return min(cap, max(abs(a), profile.peak_deceleration_mps2, peak))


def _model_report(model, attempts, candidate=None):
    return CurvedBrakingReport(
        model.model_id, model.heading_rate_radps, model.path_curvature_per_m,
        model.heading_curvature_per_m, model.sideslip_rad,
        attempts[-1].scale if candidate is not None else None,
        candidate.bounds if candidate is not None else None,
        candidate.sideslip_range_rad if candidate is not None else None,
        candidate.max_heading_step_rad if candidate is not None else None,
        attempts,
    )
