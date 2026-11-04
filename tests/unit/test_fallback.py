from dataclasses import FrozenInstanceError
import math

import numpy as np
import pytest

from hybrid_drive import Trajectory
from hybrid_drive.contracts.motion import EgoMotion
from hybrid_drive.safety.fallback import (
    BrakingConfig,
    FallbackFailure,
    FallbackKind,
    braking_fallback,
)
from hybrid_drive.safety.kinematics import KinematicLimits, KinematicViolation


def limits(interval_s=0.5, speed=100, acceleration=2, jerk=2):
    return KinematicLimits(interval_s, speed, acceleration, jerk)


def assert_path(result, expected_x, interval_s):
    assert isinstance(result.trajectory, Trajectory)
    assert result.trajectory.interval_s == interval_s
    assert result.trajectory.num_poses == len(expected_x)
    np.testing.assert_allclose(result.trajectory.poses[:, 0], expected_x,
                               rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(result.trajectory.poses[:, 1:], 0)
    assert np.isfinite(result.trajectory.poses).all()


# Values are closed-form integrals of the specified jerk phases, not a second
# implementation of the production integrator.
@pytest.mark.parametrize("velocity,acceleration,expected,stop_time,distance", [
    (2, 0, [23 / 24, 5 / 3, 47 / 24, 2, 2], 2, 2),
    (6, 0, [71 / 24, 17 / 3, 95 / 12, 29 / 3, 131 / 12,
            35 / 3, 287 / 24, 12, 12], 4, 12),
    (1, -2, [7 / 24, 1 / 3, 1 / 3, 1 / 3, 1 / 3], 1, 1 / 3),
])
def test_exact_triangular_trapezoidal_and_release_only_stops(
        velocity, acceleration, expected, stop_time, distance):
    result = braking_fallback(EgoMotion(velocity, 0, acceleration, 0),
                              limits(), BrakingConfig(2, 2), len(expected))
    assert_path(result, expected, 0.5)
    assert result.kind is FallbackKind.BRAKING
    assert result.failure_reason is None
    assert result.succeeded is True
    assert result.kinematics.within_limits is True
    assert result.stop_time_s == pytest.approx(stop_time)
    assert result.stop_distance_m == pytest.approx(distance)
    assert result.peak_speed_mps == pytest.approx(velocity)
    assert all(isinstance(value, float) for value in (
        result.stop_time_s, result.stop_distance_m, result.peak_speed_mps,
    ))


def test_initial_positive_acceleration_is_preserved_not_reset_or_clipped():
    result = braking_fallback(EgoMotion(2, 0, 1, 0), limits(),
                              BrakingConfig(2, 2), 6)
    assert_path(result, [13 / 12, 13 / 6, 3, 1737 / 512,
                         5335 / 1536, 667 / 192], 0.5)
    assert result.succeeded is True
    assert result.stop_time_s == pytest.approx(21 / 8)
    assert result.stop_distance_m == pytest.approx(667 / 192)
    assert result.peak_speed_mps == pytest.approx(9 / 4)


def test_zero_velocity_with_positive_acceleration_is_not_a_stationary_state():
    result = braking_fallback(EgoMotion(0, 0, 2, 0), limits(),
                              BrakingConfig(2, 2), 3)
    assert_path(result, [5 / 24, 2 / 3, 9 / 8], 0.5)
    assert result.kind is FallbackKind.BRAKING
    assert result.succeeded is True
    assert result.stop_time_s == pytest.approx(1 + math.sqrt(2))
    assert result.stop_distance_m == pytest.approx(2 / 3 + 1 / math.sqrt(2))
    assert result.peak_speed_mps == pytest.approx(1)


def test_stop_after_horizon_is_a_valid_untruncated_braking_profile():
    result = braking_fallback(EgoMotion(6, 0, 0, 0), limits(),
                              BrakingConfig(2, 2), 3)
    assert_path(result, [71 / 24, 17 / 3, 95 / 12], 0.5)
    assert result.succeeded is True
    assert result.stop_time_s == pytest.approx(4)
    assert result.stop_time_s > result.trajectory.horizon_s
    assert result.stop_distance_m == pytest.approx(12)


def test_release_feasibility_boundary_is_not_softened_or_clipped():
    policy = limits(acceleration=2, jerk=1)
    config = BrakingConfig(2, 1)
    below = braking_fallback(EgoMotion(math.nextafter(0.5, 0), 0, -1, 0), policy, config, 5)
    assert below.failure_reason is FallbackFailure.INFEASIBLE_STOP
    exact = braking_fallback(EgoMotion(0.5, 0, -1, 0), policy, config, 5)
    above = braking_fallback(EgoMotion(math.nextafter(0.5, math.inf), 0, -1, 0), policy, config, 5)
    assert exact.succeeded and above.succeeded
    assert exact.stop_distance_m == pytest.approx(1 / 6)
    assert above.stop_distance_m >= exact.stop_distance_m


def test_tiny_nonzero_motion_does_not_underflow_into_a_stationary_state():
    result = braking_fallback(EgoMotion(1e-200, 0, 0, 0), limits(0.25), BrakingConfig(1, 1e-200), 12)
    assert result.succeeded
    assert result.stop_time_s == pytest.approx(2)
    assert result.stop_distance_m == pytest.approx(1e-200, rel=1e-12, abs=0)
    assert result.trajectory.poses[0, 0] > 0


def test_zero_state_is_successful_stationary_braking():
    result = braking_fallback(EgoMotion(0, 0, 0, 0), limits(),
                              BrakingConfig(2, 2), 4)
    assert_path(result, [0] * 4, 0.5)
    assert result.kind is FallbackKind.BRAKING
    assert result.succeeded is True
    assert result.failure_reason is None
    assert (result.stop_time_s, result.stop_distance_m, result.peak_speed_mps) == (0, 0, 0)
    assert result.kinematics.within_limits is True


@pytest.mark.parametrize("motion,reason", [
    (EgoMotion(-1, 0, 0, 0), FallbackFailure.UNSUPPORTED_MOTION),
    (EgoMotion(1, 1e-15, 0, 0), FallbackFailure.UNSUPPORTED_MOTION),
    (EgoMotion(1, 0, 0, -1e-15), FallbackFailure.UNSUPPORTED_MOTION),
    (EgoMotion(1, 0, 2.1, 0), FallbackFailure.INITIAL_ACCELERATION),
    (EgoMotion(1, 0, -2.1, 0), FallbackFailure.INITIAL_ACCELERATION),
    (EgoMotion(0, 0, -1, 0), FallbackFailure.INFEASIBLE_STOP),
    (EgoMotion(0.9, 0, -2, 0), FallbackFailure.INFEASIBLE_STOP),
])
def test_unsupported_or_infeasible_motion_is_explicit_finite_hold(motion, reason):
    result = braking_fallback(motion, limits(acceleration=100, jerk=100),
                              BrakingConfig(2, 2), 5)
    assert_path(result, [0] * 5, 0.5)
    assert result.kind is FallbackKind.EMERGENCY_HOLD
    assert result.failure_reason is reason
    assert result.succeeded is False
    assert (result.stop_time_s, result.stop_distance_m, result.peak_speed_mps) == (None,) * 3


def test_emergency_hold_is_checked_against_the_actual_initial_motion():
    result = braking_fallback(EgoMotion(-1, 0, 0, 0),
                              limits(1, acceleration=10, jerk=10),
                              BrakingConfig(2, 2), 3)
    report = result.kinematics
    assert report.within_limits is True
    assert report.violations == ()
    assert (report.max_speed_mps, report.max_acceleration_mps2,
            report.max_jerk_mps3) == pytest.approx((1, 2, 6))
    assert result.succeeded is False


def test_unrepresentable_stop_returns_hold_instead_of_nonfinite_poses():
    result = braking_fallback(EgoMotion(1e308, 0, 0, 0),
                              limits(speed=1e308), BrakingConfig(2, 2), 5)
    assert_path(result, [0] * 5, 0.5)
    assert result.kind is FallbackKind.EMERGENCY_HOLD
    assert result.failure_reason is FallbackFailure.NUMERIC_RANGE
    assert result.succeeded is False
    assert result.stop_time_s is None
    assert result.stop_distance_m is None
    assert result.peak_speed_mps is None


def test_built_profile_is_not_discarded_when_initial_speed_exceeds_limit():
    result = braking_fallback(EgoMotion(2, 0, 0, 0), limits(speed=1),
                              BrakingConfig(2, 2), 5)
    assert_path(result, [23 / 24, 5 / 3, 47 / 24, 2, 2], 0.5)
    assert result.kind is FallbackKind.BRAKING
    assert result.failure_reason is FallbackFailure.KINEMATIC_LIMITS
    assert result.succeeded is False
    assert KinematicViolation.SPEED in result.kinematics.violations
    assert result.stop_distance_m == pytest.approx(2)


@pytest.mark.parametrize("relative_delta,accepted", [(0, True), (5e-10, True), (3e-9, False)])
def test_analytic_peak_speed_uses_limit_tolerance_even_between_samples(relative_delta, accepted):
    result = braking_fallback(EgoMotion(2, 0, 1, 0),
                              limits(1, speed=2.25 * (1 - relative_delta)),
                              BrakingConfig(2, 2), 4)
    assert result.kinematics.within_limits is True
    assert result.kinematics.max_speed_mps == pytest.approx(13 / 6)
    assert result.peak_speed_mps == pytest.approx(2.25)
    assert result.succeeded is accepted
    expected = None if accepted else FallbackFailure.KINEMATIC_LIMITS
    assert result.failure_reason is expected
    assert result.kind is FallbackKind.BRAKING


@pytest.mark.parametrize("count", [1, 2])
def test_short_grids_are_returned_but_fail_the_sampled_checker(count):
    result = braking_fallback(EgoMotion(0, 0, 0, 0), limits(),
                              BrakingConfig(2, 2), count)
    assert_path(result, [0] * count, 0.5)
    assert result.kind is FallbackKind.BRAKING
    assert result.failure_reason is FallbackFailure.KINEMATIC_LIMITS
    assert result.succeeded is False
    assert result.kinematics.violations == (KinematicViolation.INSUFFICIENT_POSES,)


@pytest.mark.parametrize("field", ["deceleration_mps2", "jerk_mps3"])
@pytest.mark.parametrize("value", [0, -1, True, False, np.bool_(True), None, "2", 2j,
                                   math.nan, math.inf, -math.inf, 10 ** 1000])
def test_config_rejects_nonpositive_nonfinite_or_nonreal_values(field, value):
    values = dict(deceleration_mps2=2, jerk_mps3=2)
    values[field] = value
    with pytest.raises(ValueError):
        BrakingConfig(**values)


@pytest.mark.parametrize("config", [BrakingConfig(2.1, 2), BrakingConfig(2, 2.1)])
def test_caps_must_fit_limits_for_validation_and_generation(config):
    with pytest.raises(ValueError):
        config.validate_for(limits())
    with pytest.raises(ValueError):
        braking_fallback(EgoMotion(0, 0, 0, 0), limits(), config, 3)


@pytest.mark.parametrize("count", [0, -1, 1.5, 2.0, True, np.bool_(False), "3", None,
                                   math.nan, math.inf, 10 ** 1000])
def test_invalid_sample_counts_raise_value_error(count):
    with pytest.raises(ValueError):
        braking_fallback(EgoMotion(0, 0, 0, 0), limits(), BrakingConfig(2, 2), count)


def test_numpy_integral_count_and_real_config_values_are_supported():
    config = BrakingConfig(np.float64(2), np.int64(2))
    assert config.validate_for(limits()) is None
    result = braking_fallback(EgoMotion(0, 0, 0, 0), limits(), config, np.int64(3))
    assert_path(result, [0] * 3, 0.5)
    assert result.succeeded is True


@pytest.mark.parametrize("index,value", [(0, None), (0, (0, 0, 0, 0)),
                                        (1, None), (1, (1, 2, 2, 2)),
                                        (2, None), (2, (2, 2))])
def test_wrong_argument_types_raise_value_error(index, value):
    args = [EgoMotion(0, 0, 0, 0), limits(), BrakingConfig(2, 2), 3]
    args[index] = value
    with pytest.raises(ValueError):
        braking_fallback(*args)


def test_result_is_frozen():
    result = braking_fallback(EgoMotion(0, 0, 0, 0), limits(), BrakingConfig(2, 2), 3)
    with pytest.raises(FrozenInstanceError):
        result.failure_reason = FallbackFailure.NUMERIC_RANGE
