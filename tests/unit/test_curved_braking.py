from dataclasses import FrozenInstanceError
from fractions import Fraction
import math

import numpy as np
import pytest

from hybrid_drive.contracts.motion import EgoMotion
from hybrid_drive.safety.curved_braking import CurvedBrakingOutcome, curved_braking_fallback
from hybrid_drive.safety.fallback import BrakingConfig, FallbackFailure, FallbackKind, unproven_fallback
from hybrid_drive.safety.kinematics import KinematicLimits, KinematicViolation


def limits(interval_s=0.5, speed=20, acceleration=2, jerk=2):
    return KinematicLimits(interval_s, speed, acceleration, jerk)


def curved(motion=EgoMotion(1, 0, 0, 0.5), rate=0.2, policy=None, config=None, count=6):
    return curved_braking_fallback(motion, rate, policy or limits(), config or BrakingConfig(2, 2), count)


def assert_failed_hold(result, reason):
    assert not result.succeeded
    assert result.failure_reason is reason
    assert result.kind is FallbackKind.EMERGENCY_HOLD
    np.testing.assert_array_equal(result.trajectory.poses, 0)
    assert (result.stop_time_s, result.stop_distance_m, result.peak_speed_mps) == (None, None, None)


def test_exact_triangular_arc_and_independent_body_heading():
    result = curved()
    # Closed-form scalar integrals for q=A_t=1, not production profile helpers.
    distances = np.array([23 / 48, 5 / 6, 47 / 48, 1, 1, 1])
    expected = np.column_stack((2 * np.sin(distances / 2),
                                4 * np.sin(distances / 4) ** 2, distances / 5))
    np.testing.assert_allclose(result.trajectory.poses, expected, rtol=1e-13, atol=1e-14)
    assert result.kind is FallbackKind.CURVED_BRAKING
    assert result.succeeded and result.kinematics.within_limits
    assert (result.stop_time_s, result.stop_distance_m, result.peak_speed_mps) == pytest.approx((2, 1, 1))
    report = result.model_report
    assert report.selected_scale == 0.5
    assert report.path_curvature_per_m == 0.5 and report.heading_curvature_per_m == 0.2
    assert report.sideslip_range_rad == pytest.approx((0, 0.3))
    assert report.continuous_bounds.acceleration_mps2 == pytest.approx(math.sqrt(5) / 2)
    assert report.continuous_bounds.jerk_mps3 == pytest.approx(math.sqrt(61) / 4)
    assert [attempt.outcome for attempt in report.attempts] == [
        CurvedBrakingOutcome.CONTINUOUS_LIMITS, CurvedBrakingOutcome.CERTIFIED,
    ]
    assert report.max_heading_step_rad >= 0.1
    assert "frozen_snapshot_coassociation_not_measurement_synchronization" in report.assumptions
    with pytest.raises(FrozenInstanceError):
        report.selected_scale = 1


def test_rotation_preserves_initial_body_heading_and_rotates_position():
    reference = curved().trajectory.poses.copy()
    reference[:, :2] = reference[:, :2] @ np.array([[0.6, 0.8], [-0.8, 0.6]])
    result = curved(EgoMotion(0.6, 0.8, -0.4, 0.3))
    assert result.succeeded
    np.testing.assert_allclose(result.trajectory.poses, reference, rtol=1e-13, atol=1e-14)
    assert result.model_report.initial_sideslip_rad == pytest.approx(math.atan2(4, 3))


def test_mirror_preserves_scalar_stop_and_full_bounds():
    left = curved(EgoMotion(1, 0.05, 0.1, 0.5), rate=-0.1)
    right = curved(EgoMotion(1, -0.05, 0.1, -0.5), rate=0.1)
    assert left.succeeded and right.succeeded
    np.testing.assert_allclose(right.trajectory.poses, left.trajectory.poses * [1, -1, -1])
    assert left.stop_time_s == right.stop_time_s
    assert left.stop_distance_m == right.stop_distance_m
    assert left.model_report.continuous_bounds == right.model_report.continuous_bounds


def test_exact_trapezoid_stop_and_stationary_endpoint():
    result = curved(EgoMotion(4, 0, 0, 0.16), rate=0.02, count=12)
    assert result.succeeded and result.model_report.selected_scale == 0.5
    assert (result.stop_time_s, result.stop_distance_m) == (5, 10)
    distances = np.array([95 / 48, 23 / 6, 131 / 24, 41 / 6, 191 / 24,
                          53 / 6, 227 / 24, 59 / 6, 479 / 48, 10, 10, 10])
    expected = np.column_stack((100 * np.sin(distances / 100),
                                200 * np.sin(distances / 200) ** 2, distances / 200))
    np.testing.assert_allclose(result.trajectory.poses, expected, rtol=1e-13, atol=1e-14)
    np.testing.assert_array_equal(result.trajectory.poses[-3:], np.tile(result.trajectory.poses[-1], (3, 1)))


def test_full_stop_metadata_and_admission_extend_beyond_output_horizon():
    result = curved(EgoMotion(4, 0, 0, 0.16), rate=0.02, count=3)
    assert result.succeeded
    assert result.stop_time_s == 5 > result.trajectory.horizon_s
    assert result.stop_distance_m == 10
    assert result.trajectory.poses[-1, 0] < result.stop_distance_m
    assert result.model_report.sideslip_range_rad[1] >= 0.05


def test_forward_body_motion_does_not_require_monotone_fixed_frame_x():
    result = curved(EgoMotion(1, 0, 0, 1), rate=1, count=12)
    assert result.succeeded
    assert result.model_report.selected_scale == 0.125
    assert result.model_report.sideslip_range_rad == (0, 0)
    assert np.any(np.diff(result.trajectory.poses[:, 0]) < 0)
    assert result.stop_distance_m == pytest.approx(2.5)
    assert result.stop_distance_m > math.hypot(*result.trajectory.poses[-1, :2])


def test_release_only_boundary_preserves_initial_braking_and_full_stop():
    result = curved(EgoMotion(0.5, 0, -1, 0.125), rate=0.1, policy=limits(0.25), count=6)
    assert result.succeeded and result.model_report.selected_scale == 0.5
    assert result.stop_time_s == 1
    assert result.stop_distance_m == pytest.approx(1 / 6)
    distances = np.array([37 / 384, 7 / 48, 21 / 128, 1 / 6, 1 / 6, 1 / 6])
    expected = np.column_stack((2 * np.sin(distances / 2),
                                4 * np.sin(distances / 4) ** 2, distances / 5))
    np.testing.assert_allclose(result.trajectory.poses, expected, rtol=1e-13, atol=1e-14)
    below = math.nextafter(0.5, 0)
    failed = curved(EgoMotion(below, 0, -1, 0.5 * below**2), rate=0.1, policy=limits(0.25))
    assert_failed_hold(failed, FallbackFailure.NO_CERTIFIED_PROFILE)
    assert failed.model_report.attempts[1].outcome is CurvedBrakingOutcome.SCALAR_INFEASIBLE


def test_positive_acceleration_is_preserved_and_continuous_peak_is_reported():
    result = curved(EgoMotion(1, 0, 1, 0.1), rate=0.2, count=8)
    assert result.succeeded
    assert result.model_report.selected_scale == 0.5
    assert result.stop_time_s == pytest.approx(3.5)
    assert result.stop_distance_m == pytest.approx(77 / 24)
    assert result.peak_speed_mps == 1.5
    assert result.model_report.continuous_bounds.speed_mps >= 1.5
    assert result.kinematics.max_speed_mps < 1.49


def test_hidden_speed_peak_rejects_candidate_even_when_coarse_means_would_pass():
    result = curved(EgoMotion(1, 0, 1, 0.1), policy=limits(speed=1.49), count=8)
    assert_failed_hold(result, FallbackFailure.NO_CERTIFIED_PROFILE)
    assert result.model_report.selected_scale is None
    assert [attempt.scale for attempt in result.model_report.attempts] == [2.0 ** -i for i in range(13)]
    assert result.model_report.attempts[0].outcome is CurvedBrakingOutcome.CONTINUOUS_LIMITS
    assert result.model_report.attempts[1].outcome is CurvedBrakingOutcome.CONTINUOUS_LIMITS
    assert all(attempt.outcome is CurvedBrakingOutcome.SCALAR_BUDGET
               for attempt in result.model_report.attempts[2:])


def test_search_checks_original_full_config_not_only_larger_kinematic_limits():
    result = curved(policy=limits(acceleration=8, jerk=20))
    assert result.succeeded and result.model_report.selected_scale == 0.5
    assert result.model_report.continuous_bounds.acceleration_mps2 < 2
    assert result.model_report.continuous_bounds.jerk_mps3 < 2
    assert result.model_report.attempts[0].outcome is CurvedBrakingOutcome.CONTINUOUS_LIMITS


def test_full_stop_sideslip_rejects_even_if_short_output_horizon_is_forward():
    result = curved(EgoMotion(4, 0, 0, 0.16), rate=-0.8, policy=limits(0.1), count=3)
    assert_failed_hold(result, FallbackFailure.NO_CERTIFIED_PROFILE)
    assert any(attempt.outcome is CurvedBrakingOutcome.SIDESLIP for attempt in result.model_report.attempts)
    assert all(attempt.outcome is not CurvedBrakingOutcome.CERTIFIED for attempt in result.model_report.attempts)


@pytest.mark.parametrize("margin,success", [(-1e-6, True), (0, False), (1e-6, False)])
def test_body_forward_sideslip_boundary_is_strict(margin, success):
    result = curved(rate=0.5 + math.pi / 2 + margin, policy=limits(0.1), count=25)
    assert result.succeeded is success
    if success:
        assert result.model_report.sideslip_range_rad[0] > -math.pi / 2
    else:
        assert_failed_hold(result, FallbackFailure.NO_CERTIFIED_PROFILE)
        assert any(attempt.outcome is CurvedBrakingOutcome.SIDESLIP for attempt in result.model_report.attempts)


def test_heading_alias_bound_rejects_without_changing_limits_or_zeroing_yaw():
    motion = EgoMotion(0.001, 0, 0, 0.01)
    coarse = curved(motion, rate=10, policy=limits(0.5, acceleration=1, jerk=1), config=BrakingConfig(1, 1))
    assert_failed_hold(coarse, FallbackFailure.NO_CERTIFIED_PROFILE)
    assert any(attempt.outcome is CurvedBrakingOutcome.HEADING_ALIAS for attempt in coarse.model_report.attempts)
    fine = curved(motion, rate=10, policy=limits(0.1, acceleration=1, jerk=1), config=BrakingConfig(1, 1))
    assert fine.succeeded
    assert fine.model_report.max_heading_step_rad < math.pi


def test_initial_velocity_acceleration_and_independent_heading_rate_are_preserved():
    motion = EgoMotion(2, 0.3, 0.4, -0.7)
    h = 1e-4
    result = curved(motion, rate=0.15, policy=limits(h), count=4)
    assert result.succeeded
    poses = np.vstack((np.zeros((1, 3)), result.trajectory.poses))
    velocity = (-3 * poses[0] + 4 * poses[1] - poses[2]) / (2 * h)
    acceleration = (2 * poses[0] - 5 * poses[1] + 4 * poses[2] - poses[3]) / h**2
    np.testing.assert_allclose(velocity, [2, 0.3, 0.15], rtol=0, atol=1e-8)
    np.testing.assert_allclose(acceleration[:2], [0.4, -0.7], rtol=0, atol=1e-6)


def test_exact_rest_needs_exact_zero_angular_motion():
    result = curved(EgoMotion(0, 0, 0, 0), rate=0)
    assert result.succeeded and result.kind is FallbackKind.CURVED_BRAKING
    np.testing.assert_array_equal(result.trajectory.poses, 0)
    assert (result.stop_time_s, result.stop_distance_m, result.peak_speed_mps) == (0, 0, 0)
    assert result.model_report.attempts == ()
    assert result.model_report.model_id == "exact-rest-v1"
    assert_failed_hold(curved(EgoMotion(0, 0, 0, 0), rate=1e-200), FallbackFailure.UNSUPPORTED_HEADING)


@pytest.mark.parametrize("motion", [
    EgoMotion(-1e-200, 0, 0, 0), EgoMotion(0, 1, 0, 0),
    EgoMotion(0, 0, 1e-200, 0), EgoMotion(0, 0, 0, 1e-200),
])
def test_reverse_lateral_and_undefined_zero_speed_are_explicitly_unsupported(motion):
    assert_failed_hold(curved(motion, rate=0), FallbackFailure.UNSUPPORTED_MOTION)


def test_tiny_nonzero_motion_is_not_accepted_as_stationary_hold():
    result = curved(EgoMotion(1e-200, 0, 0, 0), rate=0,
                    policy=limits(0.25, acceleration=1, jerk=1e-200), config=BrakingConfig(1, 1e-200))
    assert result.succeeded
    assert result.stop_time_s > 0 and result.stop_distance_m > 0
    assert result.trajectory.poses[0, 0] > 0


def test_tiny_lateral_state_and_supplied_heading_rate_are_not_zeroed():
    result = curved(EgoMotion(1, 1e-15, 0, -1e-15), rate=1e-15)
    assert result.succeeded
    assert np.all(result.trajectory.poses[:, 1] > 0)
    assert np.all(result.trajectory.poses[:, 2] > 0)
    assert result.model_report.path_curvature_per_m < 0
    assert result.model_report.heading_curvature_per_m > 0


@pytest.mark.parametrize("rate,reason", [
    (None, FallbackFailure.MISSING_HEADING_RATE),
    (math.nan, FallbackFailure.UNSUPPORTED_HEADING),
    (math.inf, FallbackFailure.UNSUPPORTED_HEADING),
    (True, FallbackFailure.UNSUPPORTED_HEADING),
])
def test_missing_or_undefined_heading_is_not_reconstructed_from_course(rate, reason):
    assert_failed_hold(curved(rate=rate), reason)


def test_initial_full_acceleration_is_not_replaced_by_its_tangential_component():
    result = curved(EgoMotion(1, 0, 0, math.nextafter(2, math.inf)), rate=0)
    assert_failed_hold(result, FallbackFailure.INITIAL_ACCELERATION)


def test_scalar_infeasible_candidates_do_not_imply_global_infeasibility():
    result = curved(EgoMotion(0.01, 0, -1, 0), rate=0)
    assert_failed_hold(result, FallbackFailure.NO_CERTIFIED_PROFILE)
    outcomes = {attempt.outcome for attempt in result.model_report.attempts}
    assert outcomes == {CurvedBrakingOutcome.SCALAR_INFEASIBLE, CurvedBrakingOutcome.SCALAR_BUDGET}
    assert result.failure_reason is not FallbackFailure.INFEASIBLE_STOP


def test_unrepresentable_model_is_a_numeric_failure_with_finite_placeholder():
    result = curved(EgoMotion(1e-200, 0, 0, 1), rate=0)
    assert_failed_hold(result, FallbackFailure.NUMERIC_RANGE)


def test_all_unrepresentable_profiles_are_uncertified_not_a_fictional_stop():
    result = curved(EgoMotion(1e308, 0, 0, 0), rate=0, policy=limits(speed=1e308))
    assert_failed_hold(result, FallbackFailure.NO_CERTIFIED_PROFILE)
    assert len(result.model_report.attempts) == 13
    assert all(attempt.outcome is CurvedBrakingOutcome.NUMERIC_RANGE for attempt in result.model_report.attempts)


def test_numeric_failure_of_one_budget_does_not_prevent_later_certification():
    result = curved(EgoMotion(2, 0, 0, 0), rate=0,
                    policy=limits(acceleration=1e308, jerk=1e308), config=BrakingConfig(1e308, 1e308))
    assert result.succeeded
    assert [attempt.outcome for attempt in result.model_report.attempts] == [
        CurvedBrakingOutcome.NUMERIC_RANGE, CurvedBrakingOutcome.CERTIFIED,
    ]
    assert result.model_report.selected_scale == 0.5
    assert result.stop_distance_m > 0


@pytest.mark.parametrize("count", [1, 2])
def test_insufficient_grid_never_becomes_a_successful_curved_profile(count):
    result = curved(count=count)
    assert_failed_hold(result, FallbackFailure.NO_CERTIFIED_PROFILE)
    assert result.kinematics.violations == (KinematicViolation.INSUFFICIENT_POSES,)
    assert any(attempt.outcome is CurvedBrakingOutcome.SAMPLED_LIMITS for attempt in result.model_report.attempts)


def test_owner_dispatch_can_return_a_typed_unproven_placeholder():
    result = unproven_fallback(EgoMotion(1, 0.1, 0, 0), limits(), 8, FallbackFailure.OBSERVATION_MISMATCH)
    assert_failed_hold(result, FallbackFailure.OBSERVATION_MISMATCH)
    assert result.trajectory.num_poses == 8 and result.model_report is None


def test_peak_guard_does_not_round_real_speed_rise_down_to_the_limit():
    motion = EgoMotion(1, 0, 1e-8, 0)
    result = curved(motion, rate=0, policy=limits(speed=1), count=8)
    assert_failed_hold(result, FallbackFailure.NO_CERTIFIED_PROFILE)
    assert all(attempt.outcome is CurvedBrakingOutcome.CONTINUOUS_LIMITS for attempt in result.model_report.attempts)
    exact_peak = Fraction(1) + Fraction(1e-8) ** 2 / 4
    assert exact_peak > 1 and float(exact_peak) == 1
