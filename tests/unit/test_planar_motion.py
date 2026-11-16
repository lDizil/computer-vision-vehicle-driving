from dataclasses import FrozenInstanceError
from fractions import Fraction
import math

import numpy as np
import pytest

from hybrid_drive.contracts.motion import EgoMotion
from hybrid_drive.safety.planar_motion import (
    PlanarMotionError,
    PlanarMotionFailure,
    PlanarMotionHypothesis,
)


def hypothesis(motion=EgoMotion(1, 0, 0, 0.5), rate=0.2):
    return PlanarMotionHypothesis.from_motion(motion, rate)


def test_exact_arc_fixture_and_independent_body_heading():
    model = hypothesis()
    distances = np.array([0, 23 / 48, 5 / 6, 47 / 48, 1])
    poses = model.poses_at_distances(distances)
    expected = np.column_stack((2 * np.sin(distances / 2),
                                4 * np.sin(distances / 4) ** 2, distances / 5))
    np.testing.assert_allclose(poses, expected, rtol=1e-14, atol=1e-15)
    assert (model.speed_mps, model.sideslip_rad, model.tangential_acceleration_mps2,
            model.normal_acceleration_mps2, model.path_curvature_per_m,
            model.heading_curvature_per_m) == pytest.approx((1, 0, 0, 0.5, 0.5, 0.2))
    assert model.sideslip_range(1) == pytest.approx((0, 0.3))
    with pytest.raises(FrozenInstanceError):
        model.heading_rate_radps = 0.5


def test_rotation_does_not_rotate_initial_body_heading():
    base = hypothesis()
    rotated = hypothesis(EgoMotion(0.6, 0.8, -0.4, 0.3))
    distances = [0, 0.25, 0.5, 1]
    expected = base.poses_at_distances(distances)
    expected[:, :2] = expected[:, :2] @ np.array([[0.6, 0.8], [-0.8, 0.6]])
    np.testing.assert_allclose(rotated.poses_at_distances(distances), expected, rtol=1e-14, atol=1e-15)
    assert rotated.sideslip_range(1) == pytest.approx((math.atan2(4, 3), math.atan2(4, 3) + 0.3))


def test_mirror_changes_lateral_motion_and_yaw_not_magnitudes():
    left = hypothesis(EgoMotion(3, 0.2, -0.3, 0.5), rate=-0.1)
    right = hypothesis(EgoMotion(3, -0.2, -0.3, -0.5), rate=0.1)
    expected = left.poses_at_distances([0, 0.5, 2]) * [1, -1, -1]
    np.testing.assert_allclose(right.poses_at_distances([0, 0.5, 2]), expected)
    assert left.continuous_bounds(4, 2, 1) == right.continuous_bounds(4, 2, 1)
    low, high = left.sideslip_range(2)
    assert right.sideslip_range(2) == pytest.approx((-high, -low))


def test_initial_position_derivatives_match_lateral_state_and_supplied_yaw():
    motion = EgoMotion(2, 0.3, 0.4, -0.7)
    model = hypothesis(motion, rate=0.15)
    h = 1e-4
    times = np.arange(4) * h
    distances = model.speed_mps * times + model.tangential_acceleration_mps2 * times**2 / 2
    poses = model.poses_at_distances(distances)
    velocity = (-3 * poses[0] + 4 * poses[1] - poses[2]) / (2 * h)
    acceleration = (2 * poses[0] - 5 * poses[1] + 4 * poses[2] - poses[3]) / h**2
    np.testing.assert_allclose(velocity, [motion.vx_mps, motion.vy_mps, 0.15], atol=1e-8)
    np.testing.assert_allclose(acceleration[:2], [motion.ax_mps2, motion.ay_mps2], atol=1e-7)
    assert acceleration[2] == pytest.approx(model.heading_curvature_per_m * model.tangential_acceleration_mps2,
                                             rel=1e-7)


def test_global_full_vector_bounds_and_heading_derivative_bounds():
    bounds = hypothesis().continuous_bounds(1, 1, 1)
    assert bounds.acceleration_mps2 >= math.sqrt(5) / 2
    assert bounds.jerk_mps3 >= math.sqrt(61) / 4
    assert bounds.acceleration_mps2 == pytest.approx(math.sqrt(5) / 2, rel=1e-14)
    assert bounds.jerk_mps3 == pytest.approx(math.sqrt(61) / 4, rel=1e-14)
    assert (bounds.heading_rate_radps, bounds.heading_acceleration_radps2,
            bounds.heading_jerk_radps3, bounds.sideslip_rate_radps) == pytest.approx((0.2, 0.2, 0.2, 0.3))
    # At t=0+ the scalar jerk -1 gives full jerk (-1.25,0).
    assert bounds.jerk_mps3 >= 1.25 > 1
    assert bounds.acceleration_mps2 < 2 and bounds.jerk_mps3 < 2
    assert "prescribed_variable_sideslip_not_vehicle_dynamics" in bounds.assumptions


@pytest.mark.parametrize("curvature,rate", [(1, -1e-16), (-1, 1e-16)])
def test_sideslip_rate_bound_covers_rounded_curvature_subtraction(curvature, rate):
    model = hypothesis(EgoMotion(1, 0, 0, curvature), rate=rate)
    exact = abs(Fraction(model.path_curvature_per_m) - Fraction(model.heading_curvature_per_m))
    bounds = model.continuous_bounds(1, 1, 1)
    assert exact > 1
    assert Fraction(bounds.sideslip_rate_radps) >= exact


@pytest.mark.parametrize("motion,rate,distance", [
    (EgoMotion(1, 0, 0, 1), -1e-16, 1),
    (EgoMotion(1, 0, 0, -1), 1e-16, 1),
    (EgoMotion(2, 0.3, 0.4, -0.7), 0.15, 1.7),
    (EgoMotion(2, -0.3, 0.4, 0.7), -0.15, 1.7),
])
def test_sideslip_range_encloses_exact_stored_coefficient_endpoint(motion, rate, distance):
    model = hypothesis(motion, rate)
    start = Fraction(model.sideslip_rad)
    end = start + (Fraction(model.path_curvature_per_m) - Fraction(model.heading_curvature_per_m)) * Fraction(distance)
    lower, upper = model.sideslip_range(distance)
    assert Fraction(lower) <= min(start, end)
    assert Fraction(upper) >= max(start, end)
    assert model.sideslip_range(0) == (model.sideslip_rad, model.sideslip_rad)


def test_exact_straight_bounds_do_not_artificially_exceed_full_caps():
    bounds = hypothesis(EgoMotion(2, 0, 1, 0), rate=0).continuous_bounds(3, 2, 4)
    assert (bounds.speed_mps, bounds.acceleration_mps2, bounds.jerk_mps3) == (3, 2, 4)
    assert (bounds.heading_rate_radps, bounds.heading_acceleration_radps2,
            bounds.heading_jerk_radps3, bounds.sideslip_rate_radps) == (0, 0, 0, 0)


def test_small_nonzero_curvature_cannot_be_certified_as_exact_straight_caps():
    bounds = hypothesis(EgoMotion(1, 0, 0, 1e-200), rate=0).continuous_bounds(1, 1, 1)
    assert bounds.acceleration_mps2 > 1
    assert bounds.jerk_mps3 > 1


@pytest.mark.parametrize("kappa", [0.5, -0.5, 20])
def test_full_profile_analytic_derivatives_stay_within_global_bounds(kappa):
    model = hypothesis(EgoMotion(1, 0, 0, kappa), rate=-0.3)
    bounds = model.continuous_bounds(1, 1, 1)
    for t in np.linspace(0, 2, 401):
        if t <= 1:
            speed, acceleration, jerk = 1 - t**2 / 2, -t, -1
        else:
            speed, acceleration, jerk = (2 - t)**2 / 2, t - 2, 1
        full_acceleration = math.hypot(acceleration, kappa * speed**2)
        full_jerk = math.hypot(jerk - kappa**2 * speed**3, 3 * kappa * speed * acceleration)
        assert full_acceleration <= bounds.acceleration_mps2
        assert full_jerk <= bounds.jerk_mps3
        assert abs(-0.3 * speed) <= bounds.heading_rate_radps
        assert abs(-0.3 * acceleration) <= bounds.heading_acceleration_radps2


def test_positive_acceleration_requires_complete_profile_peak_not_initial_speed():
    model = hypothesis(EgoMotion(1, 0, 1, 0.1), rate=0.2)
    initial_only = model.continuous_bounds(1, 1, 1)
    complete = model.continuous_bounds(1.5, 1, 1)
    assert complete.speed_mps == 1.5
    assert complete.acceleration_mps2 > initial_only.acceleration_mps2
    assert complete.jerk_mps3 > initial_only.jerk_mps3
    assert complete.acceleration_mps2 >= math.hypot(1, 0.1 * 1.5**2)


def test_tiny_lateral_components_and_heading_rate_are_not_zeroed():
    model = hypothesis(EgoMotion(1, 1e-15, 0, -1e-15), rate=1e-15)
    assert model.sideslip_rad > 0
    assert model.path_curvature_per_m < 0
    assert model.heading_curvature_per_m > 0
    poses = model.poses_at_distances([0.25, 0.5])
    assert np.all(poses[:, 1] > 0)
    assert np.all(poses[:, 2] > 0)


@pytest.mark.parametrize("curvature", [0, 1e-15, -1e-15])
def test_straight_limit_retains_small_normal_displacement(curvature):
    model = hypothesis(EgoMotion(1, 0, 0, curvature), rate=0)
    pose = model.poses_at_distances([1])[0]
    assert pose[0] == pytest.approx(1, rel=1e-15)
    assert pose[1] == pytest.approx(curvature / 2, rel=1e-15, abs=0)
    assert pose[2] == 0


def test_heading_and_sideslip_remain_unwrapped_without_hidden_admission_threshold():
    model = hypothesis(rate=-2)
    assert model.poses_at_distances([4])[0, 2] == -8
    low, high = model.sideslip_range(4)
    assert low == 0 and high >= 10
    assert high == pytest.approx(10, rel=1e-14)


def test_zero_supplied_body_rate_does_not_become_nonzero_course_rate():
    model = hypothesis(rate=0)
    poses = model.poses_at_distances([0.5, 1])
    assert np.all(poses[:, 1] > 0)
    np.testing.assert_array_equal(poses[:, 2], 0)
    assert model.sideslip_range(1) == (0, 0.5)


def test_arc_mapping_is_continuous_across_stable_formula_branches():
    distances = np.array([math.nextafter(2, 0), 2, math.nextafter(2, math.inf), 4, 10])
    poses = hypothesis().poses_at_distances(distances)
    np.testing.assert_allclose(poses[:, 0], 2 * np.sin(distances / 2), rtol=1e-14)
    np.testing.assert_allclose(poses[:, 1], 4 * np.sin(distances / 4) ** 2, rtol=1e-14)


def test_unrepresentable_normal_displacement_is_not_returned_as_straight():
    model = hypothesis(EgoMotion(1, 0, 0, math.ulp(0)), rate=0)
    with pytest.raises(PlanarMotionError) as error:
        model.poses_at_distances([1])
    assert error.value.reason is PlanarMotionFailure.NUMERIC_RANGE


@pytest.mark.parametrize("motion", [
    EgoMotion(-1e-200, 0, 0, 0), EgoMotion(0, 1, 0, 0),
    EgoMotion(0, 0, 0, 0), EgoMotion(0, 0, 1, 0),
])
def test_reverse_lateral_or_undefined_zero_speed_are_not_projected(motion):
    with pytest.raises(PlanarMotionError) as error:
        hypothesis(motion)
    assert error.value.reason is PlanarMotionFailure.UNSUPPORTED_MOTION


@pytest.mark.parametrize("rate", [None, True, float("nan"), float("inf")])
def test_unknown_or_invalid_heading_rate_is_not_inferred_from_motion(rate):
    with pytest.raises(ValueError):
        hypothesis(rate=rate)


@pytest.mark.parametrize("motion,rate", [
    (EgoMotion(1.4e308, 1.4e308, 0, 0), 0),
    (EgoMotion(1e-200, 0, 0, 1), 0),
    (EgoMotion(1e200, 0, 0, 1e-200), 0),
    (EgoMotion(1e-200, 0, 0, 0), 1e200),
    (EgoMotion(1e200, 0, 0, 0), 1e-200),
])
def test_unrepresentable_norm_curvature_or_heading_rate_is_numeric_failure(motion, rate):
    with pytest.raises(PlanarMotionError) as error:
        hypothesis(motion, rate)
    assert error.value.reason is PlanarMotionFailure.NUMERIC_RANGE


def test_tiny_nonzero_forward_speed_is_not_stationary():
    model = hypothesis(EgoMotion(1e-200, 0, 0, 0), rate=0)
    assert model.speed_mps == 1e-200
    np.testing.assert_array_equal(model.poses_at_distances([1e-200]), [[1e-200, 0, 0]])


@pytest.mark.parametrize("operation", [
    lambda model: model.poses_at_distances([1e308]),
    lambda model: model.continuous_bounds(1e308, 1, 1),
    lambda model: model.sideslip_range(1e308),
])
def test_overflow_is_explicit_numeric_failure(operation):
    model = hypothesis(EgoMotion(1, 0, 0, 20), rate=-20)
    with pytest.raises(PlanarMotionError) as error:
        operation(model)
    assert error.value.reason is PlanarMotionFailure.NUMERIC_RANGE


@pytest.mark.parametrize("speed,acceleration,jerk", [(0.5, 1, 1), (2, 0.5, 1), (2, 1, -1)])
def test_bound_inputs_cannot_exclude_measured_initial_state(speed, acceleration, jerk):
    with pytest.raises(ValueError):
        hypothesis(EgoMotion(1, 0, 1, 0)).continuous_bounds(speed, acceleration, jerk)
