import math

import pytest

from hybrid_drive import Trajectory
from hybrid_drive.safety.kinematics import (
    EgoMotion,
    KinematicLimits,
    KinematicViolation,
    check_kinematics,
)


# Expected values come from the analytic samples, not production differences.
SAMPLES = {
    "standing": ([(0, 0)] * 3, (0, 0, 0, 0), (0, 0, 0), 1),
    "velocity": ([(3, 4), (6, 8), (9, 12)], (3, 4, 0, 0), (5, 0, 0), 1),
    "acceleration": ([(2, 0), (6, 0), (12, 0)], (1, 0, 2, 0), (6, 2, 0), 1),
    "cubic": (
        [(1 / 6, 0), (8 / 6, 0), (27 / 6, 0)],
        (0, 0, 0, 0), (19 / 6, 2, 1), 1,
    ),
    "two_dimensional": (
        [(2, -0.5), (6, 2), (12, 7.5)],
        (1, -2, 2, 3), (math.sqrt(66.25), math.sqrt(13), 0), 1,
    ),
    "braking": (
        [(2.375, 0), (4.5, 0), (6.375, 0), (8, 0)],
        (5, 0, -1, 0), (5, 1, 0), 0.5,
    ),
    "interior_derivatives": (
        [(0, 0), (0, 0), (1, 0)], (0, 0, 0, 0), (1, 1, 1), 1,
    ),
    "start_velocity_mismatch": (
        [(1, 0), (2, 0), (3, 0)], (0, 0, 0, 0), (1, 2, 6), 1,
    ),
    "initial_acceleration": (
        [(0, 0)] * 3, (0, 0, 3, 4), (0, 5, 15), 1,
    ),
}


def trajectory(points, interval_s=1):
    return Trajectory([[x, y, 0] for x, y in points], interval_s)


def limits(interval_s=1, **overrides):
    values = dict(interval_s=interval_s, max_speed_mps=100,
                  max_acceleration_mps2=100, max_jerk_mps3=100)
    values.update(overrides)
    return KinematicLimits(**values)


def assert_report(report, expected, violations=()):
    assert report.within_limits is (not violations)
    assert isinstance(report.violations, tuple)
    assert set(report.violations) == set(violations)
    assert len(report.violations) == len(violations)
    actual = (report.max_speed_mps, report.max_acceleration_mps2, report.max_jerk_mps3)
    if expected is None:
        assert actual == (None, None, None)
    else:
        assert all(isinstance(value, float) for value in actual)
        assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("name", SAMPLES)
def test_analytic_samples(name):
    points, motion, expected, interval_s = SAMPLES[name]
    report = check_kinematics(trajectory(points, interval_s), EgoMotion(*motion),
                              limits(interval_s))
    assert_report(report, expected)


def test_all_limit_violations_are_reported():
    report = check_kinematics(trajectory([(1, 0), (2, 0), (3, 0)]),
                              EgoMotion(0, 0, 0, 0),
                              KinematicLimits(1, 0.5, 1, 5))
    assert_report(report, (1, 2, 6), (KinematicViolation.SPEED,
                  KinematicViolation.ACCELERATION, KinematicViolation.JERK))


def test_initial_motion_mismatch_can_violate_jerk_alone():
    report = check_kinematics(trajectory([(1, 0), (2, 0), (3, 0)]),
                              EgoMotion(0, 0, 0, 0), limits(max_jerk_mps3=5))
    assert_report(report, (1, 2, 6), (KinematicViolation.JERK,))


def test_heading_does_not_enter_translational_kinematics():
    path = Trajectory([[3, 4, 0.7], [6, 8, -1], [9, 12, 2]], 1)
    assert_report(check_kinematics(path, EgoMotion(3, 4, 0, 0), limits()), (5, 0, 0))


@pytest.mark.parametrize("kind,peak,violation", [
    ("speed", 1, KinematicViolation.SPEED),
    ("acceleration", 2, KinematicViolation.ACCELERATION),
    ("jerk", 1, KinematicViolation.JERK),
])
@pytest.mark.parametrize("relative_delta,accepted", [(0, True), (5e-10, True), (3e-9, False)])
def test_inclusive_threshold_and_relative_tolerance(kind, peak, violation,
                                                   relative_delta, accepted):
    name = {"speed": "velocity", "acceleration": "acceleration", "jerk": "cubic"}[kind]
    points, motion, expected, interval_s = SAMPLES[name]
    if kind == "speed":
        points, motion, expected = [(1, 0), (2, 0), (3, 0)], (1, 0, 0, 0), (1, 0, 0)
    field = {"speed": "max_speed_mps", "acceleration": "max_acceleration_mps2",
             "jerk": "max_jerk_mps3"}[kind]
    policy = limits(interval_s, **{field: peak * (1 - relative_delta)})
    report = check_kinematics(trajectory(points, interval_s), EgoMotion(*motion), policy)
    assert_report(report, expected, () if accepted else (violation,))


@pytest.mark.parametrize("speed,accepted", [(5e-10, True), (3e-9, False)])
def test_absolute_threshold_tolerance(speed, accepted):
    path = trajectory([(speed, 0), (2 * speed, 0), (3 * speed, 0)])
    report = check_kinematics(path, EgoMotion(speed, 0, 0, 0), limits(max_speed_mps=1e-12))
    assert_report(report, (speed, 0, 0), () if accepted else (KinematicViolation.SPEED,))


@pytest.mark.parametrize("count,interval_s,expected", [
    (1, 1, (KinematicViolation.INSUFFICIENT_POSES,)),
    (2, 1, (KinematicViolation.INSUFFICIENT_POSES,)),
    (3, 2, (KinematicViolation.TIME_GRID,)),
    (2, 2, (KinematicViolation.TIME_GRID, KinematicViolation.INSUFFICIENT_POSES)),
])
def test_invalid_grid_and_insufficient_samples(count, interval_s, expected):
    report = check_kinematics(trajectory([(0, 0)] * count, interval_s),
                              EgoMotion(0, 0, 0, 0), limits())
    assert_report(report, None, expected)


@pytest.mark.parametrize("interval_s,accepted", [(1 + 5e-13, True), (1 + 2e-12, False)])
def test_time_grid_relative_tolerance(interval_s, accepted):
    report = check_kinematics(trajectory([(0, 0)] * 3, interval_s),
                              EgoMotion(0, 0, 0, 0), limits())
    assert_report(report, (0, 0, 0) if accepted else None,
                  () if accepted else (KinematicViolation.TIME_GRID,))


def test_time_grid_has_no_absolute_tolerance():
    report = check_kinematics(trajectory([(0, 0)] * 3, 0.5e-9),
                              EgoMotion(0, 0, 0, 0), limits(1e-9))
    assert_report(report, None, (KinematicViolation.TIME_GRID,))


@pytest.mark.parametrize("points,interval_s", [
    ([(1e308, 0), (-1e308, 0), (1e308, 0)], 1),
    ([(1, 0), (2, 0), (3, 0)], 1e-300),
])
def test_arithmetic_overflow_returns_numeric_range(points, interval_s):
    report = check_kinematics(trajectory(points, interval_s),
                              EgoMotion(0, 0, 0, 0), limits(interval_s))
    assert_report(report, None, (KinematicViolation.NUMERIC_RANGE,))


def test_large_finite_euclidean_norm_is_not_a_numeric_error():
    speed = float(2 ** 600)
    path = trajectory([(speed, speed), (2 * speed, 2 * speed), (3 * speed, 3 * speed)])
    report = check_kinematics(path, EgoMotion(speed, speed, 0, 0),
                              KinematicLimits(1, 1e308, 1e308, 1e308))
    assert_report(report, (math.hypot(speed, speed), 0, 0))


@pytest.mark.parametrize("field", ["vx_mps", "vy_mps", "ax_mps2", "ay_mps2"])
@pytest.mark.parametrize("invalid", [True, False, None, "1", 1 + 0j,
                                     math.nan, math.inf, -math.inf])
def test_motion_rejects_nonfinite_or_nonreal_values(field, invalid):
    values = dict(vx_mps=0, vy_mps=0, ax_mps2=0, ay_mps2=0)
    values[field] = invalid
    with pytest.raises(ValueError):
        EgoMotion(**values)


@pytest.mark.parametrize("field", ["interval_s", "max_speed_mps",
                                    "max_acceleration_mps2", "max_jerk_mps3"])
@pytest.mark.parametrize("invalid", [0, -1, True, False, None, "1", 1 + 0j,
                                     math.nan, math.inf, -math.inf])
def test_limits_reject_nonpositive_nonfinite_or_nonreal_values(field, invalid):
    values = dict(interval_s=1, max_speed_mps=1, max_acceleration_mps2=1, max_jerk_mps3=1)
    values[field] = invalid
    with pytest.raises(ValueError):
        KinematicLimits(**values)


@pytest.mark.parametrize("index,invalid", [(0, None), (0, [[1, 0, 0]]),
                                          (1, None), (1, (0, 0, 0, 0)),
                                          (2, None), (2, (1, 1, 1, 1))])
def test_checker_rejects_wrong_argument_types_with_value_error(index, invalid):
    args = [trajectory([(0, 0)] * 3), EgoMotion(0, 0, 0, 0), limits()]
    args[index] = invalid
    with pytest.raises(ValueError):
        check_kinematics(*args)
