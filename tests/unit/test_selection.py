from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from hybrid_drive import Pose2D, Trajectory
from hybrid_drive.contracts.motion import EgoMotion
from hybrid_drive.contracts.planning import Candidate, CandidateSet
from hybrid_drive.safety.fallback import BrakingConfig, FallbackFailure, FallbackKind
from hybrid_drive.safety.kinematics import KinematicLimits, KinematicViolation
from hybrid_drive.safety.selection import DecisionStatus, select_trajectory


def trajectory(xs, interval_s=0.5, heading=0):
    return Trajectory([[x, 0, heading] for x in xs], interval_s)


def plan(*candidates):
    return CandidateSet("plan", "snapshot", 123, Pose2D(10, -20, 1.5), candidates)


def limits(interval_s=0.5, speed=10, acceleration=2, jerk=2):
    return KinematicLimits(interval_s, speed, acceleration, jerk)


def assert_braking_path(decision):
    expected = [[23 / 24, 0, 0], [5 / 3, 0, 0], [47 / 24, 0, 0],
                [2, 0, 0], [2, 0, 0]]
    np.testing.assert_allclose(decision.trajectory.poses, expected, rtol=1e-12, atol=1e-12)
    assert decision.trajectory.num_poses == 5
    assert decision.trajectory.interval_s == 0.5
    assert decision.candidate_id is None
    assert decision.fallback.kind is FallbackKind.BRAKING
    assert decision.fallback.stop_time_s == pytest.approx(2)
    assert decision.fallback.stop_distance_m == pytest.approx(2)
    assert decision.fallback.peak_speed_mps == pytest.approx(2)
    np.testing.assert_array_equal(decision.trajectory.poses, decision.fallback.trajectory.poses)


def test_first_eligible_candidate_in_plan_order_is_selected_and_all_are_assessed():
    request = plan(
        Candidate("too-fast", trajectory([1, 2, 3], 1)),
        Candidate("z-first", trajectory([0, 0, 0], 1, heading=1)),
        Candidate("a-later", trajectory([0, 0, 0], 1, heading=-1)),
        Candidate("bad-after-selection", trajectory([0, 0, 1], 1)),
    )
    decision = select_trajectory(request, EgoMotion(0, 0, 0, 0),
                                 limits(1, speed=0.5, acceleration=2, jerk=6),
                                 BrakingConfig(2, 2))
    assert decision.status is DecisionStatus.CANDIDATE
    assert decision.failed is False
    assert decision.candidate_id == "z-first"
    assert decision.plan is request
    assert decision.fallback is None
    np.testing.assert_array_equal(decision.trajectory.poses, [[0, 0, 1]] * 3)
    assert isinstance(decision.assessments, tuple)
    assert [item.candidate_id for item in decision.assessments] == [
        "too-fast", "z-first", "a-later", "bad-after-selection",
    ]
    assert [item.vetoed for item in decision.assessments] == [False] * 4
    assert [item.report.within_limits for item in decision.assessments] == [False, True, True, False]
    first, _, _, last = decision.assessments
    assert first.report.violations == (KinematicViolation.SPEED,)
    assert (first.report.max_speed_mps, first.report.max_acceleration_mps2,
            first.report.max_jerk_mps3) == pytest.approx((1, 2, 6))
    assert last.report.violations == (KinematicViolation.SPEED,)
    assert (last.report.max_speed_mps, last.report.max_acceleration_mps2,
            last.report.max_jerk_mps3) == pytest.approx((1, 1, 1))


def test_vetoed_candidate_is_still_checked_and_does_not_hide_the_next_eligible_one():
    request = plan(Candidate("first", trajectory([1, 2, 3], 1)),
                   Candidate("second", trajectory([0, 0, 0], 1)))
    decision = select_trajectory(request, EgoMotion(0, 0, 0, 0),
                                 limits(1, speed=0.5, acceleration=2, jerk=6),
                                 BrakingConfig(2, 2), vetoed_candidate_ids=("first",))
    assert decision.status is DecisionStatus.CANDIDATE
    assert decision.candidate_id == "second"
    assert [item.vetoed for item in decision.assessments] == [True, False]
    assert decision.assessments[0].report.violations == (KinematicViolation.SPEED,)
    assert decision.assessments[1].report.within_limits is True
    assert decision.fallback is None


def test_veto_skips_a_kinematically_valid_first_choice():
    request = plan(Candidate("first", trajectory([1, 2, 3])),
                   Candidate("second", trajectory([1, 2, 3], heading=0.2)))
    decision = select_trajectory(request, EgoMotion(2, 0, 0, 0), limits(),
                                 BrakingConfig(2, 2), vetoed_candidate_ids=["first"])
    assert decision.status is DecisionStatus.CANDIDATE
    assert decision.candidate_id == "second"
    assert [item.report.within_limits for item in decision.assessments] == [True, True]
    np.testing.assert_allclose(decision.trajectory.poses[:, 2], [0.2] * 3, atol=1e-12)


def test_all_vetoed_candidates_produce_local_braking_on_the_same_grid():
    request = plan(Candidate("first", trajectory([1, 2, 3, 4, 5])),
                   Candidate("second", trajectory([1, 2, 3, 4, 5], heading=0.5)))
    vetoes = ["second", "first"]
    decision = select_trajectory(request, EgoMotion(2, 0, 0, 0), limits(),
                                 BrakingConfig(2, 2), vetoed_candidate_ids=vetoes)
    vetoes.clear()
    assert decision.status is DecisionStatus.FALLBACK
    assert decision.failed is False
    assert decision.plan is request
    assert decision.fallback.succeeded is True
    assert decision.fallback.failure_reason is None
    assert [item.candidate_id for item in decision.assessments] == ["first", "second"]
    assert [item.vetoed for item in decision.assessments] == [True, True]
    assert all(item.report.within_limits for item in decision.assessments)
    assert_braking_path(decision)


def test_kinematic_rejections_without_veto_also_produce_braking():
    request = plan(Candidate("standstill", trajectory([0] * 5)))
    decision = select_trajectory(request, EgoMotion(2, 0, 0, 0), limits(),
                                 BrakingConfig(2, 2))
    assert decision.status is DecisionStatus.FALLBACK
    assert decision.failed is False
    assert len(decision.assessments) == 1
    assessment = decision.assessments[0]
    assert assessment.candidate_id == "standstill"
    assert assessment.vetoed is False
    assert set(assessment.report.violations) == {
        KinematicViolation.ACCELERATION, KinematicViolation.JERK,
    }
    assert (assessment.report.max_speed_mps, assessment.report.max_acceleration_mps2,
            assessment.report.max_jerk_mps3) == pytest.approx((2, 8, 48))
    assert_braking_path(decision)


def test_failed_unsupported_fallback_retains_assessments_and_a_finite_trajectory():
    request = plan(Candidate("only", trajectory([0] * 5)))
    decision = select_trajectory(request, EgoMotion(2, 1, 0, 0),
                                 limits(acceleration=100, jerk=100), BrakingConfig(2, 2),
                                 vetoed_candidate_ids=("only",))
    assert decision.status is DecisionStatus.FAILURE
    assert decision.failed is True
    assert decision.candidate_id is None
    assert decision.fallback.succeeded is False
    assert decision.fallback.kind is FallbackKind.EMERGENCY_HOLD
    assert decision.fallback.failure_reason is FallbackFailure.UNSUPPORTED_MOTION
    assert decision.trajectory.num_poses == 5
    assert decision.trajectory.interval_s == 0.5
    np.testing.assert_array_equal(decision.trajectory.poses, np.zeros((5, 3)))
    assert np.isfinite(decision.trajectory.poses).all()
    assert len(decision.assessments) == 1
    assert decision.assessments[0].candidate_id == "only"
    assert decision.assessments[0].vetoed is True


def test_failure_due_to_analytic_peak_keeps_built_braking_profile():
    request = plan(Candidate("only", trajectory([2, 4, 6, 8], 1)))
    decision = select_trajectory(request, EgoMotion(2, 0, 1, 0), limits(1, speed=2.2),
                                 BrakingConfig(2, 2), vetoed_candidate_ids=("only",))
    assert decision.status is DecisionStatus.FAILURE
    assert decision.failed is True
    assert decision.candidate_id is None
    assert decision.fallback.kind is FallbackKind.BRAKING
    assert decision.fallback.failure_reason is FallbackFailure.KINEMATIC_LIMITS
    assert decision.fallback.kinematics.within_limits is True
    assert decision.fallback.peak_speed_mps == pytest.approx(2.25)
    np.testing.assert_allclose(decision.trajectory.poses[:, 0],
                               [13 / 6, 1737 / 512, 667 / 192, 667 / 192], atol=1e-12)


@pytest.mark.parametrize("count", [1, 2])
def test_short_grid_exposes_failed_fallback_instead_of_candidate_or_success(count):
    decision = select_trajectory(plan(Candidate("short", trajectory([0] * count))),
                                 EgoMotion(0, 0, 0, 0), limits(), BrakingConfig(2, 2))
    assert decision.status is DecisionStatus.FAILURE
    assert decision.failed is True
    assert decision.trajectory.num_poses == count
    assert decision.fallback.failure_reason is FallbackFailure.KINEMATIC_LIMITS
    assert decision.assessments[0].report.violations == (KinematicViolation.INSUFFICIENT_POSES,)


@pytest.mark.parametrize("vetoes", [("missing",), ("only", "only"), ("",), (" \t",),
                                    (None,), (1,), "only", b"only", None])
def test_invalid_veto_collections_raise_value_error(vetoes):
    request = plan(Candidate("only", trajectory([0] * 3)))
    with pytest.raises(ValueError):
        select_trajectory(request, EgoMotion(0, 0, 0, 0), limits(), BrakingConfig(2, 2),
                          vetoed_candidate_ids=vetoes)


@pytest.mark.parametrize("relative_delta,accepted", [(5e-13, True), (2e-12, False)])
def test_time_grid_relative_tolerance_never_resamples_a_candidate(relative_delta, accepted):
    interval = 1 + relative_delta
    request = plan(Candidate("only", trajectory([0] * 3, interval)))
    if not accepted:
        with pytest.raises(ValueError):
            select_trajectory(request, EgoMotion(0, 0, 0, 0), limits(1), BrakingConfig(2, 2))
        return
    decision = select_trajectory(request, EgoMotion(0, 0, 0, 0), limits(1), BrakingConfig(2, 2))
    assert decision.status is DecisionStatus.CANDIDATE
    assert decision.trajectory.interval_s == interval
    np.testing.assert_array_equal(decision.trajectory.poses, request.candidates[0].trajectory.poses)


def test_tiny_grid_mismatch_is_not_accepted_by_an_absolute_tolerance():
    request = plan(Candidate("only", trajectory([0] * 3, 0.5e-9)))
    with pytest.raises(ValueError):
        select_trajectory(request, EgoMotion(0, 0, 0, 0), limits(1e-9), BrakingConfig(2, 2))


def test_fallback_preserves_the_exact_plan_grid_within_matching_tolerance():
    interval = 1 + 5e-13
    request = plan(Candidate("only", trajectory([0] * 3, interval)))
    decision = select_trajectory(request, EgoMotion(0, 0, 0, 0), limits(1), BrakingConfig(2, 2), ("only",))
    assert decision.status is DecisionStatus.FALLBACK
    assert decision.trajectory.interval_s == interval
    assert decision.trajectory.horizon_s == request.candidates[0].trajectory.horizon_s


@pytest.mark.parametrize("config", [BrakingConfig(2.1, 2), BrakingConfig(2, 2.1)])
def test_config_is_validated_even_when_an_eligible_candidate_exists(config):
    request = plan(Candidate("only", trajectory([0] * 3)))
    with pytest.raises(ValueError):
        select_trajectory(request, EgoMotion(0, 0, 0, 0), limits(), config)


@pytest.mark.parametrize("index,value", [(0, None), (0, ()), (1, None),
                                        (1, (0, 0, 0, 0)), (2, None), (3, None)])
def test_wrong_argument_types_raise_value_error(index, value):
    args = [plan(Candidate("only", trajectory([0] * 3))),
            EgoMotion(0, 0, 0, 0), limits(), BrakingConfig(2, 2)]
    args[index] = value
    with pytest.raises(ValueError):
        select_trajectory(*args)


def test_decision_cannot_be_mutated():
    decision = select_trajectory(plan(Candidate("only", trajectory([0] * 3))),
                                 EgoMotion(0, 0, 0, 0), limits(), BrakingConfig(2, 2))
    with pytest.raises(FrozenInstanceError):
        decision.status = DecisionStatus.FAILURE
