from dataclasses import replace
import math

import numpy as np
import pytest

from hybrid_drive.contracts.motion import EgoMotion
from hybrid_drive.contracts.planning import Candidate, CandidateSet
from hybrid_drive.contracts.slow import RiskStatus, SlowRequest, SlowResult
from hybrid_drive.contracts.trajectory import Pose2D, Trajectory
from hybrid_drive.safety.arbiter import SafetyArbiter, SlowDisposition
from hybrid_drive.safety.fallback import BrakingConfig, FallbackFailure, FallbackKind
from hybrid_drive.safety.fallback_policy import FallbackModel, FallbackObservation
from hybrid_drive.safety.kinematics import KinematicLimits
from hybrid_drive.safety.selection import DecisionStatus, select_trajectory


MOTION = EgoMotion(1, 0, 0, .5)
LIMITS = KinematicLimits(.5, 10, 2, 2)
CONFIG = BrakingConfig(2, 2)


def fixture_plan(**changes):
    # Independent triangular scalar stop: q=1, V0=1, kappa=.5, rho=.2.
    distances = [23/48, 5/6, 47/48, 1, 1]
    trajectory = Trajectory([[2*math.sin(s/2), 4*math.sin(s/4)**2, .2*s] for s in distances], .5)
    values = dict(plan_id="plan", snapshot_id="snapshot", t_ref_s=123,
                  reference_pose=Pose2D(0, 0, 0), candidates=(Candidate("candidate", trajectory),))
    return CandidateSet(**(values | changes))


def observation(plan, **changes):
    values = dict(snapshot_id=plan.snapshot_id, t_ref_s=plan.t_ref_s, motion=MOTION,
                  heading_rate_radps=.2, source_id="immutable-motion-source")
    return FallbackObservation(**(values | changes))


def select(plan, **kwargs):
    return select_trajectory(plan, MOTION, LIMITS, CONFIG, ("candidate",), **kwargs)


def test_curved_is_opt_in_and_default_legacy_failure_is_preserved():
    plan = fixture_plan()
    legacy = select(plan)
    explicit = select(plan, fallback_model=FallbackModel.LONGITUDINAL, fallback_observation=observation(plan))
    assert legacy.failed and explicit.failed
    assert legacy.fallback.failure_reason is FallbackFailure.UNSUPPORTED_MOTION
    np.testing.assert_array_equal(legacy.trajectory.poses, explicit.trajectory.poses)
    assert legacy.fallback.model_report is None
    curved = select(plan, fallback_model=FallbackModel.CURVED, fallback_observation=observation(plan))
    assert curved.status is DecisionStatus.FALLBACK
    assert curved.fallback.kind is FallbackKind.CURVED_BRAKING
    np.testing.assert_allclose(curved.trajectory.poses, plan.candidates[0].trajectory.poses)


@pytest.mark.parametrize("changes", [
    {"snapshot_id":"previous"}, {"t_ref_s":122.5}, {"motion":EgoMotion(1, 0, 0, .4)},
])
def test_another_snapshot_time_or_state_cannot_supply_fallback_heading(changes):
    plan = fixture_plan()
    result = select(plan, fallback_model=FallbackModel.CURVED,
                    fallback_observation=observation(plan, **changes))
    assert result.failed
    assert result.fallback.failure_reason is FallbackFailure.OBSERVATION_MISMATCH
    assert result.fallback.kind is FallbackKind.EMERGENCY_HOLD


def test_missing_heading_does_not_become_course_rate_or_zero():
    plan = fixture_plan()
    result = select(plan, fallback_model=FallbackModel.CURVED)
    assert result.failed
    assert result.fallback.failure_reason is FallbackFailure.MISSING_HEADING_RATE


def test_candidate_acceptance_does_not_require_unused_fallback_observation():
    plan = fixture_plan()
    result = select_trajectory(plan, MOTION, LIMITS, CONFIG, fallback_model=FallbackModel.CURVED)
    assert result.status is DecisionStatus.CANDIDATE
    assert result.fallback is None


def test_zero_observed_body_rate_is_not_inferred_from_curved_course():
    plan = fixture_plan()
    result = select(plan, fallback_model=FallbackModel.CURVED,
                    fallback_observation=observation(plan, heading_rate_radps=0))
    assert not result.failed
    assert np.any(result.trajectory.poses[:, 1] != 0)
    np.testing.assert_array_equal(result.trajectory.poses[:, 2], 0)


def test_observation_is_snapshot_bound_not_tied_to_one_candidate_list():
    plan = fixture_plan()
    revised = replace(plan, plan_id="replanned")
    result = select(revised, fallback_model=FallbackModel.CURVED, fallback_observation=observation(plan))
    assert not result.failed


def test_arbiter_applies_only_current_slow_veto_and_current_motion_observation():
    plan = fixture_plan()
    request = SlowRequest("request", plan, 10, 12, 123.5, "model")
    reply = SlowResult(request.request_id, plan.snapshot_id, plan.plan_id, plan.candidates_hash,
                       "model", RiskStatus.VETO, ("candidate",))
    engine = SafetyArbiter(LIMITS, CONFIG, FallbackModel.CURVED)
    kwargs = dict(now_monotonic_s=11, now_sim_s=123, request=request, result=reply,
                  received_at_monotonic_s=10.5, fallback_observation=observation(plan))
    current = engine.decide(plan, MOTION, **kwargs)
    assert current.slow.disposition is SlowDisposition.APPLIED
    assert current.selection.status is DecisionStatus.FALLBACK
    stale = engine.decide(plan, MOTION, **(kwargs | {"now_sim_s":124}))
    assert stale.slow.disposition is SlowDisposition.SIM_EXPIRED
    assert stale.selection.status is DecisionStatus.CANDIDATE
    updated = fixture_plan(snapshot_id="new-snapshot", t_ref_s=123.5)
    stale_plan = engine.decide(updated, MOTION, **(kwargs | {"now_sim_s":123.5}))
    assert stale_plan.slow.disposition is SlowDisposition.PLAN_MISMATCH
    assert stale_plan.selection.status is DecisionStatus.CANDIDATE


@pytest.mark.parametrize("rate", [None, True, float("nan"), float("inf")])
def test_non_numeric_or_unknown_rate_is_not_a_usable_observation(rate):
    with pytest.raises(ValueError):
        observation(fixture_plan(), heading_rate_radps=rate)
