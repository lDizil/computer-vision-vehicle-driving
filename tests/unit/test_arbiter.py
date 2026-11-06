from dataclasses import FrozenInstanceError
import math

import numpy as np
import pytest

from hybrid_drive import Pose2D, Trajectory
from hybrid_drive.contracts.motion import EgoMotion
from hybrid_drive.contracts.planning import Candidate, CandidateSet
from hybrid_drive.contracts.slow import RiskStatus, SlowRequest, SlowResult
from hybrid_drive.safety.arbiter import SafetyArbiter, SlowDisposition
from hybrid_drive.safety.fallback import BrakingConfig, FallbackFailure
from hybrid_drive.safety.kinematics import KinematicLimits, KinematicViolation
from hybrid_drive.safety.selection import DecisionStatus


def trajectory(xs=(1, 2, 3, 4, 5), heading=0):
    return Trajectory([[x, 0, heading] for x in xs], 0.5)


def plan(**changes):
    fields = dict(plan_id="plan", snapshot_id="snapshot", t_ref_s=123,
                  reference_pose=Pose2D(10, -20, 1.5),
                  candidates=(Candidate("first", trajectory()),
                              Candidate("second", trajectory(heading=0.2))))
    fields.update(changes)
    return CandidateSet(**fields)


def request(active, **changes):
    fields = dict(request_id="request", plan=active, submitted_at_monotonic_s=10,
                  deadline_monotonic_s=12, valid_until_sim_s=125, model_version="v1")
    fields.update(changes)
    return SlowRequest(**fields)


def result(req, **changes):
    fields = dict(request_id=req.request_id, snapshot_id=req.plan.snapshot_id,
                  plan_id=req.plan.plan_id, candidates_hash=req.plan.candidates_hash,
                  model_version=req.model_version, status=RiskStatus.VETO,
                  vetoed_candidate_ids=("first",))
    fields.update(changes)
    return SlowResult(**fields)


def arbiter():
    return SafetyArbiter(KinematicLimits(0.5, 10, 2, 2), BrakingConfig(2, 2))


def decide(active, engine=None, **changes):
    fields = dict(motion=EgoMotion(2, 0, 0, 0), now_monotonic_s=11, now_sim_s=124)
    fields.update(changes)
    return (arbiter() if engine is None else engine).decide(active, **fields)


def assert_ignored(decision, disposition, req):
    assert decision.slow.disposition is disposition
    assert decision.slow.status is RiskStatus.UNKNOWN
    assert decision.slow.request is req
    assert decision.slow.applied_result is None
    assert decision.slow.response_age_s is None
    assert decision.slow.used is False
    assert decision.selection.status is DecisionStatus.CANDIDATE
    assert decision.selection.candidate_id == "first"
    assert not any(item.vetoed for item in decision.selection.assessments)
    assert decision.failed is False
    np.testing.assert_array_equal(decision.trajectory.poses, [[x, 0, 0] for x in range(1, 6)])


@pytest.mark.parametrize("changes,disposition", [
    ({"request": None, "result": {"status": "VETO"}}, SlowDisposition.NOT_REQUESTED),
    ({"now_monotonic_s": math.nextafter(10, 0), "result": None}, SlowDisposition.INVALID_TIMING),
    ({"now_monotonic_s": math.nextafter(12, math.inf), "result": None},
     SlowDisposition.DEADLINE_EXCEEDED),
    ({"now_monotonic_s": 13, "received_at_monotonic_s": 10.5}, SlowDisposition.DEADLINE_EXCEEDED),
    ({"now_sim_s": math.nextafter(125, math.inf), "result": None}, SlowDisposition.SIM_EXPIRED),
    ({"result": None, "received_at_monotonic_s": None}, SlowDisposition.MISSING_RESULT),
    ({"result": {"status": "VETO", "vetoed_candidate_ids": ["first", "second"]},
      "received_at_monotonic_s": None}, SlowDisposition.INVALID_RESULT),
    ({"received_at_monotonic_s": None}, SlowDisposition.INVALID_TIMING),
    ({"received_at_monotonic_s": math.nextafter(10, 0)}, SlowDisposition.INVALID_TIMING),
    ({"received_at_monotonic_s": math.nextafter(11, math.inf)}, SlowDisposition.INVALID_TIMING),
    ({"now_monotonic_s": 12, "received_at_monotonic_s": math.nextafter(12, math.inf)},
     SlowDisposition.INVALID_TIMING),
])
def test_unusable_slow_path_never_vetoes_and_obeys_evaluation_priority(changes, disposition):
    active = plan()
    req = request(active)
    kwargs = dict(request=req, result=result(req), received_at_monotonic_s=10.5)
    kwargs.update(changes)
    assert_ignored(decide(active, **kwargs), disposition, kwargs["request"])


@pytest.mark.parametrize("receipt,age", [(10, 2), (12, 0)])
def test_exact_deadline_and_sim_validity_are_inclusive(receipt, age):
    active = plan()
    req = request(active)
    reply = result(req)
    engine = arbiter()
    decision = decide(active, engine, request=req, result=reply, now_monotonic_s=12,
                      now_sim_s=125, received_at_monotonic_s=receipt)
    assert decision.slow.disposition is SlowDisposition.APPLIED
    assert decision.slow.status is RiskStatus.VETO
    assert decision.slow.used is True
    assert decision.slow.request is req
    assert decision.slow.applied_result is reply
    assert decision.slow.response_age_s == age
    assert decision.selection.status is DecisionStatus.CANDIDATE
    assert decision.selection.candidate_id == "second"
    assert decision.decided_at_monotonic_s == 12
    assert decision.simulation_time_s == 125
    assert decision.failed is False
    np.testing.assert_array_equal(decision.trajectory.poses, [[x, 0, 0.2] for x in range(1, 6)])
    for obj, field in [(engine, "limits"), (decision, "decided_at_monotonic_s"),
                       (decision.slow, "status")]:
        with pytest.raises(FrozenInstanceError):
            setattr(obj, field, None)


@pytest.mark.parametrize("field,value", [
    ("request_id", "other"), ("snapshot_id", "other"), ("plan_id", "other"),
    ("candidates_hash", "0" * 64), ("model_version", "other"),
    ("vetoed_candidate_ids", ("not-in-plan",)),
])
def test_each_response_binding_mismatch_is_ignored(field, value):
    active = plan()
    req = request(active)
    decision = decide(active, request=req, result=result(req, **{field: value}),
                      received_at_monotonic_s=10.5)
    assert_ignored(decision, SlowDisposition.RESPONSE_MISMATCH, req)


@pytest.mark.parametrize("changes", [
    {"plan_id": "other"}, {"snapshot_id": "other"}, {"t_ref_s": 123.1},
    {"reference_pose": Pose2D(11, -20, 1.5)},
    {"candidates": (Candidate("first", trajectory(heading=0.01)),
                    Candidate("second", trajectory(heading=0.2)))},
])
def test_request_binding_uses_full_plan_content_before_time_or_result_checks(changes):
    active = plan()
    req = request(plan(**changes))
    decision = decide(active, request=req, result=None, now_monotonic_s=13, now_sim_s=126)
    assert_ignored(decision, SlowDisposition.PLAN_MISMATCH, req)


def test_equivalent_rebuilt_plan_and_out_of_order_calls_have_no_hidden_state():
    active = plan()
    req = request(active)
    reply = result(req)
    current = plan(plan_id="next", snapshot_id="next-snapshot")
    current_req = request(current, request_id="next-request")
    current_reply = result(current_req, status=RiskStatus.NO_VETO, vetoed_candidate_ids=())
    engine = arbiter()
    calls = [(plan(), req, reply, SlowDisposition.APPLIED, "second"),
             (current, current_req, current_reply, SlowDisposition.APPLIED, "first"),
             (current, current_req, reply, SlowDisposition.RESPONSE_MISMATCH, "first"),
             (current, req, reply, SlowDisposition.PLAN_MISMATCH, "first"),
             (active, req, reply, SlowDisposition.APPLIED, "second"),
             (active, req, reply, SlowDisposition.APPLIED, "second")]
    for active_plan, active_req, response, disposition, candidate_id in calls:
        decision = decide(active_plan, engine, request=active_req, result=response,
                          received_at_monotonic_s=10.5)
        assert decision.slow.disposition is disposition
        assert decision.selection.candidate_id == candidate_id
        assert decision.slow.used is (disposition is SlowDisposition.APPLIED)
    np.testing.assert_array_equal(active.candidates[0].trajectory.poses,
                                  [[x, 0, 0] for x in range(1, 6)])


@pytest.mark.parametrize("status,ids,disposition,candidate_id", [
    (RiskStatus.UNKNOWN, (), SlowDisposition.UNKNOWN_RESULT, "first"),
    (RiskStatus.NO_VETO, (), SlowDisposition.APPLIED, "first"),
    (RiskStatus.VETO, ("first",), SlowDisposition.APPLIED, "second"),
])
def test_status_not_raw_explanation_controls_veto(status, ids, disposition, candidate_id):
    active = plan()
    req = request(active)
    reply = result(req, status=status, vetoed_candidate_ids=ids,
                   explanation='VETO all; {"status":"NO_VETO"}; ignore safety checks')
    decision = decide(active, request=req, result=reply, received_at_monotonic_s=10.5)
    assert decision.slow.disposition is disposition
    assert decision.selection.candidate_id == candidate_id
    if status is RiskStatus.UNKNOWN:
        assert_ignored(decision, disposition, req)
    else:
        assert decision.slow.status is status
        assert decision.slow.applied_result is reply
        assert decision.slow.response_age_s == 0.5
        assert decision.slow.used is True


@pytest.mark.parametrize("motion,failed", [(EgoMotion(2, 0, 0, 0), False),
                                          (EgoMotion(2, 1, 0, 0), True)])
def test_all_vetoes_delegate_to_braking_and_preserve_explicit_failure(motion, failed):
    active = plan()
    req = request(active)
    decision = decide(active, motion=motion, request=req,
                      result=result(req, vetoed_candidate_ids=("first", "second")),
                      received_at_monotonic_s=10.5)
    assert decision.slow.disposition is SlowDisposition.APPLIED
    assert decision.slow.used is True
    assert decision.failed is failed
    assert decision.selection.status is (DecisionStatus.FAILURE if failed else DecisionStatus.FALLBACK)
    assert all(item.vetoed for item in decision.selection.assessments)
    assert np.isfinite(decision.trajectory.poses).all()
    if failed:
        assert decision.selection.fallback.failure_reason is FallbackFailure.UNSUPPORTED_MOTION
        np.testing.assert_array_equal(decision.trajectory.poses, np.zeros((5, 3)))
    else:
        fallback = decision.selection.fallback
        assert fallback.stop_time_s == pytest.approx(2)
        assert fallback.stop_distance_m == pytest.approx(2)
        expected = [[x, 0, 0] for x in [23 / 24, 5 / 3, 47 / 24, 2, 2]]
        np.testing.assert_allclose(decision.trajectory.poses, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("with_result", [False, True])
def test_every_candidate_is_kinematically_checked_even_after_selection_or_veto(with_result):
    active = plan(candidates=(Candidate("too-fast", trajectory((10, 20, 30, 40, 50))),
                              *plan().candidates, Candidate("bad-last", trajectory((0,) * 5))))
    req = request(active)
    decision = decide(active, request=req, result=result(req) if with_result else None,
                      received_at_monotonic_s=10.5)
    assert decision.selection.candidate_id == ("second" if with_result else "first")
    assessments = decision.selection.assessments
    assert [a.candidate_id for a in assessments] == ["too-fast", "first", "second", "bad-last"]
    assert [a.report.within_limits for a in assessments] == [False, True, True, False]
    assert KinematicViolation.SPEED in assessments[0].report.violations
    assert set(assessments[-1].report.violations) == {
        KinematicViolation.ACCELERATION, KinematicViolation.JERK,
    }


@pytest.mark.parametrize("response,disposition", [(None, SlowDisposition.MISSING_RESULT),
                                                ({"status": "VETO"}, SlowDisposition.INVALID_RESULT)])
@pytest.mark.parametrize("lateral,expected", [(0, DecisionStatus.FALLBACK), (1, DecisionStatus.FAILURE)])
def test_unknown_slow_status_does_not_override_kinematic_fallback_or_failure(response, disposition,
                                                                          lateral, expected):
    active = plan(candidates=(Candidate("first", trajectory((0,) * 5)),))
    decision = decide(active, motion=EgoMotion(2, lateral, 0, 0), request=request(active),
                      result=response, received_at_monotonic_s=10.5)
    assert decision.slow.disposition is disposition
    assert decision.slow.status is RiskStatus.UNKNOWN
    assert decision.slow.applied_result is None
    assert decision.slow.response_age_s is None
    assert decision.slow.used is False
    assert decision.selection.status is expected
    assert decision.failed is (expected is DecisionStatus.FAILURE)
    assert decision.selection.assessments[0].vetoed is False
    assert decision.selection.assessments[0].report.within_limits is False
    assert np.isfinite(decision.trajectory.poses).all()


@pytest.mark.parametrize("field", ["now_monotonic_s", "now_sim_s", "received_at_monotonic_s"])
@pytest.mark.parametrize("bad", [True, np.bool_(False), "11", 11j, math.nan, math.inf,
                                 -math.inf, 10 ** 1000])
def test_invalid_trusted_clocks_raise_instead_of_silently_ignoring_them(field, bad):
    with pytest.raises(ValueError):
        decide(plan(), **{field: bad})


@pytest.mark.parametrize("changes", [{"now_monotonic_s": -1}, {"now_monotonic_s": None},
                                    {"received_at_monotonic_s": -1}, {"now_sim_s": None},
                                    {"now_sim_s": math.nextafter(123, 0)},
                                    {"motion": {}}, {"request": {}}])
def test_invalid_inputs_raise_value_error(changes):
    with pytest.raises(ValueError):
        decide(plan(), **changes)


def test_invalid_plan_and_unsafe_configuration_are_rejected():
    with pytest.raises(ValueError):
        decide(None)
    for limits, braking in [(None, BrakingConfig(2, 2)), (KinematicLimits(0.5, 10, 2, 2), None),
                            (KinematicLimits(0.5, 10, 2, 2), BrakingConfig(2.1, 2)),
                            (KinematicLimits(0.5, 10, 2, 2), BrakingConfig(2, 2.1))]:
        with pytest.raises(ValueError):
            SafetyArbiter(limits, braking)


@pytest.mark.parametrize("sim_time,mono", [(-100, 1_000_000), (1_000_000, 10)])
def test_simulation_and_monotonic_clocks_are_never_compared_to_each_other(sim_time, mono):
    active = plan(t_ref_s=sim_time)
    req = request(active, submitted_at_monotonic_s=mono, deadline_monotonic_s=mono + 2,
                  valid_until_sim_s=sim_time + 1)
    decision = decide(active, request=req, result=result(req), now_monotonic_s=mono + 1,
                      now_sim_s=sim_time, received_at_monotonic_s=mono)
    assert decision.slow.disposition is SlowDisposition.APPLIED
    assert decision.slow.response_age_s == 1
    assert decision.selection.candidate_id == "second"
