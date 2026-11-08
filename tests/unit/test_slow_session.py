from dataclasses import replace

import numpy as np
import pytest

from hybrid_drive.contracts.motion import EgoMotion
from hybrid_drive.contracts.planning import Candidate, CandidateSet
from hybrid_drive.contracts.slow import RiskStatus, SlowRequest, SlowResult
from hybrid_drive.contracts.trajectory import Pose2D, Trajectory
from hybrid_drive.safety.arbiter import SafetyArbiter, SlowDisposition
from hybrid_drive.safety.fallback import BrakingConfig
from hybrid_drive.safety.kinematics import KinematicLimits
from hybrid_drive.slow.context import CAMERA_NAMES, CameraFrame, SlowContext
from hybrid_drive.slow.session import SlowPlanningSession
from hybrid_drive.slow.worker import SlowCompletion


def context(request_id="request", *, pixel=0):
    candidates = tuple(Candidate(name, Trajectory([[x, 0, heading] for x in (1, 2, 3)], 0.5))
                       for name, heading in (("first", 0), ("second", 0.1)))
    plan = CandidateSet("plan-" + request_id, "snapshot", 100, Pose2D(0, 0, 0), candidates)
    request = SlowRequest(request_id, plan, 10, 12, 102, "v1")
    cameras = tuple(CameraFrame(name, np.full((2, 3, 3), pixel, dtype=np.uint8)) for name in CAMERA_NAMES)
    return SlowContext(request, EgoMotion(2, 0, 0, 0), cameras)


def completion(supplied, *, outcome="completed", received=11, status=RiskStatus.VETO):
    request = supplied.request
    plan = request.plan
    result = SlowResult(request.request_id, plan.snapshot_id, plan.plan_id,
                        plan.candidates_hash, request.model_version, status,
                        ("first",) if status is RiskStatus.VETO else (), "visible evidence")
    return SlowCompletion(supplied, result, received, outcome)


class FakeWorker:
    def __init__(self):
        self.now = 11
        self.events = []
        self.submitted = []
        self.closed = False
        self.fail_submit = False

    def now_monotonic_s(self):
        return self.now

    def submit(self, supplied):
        if self.fail_submit:
            raise ValueError("rejected")
        self.submitted.append(supplied)
        return "started"

    def poll(self):
        events, self.events = tuple(self.events), []
        return events

    def close(self):
        self.closed = True


def session():
    worker = FakeWorker()
    arbiter = SafetyArbiter(KinematicLimits(0.5, 10, 2, 2), BrakingConfig(2, 2))
    return SlowPlanningSession(arbiter, worker), worker


def test_matching_result_uses_trusted_receive_time_and_vetoes_actual_candidate():
    runtime, worker = session()
    supplied = context()
    assert runtime.activate(supplied) == "started"
    worker.events = [completion(supplied, received=10.5)]
    decision = runtime.decide(now_sim_s=101)
    assert decision.slow.disposition is SlowDisposition.APPLIED
    assert decision.selection.candidate_id == "second"
    assert decision.slow.response_age_s == 0.5
    assert decision.decided_at_monotonic_s == 11
    assert runtime.current_context is supplied
    assert runtime.last_completion.context is supplied


def test_first_decision_is_final_even_when_late_result_arrives():
    runtime, worker = session()
    supplied = context()
    runtime.activate(supplied)
    original = runtime.decide(now_sim_s=100)
    assert original.slow.disposition is SlowDisposition.MISSING_RESULT
    assert original.selection.candidate_id == "first"
    reply = completion(supplied)
    worker.events = [reply]
    worker.now = 11.5
    assert runtime.decide(now_sim_s=101) is original
    assert runtime.last_completion is reply
    assert original.selection.candidate_id == "first"


def test_activation_resets_old_reply_and_out_of_order_old_result_cannot_veto_new_plan():
    runtime, worker = session()
    old, new = context("old"), context("new")
    runtime.activate(old)
    worker.events = [completion(old)]
    assert runtime.decide(now_sim_s=100).selection.candidate_id == "second"
    runtime.activate(new)
    worker.events = [completion(old)]
    decision = runtime.decide(now_sim_s=100)
    assert decision.slow.disposition is SlowDisposition.MISSING_RESULT
    assert decision.selection.candidate_id == "first"
    assert decision.selection.plan is new.request.plan


def test_reordered_events_and_conflicting_duplicate_keep_first_matching_result():
    runtime, worker = session()
    old, new = context("old"), context("new")
    runtime.activate(new)
    veto = completion(new)
    worker.events = [veto, completion(new, status=RiskStatus.NO_VETO), completion(old)]
    decision = runtime.decide(now_sim_s=100)
    assert decision.selection.candidate_id == "second"
    assert decision.slow.applied_result is veto.result
    assert runtime.last_completion.context is old


@pytest.mark.parametrize("which", ["hash", "model", "result", "outcome"])
def test_context_and_response_bindings_are_both_required(which):
    runtime, worker = session()
    supplied = context()
    runtime.activate(supplied)
    reply = completion(supplied)
    if which == "hash":
        reply = replace(reply, context=context(pixel=1))
    elif which == "model":
        changed = replace(supplied.request, model_version="other")
        reply = replace(reply, context=replace(supplied, request=changed))
    elif which == "result":
        reply = replace(reply, result=replace(reply.result, model_version="other"))
    else:
        reply = replace(reply, outcome="invalid_output")
    worker.events = [reply]
    decision = runtime.decide(now_sim_s=100)
    assert decision.slow.disposition is SlowDisposition.MISSING_RESULT
    assert decision.selection.candidate_id == "first"


def test_equivalent_rebuilt_context_is_accepted_not_object_identity():
    runtime, worker = session()
    runtime.activate(context())
    worker.events = [completion(context())]
    assert runtime.decide(now_sim_s=100).slow.disposition is SlowDisposition.APPLIED


@pytest.mark.parametrize("now,sim,receipt,expected", [
    (12, 102, 12, SlowDisposition.APPLIED),
    (12.001, 100, 11, SlowDisposition.DEADLINE_EXCEEDED),
    (11, 102.001, 11, SlowDisposition.SIM_EXPIRED),
    (11, 100, 11.001, SlowDisposition.INVALID_TIMING),
    (11, 100, 9.999, SlowDisposition.INVALID_TIMING),
])
def test_clock_domains_deadlines_and_receive_validation_delegate_to_arbiter(now, sim, receipt, expected):
    runtime, worker = session()
    supplied = context()
    runtime.activate(supplied)
    worker.now = now
    worker.events = [completion(supplied, received=receipt)]
    assert runtime.decide(now_sim_s=sim).slow.disposition is expected


def test_activation_validates_and_failed_submission_keeps_previous_association():
    runtime, worker = session()
    with pytest.raises(RuntimeError, match="Activate"):
        runtime.decide(now_sim_s=100)
    with pytest.raises(ValueError, match="context"):
        runtime.activate(None)
    original = context()
    runtime.activate(original)
    with pytest.raises(ValueError, match="new request_id"):
        runtime.activate(original)
    worker.fail_submit = True
    with pytest.raises(ValueError, match="rejected"):
        runtime.activate(context("new"))
    assert runtime.current_context is original


def test_context_manager_close_clears_association_and_forwards_lifecycle():
    runtime, worker = session()
    with runtime:
        runtime.activate(context())
    assert worker.closed
    assert runtime.current_context is None
    with pytest.raises(RuntimeError, match="closed"):
        runtime.activate(context("new"))
    with pytest.raises(RuntimeError, match="closed"):
        runtime.decide(now_sim_s=100)
    runtime.close()
