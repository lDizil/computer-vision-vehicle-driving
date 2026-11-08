"""Synthetic temporal contract checks, not inference or road-safety evaluation."""

from collections import Counter
from dataclasses import dataclass
import json

import numpy as np

from hybrid_drive.contracts.motion import EgoMotion
from hybrid_drive.contracts.planning import Candidate, CandidateSet
from hybrid_drive.contracts.slow import SlowRequest
from hybrid_drive.contracts.trajectory import Pose2D, Trajectory
from hybrid_drive.safety.arbiter import SafetyArbiter
from hybrid_drive.safety.fallback import BrakingConfig
from hybrid_drive.safety.kinematics import KinematicLimits

from .context import CAMERA_NAMES, CameraFrame, SlowContext
from .protocol import parse_response
from .session import SlowPlanningSession
from .worker import SlowCompletion


@dataclass(frozen=True)
class _Step:
    action: str
    at_s: float
    context: str = "main"
    value: str | float | None = None


@dataclass(frozen=True)
class _Case:
    name: str
    steps: tuple[_Step, ...]
    expected: tuple[dict, ...]
    counts: dict
    reverse_motion: bool = False


class _DeliveryWorker:
    """A scripted receive boundary: no process, queue policy, model or sleeps."""

    def __init__(self):
        self.now = 10.0
        self.submitted = []
        self.delivered = []
        self._ready = []
        self.closed = False

    def now_monotonic_s(self):
        return self.now

    def submit(self, context):
        self.submitted.append(context.request.request_id)
        return "scripted"

    def deliver(self, context, raw):
        try:
            result = parse_response(raw, context)
            outcome = "completed"
        except ValueError:
            result, outcome = None, "invalid_output"
        self._ready.append(SlowCompletion(context, result, self.now, outcome))
        request = context.request
        self.delivered.append({
            "request_id": request.request_id,
            "context_hash": context.content_hash,
            "received_at_monotonic_s": self.now,
            "submission_to_receive_s": self.now - request.submitted_at_monotonic_s,
            "after_deadline": self.now > request.deadline_monotonic_s,
            "outcome": outcome,
        })

    def poll(self):
        ready, self._ready = tuple(self._ready), []
        return ready

    def close(self):
        self.closed = True


def run_replay() -> dict:
    """Run fixed traces through the production parser, session and arbiter."""
    scenarios = [_run_case(case) for case in _cases()]
    totals = Counter()
    outcomes = Counter()
    response_ages = []
    for scenario in scenarios:
        totals.update(scenario["counts"])
        outcomes.update(delivery["outcome"] for delivery in scenario["deliveries"])
        response_ages.extend(scenario["response_age_s"])
    return {
        "schema": "hybrid-drive-slow-replay-v1",
        "passed": all(scenario["passed"] for scenario in scenarios),
        "execution": {
            "kind": "deterministic_synthetic_event_trace",
            "clock": "explicit monotonic and simulation event times; no wall-clock measurements",
            "components": ["parse_response", "SlowPlanningSession", "SafetyArbiter", "select_trajectory"],
            "inference_executed": False,
            "inputs": "Synthetic five-point straight candidate pair, ego-motion fixture and three zero RGB 3x2 images.",
            "claim": "Temporal/protocol contract checks only; no safety-quality or real-time claim.",
        },
        "metric_definitions": {
            "submitted": "Session activations forwarded to the scripted delivery worker.",
            "delivered": "Scripted receive events, including invalid outputs, old replies and duplicates.",
            "applied": "First decisions per activation with slow.used; repeated frozen calls count once.",
            "degradations": "First decisions per activation without an applied slow result, including explicit UNKNOWN.",
            "response_age_s": "Decision time minus trusted scripted receive time for applied first decisions only.",
            "deadline_misses": "First decisions whose domain disposition is deadline_exceeded.",
            "late_deliveries": "Receive events strictly after their own request deadline; equality is timely.",
        },
        "counts": dict(totals),
        "delivery_outcomes": dict(outcomes),
        "response_age_s": response_ages,
        "runtime_process_tests": {
            "status": "not_run_by_replay",
            "command": "python -m pytest -q tests/unit/test_slow_worker.py tests/unit/test_slow_session.py",
            "scope": "Subprocess cancellation, bounded active/pending work and latest-wins need separate real-worker tests.",
        },
        "scenarios": scenarios,
    }


def _run_case(case):
    worker = _DeliveryWorker()
    arbiter = SafetyArbiter(KinematicLimits(0.5, 10, 2, 2), BrakingConfig(2, 2))
    session = SlowPlanningSession(arbiter, worker)
    contexts, first_decisions, decisions = {}, {}, []
    frozen, original_plan = True, True
    error = None
    try:
        for step in case.steps:
            worker.now = step.at_s
            if step.action == "activate":
                context = _context(case.name + "-" + step.context, step.at_s, case.reverse_motion)
                contexts[step.context] = context
                session.activate(context)
            elif step.action == "deliver":
                context = contexts[step.context]
                worker.deliver(context, _response(context, step.value))
            elif step.action == "decide":
                decision = session.decide(now_sim_s=step.value)
                key = session.current_context.request.request_id
                if key in first_decisions:
                    frozen = frozen and decision is first_decisions[key]
                else:
                    first_decisions[key] = decision
                original_plan = original_plan and decision.selection.plan is session.current_context.request.plan
                decisions.append(_decision_record(decision))
            else:
                raise ValueError("Unknown replay step")
    except Exception as exc:
        error = type(exc).__name__
    finally:
        session.close()
    unique = list(first_decisions.values())
    counts = {
        "submitted": len(worker.submitted), "delivered": len(worker.delivered),
        "decision_calls": len(decisions), "decisions": len(unique),
        "applied": sum(decision.slow.used for decision in unique),
        "degradations": sum(not decision.slow.used for decision in unique),
        "deadline_misses": sum(decision.slow.disposition.value == "deadline_exceeded" for decision in unique),
        "late_deliveries": sum(delivery["after_deadline"] for delivery in worker.delivered),
        "fallbacks": sum(decision.selection.status.value == "fallback" for decision in unique),
        "failures": sum(decision.failed for decision in unique),
    }
    signatures = tuple({key: record[key] for key in expected} for record, expected in zip(decisions, case.expected))
    checks = {
        "expected_decisions": len(decisions) == len(case.expected) and signatures == case.expected,
        "expected_counts": all(counts[key] == value for key, value in case.counts.items()),
        "frozen_decision_identity": frozen,
        "original_plan_identity": original_plan,
        "worker_closed": worker.closed,
        "no_execution_error": error is None,
    }
    return {
        "name": case.name, "passed": all(checks.values()), "checks": checks,
        "expected": {"decisions": case.expected, "counts": case.counts},
        "trace": [{"action": step.action, "at_monotonic_s": step.at_s, "context": step.context,
                   "response_variant_or_sim_time_s": step.value} for step in case.steps],
        "contexts": {key: {
            "request_id": context.request.request_id,
            "plan_id": context.request.plan.plan_id,
            "snapshot_id": context.request.plan.snapshot_id,
            "candidates_hash": context.request.plan.candidates_hash,
            "context_hash": context.content_hash,
            "model_version": context.request.model_version,
            "submitted_at_monotonic_s": context.request.submitted_at_monotonic_s,
            "deadline_monotonic_s": context.request.deadline_monotonic_s,
            "valid_until_sim_s": context.request.valid_until_sim_s,
        } for key, context in contexts.items()},
        "counts": counts, "deliveries": worker.delivered, "decisions": decisions,
        "response_age_s": [decision.slow.response_age_s for decision in unique if decision.slow.used],
        "execution_error": error,
    }


def _context(name, submitted_at_s, reverse_motion):
    trajectory = Trajectory([[x, 0, 0] for x in range(1, 6)], 0.5)
    plan = CandidateSet(name, "snapshot-" + name, 100, Pose2D(0, 0, 0), (
        Candidate("first", trajectory), Candidate("second", trajectory),
    ))
    request = SlowRequest("request-" + name, plan, submitted_at_s, submitted_at_s + 2, 102, "scripted-v1")
    motion = EgoMotion(-1 if reverse_motion else 2, 0, 0, 0)
    cameras = tuple(CameraFrame(name, np.zeros((2, 3, 3), dtype=np.uint8)) for name in CAMERA_NAMES)
    return SlowContext(request, motion, cameras)


def _response(context, variant):
    if variant == "invalid_json":
        return "{not JSON"
    status = variant if variant in ("UNKNOWN", "NO_VETO") else "VETO"
    ids = [] if status != "VETO" else ["first"]
    if variant == "all_veto":
        ids = ["first", "second"]
    if variant == "unknown_candidate":
        ids = ["absent"]
    return json.dumps({
        "context_hash": "0" * 64 if variant == "wrong_hash" else context.content_hash,
        "status": status, "vetoed_candidate_ids": ids, "explanation": "Scripted protocol fixture, not image inference.",
    })


def _decision_record(decision):
    fallback = decision.selection.fallback
    return {
        "request_id": decision.slow.request.request_id,
        "plan_id": decision.selection.plan.plan_id,
        "decided_at_monotonic_s": decision.decided_at_monotonic_s,
        "simulation_time_s": decision.simulation_time_s,
        "slow_disposition": decision.slow.disposition.value,
        "slow_status": decision.slow.status.value,
        "selection_status": decision.selection.status.value,
        "candidate_id": decision.selection.candidate_id,
        "fallback_failure": None if fallback is None or fallback.failure_reason is None else fallback.failure_reason.value,
        "failed": decision.failed,
        "response_age_s": decision.slow.response_age_s,
    }


def _expect(disposition, candidate="first", *, status="UNKNOWN", selection="candidate", failure=None):
    return {
        "slow_disposition": disposition, "slow_status": status, "selection_status": selection,
        "candidate_id": candidate, "fallback_failure": failure, "failed": selection == "failure",
    }


def _single(name, variant, expected, *, arrival=11, decision_at=11.5, sim=101, reverse=False):
    applied = expected["slow_disposition"] == "applied"
    return _Case(name, (
        _Step("activate", 10), _Step("deliver", arrival, value=variant),
        _Step("decide", decision_at, value=sim),
    ), (expected,), {
        "submitted": 1, "delivered": 1, "decisions": 1, "decision_calls": 1,
        "applied": int(applied), "degradations": int(not applied),
        "deadline_misses": int(expected["slow_disposition"] == "deadline_exceeded"),
        "late_deliveries": int(arrival > 12),
        "fallbacks": int(expected["selection_status"] == "fallback"),
        "failures": int(expected["failed"]),
    }, reverse)


def _cases():
    veto = _expect("applied", "second", status="VETO")
    no_veto = _expect("applied", status="NO_VETO")
    missing = _expect("missing_result")
    cases = [
        _single("timely_veto", "VETO", veto),
        _single("timely_no_veto", "NO_VETO", no_veto),
        _single("timely_unknown", "UNKNOWN", _expect("unknown_result")),
        _single("late_answer", "VETO", _expect("deadline_exceeded"), arrival=12.25, decision_at=12.5),
        _single("exact_deadline_and_sim_validity", "VETO", veto, arrival=12, decision_at=12, sim=102),
        _single("decision_after_deadline", "VETO", _expect("deadline_exceeded"), decision_at=12.5),
        _single("simulation_expired", "VETO", _expect("sim_expired"), sim=102.25),
        *[_single(name, variant, missing) for name, variant in (
            ("invalid_json", "invalid_json"), ("wrong_context_hash", "wrong_hash"),
            ("unknown_candidate", "unknown_candidate"),
        )],
        _single("all_candidates_vetoed", "all_veto", _expect("applied", None, status="VETO", selection="fallback")),
        _single("fallback_failure", "all_veto", _expect("applied", None, status="VETO", selection="failure",
                                                       failure="unsupported_motion"), reverse=True),
        _Case("duplicate_answers_first_completion_wins", (
            _Step("activate", 10), _Step("deliver", 11, value="VETO"),
            _Step("deliver", 11.25, value="NO_VETO"), _Step("decide", 11.5, value=101),
        ), (veto,), {"submitted": 1, "delivered": 2, "decisions": 1, "applied": 1, "degradations": 0}),
        _Case("reordered_old_plan_arrives_last", (
            _Step("activate", 10, "old"), _Step("activate", 10.25, "new"),
            _Step("deliver", 10.75, "new", "NO_VETO"), _Step("deliver", 11, "old", "VETO"),
            _Step("decide", 11.5, "new", 101),
        ), (no_veto,), {"submitted": 2, "delivered": 2, "decisions": 1, "applied": 1, "degradations": 0}),
        _Case("old_plan_only", (
            _Step("activate", 10, "old"), _Step("activate", 10.25, "new"),
            _Step("deliver", 11, "old", "VETO"), _Step("decide", 11.5, "new", 101),
        ), (missing,), {"submitted": 2, "delivered": 1, "decisions": 1, "applied": 0, "degradations": 1}),
        _Case("frozen_fast_tick_then_new_plan", (
            _Step("activate", 10, "old"), _Step("decide", 10.1, "old", 100),
            _Step("deliver", 11, "old", "VETO"), _Step("decide", 11.1, "old", 101),
            _Step("activate", 11.25, "new"), _Step("deliver", 11.4, "old", "NO_VETO"),
            _Step("deliver", 11.5, "new", "VETO"), _Step("decide", 11.75, "new", 101.5),
        ), (missing, missing, veto), {"submitted": 2, "delivered": 3, "decisions": 2, "decision_calls": 3,
                                    "applied": 1, "degradations": 1}),
    ]
    return cases
