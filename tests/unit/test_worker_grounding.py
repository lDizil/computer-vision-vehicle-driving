from dataclasses import dataclass
import json
import time

import pytest

from hybrid_drive.qualification.cases import build_suite
from hybrid_drive.slow.grounding import PROTOCOL_NAME, parse_projected_trace
from hybrid_drive.slow.protocol import BackendFailureWithTrace
from hybrid_drive.slow.worker import AsyncSlowWorker


@dataclass
class TraceFactory:
    kind: str = "valid"
    model_version: str = "fixture-v1"

    def __call__(self):
        return self

    def generate(self, context):
        visibility = json.dumps({"scene": "visible", "context_hash": context.content_hash})
        overlap = json.dumps({"overlap": "cyan", "context_hash": context.content_hash})
        if self.kind == "malformed":
            overlap = "not-json"
        trace = json.dumps({"protocol": PROTOCOL_NAME, "visibility_raw": visibility,
                            "overlap_raw": None if self.kind == "technical" else overlap})
        if self.kind == "technical":
            raise BackendFailureWithTrace("private exception text must not cross IPC", trace)
        if self.kind == "generic":
            raise RuntimeError("private exception text must not cross IPC")
        if self.kind == "oversized":
            raise BackendFailureWithTrace("private", "x" * 8193)
        return trace


def supplied_context():
    now = time.monotonic()
    return build_suite().cases[0].to_context("fixture-v1", request_id="grounding-worker",
                                           submitted_at_monotonic_s=now,
                                           deadline_monotonic_s=now + 10)


def completion(worker):
    until = time.monotonic() + 8
    while time.monotonic() < until:
        events = worker.poll()
        if events:
            return events[0]
        time.sleep(.005)
    pytest.fail("No grounding worker completion within the test budget")


@pytest.mark.parametrize("kind,outcome", [("valid", "completed"), ("malformed", "invalid_output"),
                                          ("technical", "backend_error"), ("generic", "backend_error"),
                                          ("oversized", "backend_error")])
def test_actual_spawn_preserves_grounding_trace_and_failure_classification(kind, outcome):
    context = supplied_context()
    with AsyncSlowWorker(TraceFactory(kind), "fixture-v1", response_parser=parse_projected_trace) as worker:
        worker.submit(context)
        event = completion(worker)
        assert event.outcome == outcome
        assert worker.outcome_counts == {outcome: 1}
        if kind in ("generic", "oversized"):
            assert event.raw_response is None
        else:
            trace = json.loads(event.raw_response)
            assert trace["protocol"] == PROTOCOL_NAME
            assert json.loads(trace["visibility_raw"])["context_hash"] == context.content_hash
            assert "private exception text" not in event.raw_response
        if kind == "valid":
            assert event.result.status.value == "VETO"
            assert event.result.vetoed_candidate_ids == (context.request.plan.candidates[0].candidate_id,)
        else:
            assert event.result is None
        if kind == "technical":
            assert trace["overlap_raw"] is None
            assert json.loads(trace["visibility_raw"])["scene"] == "visible"


def broken_parser(raw, context):
    raise RuntimeError("parser fault")


@pytest.mark.parametrize("parser", [lambda raw, context: None, broken_parser])
def test_broken_parser_is_technical_failure_not_a_measured_model_rejection(parser):
    context = supplied_context()
    with AsyncSlowWorker(TraceFactory(), "fixture-v1", response_parser=parser) as worker:
        worker.submit(context)
        event = completion(worker)
        assert event.outcome == "backend_error" and event.result is None
        assert json.loads(event.raw_response)["protocol"] == PROTOCOL_NAME
