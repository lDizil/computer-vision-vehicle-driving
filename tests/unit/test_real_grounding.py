from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest

from hybrid_drive import Pose2D, Trajectory
from hybrid_drive.contracts.motion import EgoMotion
from hybrid_drive.contracts.planning import Candidate, CandidateSet
from hybrid_drive.contracts.slow import SlowRequest
from hybrid_drive.slow.context import CAMERA_NAMES, CameraFrame, SlowContext
from hybrid_drive.slow import real_grounding as real
from hybrid_drive.slow.protocol import BackendFailureWithTrace, InvalidSlowOutput


@pytest.fixture
def context():
    plan = CandidateSet("plan", "snapshot", 1, Pose2D(0, 0, 0), (Candidate("actual-ltf", Trajectory([[2, 0, 0]], .5)),))
    return SlowContext(SlowRequest("request", plan, 10, 20, 2, "fixture-real-path-v1"),
                       EgoMotion(2, 0, 0, 0), tuple(CameraFrame(n, np.zeros((2, 2, 3), np.uint8)) for n in CAMERA_NAMES))


def stage(context, key, value):
    return json.dumps({key: value, "context_hash": context.content_hash})


def trace(context, scene="visible", path="blocked"):
    return json.dumps({"protocol": real.PROTOCOL_NAME, "visibility_raw": stage(context, "scene", scene),
                       "path_state_raw": None if path is None else stage(context, "path_state", path)})


@pytest.fixture
def runtime(monkeypatch):
    state = SimpleNamespace(outputs=[], calls=[], closed=False)
    class Transport:
        def __init__(self, config): pass
        def complete(self, context, *, prompt, schema):
            state.calls.append((prompt, schema))
            value = state.outputs.pop(0)
            if isinstance(value, Exception): raise value
            return value
        def close(self): state.closed = True
    monkeypatch.setattr(real, "LlamaVisionBackend", Transport)
    state.factory = real.RealPathGroundingFactory("unused", {"model_version": "fixture"})
    return state


@pytest.mark.parametrize("scene,path,status,ids", [
    ("visible", "blocked", "VETO", ("actual-ltf",)), ("visible", "clear", "NO_VETO", ()),
    ("visible", "unknown", "UNKNOWN", ()), ("missing", None, "UNKNOWN", ()), ("uncertain", None, "UNKNOWN", ()),
])
def test_single_actual_candidate_mapping(context, scene, path, status, ids):
    result = real.parse_real_path_trace(trace(context, scene, path), context)
    result.validate_for(context.request)
    assert (result.status.value, result.vetoed_candidate_ids) == (status, ids)


def test_actual_two_stage_trace_and_decision_before_hash(context, runtime):
    runtime.outputs = [" " + stage(context, "scene", "visible"), stage(context, "path_state", "blocked") + "\n"]
    expected = runtime.outputs.copy()
    backend = runtime.factory()
    raw = backend.generate(context)
    assert json.loads(raw) == {"protocol": real.PROTOCOL_NAME, "visibility_raw": expected[0], "path_state_raw": expected[1]}
    assert [p for p, _ in runtime.calls] == [real.VISIBILITY_PROMPT, real.PATH_PROMPT]
    for (_, schema), key in zip(runtime.calls, ("scene", "path_state")):
        assert list(schema["properties"]) == [key, "context_hash"]
        assert schema["properties"]["context_hash"]["const"] == context.content_hash
    backend.close()
    assert runtime.closed


@pytest.mark.parametrize("scene", ["missing", "uncertain"])
def test_absent_evidence_skips_second_model_call(context, runtime, scene):
    runtime.outputs = [stage(context, "scene", scene)]
    raw = runtime.factory().generate(context)
    assert len(runtime.calls) == 1
    assert json.loads(raw)["path_state_raw"] is None


@pytest.mark.parametrize("change", [
    lambda x: x.update(protocol="projected-overlap-v1"),
    lambda x: x.update(path_state_raw=None),
    lambda x: x.update(path_state_raw='{"path_state":"blocked","context_hash":"foreign"}'),
    lambda x: x.update(path_state_raw='{"path_state":"blocked","path_state":"clear"}'),
    lambda x: x.update(path_state_raw='{"path_state":NaN}'),
    lambda x: x.update(extra=True),
])
def test_rejects_wrong_binding_protocol_or_malformed_response(context, change):
    data = json.loads(trace(context)); change(data)
    with pytest.raises(ValueError): real.parse_real_path_trace(json.dumps(data), context)


def test_real_protocol_never_accepts_fictitious_second_path(context, runtime):
    extra = Candidate("invented", context.request.plan.candidates[0].trajectory)
    plan = replace(context.request.plan, candidates=context.request.plan.candidates + (extra,))
    changed = replace(context, request=replace(context.request, plan=plan))
    with pytest.raises(ValueError, match="exactly one"): runtime.factory().generate(changed)
    assert not runtime.calls


def test_keeps_partial_actual_output_but_does_not_convert_transport_error_to_prediction(context, runtime):
    visibility = stage(context, "scene", "visible")
    runtime.outputs = [visibility, OSError("private provider text")]
    with pytest.raises(BackendFailureWithTrace) as exc: runtime.factory().generate(context)
    assert json.loads(exc.value.raw_response) == {"protocol": real.PROTOCOL_NAME, "visibility_raw": visibility, "path_state_raw": None}
    with pytest.raises(ValueError): runtime.factory.parse_output(exc.value.raw_response, context)


def test_malformed_model_stage_is_not_repaired(context, runtime):
    runtime.outputs = [stage(context, "scene", "visible"), "not-json"]
    with pytest.raises(InvalidSlowOutput) as exc: runtime.factory().generate(context)
    assert json.loads(exc.value.raw_response)["path_state_raw"] == "not-json"


@pytest.mark.parametrize("scene", ["missing", "uncertain"])
def test_invalid_conditional_branch_is_rejected(context, scene):
    with pytest.raises(ValueError): real.parse_real_path_trace(trace(context, scene, "clear"), context)
