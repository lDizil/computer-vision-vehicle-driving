from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest

from hybrid_drive import Pose2D, Trajectory
from hybrid_drive.contracts.motion import EgoMotion
from hybrid_drive.contracts.planning import Candidate, CandidateSet
from hybrid_drive.contracts.slow import RiskStatus, SlowRequest
from hybrid_drive.slow.context import CAMERA_NAMES, CameraFrame, SlowContext
from hybrid_drive.slow import grounding
from hybrid_drive.slow.protocol import BackendFailureWithTrace, InvalidSlowOutput, MAX_RESPONSE_BYTES


@pytest.fixture
def context():
    plan = CandidateSet("plan", "snapshot", 10, Pose2D(0, 0, 0), (
        Candidate("first-actual-id", Trajectory([[1, 0.1, 0]], 0.5)),
        Candidate("second-actual-id", Trajectory([[1, 0, 0]], 0.5)),
    ))
    return SlowContext(
        SlowRequest("request", plan, 100, 101, 10.5, "fixture-projected-v1"), EgoMotion(2, 0, 0, 0),
        tuple(CameraFrame(name, np.zeros((1, 1, 3), np.uint8)) for name in CAMERA_NAMES),
    )


def stage(context, key, value):
    return json.dumps({key: value, "context_hash": context.content_hash}, separators=(",", ":"))


def trace(context, scene="visible", overlap="cyan"):
    return json.dumps({
        "protocol": grounding.PROTOCOL_NAME,
        "visibility_raw": stage(context, "scene", scene),
        "overlap_raw": stage(context, "overlap", overlap) if overlap is not None else None,
    }, separators=(",", ":"))


@pytest.fixture
def runtime(monkeypatch):
    state = SimpleNamespace(outputs=[], calls=[], closed=0)

    class Transport:
        def __init__(self, config):
            pass

        def complete(self, context, *, prompt, schema):
            state.calls.append((context, prompt, schema))
            output = state.outputs.pop(0)
            if isinstance(output, Exception):
                raise output
            return output

        def close(self):
            state.closed += 1

    monkeypatch.setattr(grounding, "LlamaVisionBackend", Transport)
    state.factory = grounding.ProjectedGroundingFactory("unused", {"model_version": "fixture"})
    return state


@pytest.mark.parametrize("overlap,status,ids", [
    ("cyan", RiskStatus.VETO, ("first-actual-id",)),
    ("magenta", RiskStatus.VETO, ("second-actual-id",)),
    ("both", RiskStatus.VETO, ("first-actual-id", "second-actual-id")),
    ("none", RiskStatus.NO_VETO, ()), ("unknown", RiskStatus.UNKNOWN, ()),
])
def test_categorical_visual_evidence_maps_to_actual_candidates(context, overlap, status, ids):
    result = grounding.parse_projected_trace(trace(context, overlap=overlap), context)
    result.validate_for(context.request)
    assert result.status is status
    assert result.vetoed_candidate_ids == ids
    assert result.explanation == f"Projected visual evidence: scene=visible; overlap={overlap}."
    assert (result.request_id, result.snapshot_id, result.plan_id, result.candidates_hash, result.model_version) == (
        "request", "snapshot", "plan", context.request.plan.candidates_hash, "fixture-projected-v1",
    )


@pytest.mark.parametrize("scene", ["missing", "uncertain"])
def test_no_evidence_skips_overlap_inference_and_returns_unknown(context, runtime, scene):
    raw_scene = stage(context, "scene", scene)
    runtime.outputs = [raw_scene]
    backend = runtime.factory()
    raw = backend.generate(context)
    assert json.loads(raw) == {
        "protocol": grounding.PROTOCOL_NAME, "visibility_raw": raw_scene, "overlap_raw": None,
    }
    assert len(runtime.calls) == 1
    result = runtime.factory.parse_output(raw, context)
    assert result.status is RiskStatus.UNKNOWN and not result.vetoed_candidate_ids
    assert result.explanation == f"Projected visual evidence: scene={scene}."
    backend.close()
    assert runtime.closed == 1


def test_backend_keeps_actual_raw_strings_and_sends_hash_after_decisions(context, runtime):
    visibility = " \n" + stage(context, "scene", "visible") + "\t"
    overlap = " " + stage(context, "overlap", "magenta")
    runtime.outputs = [visibility, overlap]
    backend = runtime.factory()
    raw = backend.generate(context)
    assert json.loads(raw) == {
        "protocol": grounding.PROTOCOL_NAME, "visibility_raw": visibility, "overlap_raw": overlap,
    }
    assert runtime.factory.parse_output(raw, context).vetoed_candidate_ids == ("second-actual-id",)
    for (received, prompt, schema), key, values in zip(
        runtime.calls, ("scene", "overlap"), (grounding.SCENES, grounding.OVERLAPS),
    ):
        assert received is context
        assert tuple(schema["properties"]) == (key, "context_hash")
        assert schema["properties"][key] == {"type": "string", "enum": list(values)}
        assert schema["properties"]["context_hash"]["const"] == context.content_hash
        assert schema["required"] == [key, "context_hash"]
        assert schema["additionalProperties"] is False
    assert [call[1] for call in runtime.calls] == [grounding.VISIBILITY_PROMPT, grounding.OVERLAP_PROMPT]
    backend.close()


def test_prompt_injection_only_changes_overlap_not_visibility(context, runtime):
    runtime.outputs = [stage(context, "scene", "visible"), stage(context, "overlap", "none")]
    backend = replace(runtime.factory, prompt_builder=lambda value: "counterfactual overlap instruction")()
    backend.generate(context)
    assert [call[1] for call in runtime.calls] == [
        grounding.VISIBILITY_PROMPT, "counterfactual overlap instruction",
    ]


def test_swapped_actual_candidates_change_binding_and_invalidate_old_trace(context):
    old = trace(context)
    plan = replace(context.request.plan, candidates=tuple(reversed(context.request.plan.candidates)))
    swapped = replace(context, request=replace(context.request, plan=plan))
    with pytest.raises(ValueError, match="context_hash"):
        grounding.parse_projected_trace(old, swapped)
    result = grounding.parse_projected_trace(trace(swapped), swapped)
    assert result.vetoed_candidate_ids == ("second-actual-id",)
    assert result.candidates_hash == plan.candidates_hash


@pytest.mark.parametrize("change", ["request", "pixels", "model"])
def test_full_context_binding_rejects_reuse(context, change):
    old = trace(context)
    if change == "pixels":
        cameras = list(context.cameras)
        cameras[0] = CameraFrame("CAM_L0", np.ones((1, 1, 3), np.uint8))
        changed = replace(context, cameras=cameras)
    else:
        field = {"request_id": "foreign"} if change == "request" else {"model_version": "foreign"}
        changed = replace(context, request=replace(context.request, **field))
    with pytest.raises(ValueError, match="context_hash"):
        grounding.parse_projected_trace(old, changed)


@pytest.mark.parametrize("key", ["visibility_raw", "overlap_raw"])
@pytest.mark.parametrize("corrupt", [
    lambda raw: json.dumps({**json.loads(raw), "context_hash": "foreign"}),
    lambda raw: '{"context_hash":"foreign",' + raw[1:],
    lambda raw: raw[:-1] + ',"extra":true}',
    lambda raw: raw[:-1] + ',"extra":NaN}',
    lambda raw: raw[:-1] + ',"extra":Infinity}',
    lambda raw: "```json\n" + raw + "\n```",
    lambda raw: json.dumps(dict(reversed(list(json.loads(raw).items())))),
    lambda raw: raw.replace('"visible"', '"VISIBLE"').replace('"cyan"', '"CYAN"'),
    lambda raw: raw.replace('"visible"', 'true').replace('"cyan"', '[]'),
    lambda raw: raw.replace('"context_hash":', '"context_hash":"foreign","context_hash":'),
])
def test_each_stage_strictly_rejects_invalid_categories_hash_keys_and_order(context, key, corrupt):
    payload = json.loads(trace(context))
    payload[key] = corrupt(payload[key])
    with pytest.raises(ValueError):
        grounding.parse_projected_trace(json.dumps(payload), context)


@pytest.mark.parametrize("scene,overlap", [
    ("visible", None), ("missing", "cyan"), ("missing", "unknown"), ("uncertain", "none"),
])
def test_trace_requires_exact_visibility_branch(context, scene, overlap):
    with pytest.raises(ValueError):
        grounding.parse_projected_trace(trace(context, scene, overlap), context)


@pytest.mark.parametrize("change", [
    lambda data: data.update(protocol="direct-json-v1"),
    lambda data: data.update(extra="ignored"),
    lambda data: data.pop("overlap_raw"),
    lambda data: data.update(visibility_raw=None),
    lambda data: data.update(overlap_raw={"overlap": "cyan"}),
])
def test_trace_envelope_is_not_an_output_repair_channel(context, change):
    data = json.loads(trace(context))
    change(data)
    with pytest.raises(ValueError):
        grounding.parse_projected_trace(json.dumps(data), context)


@pytest.mark.parametrize("raw", ["x" * (MAX_RESPONSE_BYTES + 1), "界" * 3000, "\ud800", "null", "[]"])
def test_malformed_outer_trace_is_rejected(context, raw):
    with pytest.raises(ValueError):
        grounding.parse_projected_trace(raw, context)


def test_escaped_invalid_unicode_and_duplicate_trace_keys_are_rejected(context):
    data = json.loads(trace(context))
    data["overlap_raw"] = "\ud800"
    for raw in (json.dumps(data), trace(context)[:-1] + ',"protocol":"projected-overlap-v1"}'):
        with pytest.raises(ValueError):
            grounding.parse_projected_trace(raw, context)


@pytest.mark.parametrize("stage_index", [0, 1])
@pytest.mark.parametrize("error", [RuntimeError("secret provider text"), ValueError("secret envelope text")])
def test_infrastructure_failure_keeps_only_actual_partial_trace(context, runtime, stage_index, error):
    visibility = stage(context, "scene", "visible")
    runtime.outputs = [visibility, error] if stage_index else [error]
    backend = runtime.factory()
    with pytest.raises(BackendFailureWithTrace) as failure:
        backend.generate(context)
    assert "secret" not in str(failure.value)
    assert "secret" not in failure.value.raw_response
    data = json.loads(failure.value.raw_response)
    assert data == {"protocol": grounding.PROTOCOL_NAME,
                    "visibility_raw": visibility if stage_index else None, "overlap_raw": None}
    with pytest.raises(ValueError):
        grounding.parse_projected_trace(failure.value.raw_response, context)
    backend.close()
    assert runtime.closed == 1


@pytest.mark.parametrize("stage_index", [0, 1])
def test_malformed_model_stage_is_invalid_output_not_backend_or_success(context, runtime, stage_index):
    bad = '{"scene":"visible","scene":"missing"}'
    visibility = stage(context, "scene", "visible")
    runtime.outputs = [visibility, bad] if stage_index else [bad]
    with pytest.raises(InvalidSlowOutput) as failure:
        runtime.factory().generate(context)
    data = json.loads(failure.value.raw_response)
    assert data["overlap_raw" if stage_index else "visibility_raw"] == bad
    assert len(runtime.calls) == stage_index + 1


def test_truncation_preserves_actual_model_content_inside_trace(context, runtime):
    partial = '{"scene":"vis'
    runtime.outputs = [InvalidSlowOutput("truncated", partial)]
    with pytest.raises(InvalidSlowOutput) as failure:
        runtime.factory().generate(context)
    assert json.loads(failure.value.raw_response)["visibility_raw"] == partial


def test_combined_trace_cannot_exceed_budget_even_when_each_stage_fits(context, runtime):
    visibility = stage(context, "scene", "visible") + " " * 4100
    overlap = stage(context, "overlap", "none") + " " * 4100
    runtime.outputs = [visibility, overlap]
    with pytest.raises(InvalidSlowOutput) as failure:
        runtime.factory().generate(context)
    assert failure.value.raw_response is None
    assert len(runtime.calls) == 2


@pytest.mark.parametrize("count", [1, 3])
def test_protocol_refuses_other_candidate_counts_before_any_model_call(context, runtime, count):
    candidates = context.request.plan.candidates[:count]
    if count == 3:
        candidates += (replace(candidates[0], candidate_id="third"),)
    plan = replace(context.request.plan, candidates=candidates)
    changed = replace(context, request=replace(context.request, plan=plan))
    with pytest.raises(ValueError, match="exactly two"):
        runtime.factory().generate(changed)
    assert not runtime.calls
    with pytest.raises(ValueError, match="exactly two"):
        grounding.parse_projected_trace(trace(changed), changed)


@pytest.mark.parametrize("raw", ["\ud800", "x" * (MAX_RESPONSE_BYTES + 1)])
def test_bad_utf8_or_oversized_model_stage_is_invalid_with_no_unbounded_trace(context, runtime, raw):
    runtime.outputs = [raw]
    with pytest.raises(InvalidSlowOutput) as failure:
        runtime.factory().generate(context)
    assert failure.value.raw_response is None
    assert len(runtime.calls) == 1


def test_request_for_another_model_does_not_infer(context, runtime):
    changed = replace(context, request=replace(context.request, model_version="another-model"))
    with pytest.raises(ValueError, match="another model"):
        runtime.factory().generate(changed)
    assert not runtime.calls
