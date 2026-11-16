"""Bounded transport trace shared by explicit two-stage visual protocols."""

import json

from .protocol import BackendFailureWithTrace, InvalidSlowOutput, MAX_RESPONSE_BYTES, decode_response_object


SCENES = ("visible", "missing", "uncertain")


def run_visual_stages(transport, context, prompts, *, protocol, second_key, second_values):
    trace_key = second_key + "_raw"
    trace = {"protocol": protocol, "visibility_raw": None, trace_key: None}
    scene = _read_stage(transport, context, prompts[0], "visibility_raw", "scene", SCENES, trace)
    _require_trace(trace)
    if scene == "visible":
        _read_stage(transport, context, prompts[1], trace_key, second_key, second_values, trace)
    return _require_trace(trace)


def _read_stage(transport, context, prompt, trace_key, key, values, trace):
    try:
        trace[trace_key] = transport.complete(context, prompt=prompt, schema=_stage_schema(context, key, values))
    except InvalidSlowOutput as exc:
        trace[trace_key] = exc.raw_response
        raise InvalidSlowOutput("Invalid visual stage output", _bounded_trace(trace)) from exc
    except Exception as exc:
        raise BackendFailureWithTrace("Visual stage transport failed", _bounded_trace(trace)) from exc
    try:
        return _parse_stage(trace[trace_key], context, key, values)
    except ValueError as exc:
        raise InvalidSlowOutput("Invalid visual stage output", _bounded_trace(trace)) from exc


def _stage_schema(context, key, values):
    return {
        "type": "object",
        "properties": {key: {"type": "string", "enum": list(values)},
                       "context_hash": {"type": "string", "const": context.content_hash}},
        "required": [key, "context_hash"], "additionalProperties": False,
    }


def _parse_stage(raw, context, key, values):
    payload = decode_response_object(raw)
    if tuple(payload) != (key, "context_hash"):
        raise ValueError("Visual stage must contain its decision then context_hash, without other keys")
    if payload["context_hash"] != context.content_hash:
        raise ValueError("Visual stage context_hash does not match")
    value = payload[key]
    if not isinstance(value, str) or value not in values:
        raise ValueError("Visual stage has an unknown categorical value")
    return value


def _bounded_trace(trace):
    try:
        raw = json.dumps(trace, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        if len(raw.encode("utf-8")) <= MAX_RESPONSE_BYTES:
            return raw
    except (TypeError, ValueError, UnicodeError):
        pass
    return None


def _require_trace(trace):
    raw = _bounded_trace(trace)
    if raw is None:
        raise InvalidSlowOutput("Visual trace exceeds the response byte budget")
    return raw


def parse_visual_stages(raw, context, *, protocol, second_key, second_values):
    trace = decode_response_object(raw)
    trace_key = second_key + "_raw"
    if set(trace) != {"protocol", "visibility_raw", trace_key} or trace["protocol"] != protocol:
        raise ValueError("Visual trace has an invalid protocol or fields")
    scene = _parse_stage(trace["visibility_raw"], context, "scene", SCENES)
    decision = None
    if scene == "visible":
        decision = _parse_stage(trace[trace_key], context, second_key, second_values)
    elif trace[trace_key] is not None:
        raise ValueError("Missing or uncertain scenes must not have a second stage")
    return scene, decision
