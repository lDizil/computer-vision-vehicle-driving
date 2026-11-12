"""Two-step visual qualification for calibrated two-candidate projected overlays.

Slot 0 must be drawn cyan and slot 1 magenta in every supplied view. This
protocol is not an uncalibrated real-road scene interpreter or a driving veto.
"""

from dataclasses import dataclass
import json
from typing import Callable

from hybrid_drive.contracts.slow import RiskStatus, SlowResult
from .context import SlowContext
from .llama import LlamaServerFactory, LlamaVisionBackend
from .protocol import (
    BackendFailureWithTrace, InvalidSlowOutput, MAX_RESPONSE_BYTES,
    decode_response_object,
)


PROTOCOL_NAME = "projected-overlap-v1"
VISIBILITY_PROMPT = (
    "Look at the BACKGROUND behind the coloured lines in these three views. "
    "Is a road/grid scene visible, or is the background black/missing? "
    "Choose visible for road/grid/background scene; missing for black background containing only coloured lines; "
    "uncertain if you cannot tell. Return the required JSON scene and context_hash."
)
OVERLAP_PROMPT = (
    "Which coloured line overlaps a tan solid box in these three views of the SAME scene? "
    "Choose cyan, magenta, both, or none. "
    "If the scene is missing and only coloured lines on black are visible, choose unknown. "
    "Inspect the pictures; do not invent boxes. Return only the required JSON overlap and context_hash."
)
SCENES = ("visible", "missing", "uncertain")
OVERLAPS = ("cyan", "magenta", "both", "none", "unknown")


def _overlap_prompt(context: SlowContext) -> str:
    return OVERLAP_PROMPT


@dataclass(frozen=True)
class ProjectedGroundingFactory(LlamaServerFactory):
    prompt_builder: Callable[[SlowContext], str] = _overlap_prompt

    def __post_init__(self):
        if self.decision_first:
            raise ValueError("decision_first is not meaningful for the projected protocol")

    @property
    def model_version(self):
        return self.sources["model_version"] + "-projected-v1"

    @property
    def protocol_name(self):
        return PROTOCOL_NAME

    @property
    def expected_output_fields(self):
        return None

    def prompts_for(self, context):
        _validate_context(context)
        return VISIBILITY_PROMPT, self.prompt_builder(context)

    def parse_output(self, raw, context):
        return parse_projected_trace(raw, context)

    def __call__(self):
        return ProjectedGroundingBackend(self)


class ProjectedGroundingBackend:
    def __init__(self, config: ProjectedGroundingFactory):
        self.config = config
        self._transport = LlamaVisionBackend(config)

    def generate(self, context):
        _validate_context(context)
        if context.request.model_version != self.config.model_version:
            raise ValueError("Slow request targets another model version")
        prompts = self.config.prompts_for(context)
        trace = {"protocol": PROTOCOL_NAME, "visibility_raw": None, "overlap_raw": None}
        scene = self._read_stage(context, prompts[0], "visibility_raw", "scene", SCENES, trace)
        self._require_trace(trace)
        if scene == "visible":
            self._read_stage(context, prompts[1], "overlap_raw", "overlap", OVERLAPS, trace)
        return self._require_trace(trace)

    def _read_stage(self, context, prompt, trace_key, key, values, trace):
        try:
            trace[trace_key] = self._transport.complete(
                context, prompt=prompt, schema=_stage_schema(context, key, values))
        except InvalidSlowOutput as exc:
            trace[trace_key] = exc.raw_response
            raise InvalidSlowOutput("Invalid projected model output", _bounded_trace(trace)) from exc
        except Exception as exc:
            raise BackendFailureWithTrace(
                "Projected model transport failed", _bounded_trace(trace)) from exc
        try:
            return _parse_stage(trace[trace_key], context, key, values)
        except ValueError as exc:
            raise InvalidSlowOutput("Invalid projected model output", _bounded_trace(trace)) from exc

    @staticmethod
    def _require_trace(trace):
        raw = _bounded_trace(trace)
        if raw is None:
            raise InvalidSlowOutput("Projected trace exceeds the response byte budget")
        return raw

    def close(self):
        self._transport.close()


def _validate_context(context):
    if not isinstance(context, SlowContext):
        raise ValueError("context must be SlowContext")
    if len(context.request.plan.candidates) != 2:
        raise ValueError("Projected overlays require exactly two actual candidates")


def _stage_schema(context, key, values):
    return {
        "type": "object",
        "properties": {
            key: {"type": "string", "enum": list(values)},
            "context_hash": {"type": "string", "const": context.content_hash},
        },
        "required": [key, "context_hash"],
        "additionalProperties": False,
    }


def _parse_stage(raw, context, key, values):
    payload = decode_response_object(raw)
    if tuple(payload) != (key, "context_hash"):
        raise ValueError("Projected stage must contain its decision then context_hash, without other keys")
    if payload["context_hash"] != context.content_hash:
        raise ValueError("Projected stage context_hash does not match")
    value = payload[key]
    if not isinstance(value, str) or value not in values:
        raise ValueError("Projected stage has an unknown categorical value")
    return value


def _bounded_trace(trace):
    try:
        raw = json.dumps(trace, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        if len(raw.encode("utf-8")) <= MAX_RESPONSE_BYTES:
            return raw
    except (TypeError, ValueError, UnicodeError):
        pass
    return None


def parse_projected_trace(raw: str, context: SlowContext) -> SlowResult:
    _validate_context(context)
    trace = decode_response_object(raw)
    if set(trace) != {"protocol", "visibility_raw", "overlap_raw"} or trace["protocol"] != PROTOCOL_NAME:
        raise ValueError("Projected trace has an invalid protocol or fields")
    scene = _parse_stage(trace["visibility_raw"], context, "scene", SCENES)
    overlap = None
    if scene == "visible":
        overlap = _parse_stage(trace["overlap_raw"], context, "overlap", OVERLAPS)
    elif trace["overlap_raw"] is not None:
        raise ValueError("Missing or uncertain scenes must not have an overlap stage")
    status = RiskStatus.UNKNOWN
    slots = ()
    if scene == "visible":
        if overlap == "none":
            status = RiskStatus.NO_VETO
        elif overlap != "unknown":
            status = RiskStatus.VETO
            slots = {"cyan": (0,), "magenta": (1,), "both": (0, 1)}[overlap]
    request = context.request
    plan = request.plan
    evidence = f"scene={scene}" + (f"; overlap={overlap}" if overlap is not None else "")
    return SlowResult(
        request_id=request.request_id, snapshot_id=plan.snapshot_id, plan_id=plan.plan_id,
        candidates_hash=plan.candidates_hash, model_version=request.model_version,
        status=status, vetoed_candidate_ids=tuple(plan.candidates[slot].candidate_id for slot in slots),
        explanation=f"Projected visual evidence: {evidence}.",
    )
