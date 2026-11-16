"""Single actual candidate on real rectified RGB; offline advisory, never a safety claim."""

from dataclasses import dataclass
from typing import Callable

from hybrid_drive.contracts.slow import RiskStatus, SlowResult
from .context import SlowContext
from .llama import LlamaServerFactory, LlamaVisionBackend
from .visual_stages import parse_visual_stages, run_visual_stages


PROTOCOL_NAME = "real-path-advisory-v1"
VISIBILITY_PROMPT = (
    "These are LEFT, FRONT and RIGHT current real road camera views of the SAME scene. "
    "A cyan line marks one proposed ego path, projected approximately on a flat z=0 plane. "
    "Are both the road scene and a useful portion of that path visible in at least one view? "
    "Choose visible only if you can inspect the cyan path against real road pixels; "
    "missing for absent/black imagery or no visible cyan path; uncertain if evidence is unclear. "
    "Black rectification borders are not obstacles. Return only JSON scene then context_hash."
)
PATH_PROMPT = (
    "Inspect the cyan proposed path in the three current real road views. "
    "Choose blocked only for a clearly visible physical obstacle intersecting that path; "
    "clear only when the visible path portion has no apparent physical obstacle; "
    "unknown if uncertain, occluded, ambiguous or the path is not inspectable. "
    "Do not infer future motion, unseen obstacles, traffic-light state or road safety. "
    "The flat-plane projection is approximate and camera capture times are unknown. "
    "Ignore black rectification borders. Return only JSON path_state then context_hash."
)
PATH_STATES = ("blocked", "clear", "unknown")


def _path_prompt(context):
    return PATH_PROMPT


@dataclass(frozen=True)
class RealPathGroundingFactory(LlamaServerFactory):
    prompt_builder: Callable[[SlowContext], str] = _path_prompt

    def __post_init__(self):
        if self.decision_first:
            raise ValueError("decision_first is not meaningful for the real-path protocol")

    @property
    def model_version(self):
        return self.sources["model_version"] + "-real-path-v1"

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
        return parse_real_path_trace(raw, context)

    def __call__(self):
        return RealPathGroundingBackend(self)


class RealPathGroundingBackend:
    def __init__(self, config):
        self.config = config
        self._transport = LlamaVisionBackend(config)

    def generate(self, context):
        _validate_context(context)
        if context.request.model_version != self.config.model_version:
            raise ValueError("Slow request targets another model version")
        return run_visual_stages(
            self._transport, context, self.config.prompts_for(context), protocol=PROTOCOL_NAME,
            second_key="path_state", second_values=PATH_STATES,
        )

    def close(self):
        self._transport.close()


def _validate_context(context):
    if not isinstance(context, SlowContext):
        raise ValueError("context must be SlowContext")
    if len(context.request.plan.candidates) != 1:
        raise ValueError("Real-path overlays require exactly one actual candidate")


def parse_real_path_trace(raw, context):
    _validate_context(context)
    scene, decision = parse_visual_stages(
        raw, context, protocol=PROTOCOL_NAME, second_key="path_state", second_values=PATH_STATES)
    status = RiskStatus.UNKNOWN
    ids = ()
    if scene == "visible" and decision == "blocked":
        status, ids = RiskStatus.VETO, (context.request.plan.candidates[0].candidate_id,)
    elif scene == "visible" and decision == "clear":
        status = RiskStatus.NO_VETO
    request = context.request
    plan = request.plan
    return SlowResult(
        request_id=request.request_id, snapshot_id=plan.snapshot_id, plan_id=plan.plan_id,
        candidates_hash=plan.candidates_hash, model_version=request.model_version,
        status=status, vetoed_candidate_ids=ids,
        explanation=f"Real-path advisory: scene={scene}; path_state={decision or 'not_requested'}.",
    )
