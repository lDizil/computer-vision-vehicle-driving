"""Two-step visual qualification for calibrated two-candidate projected overlays.

Slot 0 must be drawn cyan and slot 1 magenta in every supplied view. This
protocol is not an uncalibrated real-road scene interpreter or a driving veto.
"""

from dataclasses import dataclass
from typing import Callable

from hybrid_drive.contracts.slow import RiskStatus, SlowResult
from .context import SlowContext
from .llama import LlamaServerFactory, LlamaVisionBackend
from .visual_stages import SCENES, parse_visual_stages, run_visual_stages


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
        return run_visual_stages(
            self._transport, context, self.config.prompts_for(context), protocol=PROTOCOL_NAME,
            second_key="overlap", second_values=OVERLAPS,
        )

    def close(self):
        self._transport.close()


def _validate_context(context):
    if not isinstance(context, SlowContext):
        raise ValueError("context must be SlowContext")
    if len(context.request.plan.candidates) != 2:
        raise ValueError("Projected overlays require exactly two actual candidates")


def parse_projected_trace(raw: str, context: SlowContext) -> SlowResult:
    _validate_context(context)
    scene, overlap = parse_visual_stages(
        raw, context, protocol=PROTOCOL_NAME, second_key="overlap", second_values=OVERLAPS)
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
