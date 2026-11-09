"""A fixed prompt for explicitly projected candidate overlays, without fixture labels."""

import json

from hybrid_drive.slow.context import SlowContext
from hybrid_drive.slow.protocol import MAX_PROMPT_BYTES


PROMPT_VERSION = "projected-candidates-v2"


def projected_prompt(context: SlowContext) -> str:
    if not isinstance(context, SlowContext) or len(context.request.plan.candidates) != 2:
        raise ValueError("Projected qualification prompt requires exactly two candidates")
    plan = context.request.plan
    candidates = []
    for candidate, color in zip(plan.candidates, ("cyan", "magenta")):
        candidates.append({"id": candidate.candidate_id, "overlay_color": color,
                           "times_s": candidate.trajectory.times_s.tolist(),
                           "poses_xy_heading": candidate.trajectory.poses.tolist()})
    payload = {
        "axes": "ego x forward, y left, z up; metres, seconds, heading radians",
        "candidates": candidates,
        "motion": {key: getattr(context.motion, key) for key in ("vx_mps", "vy_mps", "ax_mps2", "ay_mps2")},
    }
    text = (
        "Look at these three calibrated schematic views of the SAME static road scene. "
        "The left, front and right images include TWO projected ground paths: first candidate CYAN, "
        "second MAGENTA. Solid boxes are physical obstacles; path colours are NOT risk labels. "
        "Assess each actual coloured path for visible intersection with a solid obstacle. "
        "VETO only candidates with a visible obstacle on their path. NO_VETO if all supplied paths "
        "are visibly unobstructed. UNKNOWN if views lack scene evidence, such as black images with "
        "only path overlays. This is a static schematic control, NOT a real-road safety decision. "
        "Do not infer hidden objects or future observations.\n"
        "Return only one JSON object with status (UNKNOWN, NO_VETO or VETO), "
        "vetoed_candidate_ids (unique IDs from input; empty unless VETO), and explanation "
        "(a short description of what is actually visible). The runtime constrains the final "
        "context_hash field; it has no driving meaning. No example answer is supplied.\n"
        + json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    )
    if len(text.encode("utf-8")) > MAX_PROMPT_BYTES:
        raise ValueError("Projected prompt exceeds byte budget")
    return text
