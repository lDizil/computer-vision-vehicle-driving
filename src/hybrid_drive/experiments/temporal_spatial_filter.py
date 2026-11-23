"""Bounded causal alternatives for replaying one clip of spatial evidence."""

from dataclasses import dataclass
import math
from types import MappingProxyType

import numpy as np

from .spatial_decision_policy import PRESENT, UNCERTAIN, ZoneDecisionPolicy
from .temporal_spatial_types import (
    FrameDecision,
    FrameObservation,
    TemporalInputError,
    _valid_uint63,
    _valid_zone_box,
)


DEFAULT_MAX_GAP_US = 800_000
DEFAULT_MAX_AGE_US = 350_000
_EMA_TAU_US = 500_000
_ASSOCIATION_IOU = 0.15


@dataclass(frozen=True, slots=True)
class TemporalRecipe:
    confirmation_frames: int = 1
    hold_uncertain: bool = False
    score_tau_us: int | None = None
    box_tau_us: int | None = None

    def __post_init__(self):
        if not _valid_uint63(self.confirmation_frames) or self.confirmation_frames == 0:
            raise ValueError("confirmation_frames must be a positive integer")
        if type(self.hold_uncertain) is not bool:
            raise ValueError("hold_uncertain must be a boolean")
        for tau in (self.score_tau_us, self.box_tau_us):
            if tau is not None and (not _valid_uint63(tau) or tau == 0):
                raise ValueError("EMA time constants must be positive integer microseconds or None")


RECIPES = MappingProxyType({
    "framewise": TemporalRecipe(),
    "confirm2": TemporalRecipe(confirmation_frames=2),
    "confirm3": TemporalRecipe(confirmation_frames=3),
    "hold1": TemporalRecipe(hold_uncertain=True),
    "score_ema": TemporalRecipe(score_tau_us=_EMA_TAU_US),
    "box_ema": TemporalRecipe(box_tau_us=_EMA_TAU_US),
    "score_box": TemporalRecipe(score_tau_us=_EMA_TAU_US, box_tau_us=_EMA_TAU_US),
    "confirm_box": TemporalRecipe(confirmation_frames=2, box_tau_us=_EMA_TAU_US),
    "hold_box": TemporalRecipe(hold_uncertain=True, box_tau_us=_EMA_TAU_US),
})


@dataclass(frozen=True, slots=True)
class _History:
    decision: FrameDecision
    available_at_us: int
    positive_runs: tuple[int, ...]
    held: tuple[bool, ...]


class TemporalSpatialFilter:
    """Accept frames in source order; one instance owns one recording/clip.

    Replay delivery is on the capture clock, never computer wall time. No
    output is generated for a missing frame. A gap or skipped source index
    clears all history before classifying the new frame.
    """

    def __init__(
        self,
        policy: ZoneDecisionPolicy,
        recipe: TemporalRecipe,
        *,
        max_gap_us: int = DEFAULT_MAX_GAP_US,
        max_age_us: int = DEFAULT_MAX_AGE_US,
    ):
        if not isinstance(policy, ZoneDecisionPolicy) or not isinstance(recipe, TemporalRecipe):
            raise ValueError("Expected a ZoneDecisionPolicy and TemporalRecipe")
        if not _valid_uint63(max_gap_us) or max_gap_us == 0:
            raise ValueError("max_gap_us must be a positive integer")
        if not _valid_uint63(max_age_us):
            raise ValueError("max_age_us must be a nonnegative integer")
        self._policy = policy
        self._recipe = recipe
        self._max_gap_us = max_gap_us
        self._max_age_us = max_age_us
        self._last: _History | None = None

    def push(self, observation: FrameObservation, *, available_at_us: int | None = None) -> FrameDecision:
        if not isinstance(observation, FrameObservation):
            raise TemporalInputError("invalid_observation", "Expected a FrameObservation")
        if available_at_us is None:
            available_at_us = observation.timestamp_us
        if not _valid_uint63(available_at_us):
            raise TemporalInputError("invalid_availability", "available_at_us must be a nonnegative int64 value")
        previous, reset = self._admit(observation, available_at_us)
        updated = self._classify(observation, available_at_us, previous, reset)
        self._last = updated
        return updated.decision

    def _admit(self, frame, available_at_us):
        previous = self._last
        if previous is not None:
            observed = previous.decision.observation
            if (frame.recording_id, frame.clip_id) != (observed.recording_id, observed.clip_id):
                raise TemporalInputError("wrong_clip", "The filter is already bound to another recording/clip")
            if (
                frame.source_index <= observed.source_index
                or frame.timestamp_us <= observed.timestamp_us
                or frame.sample_id == observed.sample_id
                or available_at_us < previous.available_at_us
            ):
                raise TemporalInputError("chronology", "Expected advancing source order and monotonic delivery")

        age_us = available_at_us - frame.timestamp_us
        if age_us < 0:
            raise TemporalInputError("early", "Evidence cannot arrive before its source frame")
        if age_us > self._max_age_us:
            raise TemporalInputError("stale", "Evidence exceeded the allowed source-clock age")
        if previous is None:
            return None, "initial"
        if (
            frame.source_index != observed.source_index + 1
            or frame.timestamp_us - observed.timestamp_us > self._max_gap_us
        ):
            return None, "gap"
        return previous, None

    def _classify(self, frame, availability_us, previous, reset):
        recipe = self._recipe
        elapsed_us = frame.timestamp_us - previous.decision.observation.timestamp_us if previous is not None else 0
        score_history = previous is not None and recipe.score_tau_us is not None
        boxes, scores = frame.boxes, frame.scores
        if score_history:
            weight = _ema_weight(elapsed_us, recipe.score_tau_us)
            scores = tuple(_blend(old, now, weight) for old, now in zip(previous.decision.scores, scores))
        candidates = _policy_states(self._policy, scores)
        raw_states = _policy_states(self._policy, frame.scores) if recipe.hold_uncertain and score_history else candidates
        box_weight = (
            _ema_weight(elapsed_us, recipe.box_tau_us)
            if previous is not None and recipe.box_tau_us is not None else None
        )

        states, adjusted, reasons, runs, held = [], [], [], [], []
        for band, (candidate, box) in enumerate(zip(candidates, boxes)):
            run = _positive_run(candidate, previous, band, recipe.confirmation_frames)
            is_held = False
            reason = ["score_ema"] if score_history else []
            state = candidate
            if candidate == PRESENT and recipe.confirmation_frames > 1:
                if run < recipe.confirmation_frames:
                    state = UNCERTAIN
                    reason.append("pending_confirmation")
                else:
                    reason.append("confirmed")

            maybe_hold = recipe.hold_uncertain and candidate == UNCERTAIN and raw_states[band] == UNCERTAIN
            maybe_smooth = box_weight is not None and state == PRESENT
            associated = (
                (maybe_hold or maybe_smooth)
                and previous is not None
                and previous.decision.states[band] == PRESENT
                and not previous.held[band]
                and _iou(previous.decision.observation.boxes[band], box) >= _ASSOCIATION_IOU
            )
            if associated and maybe_hold:
                # Report the current box, not geometry from a past positive.
                state, is_held = PRESENT, True
                reason.append("held_uncertain")
            if associated and maybe_smooth:
                smoothed = tuple(_blend(old, now, box_weight) for old, now in zip(previous.decision.boxes[band], box))
                # FP32 center rounding at a quarter boundary can invalidate even a convex blend.
                if _valid_zone_box(smoothed, band):
                    box = smoothed
                    reason.append("box_ema")

            states.append(state)
            adjusted.append(box)
            reasons.append("+".join(reason) or "raw")
            runs.append(run)
            held.append(is_held)
        decision = FrameDecision(frame, tuple(states), scores, tuple(adjusted), tuple(reasons), reset)
        return _History(decision, availability_us, tuple(runs), tuple(held))


def _policy_states(policy, scores):
    return tuple(int(x) for x in policy.decide(np.asarray([scores], dtype=np.float64))[0])


def _positive_run(candidate, history, band, required):
    if candidate != PRESENT:
        return 0
    count = history.positive_runs[band] if history is not None else 0
    return min(count + 1, required)


def _ema_weight(delta_us, tau_us):
    return -math.expm1(-delta_us / tau_us)


def _blend(previous, current, weight):
    if previous == current or weight == 1:
        return current
    # An opposite-sign subtraction can overflow with otherwise finite logits.
    if (previous < 0 < current) or (current < 0 < previous):
        return (1 - weight) * previous + weight * current
    return previous + weight * (current - previous)


def _iou(a, b):
    w = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    common = w * h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - common
    return common / union
