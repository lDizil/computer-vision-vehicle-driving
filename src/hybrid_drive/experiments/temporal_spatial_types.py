"""Immutable frame evidence and per-zone answers for offline temporal replay."""

from dataclasses import dataclass
import math
from numbers import Integral, Real

import numpy as np

from hybrid_drive.assets.spatial_manifest import BAND_COUNT, box_horizontal_band

from .spatial_decision_policy import ABSENT, PRESENT, UNCERTAIN


Scores = tuple[float, float, float, float]
Box = tuple[float, float, float, float]
Boxes = tuple[Box, Box, Box, Box]
States = tuple[int, int, int, int]
Reasons = tuple[str, str, str, str]

_MAX_INT64 = (1 << 63) - 1
_MAX_IDENTIFIER = 256


class TemporalInputError(ValueError):
    """Rejected replay input with a machine-readable reason."""

    def __init__(self, reason: str, message: str):
        self.reason = reason
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class FrameObservation:
    recording_id: str
    clip_id: str
    sample_id: str
    source_index: int
    timestamp_us: int
    scores: Scores
    boxes: Boxes

    def __post_init__(self):
        try:
            _identifier(self.recording_id, "recording_id")
            _identifier(self.clip_id, "clip_id")
            _identifier(self.sample_id, "sample_id")
            if not _valid_uint63(self.source_index) or not _valid_uint63(self.timestamp_us):
                raise ValueError("source_index and timestamp_us must be nonnegative int64 values")
            scores, boxes = _checked_scores(self.scores), _checked_boxes(self.boxes)
        except ValueError as exc:
            raise TemporalInputError("invalid_observation", str(exc)) from None
        object.__setattr__(self, "scores", scores)
        object.__setattr__(self, "boxes", boxes)


@dataclass(frozen=True, slots=True)
class FrameDecision:
    observation: FrameObservation
    states: States
    scores: Scores
    boxes: Boxes
    reasons: Reasons
    reset_reason: str | None

    def __post_init__(self):
        if not isinstance(self.observation, FrameObservation):
            raise ValueError("Expected a FrameObservation for the decision")
        states = _read_four(self.states, "states")
        if any(
            not isinstance(state, Integral)
            or isinstance(state, bool)
            or state not in (ABSENT, UNCERTAIN, PRESENT)
            for state in states
        ):
            raise ValueError("states must be ABSENT, UNCERTAIN or PRESENT")
        reasons = _read_four(self.reasons, "reasons")
        for reason in reasons:
            _identifier(reason, "reason")
        if self.reset_reason not in (None, "initial", "gap"):
            raise ValueError("reset_reason must be initial, gap or None")
        object.__setattr__(self, "states", tuple(map(int, states)))
        object.__setattr__(self, "scores", _checked_scores(self.scores))
        object.__setattr__(self, "boxes", _checked_boxes(self.boxes))
        object.__setattr__(self, "reasons", reasons)


def _identifier(value, name):
    if (
        not isinstance(value, str)
        or not 0 < len(value) <= _MAX_IDENTIFIER
        or not value.strip()
        or any(char in value for char in ("\0", "\n", "\r"))
    ):
        raise ValueError(f"{name} must be a nonempty bounded string")


def _valid_uint63(value):
    return type(value) is int and 0 <= value <= _MAX_INT64


def _read_four(values, name):
    if (
        not isinstance(values, (list, tuple, np.ndarray))
        or isinstance(values, np.ndarray) and values.ndim == 0
        or len(values) != BAND_COUNT
    ):
        raise ValueError(f"{name} must have exactly four entries")
    return tuple(values)


def _finite(value):
    if not isinstance(value, Real) or isinstance(value, bool):
        raise ValueError("scores and box coordinates must be finite real numbers")
    try:
        number = float(value)
    except OverflowError:
        raise ValueError("scores and box coordinates must be finite float64 values") from None
    if not math.isfinite(number):
        raise ValueError("scores and box coordinates must be finite float64 values")
    return number


def _checked_scores(scores):
    return tuple(_finite(value) for value in _read_four(scores, "scores"))


def _checked_boxes(boxes):
    checked = tuple(tuple(_finite(v) for v in _read_four(box, "box")) for box in _read_four(boxes, "boxes"))
    for band, box in enumerate(checked):
        if not _valid_zone_box(box, band):
            raise ValueError(f"box {band} must be a normalized, positive-area xyxy box in its FP32 quarter")
    return checked


def _valid_zone_box(box, band):
    x1, y1, x2, y2 = box
    return (
        0 <= x1 < x2 <= 1
        and 0 <= y1 < y2 <= 1
        and np.float32(x1) < np.float32(x2)
        and np.float32(y1) < np.float32(y2)
        and box_horizontal_band(box) == band
    )
