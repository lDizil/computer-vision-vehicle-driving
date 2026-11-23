from dataclasses import FrozenInstanceError, replace
import math

import numpy as np
import pytest

from hybrid_drive.experiments.temporal_spatial_types import FrameDecision, FrameObservation, TemporalInputError


BOXES = tuple((band / 4 + 0.02, 0.2, band / 4 + 0.19, 0.5) for band in range(4))


def _observation(**changes):
    row = {
        "recording_id": "recording",
        "clip_id": "clip",
        "sample_id": "frame",
        "source_index": 1,
        "timestamp_us": 500_000,
        "scores": (0.2, 1.0, -0.1, 0.3),
        "boxes": BOXES,
    }
    return FrameObservation(**(row | changes))


def test_observation_copies_json_lists_to_an_immutable_payload():
    logits = [0.2, 1.0, -0.1, 0.3]
    boxes = [list(box) for box in BOXES]
    frame = _observation(scores=logits, boxes=boxes)

    logits[0] = 1.0
    boxes[0][0] = 1.0
    assert frame.scores == (0.2, 1.0, -0.1, 0.3)
    assert frame.boxes == BOXES
    with pytest.raises(FrozenInstanceError):
        frame.scores = tuple(logits)


@pytest.mark.parametrize(
    "changes",
    [
        {"recording_id": ""},
        {"clip_id": "clip\nother"},
        {"sample_id": " "},
        {"source_index": True},
        {"source_index": -1},
        {"timestamp_us": 2**63},
        {"timestamp_us": 1.0},
        {"scores": (0.2, 1.0, -0.1)},
        {"scores": (0.2, 1.0, True, 0.3)},
        {"scores": (0.2, 1.0, "0.1", 0.3)},
        {"scores": (0.2, 1.0, float("nan"), 0.3)},
        {"scores": (0.2, 1.0, float("-inf"), 0.3)},
        {"boxes": tuple(reversed(BOXES))},
        {"boxes": ((0.01, -0.2, 0.15, 0.5), *BOXES[1:])},
        {"boxes": ((0.01, 0.5, 0.15, 0.2), *BOXES[1:])},
        {"boxes": ((0.01, 0.5, 0.15, math.nextafter(0.5, 1.0)), *BOXES[1:])},
    ],
)
def test_invalid_trace_inputs_are_rejected_at_the_boundary(changes):
    with pytest.raises(TemporalInputError) as error:
        _observation(**changes)
    assert error.value.reason == "invalid_observation"


def test_box_order_follows_fp32_center_at_the_quarter_boundary():
    x0 = float(np.nextafter(np.float32(0.25), np.float32(0.0)))
    x1 = 0.25
    # Mathematically just left of the boundary; FP32 addition rounds it onto 1/4.
    assert (x0 + x1) / 2 < 0.25
    box = (x0, 0.2, x1, 0.5)
    assert _observation(boxes=(BOXES[0], box, *BOXES[2:])).boxes[1] == box
    with pytest.raises(TemporalInputError, match="FP32 quarter"):
        _observation(boxes=(box, *BOXES[1:]))


def test_decision_rejects_invalid_states_and_copies_mutable_presentation_fields():
    observation = _observation()
    mutable = [1, 0, -1, 1]
    reasons = ["raw"] * 4
    decision = FrameDecision(observation, mutable, observation.scores, observation.boxes, reasons, "initial")
    mutable[0] = -1
    reasons[1] = "changed"
    assert decision.states == (1, 0, -1, 1)
    assert decision.reasons == ("raw",) * 4
    with pytest.raises(ValueError, match="states"):
        replace(decision, states=(1, 2, -1, 1))
