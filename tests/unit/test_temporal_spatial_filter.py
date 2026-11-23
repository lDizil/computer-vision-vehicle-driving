from dataclasses import replace
import math
import sys

import pytest

from hybrid_drive.experiments.spatial_decision_policy import ABSENT, PRESENT, UNCERTAIN, ZoneDecisionPolicy
from hybrid_drive.experiments.temporal_spatial_filter import (
    DEFAULT_MAX_AGE_US,
    DEFAULT_MAX_GAP_US,
    RECIPES,
    FrameObservation,
    TemporalInputError,
    TemporalRecipe,
    TemporalSpatialFilter,
)


POLICY = ZoneDecisionPolicy(0.25, 0.75)
BOXES = tuple((band / 4 + 0.02, 0.2, band / 4 + 0.19, 0.5) for band in range(4))
DISJOINT_BOXES = tuple((x1, 0.6, x2, 0.8) for x1, _, x2, _ in BOXES)


def _frame(index, score=0.9, *, scores=None, timestamp_us=None, boxes=BOXES, clip="clip", recording="rec"):
    return FrameObservation(
        recording,
        clip,
        f"{clip}-{index}",
        index,
        1_000_000 + 500_000 * index if timestamp_us is None else timestamp_us,
        (score, 0.0, 0.0, 0.0) if scores is None else scores,
        boxes,
    )


def _filter(key, **options):
    return TemporalSpatialFilter(POLICY, RECIPES[key], **options)


def _shifted_boxes(dx):
    return tuple((x1 + dx, y1, x2 + dx, y2) for x1, y1, x2, y2 in BOXES)


def test_framewise_uses_existing_threshold_ties_and_keeps_raw_geometry():
    low, high = POLICY.absent_below, POLICY.present_at_or_above
    scores = (low, high, math.nextafter(low, -math.inf), math.nextafter(high, -math.inf))
    run = _filter("framewise")

    first = _frame(0, scores=scores)
    initial = run.push(first)
    assert initial.states == (UNCERTAIN, PRESENT, ABSENT, UNCERTAIN)
    assert initial.scores == first.scores and initial.boxes == first.boxes
    assert initial.reset_reason == "initial" and initial.reasons == ("raw",) * 4

    second = _frame(1, scores=scores, boxes=DISJOINT_BOXES)
    current = run.push(second)
    assert current.scores == second.scores and current.boxes == second.boxes
    assert current.reset_reason is None


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("confirm2", (UNCERTAIN, PRESENT, PRESENT, UNCERTAIN, UNCERTAIN, PRESENT, ABSENT, UNCERTAIN)),
        ("confirm3", (UNCERTAIN, UNCERTAIN, PRESENT, UNCERTAIN, UNCERTAIN, UNCERTAIN, ABSENT, UNCERTAIN)),
    ],
)
def test_only_consecutive_positive_frames_confirm_and_absence_is_immediate(key, expected):
    stream = _filter(key)
    scores = (0.9, 0.9, 0.9, 0.5, 0.9, 0.9, 0.1, 0.9)
    out = [stream.push(_frame(idx, score)) for idx, score in enumerate(scores)]
    assert tuple(item.states[0] for item in out) == expected
    assert out[0].reasons[0] == "pending_confirmation"
    assert out[2].reasons[0] == "confirmed"
    assert all(item.states[1:] == (ABSENT, ABSENT, ABSENT) for item in out)


def test_confirmation_streaks_are_per_zone_and_continue_after_maturity():
    stream = _filter("confirm2")
    scores = ((0.9, 0.1, 0.1, 0.9), (0.9, 0.9, 0.1, 0.9), (0.9, 0.9, 0.9, 0.1))
    out = [stream.push(_frame(i, scores=row)) for i, row in enumerate(scores)]
    assert [x.states for x in out] == [
        (UNCERTAIN, ABSENT, ABSENT, UNCERTAIN),
        (PRESENT, UNCERTAIN, ABSENT, PRESENT),
        (PRESENT, PRESENT, UNCERTAIN, ABSENT),
    ]


@pytest.mark.parametrize("key", ["hold1", "hold_box"])
def test_hold_bridges_only_one_associated_unknown_without_reusing_an_old_box(key):
    stream = _filter(key)
    assert stream.push(_frame(0)).states[0] == PRESENT

    current = _frame(1, 0.5, boxes=_shifted_boxes(0.03))
    held = stream.push(current)
    assert held.states[0] == PRESENT and held.reasons[0] == "held_uncertain"
    assert held.boxes[0] == current.boxes[0] and held.boxes[0] != BOXES[0]
    assert held.scores[0] == current.scores[0]

    assert stream.push(_frame(2, 0.5)).states[0] == UNCERTAIN
    assert stream.push(_frame(3)).states[0] == PRESENT
    assert stream.push(_frame(4, 0.1)).states[0] == ABSENT
    assert stream.push(_frame(5, 0.5)).states[0] == UNCERTAIN


def test_hold_requires_geometric_association_and_does_not_bridge_a_gap():
    stream = _filter("hold1")
    stream.push(_frame(0))
    disconnected = stream.push(_frame(1, score=0.5, boxes=DISJOINT_BOXES))
    assert disconnected.states[0] == UNCERTAIN
    assert disconnected.boxes == DISJOINT_BOXES
    stream.push(_frame(2))
    after_missing = stream.push(_frame(4, score=0.5))
    assert after_missing.states[0] == UNCERTAIN and after_missing.reset_reason == "gap"


def test_box_ema_only_blends_two_contiguous_associated_present_boxes():
    stream = _filter("box_ema")
    start = stream.push(_frame(0))
    assert start.boxes == BOXES

    moved = _frame(1, boxes=_shifted_boxes(0.04))
    adjusted = stream.push(moved)
    weight = 1 - math.exp(-1.0)
    assert adjusted.boxes[0] == pytest.approx((0.02 + 0.04 * weight, 0.2, 0.19 + 0.04 * weight, 0.5))
    assert adjusted.reasons[0] == "box_ema" and adjusted.states == start.states
    assert adjusted.boxes[1:] == moved.boxes[1:]  # No geometry smoothing on absent zones.

    wrong = stream.push(_frame(2, boxes=DISJOINT_BOXES))
    assert wrong.boxes == DISJOINT_BOXES and wrong.reasons[0] == "raw"
    assert stream.push(_frame(3, 0.1)).boxes == BOXES
    next_detection = stream.push(_frame(4, boxes=_shifted_boxes(0.03)))
    assert next_detection.boxes == _shifted_boxes(0.03) and next_detection.reasons[0] == "raw"


def test_combined_filter_does_not_blend_with_a_previous_held_unknown():
    stream = _filter("hold_box")
    stream.push(_frame(0))
    stream.push(_frame(1, 0.5, boxes=_shifted_boxes(0.02)))
    new_box = _frame(2, boxes=_shifted_boxes(0.04))
    next_detection = stream.push(new_box)
    assert next_detection.boxes == new_box.boxes and next_detection.reasons[0] == "raw"


def test_score_ema_uses_source_delta_and_exact_repeated_score_boundaries():
    short_run, long_run = _filter("score_ema"), _filter("score_ema")
    start = _frame(0, score=0.0, scores=(0.0, 0.75, 0.25, -1.0))
    for stream in (short_run, long_run):
        stream.push(start)

    # Two 250 ms steps with a constant new score, or one 500 ms step.
    result_250ms = short_run.push(_frame(1, timestamp_us=1_250_000, scores=(1.0, 0.75, 0.25, -1.0)))
    result_short = short_run.push(_frame(2, timestamp_us=1_500_000, scores=(1.0, 0.75, 0.25, -1.0)))
    result_long = long_run.push(_frame(1, timestamp_us=1_500_000, scores=(1.0, 0.75, 0.25, -1.0)))
    assert result_250ms.scores[0] == pytest.approx(1 - math.exp(-0.5))
    assert result_short.scores[0] == pytest.approx(result_long.scores[0])
    assert result_long.scores[0] == pytest.approx(1 - math.exp(-1.0))
    assert result_short.states == (UNCERTAIN, PRESENT, UNCERTAIN, ABSENT)
    assert result_short.scores[1:] == (0.75, 0.25, -1.0)


def test_score_ema_keeps_finite_extreme_logits_finite():
    maximum = sys.float_info.max
    stream = _filter("score_ema")
    stream.push(_frame(0, scores=(-maximum, maximum, maximum, -maximum)))
    result = stream.push(_frame(1, scores=(maximum, -maximum, math.nextafter(maximum, 0), -maximum)))
    assert all(math.isfinite(score) for score in result.scores)
    assert result.scores[0] > 0 and result.scores[1] < 0


def test_gap_threshold_is_inclusive_and_either_time_or_an_index_gap_resets():
    stream = _filter("confirm2")
    t = 1_000_000
    assert stream.push(_frame(3, timestamp_us=t)).states[0] == UNCERTAIN
    t += DEFAULT_MAX_GAP_US
    at_limit = stream.push(_frame(4, timestamp_us=t))
    assert at_limit.states[0] == PRESENT and at_limit.reset_reason is None

    t += DEFAULT_MAX_GAP_US + 1
    gap = stream.push(_frame(5, timestamp_us=t))
    assert gap.states[0] == UNCERTAIN and gap.reset_reason == "gap"
    assert stream.push(_frame(6, timestamp_us=t + 300_000)).states[0] == PRESENT
    skipped = stream.push(_frame(8, timestamp_us=t + 650_000))
    assert skipped.states[0] == UNCERTAIN and skipped.reset_reason == "gap"


@pytest.mark.parametrize(
    ("frame", "availability_us", "reason"),
    [
        (_frame(0), None, "chronology"),
        (replace(_frame(1), sample_id="clip-0"), None, "chronology"),
        (_frame(1, timestamp_us=1_000_000), None, "chronology"),
        (_frame(1, clip="other"), None, "wrong_clip"),
        (_frame(1, recording="other"), None, "wrong_clip"),
        (_frame(1), 1_499_999, "early"),
        (_frame(1), 1_500_000 + DEFAULT_MAX_AGE_US + 1, "stale"),
        (_frame(1), False, "invalid_availability"),
    ],
)
def test_rejections_do_not_change_confirmation_or_bind_to_the_wrong_clip(frame, availability_us, reason):
    stream, control = _filter("confirm2"), _filter("confirm2")
    first = _frame(0)
    assert stream.push(first) == control.push(first)
    with pytest.raises(TemporalInputError) as error:
        stream.push(frame, available_at_us=availability_us)
    assert error.value.reason == reason
    correct = _frame(1)
    assert stream.push(correct) == control.push(correct)


def test_capture_age_boundary_and_clock_order_check():
    stream = _filter("confirm2")
    first = _frame(0)
    arrival = first.timestamp_us + DEFAULT_MAX_AGE_US
    assert stream.push(first, available_at_us=arrival).reset_reason == "initial"
    current = _frame(1, timestamp_us=1_250_000)
    with pytest.raises(TemporalInputError) as error:
        stream.push(current, available_at_us=arrival - 1)
    assert error.value.reason == "chronology"
    assert stream.push(current, available_at_us=arrival).states[0] == PRESENT


def test_invalid_first_frame_does_not_claim_the_clip():
    stream = _filter("framewise")
    wrong_first = _frame(0, clip="delayed")
    with pytest.raises(TemporalInputError) as error:
        stream.push(wrong_first, available_at_us=wrong_first.timestamp_us + DEFAULT_MAX_AGE_US + 1)
    assert error.value.reason == "stale"
    assert stream.push(_frame(0, clip="correct")).reset_reason == "initial"


@pytest.mark.parametrize("key", RECIPES)
def test_past_decisions_are_causal_and_post_gap_is_independent(key):
    first, second = _filter(key), _filter(key)
    prefix = [_frame(idx, score, boxes=_shifted_boxes(0.01 * idx)) for idx, score in enumerate((0.8, 0.5, 0.9))]
    decisions = [first.push(item) for item in prefix]
    assert [second.push(item) for item in prefix] == decisions
    preserved = tuple((x.states, x.scores, x.boxes) for x in decisions)

    more = _frame(3, score=0.9)
    first.push(more)
    second.push(replace(more, scores=(0.01, 0.95, 0.0, 0.0)))
    assert tuple((x.states, x.scores, x.boxes) for x in decisions) == preserved

    after_gap = _frame(12, score=0.5)
    a, b = first.push(after_gap), second.push(after_gap)
    assert a == b and a.reset_reason == "gap"
    assert a == replace(_filter(key).push(after_gap), reset_reason="gap")


@pytest.mark.parametrize(
    "options",
    [{"confirmation_frames": 0}, {"hold_uncertain": 1}, {"score_tau_us": -1}, {"box_tau_us": True}],
)
def test_rejects_ambiguous_filter_configuration(options):
    with pytest.raises(ValueError):
        TemporalRecipe(**options)
