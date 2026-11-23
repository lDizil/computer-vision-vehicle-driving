from itertools import product
import json
from types import SimpleNamespace

import pytest

from hybrid_drive.experiments.spatial_decision_policy import ABSENT as A, PRESENT as P, UNCERTAIN as U
from hybrid_drive.experiments.temporal_spatial_metrics import (
    TemporalTarget,
    evaluate_temporal,
    select_temporal_recipe,
)


_RECTANGLES = (
    (0.01, 0.25, 0.11, 0.50),
    (0.26, 0.25, 0.36, 0.50),
    (0.51, 0.25, 0.61, 0.50),
    (0.76, 0.25, 0.86, 0.50),
)


def _decision(
    index, states=(A, A, A, A), *, recording="R", clip="a", timestamp=None,
    boxes=_RECTANGLES, sample=None,
):
    return SimpleNamespace(
        observation=SimpleNamespace(
            recording_id=recording,
            clip_id=clip,
            sample_id=sample or f"{recording}-{index}",
            source_index=index,
            timestamp_us=index * 500_000 if timestamp is None else timestamp,
        ),
        states=states,
        boxes=boxes,
    )


def _target(*positive_bands, boxes=_RECTANGLES):
    seen = tuple(band in positive_bands for band in range(4))
    return TemporalTarget(seen, tuple(box if present else None for box, present in zip(boxes, seen)))


def _measure(frames, labels, *, baseline=None, max_gap_us=800_000):
    assert len(frames) == len(labels)
    target_map = {row.observation.sample_id: value for row, value in zip(frames, labels)}
    return evaluate_temporal(frames, target_map, baseline=baseline, max_gap_us=max_gap_us)


def _shift_first(dx, dy=0):
    x0, y0, x1, y1 = _RECTANGLES[0]
    return ((x0 + dx, y0 + dy, x1 + dx, y1 + dy), *_RECTANGLES[1:])


def test_counts_abstention_flicker_confident_reversals_and_false_runs_separately():
    rows = [
        _decision(0, (P, P, A, A)),
        _decision(1, (U, P, A, A)),
        _decision(2, (P, A, A, A)),
        _decision(3, (A, P, A, A)),
    ]
    report = _measure(rows, [_target(0), _target(0), _target(), _target()])
    assert (report["frames"], report["queries"], report["positives"], report["negatives"]) == (4, 16, 2, 14)
    assert (report["true_present"], report["false_present"], report["false_absent"], report["uncertain"]) == (
        1, 4, 0, 1,
    )
    assert (report["coverage"], report["recall"], report["precision"]) == (15 / 16, 1 / 2, 1 / 5)
    assert report["localized_iou_half"] == 1
    assert report["macro_positive_iou"] == 1  # Geometry doesn't conceal uncertain positive frames.
    assert (report["stable_edges"], report["flicker_edges"], report["confident_reversal_edges"]) == (11, 4, 3)
    assert report["false_present_runs"] == 3
    assert report["per_recording"]["R"]["false_present_runs"] == 3
    json.dumps(report, allow_nan=False)


def test_localization_count_needs_visible_prediction_and_overlap():
    # The third positive is explicitly classified absent; all three still
    # enter the geometry metric independently of their decision state.
    rows = [
        _decision(0, (P, A, A, A), boxes=_shift_first(0.0, 0.5)),
        _decision(1, (P, A, A, A)),
        _decision(2, (A, A, A, A)),
    ]
    report = _measure(rows, [_target(0)] * 3)
    assert report["true_present"] == 2
    assert report["false_absent"] == 1
    assert report["localized_iou_half"] == 1
    assert report["positive_iou"] == pytest.approx(2 / 3)


@pytest.mark.parametrize("late_index, late_timestamp", [(2, 1_400_000), (3, 1_000_000)])
def test_time_gap_and_index_gap_each_cut_the_pending_appearance(late_index, late_timestamp):
    rows = [
        _decision(0),
        _decision(1, (U, A, A, A)),
        _decision(late_index, (P, A, A, A), timestamp=late_timestamp),
        _decision(late_index + 1, (P, A, A, A), timestamp=late_timestamp + 500_000, clip="another"),
        _decision(0, (P, A, A, A), recording="other", clip="another"),
    ]
    report = _measure(rows, [_target(), *([_target(0)] * 4)])
    assert report["clips"] == 3
    assert report["stable_edges"] == 3
    assert report["jitter_pairs"] == 0
    assert report["delay_appearances"] == 1
    assert report["delay_detected"] == 0
    assert report["delay_never_detected"] == 1
    assert report["delay_median_us"] is None


def test_false_runs_restart_at_clip_boundaries_or_a_dropped_index():
    rows = [
        _decision(1, (P, A, A, A)),
        _decision(2, (P, A, A, A)),
        _decision(4, (P, A, A, A), timestamp=1_500_000),
        _decision(5, (P, A, A, A), timestamp=2_000_000, clip="next"),
    ]
    report = _measure(rows, [_target()] * 4)
    assert report["false_present"] == 4
    assert report["false_present_runs"] == 3
    assert report["stable_edges"] == 4
    assert report["recall"] is None
    assert report["macro_positive_iou"] is None


def test_delay_uses_actual_microseconds_until_first_present_in_each_observed_run():
    times = (0, 400_000, 1_000_001, 1_500_000, 2_000_000, 2_500_000, 3_000_000, 3_500_000)
    presence = (False, True, True, False, True, False, True, False)
    states = (A, U, P, A, P, P, U, P)
    rows = [
        _decision(i, (state, A, A, A), timestamp=time)
        for i, (state, time) in enumerate(zip(states, times))
    ]
    report = _measure(rows, [_target(0) if present else _target() for present in presence])
    assert report["delay_appearances"] == 3
    assert report["delay_detected"] == 2
    assert report["delay_never_detected"] == 1
    assert report["delay_median_us"] == 300_000  # median of 0 and 600001; floor to integer µs


def test_a_positive_at_the_start_of_a_clip_is_not_an_appearance():
    rows = [
        _decision(0),
        _decision(1, (P, A, A, A), clip="new"),
        _decision(2, (P, A, A, A), clip="new"),
    ]
    report = _measure(rows, [_target(), _target(0), _target(0)])
    assert report["delay_appearances"] == report["delay_detected"] == 0
    assert report["delay_median_us"] is None
    assert report["jitter_pairs"] == 1


def test_jitter_penalizes_frozen_box_if_reference_moves():
    shifting = [_shift_first(i * 0.04) for i in range(3)]
    correct = [_decision(i, (P, A, A, A), boxes=boxes) for i, boxes in enumerate(shifting)]
    frozen = [_decision(i, (P, A, A, A)) for i in range(3)]
    report = _measure(correct, [_target(0, boxes=boxes) for boxes in shifting], baseline=frozen)
    assert report["jitter_pairs"] == report["paired_jitter"]["count"] == 2
    assert report["mean_jitter"] == pytest.approx(0)
    assert report["paired_jitter"]["candidate_mean"] == pytest.approx(0)
    assert report["paired_jitter"]["baseline_mean"] == pytest.approx(0.04)


def test_paired_jitter_wont_reward_hiding_the_difficult_second_edge():
    shifting = [_shift_first(i * 0.04) for i in range(3)]
    baseline = [
        _decision(0, (P, A, A, A), boxes=shifting[0]),
        _decision(1, (P, A, A, A), boxes=shifting[1]),
        _decision(2, (P, A, A, A), boxes=shifting[1]),
    ]
    candidate = [*baseline[:2], _decision(2, (U, A, A, A), boxes=shifting[1])]
    labels = [_target(0, boxes=boxes) for boxes in shifting]
    report = _measure(candidate, labels, baseline=baseline)
    assert _measure(baseline, labels)["mean_jitter"] == pytest.approx(0.02)
    assert report["paired_jitter"] == {"count": 1, "candidate_mean": 0.0, "baseline_mean": 0.0}
    assert report["recall"] == pytest.approx(2 / 3)


def test_macro_geometry_weights_recordings_not_their_number_of_positive_queries():
    rows = [
        *(_decision(i, (P, A, A, A)) for i in range(3)),
        _decision(0, (P, A, A, A), recording="other"),
        _decision(0, recording="negative-only"),
    ]
    labels = [_target(0)] * 3 + [_target(0, boxes=_shift_first(0, 0.5)), _target()]
    report = _measure(rows, labels)
    assert report["positive_iou"] == 0.75
    assert report["macro_positive_iou"] == 0.5
    assert report["per_recording"]["negative-only"]["positive_iou"] is None


@pytest.mark.parametrize(
    "rows, error",
    [
        ([_decision(0), _decision(0)], "Duplicate"),
        ([_decision(1), _decision(0, timestamp=900_000)], "increase"),
        ([_decision(0), _decision(1, timestamp=0)], "increase"),
        ([_decision(0), _decision(1, clip="b"), _decision(2)], "clip"),
        ([_decision(0), _decision(0, recording="other"), _decision(1)], "recording"),
        ([_decision(0, (P, U, 2, A))], "decision"),
        ([_decision(0, (False, U, A, A))], "decision"),
    ],
)
def test_rejects_identity_and_decision_inconsistencies(rows, error):
    with pytest.raises(ValueError, match=error):
        _measure(rows, [_target()] * len(rows))


def test_exact_target_and_baseline_alignment():
    frames = [_decision(i) for i in (1, 3)]
    targets = {"R-1": _target(), "R-3": _target()}
    with pytest.raises(ValueError, match="exactly one"):
        evaluate_temporal(frames, {**targets, "R-2": _target()})
    with pytest.raises(ValueError, match="exactly one"):
        evaluate_temporal(frames, {"R-1": _target()})
    with pytest.raises(ValueError, match="Baseline must match"):
        evaluate_temporal(frames, targets, baseline=[_decision(1), _decision(3, timestamp=1_400_000)])
    with pytest.raises(ValueError, match="Baseline must match"):
        evaluate_temporal(frames, targets, baseline=frames[:1])


@pytest.mark.parametrize(
    "present,boxes",
    [
        ((True, False, False, False), (None,) * 4),
        ((False,) * 4, _RECTANGLES),
        ((1, False, False, False), (None,) * 4),
        ((True, False, False, False), ((0.1, 0.2, 0.1, 0.9), None, None, None)),
        ((True, False, False, False), ((0.1, float("nan"), 0.2, 0.9), None, None, None)),
    ],
)
def test_bad_targets_are_not_silently_treated_as_negative(present, boxes):
    with pytest.raises(ValueError):
        TemporalTarget(present, boxes)


def test_selection_development_tie_and_actual_improvement():
    frames = [_decision(i, (state, A, A, A)) for i, state in enumerate((P, A, P, P))]
    chosen = [_decision(i, (state, A, A, A)) for i, state in enumerate((A, A, P, P))]
    extra_false = [_decision(i, (P, A, A, A)) for i in range(4)]
    labels = [_target(), _target(), _target(0), _target(0)]
    base = _measure(frames, labels)
    better = _measure(chosen, labels, baseline=frames)
    bad = _measure(extra_false, labels, baseline=frames)
    report = select_temporal_recipe({"framewise": base, "b": better, "a": better, "bad": bad})
    assert report["name"] == "a"
    assert report["summary"]["scope"] == "development_only"
    assert report["candidates"]["a"]["fewer_false_present"] == 1
    assert "more_false_present" in report["candidates"]["bad"]["reasons"]


def test_selection_falls_back_when_a_candidate_just_becomes_uncertain_on_positive_queries():
    frames = [_decision(i, (P, P, A, A)) for i in range(3)]
    candidate = [_decision(i, (U, A, A, A)) for i in range(3)]
    labels = [_target(0)] * 3
    report = select_temporal_recipe({
        "framewise": _measure(frames, labels),
        "too-cautious": _measure(candidate, labels, baseline=frames),
    })
    assert report["name"] == "framewise"
    assert "recall_loss_or_unmeasured_recall" in report["candidates"]["too-cautious"]["reasons"]


@pytest.mark.parametrize(
    "event_ends, confirm_at, reason",
    [(2, 2, "more_undetected_appearances"), (120, 3, "appearance_delay_regression")],
)
def test_a_small_recall_change_can_still_hide_a_large_event_regression(event_ends, confirm_at, reason):
    frames, alternatives, labels = [], [], []
    for i in range(120):
        realized = 1 <= i < event_ends
        labels.append(_target(0, 1) if realized else _target(0))
        frames.append(_decision(i, (P, P if realized else A, P if i == 0 else A, A)))
        alternatives.append(_decision(i, (P, P if realized and i >= confirm_at else A, A, A)))
    baseline = _measure(frames, labels)
    candidate = _measure(alternatives, labels, baseline=frames)
    assert candidate["recall"] >= baseline["recall"] - 0.02
    selection = select_temporal_recipe({"framewise": baseline, "blocked": candidate})
    assert selection["name"] == "framewise"
    assert reason in selection["candidates"]["blocked"]["reasons"]


def test_measured_jitter_gain_alone_can_qualify_but_unpaired_evidence_cant():
    moving = [_shift_first(i * 0.03) for i in range(3)]
    baseline = [_decision(i, (P, A, A, A)) for i in range(3)]
    candidate = [_decision(i, (P, A, A, A), boxes=boxes) for i, boxes in enumerate(moving)]
    labels = [_target(0, boxes=boxes) for boxes in moving]
    result = select_temporal_recipe({
        "framewise": _measure(baseline, labels),
        "paired": _measure(candidate, labels, baseline=baseline),
        "unpaired": _measure(candidate, labels),
    })
    assert result["name"] == "paired"
    assert result["candidates"]["paired"]["paired_jitter_gain"] == pytest.approx(0.03)
    assert result["candidates"]["unpaired"]["reasons"] == ["no_measured_improvement"]


def test_selection_detects_changed_targets_even_when_counts_stay_equal():
    frames = [_decision(i, (P, A, A, A)) for i in (0, 1)]
    base = _measure(frames, [_target(0), _target(0)])
    swapped = _measure(frames, [_target(0), _target(0, boxes=_shift_first(0.01))])
    assert base["positives"] == swapped["positives"] and base["frames"] == swapped["frames"]
    result = select_temporal_recipe({"framewise": base, "changed": swapped})
    assert result["name"] == "framewise"
    assert result["candidates"]["changed"]["reasons"] == ["different_evaluation_frames_labels_or_gap"]


def test_all_short_traces_account_for_each_confident_false_run_once():
    # Exhaustively enumerate 216 short three-frame combinations. Each false run
    # starts where a wrong confident presence follows a non-wrong state or cut.
    for states, labels in product(product((A, U, P), repeat=3), product((False, True), repeat=3)):
        rows = [_decision(i, (state, A, A, A)) for i, state in enumerate(states)]
        values = [_target(0) if present else _target() for present in labels]
        result = _measure(rows, values)
        wrong = [state == P and not present for state, present in zip(states, labels)]
        start_count = sum(a and not b for a, b in zip(wrong, [False, *wrong]))
        assert result["false_present"] == sum(wrong)
        assert result["false_present_runs"] == start_count


def test_selector_does_not_credit_jitter_paired_against_a_different_recipe():
    moving = [_shift_first(i * 0.03) for i in range(3)]
    base_frames = [_decision(i, (P, A, A, A), boxes=boxes) for i, boxes in enumerate(moving)]
    unrelated_frozen = [_decision(i, (P, A, A, A)) for i in range(3)]
    labels = [_target(0, boxes=boxes) for boxes in moving]
    result = select_temporal_recipe({
        "framewise": _measure(base_frames, labels),
        "bogus_gain": _measure(base_frames, labels, baseline=unrelated_frozen),
    })
    assert result["name"] == "framewise"
    assert result["candidates"]["bogus_gain"]["reasons"] == ["different_paired_baseline"]



def test_selection_does_not_trade_accurate_localization_for_fewer_ghosts():
    base = [_decision(i, (P, P, A, A)) for i in range(3)]
    # Keep the correct positive and suppress the false one, but move the
    # remaining box away from the positive target on every frame.
    bad_geometry = [_decision(i, (P, A, A, A), boxes=_shift_first(0.12)) for i in range(3)]
    targets = [_target(0)] * 3
    report = select_temporal_recipe({
        "framewise": _measure(base, targets),
        "bad": _measure(bad_geometry, targets, baseline=base),
    })
    assert report["name"] == "framewise"
    assert "box_iou_regression" in report["candidates"]["bad"]["reasons"]
