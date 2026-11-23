from collections import Counter
from dataclasses import replace
from itertools import permutations
from random import Random
from types import SimpleNamespace

import pytest

from hybrid_drive.experiments.spatial_decision_policy import ABSENT, PRESENT, UNCERTAIN
from hybrid_drive.experiments.spatial_track_metrics import (
    TrackingTarget,
    _match_frame,
    _max_rectangular_assignment,
    _maximum_cooccurrence,
    evaluate_tracking,
    select_tracking_recipe,
)
from hybrid_drive.experiments.temporal_spatial_types import FrameDecision, FrameObservation


_BOXES = tuple((0.08 + i * 0.25, 0.15, 0.21 + i * 0.25, 0.5) for i in range(4))
_EMPTY = (None,) * 4


def _tracked(index, ids, *, boxes=_BOXES, states=None, recording="r", clip="clip", time=None, sample=None):
    if states is None:
        states = tuple(ABSENT if number is None else PRESENT for number in ids)
    obs = FrameObservation(
        recording, clip, sample or f"{recording}-{index}", index,
        index * 500000 if time is None else time, (0.1, 0.2, 0.3, 0.4), boxes,
    )
    decision = FrameDecision(obs, states, obs.scores, boxes, ("raw",) * 4, None)
    return SimpleNamespace(decision=decision, track_ids=ids)


def _target(*ids, boxes=_BOXES):
    labels = ids + (None,) * (4 - len(ids))
    return TrackingTarget(labels, tuple(box if alias is not None else None for alias, box in zip(labels, boxes)))


def _measure(rows, labels, **kwargs):
    return evaluate_tracking(rows, {x.decision.observation.sample_id: y for x, y in zip(rows, labels)}, **kwargs)


def test_frame_matching_prefers_two_admissible_detections_over_greedy_iou():
    predicted = ((0.0, 0.1, 0.48, 0.4), (0.14, 0.1, 0.56, 0.4), *_BOXES[2:])
    truth = ((0.0, 0.1, 0.28, 0.4), (0.04, 0.1, 0.46, 0.4), *_BOXES[2:])
    rows = [_tracked(0, (7, 3, None, None), boxes=predicted)]
    stats = _measure(rows, [_target("a", "b", boxes=truth)])
    # Greedy takes predicted 0 -> true 1 (IoU .875), leaving no admissible
    # partner for predicted 1. The max-cardinality assignment keeps both.
    assert stats["detections_matched"] == stats["idtp"] == 2
    assert stats["fixed_match_idf1"] == 1.0


def test_frame_matching_tie_is_stable_and_iou_cutoff_is_inclusive():
    a = (0.0, 0.0, 0.5, 1.0)
    b = (0.0, 0.0, 1.0, 1.0)
    output, truth = (a, a), (b, b)
    assert _match_frame(output, (0, 1), truth, (0, 1), 0.5) == ((0, 0), (1, 1))
    assert _match_frame(output, (0, 1), truth, (0, 1), 0.500001) == ()


def test_matches_across_bands_before_assessing_identity_retention():
    # The same annotated vehicle (and its predicted box) crosses x=.25.
    # A namespace based only on zone would incorrectly reset the track.
    rows = [_tracked(0, (1, None, None, None)), _tracked(1, (None, 1, None, None))]
    targets = [_target("car"), _target(None, "car")]
    retained = _measure(rows, targets)
    switched = _measure([rows[0], _tracked(1, (None, 2, None, None))], targets)
    assert retained["evaluation_identity"] == switched["evaluation_identity"]
    assert retained["adjacent_links"] == retained["cross_zone_links"] == 1
    assert retained["cross_zone_correct"] == 1 and switched["cross_zone_correct"] == 0
    assert retained["fixed_match_idf1"] == 1.0
    assert switched["adjacent_switches"] == 1 and switched["fixed_match_idf1"] == 0.5


def test_near_boundary_matching_does_not_require_same_band():
    left = (0.08, 0.15, 0.40, 0.5)
    truth_right = (0.11, 0.15, 0.43, 0.5)
    row = _tracked(0, (1, None, None, None), boxes=(left, *_BOXES[1:]))
    score = _measure([row], [_target(None, "car", boxes=(_BOXES[0], truth_right, *_BOXES[2:]))])
    assert score["fixed_match_idf1"] == 1.0 and score["detections_matched"] == 1


def test_identity_split_penalized_even_with_perfect_frame_detections():
    rows = [_tracked(i, (track, None, None, None)) for i, track in enumerate((4, 6, 6))]
    score = _measure(rows, [_target("same")] * 3)
    assert score["detections_matched"] == 3
    assert (score["idtp"], score["idfp"], score["idfn"]) == (2, 1, 1)
    assert score["fixed_match_idf1"] == pytest.approx(2 / 3)
    assert score["adjacent_links"] == 2 and score["adjacent_switches"] == 1
    assert score["wrong_links"] == 0 and score["inspected_links"] == 1


def test_identity_merge_and_unmatched_detections_both_have_cost():
    rows = [_tracked(i, (1, None, None, None)) for i in range(3)]
    truth = [_target("a"), _target("b"), _target()]
    score = _measure(rows, truth)
    assert score["detections_matched"] == 2
    assert score["wrong_links"] == score["inspected_links"] == 1
    assert (score["idtp"], score["idfp"], score["idfn"]) == (1, 2, 1)
    assert score["fixed_match_idf1"] == pytest.approx(0.4)


def test_observed_one_frame_bridge_is_inspected_but_dropped_image_breaks_identity_scope():
    first, last = [_tracked(i, (1, None, None, None)) for i in (0, 2)]
    unknown = _tracked(1, _EMPTY, states=(UNCERTAIN, ABSENT, ABSENT, ABSENT))
    observed = _measure([first, unknown, last], [_target("a"), _target(), _target("b")])
    dropped = _measure([first, last], [_target("a"), _target("b")])
    assert observed["wrong_links"] == observed["inspected_links"] == 1
    assert observed["fixed_match_idf1"] == 0.5 and observed["adjacent_links"] == 0
    assert dropped["wrong_links"] == dropped["inspected_links"] == 0
    assert dropped["fixed_match_idf1"] == 1.0 and dropped["adjacent_links"] == 0


def test_unmatched_emitted_box_does_not_validate_an_observed_identity_link():
    rows = [_tracked(i, (1, None, None, None)) for i in range(3)]
    score = _measure(rows, [_target("a"), _target(), _target("b")])
    assert score["detections_matched"] == 2 and score["wrong_links"] == score["inspected_links"] == 0


@pytest.mark.parametrize("boundary", ["recording", "clip", "timeout", "drop"])
def test_cut_and_source_gaps_are_not_identity_errors(boundary):
    opts = {}
    second_index = 1
    if boundary == "recording":
        opts.update(recording="r2")
        second_index = 0
    if boundary == "clip":
        opts.update(clip="other")
    if boundary == "timeout":
        opts.update(time=800001)
    if boundary == "drop":
        second_index = 2
    rows = [_tracked(0, (1, None, None, None)), _tracked(second_index, (2, None, None, None), **opts)]
    score = _measure(rows, [_target("same"), _target("same")])
    assert score["idtp"] == 2 and score["fixed_match_idf1"] == 1.0
    assert score["adjacent_links"] == score["adjacent_switches"] == score["inspected_links"] == 0


def test_adjacent_presentation_at_exact_gap_limit_is_inspected():
    rows = [_tracked(0, (1, None, None, None)), _tracked(1, (2, None, None, None), time=800000)]
    s = _measure(rows, [_target("x"), _target("x")])
    assert s["adjacent_links"] == s["adjacent_switches"] == 1


def test_empty_identity_denominator_is_null_not_a_perfect_match():
    row = _tracked(0, _EMPTY)
    empty = _measure([row], [_target()])
    missed = _measure([row], [_target("missed")])
    assert empty["fixed_match_idf1"] is None and not empty["wrong_links"]
    assert missed["fixed_match_idf1"] == 0.0 and missed["idfn"] == 1
    assert empty["per_recording"]["r"]["frames"] == 1


@pytest.mark.parametrize("ids", [(None, 1, None, None), (True, None, None, None), (0, None, None, None)])
def test_rejects_invalid_track_presence_and_ids(ids):
    row = _tracked(0, ids, states=(PRESENT, ABSENT, ABSENT, ABSENT))
    with pytest.raises(ValueError, match="PRESENT|positive"):
        _measure([row], [_target("car")])


def test_duplicate_track_ids_or_target_aliases_cannot_impersonate_two_objects():
    with pytest.raises(ValueError, match="cannot occupy two zones"):
        _measure([_tracked(0, (1, 1, None, None))], [_target("a", "b")])
    with pytest.raises(ValueError, match="cannot occupy two zones"):
        _target("a", "a")
    with pytest.raises(ValueError, match="positive target zones"):
        TrackingTarget(("a", None, None, None), _EMPTY)


@pytest.mark.parametrize("threshold", [0, -0.01, 1.01, True, float("nan"), 10**400])
def test_invalid_iou_threshold_fails_closed(threshold):
    with pytest.raises(ValueError, match="minimum IoU"):
        _measure([_tracked(0, _EMPTY)], [_target()], min_iou=threshold)


def test_rejects_different_target_keys_reordered_source_cut_and_duplicate_frame():
    rows = [_tracked(0, _EMPTY), _tracked(1, _EMPTY, clip="other"), _tracked(2, _EMPTY)]
    with pytest.raises(ValueError, match="source block"):
        _measure(rows, [_target()] * 3)
    with pytest.raises(ValueError, match="increase"):
        _measure(rows[:2][::-1], [_target()] * 2)
    with pytest.raises(ValueError, match="Duplicate evaluated sample"):
        _measure([rows[0], rows[0]], [_target()] * 2)
    with pytest.raises(ValueError, match="tracking target"):
        evaluate_tracking(rows[:1], {"wrong-sample-id": _target()})


def test_trajectory_assignment_is_global_and_integer_exact():
    pairs = Counter({(1, "x"): 10, (1, "y"): 9, (2, "x"): 9})
    assert _maximum_cooccurrence(pairs) == 18
    assert _maximum_cooccurrence(Counter({(1, "x"): 8, (2, "x"): 10, (3, "x"): 4})) == 10
    assert _maximum_cooccurrence(Counter()) == 0
    rng = Random(184)
    for rows in range(1, 5):
        for cols in range(rows, 6):
            for _ in range(13):
                weights = [[rng.randrange(13) for _ in range(cols)] for _ in range(rows)]
                exhaustive = max(sum(weights[i][j] for i, j in enumerate(p))
                                 for p in permutations(range(cols), rows))
                assert _max_rectangular_assignment(weights) == exhaustive


def test_selection_improvement_is_deterministic_and_blocks_identity_merges():
    truth = [_target("a")] * 3 + [_target("b")]
    base = _measure([_tracked(i, (i + 1, None, None, None)) for i in range(4)], truth)
    clean = _measure([_tracked(i, (1 if i < 3 else 2, None, None, None)) for i in range(4)], truth)
    merge = _measure([_tracked(i, (1, None, None, None)) for i in range(4)], truth)
    assert merge["fixed_match_idf1"] > base["fixed_match_idf1"]
    result = select_tracking_recipe({"z": clean, "a": clean, "zone": base, "merged": merge})
    assert result["name"] == "a" and result["baseline_name"] == "zone"
    assert result["candidates"]["merged"]["reasons"] == ["more_wrong_links"]
    assert select_tracking_recipe({"zone": base, "merged": merge})["name"] == "zone"


def test_selection_rejects_changed_geometry_or_targets_even_at_same_counts():
    source = _tracked(0, (1, None, None, None))
    base = _measure([source], [_target("x")])
    changed_target = _measure([source], [_target("not-x")])
    changed_decision = replace(source.decision, scores=(42.0, 0.2, 0.3, 0.4))
    changed_geometry_contract = _measure([SimpleNamespace(
        decision=changed_decision, track_ids=source.track_ids,
    )], [_target("x")])
    changed_cutoff = _measure([source], [_target("x")], min_iou=0.6)
    dev = {"zone": base, "targets": changed_target, "scores": changed_geometry_contract, "cutoff": changed_cutoff}
    result = select_tracking_recipe(dev)
    assert result["name"] == "zone"
    assert all(item["reasons"] == ["different_frozen_geometry_or_targets"] for item in result["candidates"].values())


def test_rejects_noniterable_and_empty_input():
    with pytest.raises(ValueError, match="sequence"):
        evaluate_tracking(None, {})
    with pytest.raises(ValueError, match="at least one"):
        evaluate_tracking([], {})
