from dataclasses import replace

import pytest

from hybrid_drive.experiments.spatial_appearance_features import APPEARANCE_KINDS, FrameAppearance
from hybrid_drive.experiments.spatial_appearance_signal import measure_appearance_signal
from hybrid_drive.experiments.spatial_decision_policy import ABSENT, PRESENT, UNCERTAIN
from hybrid_drive.experiments.spatial_track_metrics import TrackingTarget
from hybrid_drive.experiments.temporal_spatial_types import FrameDecision, FrameObservation


_BOXES = tuple((0.08 + i * 0.25, 0.15, 0.21 + i * 0.25, 0.5) for i in range(4))
_A, _B, _C = (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)


def _decision(index, slots, *, boxes=_BOXES, clip="clip", recording="r", timestamp=None):
    states = tuple(PRESENT if i in slots else ABSENT for i in range(4))
    source = FrameObservation(recording, clip, f"{recording}-{index}", index,
                              500_000 * index if timestamp is None else timestamp, (0.0,) * 4, boxes)
    return FrameDecision(source, states, source.scores, boxes, ("raw",) * 4, None)


def _target(ids, boxes=_BOXES):
    return TrackingTarget(ids, tuple(box if alias is not None else None for box, alias in zip(boxes, ids)))


def _features(decision, vectors):
    rgb = tuple(vector + (0.0,) * (24 - len(vector)) if vector is not None else None for vector in vectors)
    return FrameAppearance(decision.observation.sample_id, rgb, vectors, vectors)


def _shuffled(frame):
    result = {}
    for profile in APPEARANCE_KINDS:
        vectors = getattr(frame, profile)
        active = [v for v in vectors if v is not None]
        moved = iter(active[1:] + active[:1])
        result[profile] = tuple(next(moved) if v is not None else None for v in vectors)
    return replace(frame, **result)


def _crossing_pair():
    before, after = _decision(0, {0}), _decision(1, {0, 1})
    targets = {before.observation.sample_id: _target(("a", None, None, None)),
               after.observation.sample_id: _target(("b", "a", None, None))}
    actual = (_features(before, (_A, None, None, None)), _features(after, (_B, _A, None, None)))
    return (before, after), targets, actual, tuple(_shuffled(f) for f in actual)


def test_same_anchor_and_competing_geometry_are_scored_in_both_controls():
    score = measure_appearance_signal(*_crossing_pair())
    assert score["eligible_geometry"] == 1
    for profile in APPEARANCE_KINDS:
        assert score["profiles"][profile] == {
            "queries": 1, "actual_correct": 1, "shuffled_correct": 0, "actual_ties": 0, "shuffled_ties": 0,
            "actual_mean_margin": 1.0, "shuffled_mean_margin": -1.0,
        }


def test_permuting_both_ends_can_retain_the_signal_for_a_persistent_pair():
    before, after = _decision(0, {0, 1}), _decision(1, {0, 1})
    target = _target(("a", "b", None, None))
    labels = {row.observation.sample_id: target for row in (before, after)}
    features = tuple(_features(row, (_A, _B, None, None)) for row in (before, after))
    result = measure_appearance_signal((before, after), labels, features, tuple(_shuffled(f) for f in features))
    assert result["eligible_geometry"] == 2
    assert result["profiles"]["dino"]["actual_correct"] == result["profiles"]["dino"]["shuffled_correct"] == 2


def test_the_best_wrong_candidate_matters_not_average_or_order():
    before, after = _decision(0, {1}), _decision(1, {0, 1, 2})
    targets = {before.observation.sample_id: _target((None, "a", None, None)),
               after.observation.sample_id: _target(("b", "a", "c", None))}
    # Cosine to the true box is 0.6. The stronger distractor scores 0.8,
    # even though a second distractor would bring the average below true.
    features = (_features(before, (None, _A, None, None)),
                _features(after, ((0.8, 0.6, 0.0), (0.6, 0.8, 0.0), _C, None)))
    score = measure_appearance_signal((before, after), targets, features, tuple(_shuffled(f) for f in features))
    assert score["eligible_geometry"] == 1
    assert score["profiles"]["dino"]["actual_correct"] == 0
    assert score["profiles"]["dino"]["actual_mean_margin"] == pytest.approx(-0.2)


def test_a_nearly_equal_cosine_is_ambiguous_not_a_success():
    decisions, targets, actual, _ = _crossing_pair()
    actual = (actual[0], _features(decisions[1], ((1.0, 0.00001, 0.0), _A, None, None)))
    score = measure_appearance_signal(decisions, targets, actual, tuple(_shuffled(f) for f in actual))
    for stats in score["profiles"].values():
        assert stats["queries"] == stats["actual_ties"] == stats["shuffled_ties"] == 1
        assert stats["actual_correct"] == stats["shuffled_correct"] == 0


def test_missing_any_candidate_vector_removes_only_that_profile_from_both_controls():
    decisions, targets, actual, _ = _crossing_pair()
    actual = (actual[0], replace(actual[1], contrast=(None, _A, None, None)))
    score = measure_appearance_signal(decisions, targets, actual, tuple(_shuffled(f) for f in actual))
    assert score["eligible_geometry"] == 1
    assert score["profiles"]["rgb"]["queries"] == score["profiles"]["dino"]["queries"] == 1
    assert score["profiles"]["contrast"] == {
        "queries": 0, "actual_correct": 0, "shuffled_correct": 0, "actual_ties": 0, "shuffled_ties": 0,
        "actual_mean_margin": None, "shuffled_mean_margin": None,
    }


@pytest.mark.parametrize("bad_box", [(0.08, 0.7, 0.21, 1.0), (0.09, 0.15, 0.12, 0.5), (0.08, 0.2, 0.21, 0.24)])
def test_competing_box_must_pass_center_width_and_height_gates(bad_box):
    decisions, targets, actual, shuffled = _crossing_pair()
    next_boxes = (bad_box, *_BOXES[1:])
    after = replace(decisions[1], boxes=next_boxes, observation=replace(decisions[1].observation, boxes=next_boxes))
    targets[after.observation.sample_id] = _target(("b", "a", None, None), next_boxes)
    score = measure_appearance_signal((decisions[0], after), targets, actual, shuffled)
    assert score["eligible_geometry"] == 0
    assert score["profiles"]["rgb"]["queries"] == 0


def test_unmatched_competitor_cannot_supply_an_identity_negative():
    decisions, targets, actual, shuffled = _crossing_pair()
    targets[decisions[-1].observation.sample_id] = _target((None, "a", None, None))
    assert measure_appearance_signal(decisions, targets, actual, shuffled)["eligible_geometry"] == 0


@pytest.mark.parametrize("changes", [{"clip_id": "cut"}, {"recording_id": "another"},
                                     {"source_index": 2}, {"timestamp_us": 800_001}])
def test_recording_cuts_and_source_gaps_do_not_form_queries(changes):
    decisions, targets, actual, shuffled = _crossing_pair()
    after = replace(decisions[1], observation=replace(decisions[1].observation, **changes))
    score = measure_appearance_signal((decisions[0], after), targets, actual, shuffled)
    assert score["eligible_geometry"] == 0
    assert score["profiles"]["dino"]["queries"] == 0


def test_accepted_raw_pair_at_exact_time_limit_is_eligible():
    decisions, targets, actual, shuffled = _crossing_pair()
    after = replace(decisions[1], observation=replace(decisions[1].observation, timestamp_us=800_000))
    assert measure_appearance_signal((decisions[0], after), targets, actual, shuffled)["eligible_geometry"] == 1


@pytest.mark.parametrize(("change", "error"), [
    ("missing_frame", "cover all"),
    ("reordered", "sample order"),
    ("missing_slot", "preserve the missing descriptor mask"),
    ("not_a_permutation", "within-frame permutation"),
    ("out_of_order", "must increase"),
    ("missing_target", "exactly the evaluated appearance frames"),
    ("not_present", "Only PRESENT"),
])
def test_comparison_fails_closed_on_unbound_inputs(change, error):
    decisions, targets, actual, shuffled = _crossing_pair()
    if change == "missing_frame":
        shuffled = shuffled[:1]
    elif change == "reordered":
        actual = actual[::-1]
    elif change == "missing_slot":
        shuffled = (shuffled[0], replace(shuffled[1], dino=(None, _B, None, None)))
    elif change == "not_a_permutation":
        shuffled = (shuffled[0], replace(shuffled[1], dino=(_A, _C, None, None)))
    elif change == "out_of_order":
        decisions, actual, shuffled = decisions[::-1], actual[::-1], shuffled[::-1]
    elif change == "missing_target":
        del targets[decisions[0].observation.sample_id]
    else:
        before = replace(decisions[0], states=(UNCERTAIN, ABSENT, ABSENT, ABSENT))
        decisions = before, decisions[1]
    with pytest.raises(ValueError, match=error):
        measure_appearance_signal(decisions, targets, actual, shuffled)
