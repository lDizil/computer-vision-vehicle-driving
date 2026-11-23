import pytest

from hybrid_drive.experiments.spatial_decision_policy import ABSENT, PRESENT, UNCERTAIN
from hybrid_drive.experiments.spatial_vehicle_tracks import (
    TRACK_RECIPES,
    TrackingRecipe,
    VehicleTracker,
    replay_vehicle_tracks,
)
from hybrid_drive.experiments.temporal_spatial_types import FrameDecision, FrameObservation, TemporalInputError


_STRICT = TRACK_RECIPES["motion_strict"]
_WEIGHT = 0.4
_RED = (1.0, 0.0)
_BLUE = (0.0, 1.0)


def _frame(index, *, present=(), uncertain=(), corners=None, clip="clip"):
    boxes = [(band / 4 + 0.04, 0.22, band / 4 + 0.14, 0.42) for band in range(4)]
    for band, (left, right) in (corners or {}).items():
        boxes[band] = (left, 0.22, right, 0.42)
    states = tuple(PRESENT if i in present else UNCERTAIN if i in uncertain else ABSENT for i in range(4))
    scores = tuple(float(state) for state in states)
    obs = FrameObservation("recording", clip, f"{clip}-{index}", index, 1_000_000 + index * 500_000, scores, boxes)
    return FrameDecision(obs, states, scores, boxes, ("raw",) * 4, None)


def _ambiguous(*, extra=False):
    before = _frame(0, present=(0, 1, 3) if extra else (0, 1), corners={0: (0, 0.25), 1: (0.25, 0.50)})
    after = _frame(1, present=(1, 3) if extra else (1,), corners={1: (0.125, 0.375)})
    return before, after


def _run(frames, slots, *, recipe=_STRICT, weight=_WEIGHT):
    inputs = {decision.observation.sample_id: row for decision, row in zip(frames, slots)}
    return replay_vehicle_tracks(frames, recipe, appearance_by_sample=inputs, appearance_weight=weight)


def test_appearance_disambiguates_geometry_without_changing_a_decision():
    frames = _ambiguous()
    baseline = replay_vehicle_tracks(frames, _STRICT)
    assert baseline[1].track_ids[1] == 3 and baseline[1].reasons[1] == "ambiguous_new_track"

    original = (_RED, _BLUE, None, None)
    red = _run(frames, (original, (None, _RED, None, None)))
    blue = _run(frames, (original, (None, _BLUE, None, None)))
    assert red[1].track_ids == (None, 1, None, None)
    assert blue[1].track_ids == (None, 2, None, None)
    assert red[1].reasons[1] == "cross_zone"
    assert all(row.decision is decision for rows in (baseline, red, blue) for row, decision in zip(rows, frames))


def test_missing_feature_even_on_an_unrelated_edge_falls_back_for_the_entire_frame():
    frames = _ambiguous(extra=True)
    baseline = replay_vehicle_tracks(frames, _STRICT)
    assert baseline[1].track_ids == (None, 4, None, 3)
    # The separate link in band 3 lacks a feature. The two ambiguous links
    # into band 1 have features, but mixing comparable and noncomparable
    # costs in one frame is not part of this experiment.
    before = (_RED, _BLUE, None, None)
    after = (None, _RED, None, _BLUE)
    assert _run(frames, (before, after)) == baseline


@pytest.mark.parametrize("destination", [2, 1])
def test_identical_visual_vectors_do_not_open_a_rejected_geometry_edge(destination):
    if destination == 2:
        before = _frame(0, present=(0,), corners={0: (0, 0.48)})
        after = _frame(1, present=(2,), corners={2: (0.25, 1.0)})
    else:
        before, after = _frame(0, present=(0,)), _frame(1, present=(1,))
    current = tuple(_RED if i == destination else None for i in range(4))
    tracked = _run((before, after), ((_RED, None, None, None), current), weight=1.0)
    assert tracked[1].track_ids[destination] == 2
    assert tracked[1].reasons[destination] == "new_track"


def test_no_visual_change_with_zero_weight_or_missing_descriptor_pack():
    frames = _ambiguous()
    features = ((_RED, _BLUE, None, None), (None, _RED, None, None))
    for recipe in TRACK_RECIPES.values():
        old = replay_vehicle_tracks(frames, recipe)
        assert _run(frames, features, recipe=recipe, weight=0) == old
        assert replay_vehicle_tracks(frames, recipe, appearance_weight=1.0) == old
        assert _run(frames, (None, (None, None, None, None)), recipe=recipe) == old


def test_missing_feature_on_a_present_frame_does_not_restore_an_older_descriptor():
    before, _ = _ambiguous()
    again = _frame(1, present=(0, 1), corners={0: (0, 0.25), 1: (0.25, 0.50)})
    after = _frame(2, present=(1,), corners={1: (0.125, 0.375)})
    frames = before, again, after
    features = ((_RED, _BLUE, None, None), (_RED, None, None, None), (None, _RED, None, None))
    result = _run(frames, features)
    assert result[1].track_ids == (1, 2, None, None)
    assert result[2].track_ids[1] == 3
    assert result == replay_vehicle_tracks(frames, _STRICT)


def test_geometry_can_bridge_one_observed_uncertainty_without_output_during_gap():
    before, _ = _ambiguous()
    frames = before, _frame(1, uncertain=(0, 1)), _frame(2, present=(1,), corners={1: (0.125, 0.375)})
    features = (_RED, _BLUE, None, None), None, (None, _BLUE, None, None)
    recipe = TrackingRecipe("motion", bridge_uncertainty=True, reject_ambiguous=True)
    baseline = replay_vehicle_tracks(frames, recipe)
    result = _run(frames, features, recipe=recipe)
    assert baseline[-1].track_ids[1] == 3 and result[-1].track_ids[1] == 2
    assert result[1].track_ids == (None,) * 4
    assert result[-1].reasons[1] == "uncertainty_bridge"
    assert all(row.decision is decision for row, decision in zip(result, frames))


def test_dimensions_across_clips_are_bound_but_visual_history_does_not_cross_cuts():
    before, _ = _ambiguous()
    after = _frame(1, present=(1,), clip="other", corners={1: (0.125, 0.375)})
    frames = before, after
    rows = (_RED, _BLUE, None, None), (None, _BLUE, None, None)
    out = _run(frames, rows)
    assert out[1].track_ids == (None, 1, None, None)
    assert out[1].reasons[1] == "new_track"
    with pytest.raises(TemporalInputError, match="dimension changed"):
        _run(frames, (rows[0], (None, (1, 0, 0), None, None)))


def test_appearance_does_not_carry_across_missing_capture_in_same_clip():
    first = _frame(0, present=(1,))
    third = _frame(2, present=(1,))
    rows = (None, _RED, None, None), (None, _RED, None, None)
    assert [out.track_ids[1] for out in _run((first, third), rows)] == [1, 2]


@pytest.mark.parametrize(
    "features, error",
    [
        ((None,) * 3, "exactly four"),
        (((1, 0), _RED, None, None), "Only a PRESENT"),
        ((None, (1, 0, 0), None, None), "dimension changed"),
        ((None, (), None, None), "nonempty"),
        ((None, (0, 0), None, None), "nonzero"),
        ((None, (float("nan"), 0), None, None), "finite"),
        ((None, (float("inf"), 0), None, None), "finite"),
        ((None, (10**1200, 0), None, None), "finite"),
        ((None, (True, 0.0), None, None), "real numbers"),
    ],
)
def test_invalid_appearance_does_not_mutate_the_live_tracker(features, error):
    before, after = _ambiguous()
    initial = (_RED, _BLUE, None, None)
    actual = VehicleTracker(_STRICT, appearance_weight=_WEIGHT)
    actual.push(before, appearance=initial)
    with pytest.raises(TemporalInputError, match=error) as rejected:
        actual.push(after, appearance=features)
    assert rejected.value.reason == "invalid_appearance"
    expected = _run((before, after), (initial, (None, _RED, None, None)))[1]
    assert actual.push(after, appearance=(None, _RED, None, None)) == expected


def test_invalid_first_frame_must_not_commit_partial_feature_dimension():
    before, after = _ambiguous()
    tracker = VehicleTracker(_STRICT, appearance_weight=_WEIGHT)
    with pytest.raises(TemporalInputError, match="dimension changed"):
        tracker.push(before, appearance=((1, 0, 0), _BLUE, None, None))
    assert tracker.push(before, appearance=(_RED, _BLUE, None, None)).track_ids[:2] == (1, 2)
    assert tracker.push(after, appearance=(None, _BLUE, None, None)).track_ids[1] == 2


@pytest.mark.parametrize("kind", ["missing", "extra", "wrong namespace", "not mapping"])
def test_replay_checks_that_appearance_covers_the_exact_frame_ids(kind):
    before, after = _ambiguous()
    rows = {before.observation.sample_id: None, after.observation.sample_id: None}
    if kind == "missing":
        rows.pop(before.observation.sample_id)
    elif kind == "extra":
        rows["alien"] = None
    elif kind == "wrong namespace":
        rows["alien"] = rows.pop(before.observation.sample_id)
    else:
        rows = tuple(rows)
    with pytest.raises(TemporalInputError) as error:
        replay_vehicle_tracks((before, after), _STRICT, appearance_by_sample=rows, appearance_weight=_WEIGHT)
    assert error.value.reason == "invalid_appearance"


@pytest.mark.parametrize("weight", [True, -0.1, 1.1, float("nan"), float("inf"), "0.3"])
def test_appearance_weight_is_finite_and_bounded_even_on_empty_replay(weight):
    with pytest.raises(ValueError, match="appearance weight"):
        VehicleTracker(_STRICT, appearance_weight=weight)
    with pytest.raises(ValueError, match="appearance weight"):
        replay_vehicle_tracks((), _STRICT, appearance_weight=weight)


def test_embedding_storage_uses_a_snapshot_and_robust_normalization():
    before, after = _ambiguous()
    own, other = [1e308, 0], [0, 1e308]
    tracker = VehicleTracker(_STRICT, appearance_weight=_WEIGHT)
    tracker.push(before, appearance=[own, other, None, None])
    own[:] = [0, 1]
    other[:] = [1, 0]
    assert tracker.push(after, appearance=(None, (5e-324, 0), None, None)).track_ids[1] == 1


def test_prefix_tracking_is_causal_and_independent_of_later_appearance():
    before, after = _ambiguous()
    last = _frame(2, present=(1,), corners={1: (0.13, 0.38)})
    data = (_RED, _BLUE, None, None), (None, _RED, None, None), (None, _BLUE, None, None)
    just_prefix = _run((before, after), data[:2])
    entire = _run((before, after, last), data)
    assert entire[:2] == just_prefix
    assert _run((before, after, last), data) == entire
