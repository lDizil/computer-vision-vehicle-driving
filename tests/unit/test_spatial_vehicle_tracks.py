from dataclasses import FrozenInstanceError, asdict, replace
from itertools import product
import random

import pytest

from hybrid_drive.experiments.spatial_decision_policy import ABSENT, PRESENT, UNCERTAIN
from hybrid_drive.experiments.spatial_vehicle_tracks import (
    TRACK_RECIPES, TrackedFrame, TrackingRecipe, VehicleTracker, _assign, replay_vehicle_tracks,
)
from hybrid_drive.experiments.temporal_spatial_types import FrameDecision, FrameObservation, TemporalInputError


def _box(left, right, y1=0.22, y2=0.42):
    return (left, y1, right, y2)


_BASE_BOXES = tuple(_box(0.04 + 0.25 * i, 0.14 + 0.25 * i) for i in range(4))


def _frame(
    index,
    *,
    shown=(),
    unsure=(),
    boxes=None,
    clip="clip-0",
    recording="recording-0",
    timestamp=None,
    sample_id=None,
    reset=None,
):
    bands = list(_BASE_BOXES)
    for band, box in (boxes or {}).items():
        bands[band] = box
    states = tuple(PRESENT if i in shown else UNCERTAIN if i in unsure else ABSENT for i in range(4))
    scores = tuple(float(i) for i in states)
    observed = FrameObservation(
        recording, clip, sample_id or f"{recording}-{clip}-{index}", index,
        1_000_000 + index * 500_000 if timestamp is None else timestamp, scores, tuple(bands),
    )
    return FrameDecision(observed, states, scores, tuple(bands), ("raw",) * 4, reset)


def test_link_follows_overlapping_box_into_adjacent_quarter_without_changing_evidence():
    rows = (
        _frame(0, shown=(0,), boxes={0: _box(0.15, 0.32)}),
        _frame(1, shown=(1,), boxes={1: _box(0.20, 0.35)}),
    )
    zone = replay_vehicle_tracks(rows, TRACK_RECIPES["zone"])
    tracks = replay_vehicle_tracks(rows, TRACK_RECIPES["iou"])
    assert zone[1].track_ids == (None, 2, None, None)
    assert tracks[1].track_ids == (None, 1, None, None)
    assert tracks[1].reasons == ("absent", "cross_zone", "absent", "absent")
    for original, output in zip(rows, tracks):
        assert output.decision is original
    with pytest.raises(FrozenInstanceError):
        tracks[1].track_ids = (None,) * 4


def test_global_match_does_not_steal_the_only_tracking_candidate_of_the_second_detection():
    rows = (
        _frame(0, shown=(0, 1), boxes={0: _box(0.03, 0.19), 1: _box(0.12, 0.46)}),
        _frame(1, shown=(0, 1), boxes={0: _box(0.14, 0.32), 1: _box(0.32, 0.46)}),
    )
    tracks = replay_vehicle_tracks(rows, TRACK_RECIPES["iou"])
    # Band 0 overlaps ID 2 more strongly than ID 1, but band 1 can
    # only use ID 2. Taking the locally stronger link loses a match.
    assert tracks[1].track_ids == (1, 2, None, None)
    assert tracks[1].reasons[:2] == ("linked", "linked")


def test_motion_link_uses_only_two_earlier_boxes_not_a_new_displayed_box():
    rows = (
        _frame(0, shown=(0,), boxes={0: _box(0.145, 0.205)}),
        _frame(1, shown=(0,), boxes={0: _box(0.185, 0.245)}),
        _frame(2, shown=(1,), boxes={1: _box(0.242, 0.302)}),
    )
    iou = replay_vehicle_tracks(rows, TRACK_RECIPES["iou"])
    motion = replay_vehicle_tracks(rows, TRACK_RECIPES["motion"])
    assert iou[2].track_ids == (None, 2, None, None)
    assert motion[2].track_ids == (None, 1, None, None)
    assert motion[2].decision.boxes[1] == (0.242, 0.22, 0.302, 0.42)
    assert motion[:2] == replay_vehicle_tracks(rows[:2], TRACK_RECIPES["motion"])


def test_ambiguous_motion_link_starts_new_identity_instead_of_guessing():
    rows = (
        _frame(0, shown=(0, 1), boxes={0: _box(0, 0.25), 1: _box(0.25, 0.50)}),
        _frame(1, shown=(1,), boxes={1: _box(0.125, 0.375)}),
    )
    normal = replay_vehicle_tracks(rows, TRACK_RECIPES["motion"])
    strict = replay_vehicle_tracks(rows, TRACK_RECIPES["motion_strict"])
    assert normal[1].track_ids == (None, 1, None, None)  # deterministic equal-cost preference
    assert strict[1].track_ids == (None, 3, None, None)
    assert strict[1].reasons[1] == "ambiguous_new_track"


@pytest.mark.parametrize("recipe", ("iou", "iou_gap", "motion", "motion_gap", "motion_strict"))
def test_two_quarter_jump_is_not_a_link_even_if_boxes_intersect(recipe):
    rows = (
        _frame(0, shown=(0,), boxes={0: _box(0, 0.48)}),
        _frame(1, shown=(2,), boxes={2: _box(0.25, 1.0)}),
    )
    observed = replay_vehicle_tracks(rows, TRACK_RECIPES[recipe])
    assert observed[1].track_ids == (None, None, 2, None)


@pytest.mark.parametrize("recipe", ("iou_gap", "motion_gap"))
def test_only_one_observed_uncertainty_can_bridge_an_id_without_emitting_a_box(recipe):
    rows = (
        _frame(0, shown=(1,)),
        _frame(1, unsure=(1,)),
        _frame(2, shown=(1,)),
        _frame(3, unsure=(1,)),
        _frame(4, unsure=(1,)),
        _frame(5, shown=(1,)),
    )
    actual = replay_vehicle_tracks(rows, TRACK_RECIPES[recipe])
    assert [row.track_ids[1] for row in actual] == [1, None, 1, None, None, 2]
    assert actual[1].reasons[1] == "uncertain"
    assert actual[2].reasons[1] == "uncertainty_bridge"
    assert actual[1].decision is rows[1]


@pytest.mark.parametrize(
    "rows",
    [
        (_frame(0, shown=(1,)), _frame(1), _frame(2, shown=(1,))),
        (_frame(0, shown=(1,)), _frame(1, unsure=(0,)), _frame(2, shown=(1,))),
        (_frame(0, shown=(1,)), _frame(2, shown=(1,), timestamp=1_700_000)),
        (_frame(0, shown=(1,)), _frame(1, shown=(1,), timestamp=1_800_001)),
        (_frame(0, shown=(1,)), _frame(1, shown=(1,), reset="gap")),
    ],
)
def test_gap_variant_never_carries_through_absence_missing_capture_or_explicit_reset(rows):
    actual = replay_vehicle_tracks(rows, TRACK_RECIPES["motion_gap"])
    assert actual[-1].track_ids[1] == 2 and actual[-1].reasons[1] == "new_track"


def test_non_gap_modes_do_not_keep_observed_uncertainty():
    rows = (_frame(0, shown=(1,)), _frame(1, unsure=(1,)), _frame(2, shown=(1,)))
    for recipe in ("zone", "iou", "motion", "motion_strict"):
        assert replay_vehicle_tracks(rows, TRACK_RECIPES[recipe])[-1].track_ids[1] == 2


def test_dormant_link_may_join_the_next_quarter_and_is_not_drawn_in_the_gap():
    rows = (
        _frame(0, shown=(0,), boxes={0: _box(0.15, 0.32)}),
        _frame(1, unsure=(0,)),
        _frame(2, shown=(1,), boxes={1: _box(0.20, 0.35)}),
    )
    actual = replay_vehicle_tracks(rows, TRACK_RECIPES["iou_gap"])
    assert actual[1].track_ids == (None,) * 4
    assert actual[2].track_ids == (None, 1, None, None)
    assert actual[2].reasons[1] == "uncertainty_bridge_cross_zone"


def test_bad_push_does_not_mutate_valid_history_or_allocate_an_id():
    recipe = TRACK_RECIPES["iou"]
    tracker = VehicleTracker(recipe)
    first, second = _frame(0, shown=(1,)), _frame(1, shown=(1, 2))
    tracker.push(first)
    for injected, reason in (
        (first, "chronology"),
        (_frame(1, shown=(0, 2), clip="outside"), "wrong_clip"),
        (_frame(1, shown=(0, 2), timestamp=999_999), "chronology"),
        (object(), "invalid_decision"),
    ):
        with pytest.raises(TemporalInputError) as error:
            tracker.push(injected)
        assert error.value.reason == reason
    assert tracker.push(second) == replay_vehicle_tracks((first, second), recipe)[1]


def test_clip_identity_is_local_and_missing_capture_does_not_reuse_intra_clip_ids():
    rows = (
        _frame(0, shown=(1, 2)),
        _frame(2, shown=(1, 2)),
        _frame(1, shown=(1, 2), recording="second-recording"),
    )
    actual = replay_vehicle_tracks(rows, TRACK_RECIPES["iou"])
    assert [r.track_ids[1:3] for r in actual] == [(1, 2), (3, 4), (1, 2)]


@pytest.mark.parametrize(
    "rows",
    [
        (_frame(1), _frame(2, clip="another"), _frame(3)),
        (_frame(1), _frame(0, clip="another")),
        (_frame(1), _frame(2, recording="another"), _frame(3, clip="different")),
        (_frame(1, sample_id="duplicate"), _frame(2), _frame(3, sample_id="duplicate")),
    ],
)
def test_replay_rejects_different_clip_and_recording_order_corruption(rows):
    with pytest.raises(TemporalInputError) as error:
        replay_vehicle_tracks(rows, TRACK_RECIPES["zone"])
    assert error.value.reason == "chronology"


def test_recipes_are_serializable_and_duplicate_ids_or_synthetic_detections_are_rejected():
    assert asdict(TRACK_RECIPES["motion_gap"]) == {
        "method": "motion", "bridge_uncertainty": True, "reject_ambiguous": False,
    }
    raw = _frame(0, shown=(0, 1), unsure=(2,))
    with pytest.raises(ValueError, match="cannot explain two"):
        TrackedFrame(raw, (1, 1, None, None), ("test",) * 4)
    with pytest.raises(ValueError, match="unconfirmed"):
        TrackedFrame(raw, (1, 2, 3, None), ("test",) * 4)
    with pytest.raises(ValueError):
        TrackingRecipe("zone", bridge_uncertainty=True)


def test_exact_match_agrees_with_independent_small_exhaustive_oracle():
    points = random.Random(341)
    bands = (0, 1, 2, 3)
    for _ in range(64):
        tracks = range(1, points.randint(2, 7))
        costs = {(t, band): points.randrange(8) / 8 for t in tracks for band in bands if points.random() < 0.5}
        outcome = _assign(costs, bands)
        if not costs:
            assert outcome == {}
            continue
        allowed = sorted({t for t, _ in costs})
        missed = max(allowed) + 1
        best = None
        for order in product([*allowed, missed], repeat=len(bands)):
            edges = [(t, b) for t, b in zip(order, bands) if t != missed]
            if len({t for t, _ in edges}) != len(edges) or any(edge not in costs for edge in edges):
                continue
            candidate = (-len(edges), sum(costs[edge] for edge in edges), order)
            if best is None or candidate < best:
                best = candidate
        assert [outcome.get(b, missed) for b in bands] == list(best[2])
