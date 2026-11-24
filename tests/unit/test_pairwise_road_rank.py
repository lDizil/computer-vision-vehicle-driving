from dataclasses import dataclass, replace
import math

import numpy as np
import pytest

from hybrid_drive.contracts.planning import Candidate
from hybrid_drive.contracts.trajectory import Trajectory
from hybrid_drive.fast.pairwise_road_rank import PairwiseRoadComparison, rank_paired_road_candidates
from hybrid_drive.fast.road_candidate_rank import RoadAssessment, assess_road


@dataclass(frozen=True)
class _Grid:
    x_min: float = -8.
    y_min: float = -10.
    step_m: float = 1.
    rows: int = 48
    cols: int = 24

    def index(self, x, y):
        row = math.floor((x - self.x_min) / self.step_m)
        col = math.floor((y - self.y_min) / self.step_m)
        return (row, col) if 0 <= row < self.rows and 0 <= col < self.cols else None


GRID = _Grid()
THRESHOLDS = {"offroad_threshold": 0.25, "onroad_threshold": 0.75}


def _candidate(name, *poses, interval_s=0.5):
    return Candidate(name, Trajectory(poses, interval_s=interval_s))


def _lane():
    probabilities = np.ones((GRID.rows, GRID.cols))
    known = np.ones_like(probabilities, dtype=bool)
    probabilities[:, GRID.index(0., 2.)[1]:] = 0.
    known[:, GRID.index(0., 7.)[1]:] = False
    return probabilities, known


def _rank(candidates, probabilities, known, *, obstacle_ids=()):
    free = {value.candidate_id: value.candidate_id not in obstacle_ids for value in candidates}
    return rank_paired_road_candidates(candidates, free, probabilities, known, GRID, **THRESHOLDS)


def test_common_blind_start_does_not_hide_observed_future_differences():
    probabilities, known = _lane()
    known[GRID.index(-1., 0.)] = False
    base = [
        _candidate("lateral", (8., 0., 0.), (12., 3., 0.)),
        _candidate("keep", (8., 0., 0.), (12., 0., 0.)),
    ]
    probabilities_before, known_before = probabilities.copy(), known.copy()

    result = _rank(base, probabilities, known)

    assert result.ordered == ("keep", "lateral")
    assert result.initial_unknown and not result.initial_offroad and not result.cycle_detected
    assert result.comparisons == (PairwiseRoadComparison(
        "lateral", "keep", different_samples=1, unknown_different_samples=0,
        comparable_samples=1, left_wins=0, right_wins=1, winner_id="keep",
    ),)
    assert result.assessments["lateral"] == RoadAssessment(3, 1, 1, 1.0)
    assert result.assessments["keep"] == RoadAssessment(3, 0, 1, None)
    assert all(not value.fully_supported for value in result.assessments.values())
    assert tuple(value.candidate_id for value in base) == ("lateral", "keep")
    np.testing.assert_array_equal(probabilities, probabilities_before)
    np.testing.assert_array_equal(known, known_before)


@pytest.mark.parametrize("uncertain", [False, True])
def test_partly_unknown_offroad_footprint_has_no_directional_vote(uncertain):
    probabilities, known = _lane()
    front_corner = GRID.index(11.5, 3.95)
    if uncertain:
        probabilities[front_corner] = 0.5
    else:
        known[front_corner] = False
    base = (_candidate("lateral", (8., 3., 0.)), _candidate("keep", (8., 0., 0.)))

    result = _rank(base, probabilities, known)

    assert result.ordered == ("lateral", "keep")
    assert result.assessments["lateral"] == RoadAssessment(2, 1, 1, 0.5)
    assert result.comparisons == (PairwiseRoadComparison("lateral", "keep", 1, 1, 0, 0, 0, None),)


def test_equal_footprint_cells_are_ignored_and_fully_visible_equal_states_tie():
    probabilities, known = _lane()
    base = (
        _candidate("second", (8.10, 0., 0.), (12., -3., 0.)),
        _candidate("first", (8.11, 0.01, 0.), (12., 0., 0.)),
    )

    result = _rank(base, probabilities, known)

    assert result.ordered == ("second", "first")
    assert result.comparisons == (PairwiseRoadComparison("second", "first", 1, 0, 1, 0, 0, None),)


def test_fully_observed_offroad_vs_offroad_does_not_create_a_winner():
    probabilities, known = _lane()
    probabilities[:, :GRID.index(0., -2.)[1]] = 0.
    base = (_candidate("left", (8., 3., 0.)), _candidate("right", (8., -3., 0.)))

    result = _rank(base, probabilities, known)

    assert result.ordered == ("left", "right")
    assert all(value.offroad_samples == 1 for value in result.assessments.values())
    assert result.comparisons == (PairwiseRoadComparison("left", "right", 1, 0, 1, 0, 0, None),)


def test_rotated_corner_at_offroad_threshold_is_directional_evidence():
    probabilities, known = _lane()
    probabilities.fill(1.)
    probabilities[GRID.index(7.05, 3.5)] = THRESHOLDS["offroad_threshold"]
    probabilities[GRID.index(8., 0.)] = THRESHOLDS["onroad_threshold"]
    base = (_candidate("turned", (8., 0., math.pi / 2)), _candidate("straight", (8., 0., 0.)))

    result = _rank(base, probabilities, known)

    assert result.ordered == ("straight", "turned")
    assert result.comparisons == (PairwiseRoadComparison("turned", "straight", 1, 0, 1, 0, 1, "straight"),)


@pytest.mark.parametrize("also_unknown", [False, True])
def test_observed_initial_offroad_blocks_reordering_even_when_other_start_cells_are_unknown(also_unknown):
    probabilities, known = _lane()
    probabilities[GRID.index(3.5, .95)] = 0.
    if also_unknown:
        known[GRID.index(-1., 0.)] = False
    base = (_candidate("lateral", (8., 3., 0.)), _candidate("keep", (8., 0., 0.)))

    result = _rank(base, probabilities, known)

    assert result.ordered == ("lateral", "keep")
    assert result.initial_offroad and result.initial_unknown == also_unknown
    assert result.assessments["keep"] == RoadAssessment(2, 1, int(also_unknown), 0.0)
    assert result.comparisons[0].winner_id == "keep"


def test_conflicting_directional_votes_abstain_even_if_one_side_has_more():
    probabilities, known = _lane()
    base = (
        _candidate("a", (8., 0., 0.), (12., 3., 0.), (16., 0., 0.)),
        _candidate("b", (8., 3., 0.), (12., 0., 0.), (16., 3., 0.)),
    )

    result = _rank(base, probabilities, known)

    assert result.ordered == ("a", "b")
    assert result.comparisons == (PairwiseRoadComparison("a", "b", 3, 0, 3, 2, 1, None),)


def test_conflicting_candidates_are_barriers_and_keep_their_full_road_assessments():
    probabilities, known = _lane()
    base = (
        _candidate("lateral", (8., 3., 0.)),
        _candidate("keep", (8., 0., 0.)),
        _candidate("overlap", (8., 3., 0.)),
        _candidate("other_keep", (8., 0., 0.)),
    )

    result = _rank(base, probabilities, known, obstacle_ids=("overlap",))

    assert result.ordered == ("keep", "lateral", "overlap", "other_keep")
    assert result.comparisons == (PairwiseRoadComparison("lateral", "keep", 1, 0, 1, 0, 1, "keep"),)
    for value in base:
        assert result.assessments[value.candidate_id] == assess_road(value.trajectory, probabilities, known, GRID, **THRESHOLDS)


def test_preferences_reorder_only_the_slots_of_their_component():
    probabilities, known = _lane()
    base = (
        _candidate("a", (8., 3., 0.), (12., 8., 0.)),
        _candidate("c", (8., 8., 0.), (12., 3., 0.)),
        _candidate("untested", (8., 8., 0.), (12., 8., 0.)),
        _candidate("b", (8., 0., 0.), (12., 8., 0.)),
        _candidate("d", (8., 8., 0.), (12., 0., 0.)),
    )

    result = _rank(base, probabilities, known)

    assert result.ordered == ("b", "d", "untested", "a", "c")
    assert {entry.winner_id for entry in result.comparisons if entry.winner_id} == {"b", "d"}
    assert result.assessments["untested"].unknown_samples == 2


def test_equally_supported_ready_candidates_keep_forecast_order_within_component():
    probabilities, known = _lane()
    base = (
        _candidate("loser", (8., 3., 0.)),
        _candidate("z", (8., 0., 0.)),
        _candidate("a", (8., 0., 0.)),
    )

    assert _rank(base, probabilities, known).ordered == ("z", "a", "loser")


def test_evidence_cycle_leaves_full_forecast_order_intact():
    probabilities, known = _lane()
    base = (
        _candidate("off", (8., 3., 0.), (12., 3., 0.), (16., 3., 0.)),
        _candidate("on", (8., 0., 0.), (12., 0., 0.), (16., 0., 0.)),
        _candidate("overlap", (8., 0., 0.), (12., 0., 0.), (16., 0., 0.)),
        _candidate("a", (8., 0., 0.), (12., 8., 0.), (16., 3., 0.)),
        _candidate("b", (8., 3., 0.), (12., 0., 0.), (16., 8., 0.)),
        _candidate("c", (8., 8., 0.), (12., 3., 0.), (16., 0., 0.)),
    )

    result = _rank(base, probabilities, known, obstacle_ids=("overlap",))

    assert result.cycle_detected
    assert result.ordered == tuple(value.candidate_id for value in base)
    assert tuple(entry.winner_id for entry in result.comparisons) == ("on", "a", "c", "b")


def test_out_of_grid_samples_are_unknown_and_never_compared():
    probabilities, known = _lane()
    base = (_candidate("outside", (1e308, 0., 0.)), _candidate("inside", (8., 0., 0.)))

    result = _rank(base, probabilities, known)

    assert result.ordered == ("outside", "inside")
    assert result.comparisons == (PairwiseRoadComparison("outside", "inside", 1, 1, 0, 0, 0, None),)


@pytest.mark.parametrize("grid", [replace(GRID, cols=True), replace(GRID, step_m=0.), replace(GRID, x_min=math.nan)])
def test_invalid_road_grid_is_rejected(grid):
    probabilities, known = _lane()
    base = (_candidate("a", (8., 0., 0.)),)
    with pytest.raises(ValueError, match="Grid|grid"):
        rank_paired_road_candidates(base, {"a": True}, probabilities, known, grid, **THRESHOLDS)


def test_candidate_ids_overlap_flags_and_time_grids_must_align():
    probabilities, known = _lane()
    a = _candidate("a", (8., 0., 0.))
    cases = [
        ((), {}, "at least one"),
        ((a, a), {"a": True}, "unique"),
        ((a,), {"wrong": True}, "candidate IDs"),
        ((a,), {"a": 1}, "boolean"),
        ((a, _candidate("b", (8., 0., 0.), interval_s=0.6)), {"a": True, "b": True}, "time grid"),
        ((a, _candidate("b", (8., 0., 0.), (12., 0., 0.))), {"a": True, "b": True}, "time grid"),
    ]
    for candidates, free, error in cases:
        with pytest.raises(ValueError, match=error):
            rank_paired_road_candidates(candidates, free, probabilities, known, GRID, **THRESHOLDS)
