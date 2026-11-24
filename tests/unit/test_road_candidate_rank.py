from dataclasses import dataclass, replace
import math

import numpy as np
import pytest

from hybrid_drive.contracts.planning import Candidate
from hybrid_drive.contracts.trajectory import Trajectory
from hybrid_drive.fast.road_candidate_rank import assess_road, rank_road_candidates


@dataclass(frozen=True)
class _Grid:
    x_min: float = -8.0
    y_min: float = -8.0
    step_m: float = 1.0
    rows: int = 16
    cols: int = 16

    def index(self, x, y):
        row = math.floor((x - self.x_min) / self.step_m)
        col = math.floor((y - self.y_min) / self.step_m)
        return (row, col) if 0 <= row < self.rows and 0 <= col < self.cols else None


GRID = _Grid()
LIMITS = {"offroad_threshold": 0.25, "onroad_threshold": 0.75}


def _trajectory(*poses, interval_s=0.5):
    return Trajectory(poses, interval_s)


def _candidate(key, pose):
    return Candidate(key, _trajectory(pose))


def _evidence(grid=GRID):
    pose_grid = (grid.rows, grid.cols)
    return np.ones(pose_grid), np.ones(pose_grid, dtype=np.bool_)


def _marked_lane():
    probability, known = _evidence()
    # In this simple fixture, only the far left of the initial lane is offroad.
    probability[:, 10:] = 0.
    return probability, known


def test_road_evidence_can_change_priority_on_a_lateral_probe_without_mutating_inputs():
    probability, known = _marked_lane()
    lateral = _candidate("lateral", (2.0, 3.0, 0.0))
    keep = _candidate("keep", (2.0, 0.0, 0.0))
    candidates = [lateral, keep]
    free = {"lateral": True, "keep": True}
    first = rank_road_candidates(candidates, free, probability, known, GRID, **LIMITS)
    opposite = rank_road_candidates(candidates, free, np.ones_like(probability), known, GRID, **LIMITS)

    assert first.ordered == ("keep", "lateral")
    assert opposite.ordered == ("lateral", "keep")
    assert tuple(item.candidate_id for item in candidates) == ("lateral", "keep")
    np.testing.assert_array_equal(probability, _marked_lane()[0])
    np.testing.assert_array_equal(known, _marked_lane()[1])
    np.testing.assert_array_equal(lateral.trajectory.poses, [[2, 3, 0]])
    assert first.assessments["lateral"].sample_count == 2
    assert first.assessments["lateral"].offroad_samples == 1
    assert first.assessments["lateral"].first_offroad_s == 0.5
    assert first.assessments["lateral"].fully_supported
    assert first.assessments["keep"].offroad_samples == 0
    assert first.assessments["keep"].first_offroad_s is None


def test_ego_rectangle_uses_rear_axle_heading_and_corners_not_just_its_center():
    probability, known = _evidence()
    # At pi/2 the front-left corner of the initial rectangle samples this
    # cell; straight pose and a rotated center at (0, 0) do not.
    probability[GRID.index(-0.95, 3.5)] = 0.25
    out = assess_road(_trajectory((0, 0, math.pi / 2)), probability, known, GRID, **LIMITS)
    straight = assess_road(_trajectory((0, 0, 0)), probability, known, GRID, **LIMITS)
    assert out.offroad_samples == 1 and out.first_offroad_s == 0.5
    assert out.unknown_samples == 0
    assert straight.offroad_samples == 0 and straight.first_offroad_s is None


def test_time_zero_is_sampled_and_a_pose_can_have_both_offroad_and_unknown_points():
    probability, known = _evidence()
    probability[GRID.index(3.5, 0.95)] = 0.0
    known[GRID.index(3.5, -0.95)] = False
    probability[GRID.index(3.5, -0.95)] = 0.0
    out = assess_road(_trajectory((0, 0, 0), interval_s=1.0), probability, known, GRID, **LIMITS)
    assert out.sample_count == out.offroad_samples == out.unknown_samples == 2
    assert out.first_offroad_s == 0.0
    assert not out.fully_supported


def test_grid_boundary_is_half_open_and_outside_is_unknown_even_under_extreme_coordinates():
    g = _Grid(x_min=-2.0, rows=8)
    probability, known = _evidence(g)
    last_inside = _trajectory((2.49, 0, 0))
    # Front bumper x=6 is at the upper boundary; rear bumper x=-2 is at the lower.
    out = assess_road(
        _trajectory((2.5, 0, 0), (-1.0, 0, 0), (-1.01, 0, 0), (1e308, 0, 0)),
        probability, known, g, **LIMITS,
    )
    assert out.sample_count == 5 and out.unknown_samples == 3
    assert out.offroad_samples == 0 and out.first_offroad_s is None
    assert assess_road(last_inside, probability, known, g, **LIMITS).fully_supported


def test_uncertain_and_unobserved_cells_do_not_substitute_for_offroad():
    probability, known = _evidence()
    at_front = GRID.index(3.5, 0.95)
    probability[at_front] = 0.5
    t = _trajectory((0, 0, 0))
    uncertain = assess_road(t, probability, known, GRID, **LIMITS)
    probability[at_front] = 0.0
    known[at_front] = False
    unobserved = assess_road(t, probability, known, GRID, **LIMITS)
    assert uncertain == unobserved
    assert uncertain.unknown_samples == 2 and uncertain.offroad_samples == 0
    probability[at_front] = 0.75
    known[at_front] = True
    assert assess_road(t, probability, known, GRID, **LIMITS).fully_supported


def test_road_ties_keep_order_and_unknown_slot_is_not_given_clear_road_priority():
    probability, known = _marked_lane()
    base = [
        _candidate("off_z", (2.0, 3.0, 0.0)),
        _candidate("unknown", (99.0, 3.0, 0.0)),
        _candidate("off_a", (2.0, 3.0, 0.0)),
        _candidate("known_z", (2.0, 0.0, 0.0)),
        _candidate("known_a", (2.0, 0.0, 0.0)),
        _candidate("overlap", (2.0, 0.0, 0.0)),
    ]
    free = {v.candidate_id: v.candidate_id != "overlap" for v in base}
    ranked = rank_road_candidates(base, free, probability, known, GRID, **LIMITS)
    assert ranked.ordered == ("known_z", "unknown", "known_a", "off_z", "off_a", "overlap")
    assert not ranked.assessments["unknown"].fully_supported
    assert ranked.assessments["overlap"].fully_supported
    all_unknown = rank_road_candidates(base, free, probability, ~known, GRID, **LIMITS)
    assert all_unknown.ordered == tuple(v.candidate_id for v in base)
    assert all(value.unknown_samples == value.sample_count for value in all_unknown.assessments.values())


def test_existing_overlaps_cannot_exchange_priority_with_free_candidates_or_other_overlaps():
    probability, known = _marked_lane()
    off = _candidate("free_offroad", (2.0, 3.0, 0.0))
    overlap_early = _candidate("overlap_offroad", (2.0, 3.0, 0.0))
    overlap_late = _candidate("overlap_onroad", (2.0, 0.0, 0.0))
    on = _candidate("free_onroad", (2.0, 0.0, 0.0))
    free = {"free_offroad": True, "overlap_offroad": False, "overlap_onroad": False, "free_onroad": True}
    # Preserve both the overlap group and its precedence with respect to the base free blocks.
    base = (off, overlap_early, overlap_late, on)
    result = rank_road_candidates(base, free, probability, known, GRID, **LIMITS)
    assert result.ordered == tuple(v.candidate_id for v in base)
    usual = rank_road_candidates((off, on, overlap_early, overlap_late), free, probability, known, GRID, **LIMITS)
    assert usual.ordered == ("free_onroad", "free_offroad", "overlap_offroad", "overlap_onroad")


@pytest.mark.parametrize("mod", ["road_shape", "mask_shape", "mask_type", "negative", "too_large", "nan", "complex"])
def test_bad_evidence_is_rejected_instead_of_silently_assuming_road(mod):
    probability, known = _evidence()
    if mod == "road_shape":
        probability = probability[:-1]
    elif mod == "mask_shape":
        known = known[:, :-1]
    elif mod == "mask_type":
        known = known.astype(int)
    elif mod == "complex":
        probability = probability.astype(np.complex128)
    else:
        probability[0, 0] = {"negative": -.01, "too_large": 1.01, "nan": np.nan}[mod]
    with pytest.raises(ValueError, match="grid shape|boolean mask|finite numbers"):
        assess_road(_trajectory((0, 0, 0)), probability, known, GRID, **LIMITS)


@pytest.mark.parametrize("thresholds", [(True, .75), (0., np.inf), (-.01, .75), (.75, .75), (.9, .75), (.25, 1.1)])
def test_invalid_confidence_thresholds(thresholds):
    probability, known = _evidence()
    with pytest.raises(ValueError, match="threshold"):
        assess_road(_trajectory((0, 0, 0)), probability, known, GRID,
                    offroad_threshold=thresholds[0], onroad_threshold=thresholds[1])


@pytest.mark.parametrize("g", [replace(GRID, rows=True), replace(GRID, step_m=0), replace(GRID, y_min=np.nan)])
def test_invalid_grid_is_not_reinterpreted_as_outside(g):
    probability, known = _evidence()
    with pytest.raises(ValueError, match="Grid|grid"):
        assess_road(_trajectory((0, 0, 0)), probability, known, g, **LIMITS)


def test_ranking_rejects_noncomparable_time_grids_stale_overlap_data_and_duplicate_names():
    probability, known = _evidence()
    c0 = _candidate("a", (0, 0, 0))
    c1 = Candidate("b", _trajectory((1, 0, 0), interval_s=1.0))
    for candidates, free, message in [
        ((c0, c1), {"a": True, "b": True}, "time grid"),
        ((c0, c0), {"a": True}, "unique"),
        ((c0,), {"b": True}, "IDs"),
        ((c0,), {"a": 1}, "boolean"),
        ((), {}, "at least one"),
    ]:
        with pytest.raises(ValueError, match=message):
            rank_road_candidates(candidates, free, probability, known, GRID, **LIMITS)
