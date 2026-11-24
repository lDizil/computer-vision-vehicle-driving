"""Relative roadmap evidence for offline experiments, never a safety check."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from graphlib import CycleError, TopologicalSorter
from heapq import heapify, heappop, heappush
from itertools import combinations, groupby
import math
from types import MappingProxyType

import numpy as np
from numpy.typing import ArrayLike

from hybrid_drive.contracts._validation import tuple_snapshot
from hybrid_drive.contracts.planning import Candidate
from hybrid_drive.fast.road_candidate_rank import (
    _FOOTPRINT,
    _OFFROAD,
    _UNKNOWN,
    _Road,
    _assess,
    _road,
    RoadAssessment,
    RoadGrid,
)


@dataclass(frozen=True)
class PairwiseRoadComparison:
    """Matched future-pose counts, ordered by the original forecast priority.

    ``comparable_samples`` counts different cell sets with both footprints
    fully observed/confident. Only on-road vs offroad samples are directional;
    the remainder of comparable samples are ties. Opposing votes always abstain.
    """

    left_id: str
    right_id: str
    different_samples: int
    unknown_different_samples: int
    comparable_samples: int
    left_wins: int
    right_wins: int
    winner_id: str | None


@dataclass(frozen=True)
class PairwiseRoadRanking:
    ordered: tuple[str, ...]
    comparisons: tuple[PairwiseRoadComparison, ...]
    assessments: Mapping[str, RoadAssessment]
    initial_unknown: bool
    initial_offroad: bool
    cycle_detected: bool


@dataclass(frozen=True)
class _FootprintSample:
    cells: frozenset[tuple[int, int]]
    unknown: bool
    offroad: bool


def rank_paired_road_candidates(
    candidates: Sequence[Candidate], obstacle_free: Mapping[str, bool],
    drivable_probabilities: ArrayLike, known: ArrayLike, grid: RoadGrid,
    *, offroad_threshold: float, onroad_threshold: float,
) -> PairwiseRoadRanking:
    """Compare road evidence in matched future samples, rather than entire drives.

    Ignore shared footprints and never treat missing or ambiguous cells as
    clear road. Compare only inside each contiguous obstacle-free block. A
    component of directional preferences can reorder its own existing slots,
    not those of unrelated candidates. Any cycle or initial offroad evidence
    leaves the full input order intact. An unknown starting footprint remains
    unknown even when a preference can be established farther along the plan.
    """
    values, ids = _ranking_inputs(candidates, obstacle_free)
    road = _road(drivable_probabilities, known, grid, offroad_threshold, onroad_threshold)
    initial = _footprint_sample(0., 0., 0., road)
    samples = {
        value.candidate_id: tuple(_footprint_sample(*pose, road) for pose in value.trajectory.poses)
        for value in values
    }
    assessments = MappingProxyType({value.candidate_id: _assess(value.trajectory, road) for value in values})
    comparisons = tuple(
        _compare(left, right, samples[left], samples[right])
        for free, block in groupby(ids, key=lambda key: obstacle_free[key]) if free
        for left, right in combinations(block, 2)
    )
    proposed_order = _order_by_evidence(ids, comparisons)
    return PairwiseRoadRanking(
        ordered=ids if initial.offroad or proposed_order is None else proposed_order,
        comparisons=comparisons,
        assessments=assessments,
        initial_unknown=initial.unknown,
        initial_offroad=initial.offroad,
        cycle_detected=proposed_order is None,
    )


def _ranking_inputs(
    candidates: Sequence[Candidate], obstacle_free: Mapping[str, bool],
) -> tuple[tuple[Candidate, ...], tuple[str, ...]]:
    values = tuple_snapshot(candidates, "candidates")
    if not values or any(not isinstance(value, Candidate) for value in values):
        raise ValueError("candidates must contain at least one Candidate")
    ids = tuple(value.candidate_id for value in values)
    if len(set(ids)) != len(ids):
        raise ValueError("candidate IDs must be unique")
    if not isinstance(obstacle_free, Mapping) or set(obstacle_free) != set(ids):
        raise ValueError("obstacle_free must cover exactly the candidate IDs")
    if any(not isinstance(value, (bool, np.bool_)) for value in obstacle_free.values()):
        raise ValueError("obstacle_free values must be boolean")
    first = values[0].trajectory
    if any(
        value.trajectory.num_poses != first.num_poses or value.trajectory.interval_s != first.interval_s
        for value in values[1:]
    ):
        raise ValueError("candidates must share the same time grid")
    return values, ids


def _footprint_sample(x: float, y: float, yaw: float, road: _Road) -> _FootprintSample:
    grid = road.grid
    c, s = math.cos(yaw), math.sin(yaw)
    cells = set()
    unknown = offroad = False
    for u, v in _FOOTPRINT:
        px, py = x + c * u - s * v, y + s * u + c * v
        if not (grid.x_min <= px < road.x_max and grid.y_min <= py < road.y_max):
            unknown = True
            continue
        cell = grid.index(px, py)
        if cell is None:
            unknown = True
            continue
        cells.add(cell)
        state = road.cells[cell]
        unknown |= state == _UNKNOWN
        offroad |= state == _OFFROAD
    return _FootprintSample(frozenset(cells), bool(unknown), bool(offroad))


def _compare(
    left_id: str, right_id: str, left_samples: tuple[_FootprintSample, ...], right_samples: tuple[_FootprintSample, ...],
) -> PairwiseRoadComparison:
    different = unknown = comparable = left_wins = right_wins = 0
    for left, right in zip(left_samples, right_samples, strict=True):
        if left.cells == right.cells:
            continue
        different += 1
        if left.unknown or right.unknown:
            unknown += 1
            continue
        comparable += 1
        left_wins += not left.offroad and right.offroad
        right_wins += not right.offroad and left.offroad

    winner_id = None
    if left_wins and not right_wins:
        winner_id = left_id
    if right_wins and not left_wins:
        winner_id = right_id
    return PairwiseRoadComparison(left_id, right_id, different, unknown, comparable, left_wins, right_wins, winner_id)


def _order_by_evidence(
    ids: tuple[str, ...], comparisons: tuple[PairwiseRoadComparison, ...],
) -> tuple[str, ...] | None:
    original_slots = {key: slot for slot, key in enumerate(ids)}
    predecessors = defaultdict(set)
    neighbors = defaultdict(set)
    for entry in comparisons:
        winner = entry.winner_id
        if winner is None:
            continue
        loser = entry.left_id if winner == entry.right_id else entry.right_id
        before, after = original_slots[winner], original_slots[loser]
        predecessors[after].add(before)
        neighbors[before].add(after)
        neighbors[after].add(before)

    ordered = list(ids)
    seen = set()
    for start in neighbors:
        if start in seen:
            continue
        component = {start}
        pending = [start]
        while pending:
            unseen = neighbors[pending.pop()] - component
            component.update(unseen)
            pending.extend(unseen)
        seen.update(component)

        slots = sorted(component)
        topology = TopologicalSorter({slot: predecessors[slot] for slot in slots})
        try:
            topology.prepare()
        except CycleError:
            return None

        ready = list(topology.get_ready())
        heapify(ready)
        for slot in slots:
            source = heappop(ready)
            ordered[slot] = ids[source]
            topology.done(source)
            for newly_ready in topology.get_ready():
                heappush(ready, newly_ready)
    return tuple(ordered)
