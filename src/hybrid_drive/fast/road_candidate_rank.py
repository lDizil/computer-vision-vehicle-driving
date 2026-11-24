"""Offline road evidence at sampled poses, not a driving-safety check."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import groupby, product
import math
from numbers import Integral
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from hybrid_drive.contracts._validation import finite_real, tuple_snapshot
from hybrid_drive.contracts.planning import Candidate
from hybrid_drive.contracts.trajectory import Trajectory
from hybrid_drive.fast.forecast_proposals import _EGO_FRONT_M, _EGO_REAR_M, _EGO_WIDTH_M


class RoadGrid(Protocol):
    """Uniform ego metric grid: rows are forward x, columns are left y."""

    @property
    def x_min(self) -> float: ...

    @property
    def y_min(self) -> float: ...

    @property
    def step_m(self) -> float: ...

    @property
    def rows(self) -> int: ...

    @property
    def cols(self) -> int: ...

    def index(self, x: float, y: float) -> tuple[int, int] | None: ...


@dataclass(frozen=True)
class RoadAssessment:
    """Pose counts including t=0: a footprint can contain both offroad and unknown points."""

    sample_count: int
    offroad_samples: int
    unknown_samples: int
    first_offroad_s: float | None

    @property
    def fully_supported(self) -> bool:
        return self.unknown_samples == 0


@dataclass(frozen=True)
class RoadRanking:
    ordered: tuple[str, ...]
    assessments: Mapping[str, RoadAssessment]


def assess_road(
    trajectory: Trajectory, drivable_probabilities: ArrayLike, known: ArrayLike, grid: RoadGrid,
    *, offroad_threshold: float, onroad_threshold: float,
) -> RoadAssessment:
    """Sample nine points on each nominal ego rectangle, including corners.

    At a known cell, p <= offroad_threshold is offroad evidence and
    p >= onroad_threshold is onroad evidence. Between thresholds, outside
    support or outside the grid means unknown. Neither the gaps between
    points or poses nor actual road compliance are checked.
    """
    if not isinstance(trajectory, Trajectory):
        raise ValueError("trajectory must be a Trajectory")
    road = _road(drivable_probabilities, known, grid, offroad_threshold, onroad_threshold)
    return _assess(trajectory, road)


def rank_road_candidates(
    candidates: Sequence[Candidate], obstacle_free: Mapping[str, bool],
    drivable_probabilities: ArrayLike, known: ArrayLike, grid: RoadGrid,
    *, offroad_threshold: float, onroad_threshold: float,
) -> RoadRanking:
    """Refine an existing forecast order without crossing an overlap slot.

    Only fully-supported candidates in a contiguous obstacle-free block can
    exchange places. Prefer fewer sampled offroad poses; ties, unknown slots
    and affected candidates retain base priority. No candidate is removed.
    """
    values = tuple_snapshot(candidates, "candidates")
    if not values or any(not isinstance(candidate, Candidate) for candidate in values):
        raise ValueError("candidates must contain at least one Candidate")
    ids = tuple(candidate.candidate_id for candidate in values)
    if len(set(ids)) != len(ids):
        raise ValueError("candidate IDs must be unique")
    if not isinstance(obstacle_free, Mapping) or set(obstacle_free) != set(ids):
        raise ValueError("obstacle_free must cover exactly the candidate IDs")
    if any(not isinstance(value, (bool, np.bool_)) for value in obstacle_free.values()):
        raise ValueError("obstacle_free values must be boolean")
    first = values[0].trajectory
    if any(
        item.trajectory.num_poses != first.num_poses or item.trajectory.interval_s != first.interval_s
        for item in values[1:]
    ):
        raise ValueError("candidates must share the same time grid")

    road = _road(drivable_probabilities, known, grid, offroad_threshold, onroad_threshold)
    evidence = {item.candidate_id: _assess(item.trajectory, road) for item in values}
    ordered = list(ids)
    for free, group in groupby(enumerate(ids), key=lambda entry: obstacle_free[entry[1]]):
        if not free:
            continue
        supported = [(i, key) for i, key in group if evidence[key].fully_supported]
        by_road = sorted(supported, key=lambda entry: evidence[entry[1]].offroad_samples)
        for (position, _), (_, key) in zip(supported, by_road):
            ordered[position] = key
    return RoadRanking(tuple(ordered), evidence)


_UNKNOWN, _OFFROAD, _ONROAD = 0, 1, 2
# Use the same nominal diagnostic rectangle as the forecast overlap checker.
_FOOTPRINT = tuple(product(
    (-_EGO_REAR_M, (_EGO_FRONT_M - _EGO_REAR_M) / 2, _EGO_FRONT_M),
    (-_EGO_WIDTH_M / 2, 0.0, _EGO_WIDTH_M / 2),
))


@dataclass(frozen=True)
class _Road:
    grid: RoadGrid
    cells: NDArray[np.int8]
    x_max: float
    y_max: float

    def state(self, x: float, y: float) -> int:
        # Reject out-of-grid positions before converting large finite poses to indexes.
        if not (self.grid.x_min <= x < self.x_max and self.grid.y_min <= y < self.y_max):
            return _UNKNOWN
        cell = self.grid.index(x, y)
        return self.cells[cell] if cell is not None else _UNKNOWN


def _road(probabilities, known, grid, offroad_threshold, onroad_threshold) -> _Road:
    off = finite_real(offroad_threshold, "offroad_threshold")
    on = finite_real(onroad_threshold, "onroad_threshold")
    if not 0 <= off < on <= 1:
        raise ValueError("Road thresholds must satisfy 0 <= offroad_threshold < onroad_threshold <= 1")

    shape = (grid.rows, grid.cols)
    if any(isinstance(n, (bool, np.bool_)) or not isinstance(n, Integral) or n <= 0 for n in shape):
        raise ValueError("Grid rows and columns must be positive integers")
    probability = np.asarray(probabilities)
    observed = np.asarray(known)
    if probability.shape != shape or observed.shape != shape:
        raise ValueError("Road probabilities and known mask must match the grid shape")
    if observed.dtype != np.bool_:
        raise ValueError("known must be a boolean mask")
    if (probability.dtype.kind not in "bfiu" or not np.all(np.isfinite(probability))
            or not np.all((0 <= probability) & (probability <= 1))):
        raise ValueError("Road probabilities must be finite numbers in [0, 1]")

    step_m = finite_real(grid.step_m, "grid step")
    x_min = finite_real(grid.x_min, "grid x_min")
    y_min = finite_real(grid.y_min, "grid y_min")
    x_max = x_min + grid.rows * step_m
    y_max = y_min + grid.cols * step_m
    if step_m <= 0 or not (x_min < x_max < math.inf and y_min < y_max < math.inf):
        raise ValueError("Grid must have finite increasing bounds and a positive step")

    cells = np.zeros(shape, dtype=np.int8)
    cells[observed & (probability <= off)] = _OFFROAD
    cells[observed & (probability >= on)] = _ONROAD
    return _Road(grid, cells, x_max, y_max)


def _assess(trajectory: Trajectory, road: _Road) -> RoadAssessment:
    offroad = unknown = 0
    first_offroad_s = None
    for i, (x, y, yaw) in enumerate(np.vstack(([0., 0., 0.], trajectory.poses))):
        c, s = math.cos(yaw), math.sin(yaw)
        sample = {road.state(x + c * u - s * v, y + s * u + c * v) for u, v in _FOOTPRINT}
        unknown += _UNKNOWN in sample
        if _OFFROAD in sample:
            offroad += 1
            if first_offroad_s is None:
                first_offroad_s = i * trajectory.interval_s
    return RoadAssessment(trajectory.num_poses + 1, offroad, unknown, first_offroad_s)
