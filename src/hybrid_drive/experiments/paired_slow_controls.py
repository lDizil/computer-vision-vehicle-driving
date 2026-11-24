"""Conditional actor crossings on an unchanged camera capture and Fast plan."""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import combinations

import numpy as np
from numpy.typing import NDArray

from hybrid_drive.contracts.planning import Candidate
from hybrid_drive.fast.forecast_proposals import ForecastObject
from hybrid_drive.fast.scene_payload import FastScene


_SAMPLE_S = .5
_N_POSES = 5
_SUBSTEPS = 10
_TIMES = np.linspace(0, _N_POSES * _SAMPLE_S, _N_POSES * _SUBSTEPS + 1)
_EGO_HALF_LENGTH, _EGO_HALF_WIDTH, _EGO_CENTER_X = 2.25, .95, 1.25
_MIN_PATH_SPREAD_M = 3.8
_HIT_DEPTH_M, _MISS_GAP_M, _START_GAP_M = .4, .6, 1.
_MAX_CONTROLS, _TRIES_PER_STRATUM = 32, 12
_MAX_VX_MPS = 11.
_MIN_SIGN_SPEED_MPS = .2


@dataclass(frozen=True)
class CrossingControl:
    """Each world blocks exactly one of these same two candidate IDs.

    Gaps index [world][candidate] and report the minimum signed SAT gap at
    discrete times. They are labels, not model inputs or distances to impact.
    """

    worlds: tuple[FastScene, FastScene]
    candidate_ids: tuple[str, str]
    blocked: tuple[str, str]
    gaps_m: tuple[tuple[float, float], tuple[float, float]]


@dataclass(frozen=True, eq=False)
class _EgoPath:
    candidate_id: str
    centers: NDArray[np.float64]
    axes: NDArray[np.float64]


def crossing_controls(scene: FastScene, *, seed: int, count: int) -> tuple[CrossingControl, ...]:
    """Generate paired velocity interventions; return fewer if no robust swap exists.

    The two worlds preserve the input snapshot, plan, rank and road. Replacing
    the box list by one hypothetical actor isolates the intervention, not a
    statement about the actual photographed scene. The label concerns only
    sampled planar overlap under a constant-velocity/constant-yaw hypothesis.

    Target half the groups with equal longitudinal velocity sign in the two
    worlds. Keep feasible alternatives if one side or sign lacks support.
    """
    if not isinstance(scene, FastScene):
        raise ValueError("Expected a FastScene for crossing controls")
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("seed must be an unsigned 64-bit integer")
    if type(count) is not int or not 1 <= count <= _MAX_CONTROLS:
        raise ValueError(f"count must be between 1 and {_MAX_CONTROLS}")

    pair = _select_paths(scene.plan.candidates)
    if pair is None:
        return ()
    rng = np.random.default_rng(seed)
    same_sign = ((1, 1), (-1, -1), (-1, 1), (1, -1))
    opposite_sign = ((0, 1), (0, -1))
    sign_flip, side_flip = rng.choice((-1, 1), size=2)
    result = []
    for i in range(count):
        primary, fallback = (opposite_sign, same_sign) if i % 2 else (same_sign, opposite_sign)
        for strata in (primary, fallback):
            shift = (i // 2) % len(strata)
            order = strata[shift:] + strata[:shift]
            adjusted = tuple((direction * sign_flip, side * side_flip) for direction, side in order)
            control = _draw_control(scene, pair, rng, adjusted)
            if control is not None:
                result.append(control)
                break
    return tuple(result)


def _draw_control(scene, pair, rng, strata):
    first, second, eligible = pair
    ids = first.candidate_id, second.candidate_id
    for direction, side in strata:
        for _ in range(_TRIES_PER_STRATUM):
            sample = int(rng.choice(eligible))
            actors = _actor_variants(
                rng, first.centers[sample], second.centers[sample], _TIMES[sample], direction, side,
            )
            if actors is None:
                continue
            margins = tuple(tuple(_axis_gaps(p, actor) for p in (first, second)) for actor in actors)
            if any(gaps[0] < _START_GAP_M for row in margins for gaps in row):
                continue
            if not all(
                row[i][sample] <= -_HIT_DEPTH_M and row[1 - i].min() >= _MISS_GAP_M
                for i, row in enumerate(margins)
            ):
                continue
            # Arm position itself must not carry the hit label.
            order = (1, 0) if rng.integers(2) else (0, 1)
            a, b = order
            min_gaps = tuple(tuple(float(values.min()) for values in margins[i]) for i in order)
            return CrossingControl(
                (replace(scene, boxes=(actors[a],)), replace(scene, boxes=(actors[b],))),
                ids, (ids[a], ids[b]), min_gaps,
            )
    return None


def _select_paths(candidates: tuple[Candidate, ...]) -> tuple[_EgoPath, _EgoPath, NDArray[np.int64]] | None:
    if not 2 <= len(candidates) <= 8:
        return None
    sampling = candidates[0].trajectory
    if sampling.num_poses != _N_POSES or sampling.interval_s != _SAMPLE_S:
        return None
    if any((np.abs(c.trajectory.poses[:, :2]) > 160.).any() for c in candidates):
        return None
    # Keep chosen comparisons independent of current rank, including ties.
    paths = tuple(_sample_path(candidate) for candidate in sorted(candidates, key=lambda c: c.candidate_id))
    anchor_indices = np.arange(2, _N_POSES + 1) * _SUBSTEPS
    best, best_spread = None, 0.
    for a, b in combinations(paths, 2):
        spread = np.linalg.norm(a.centers[anchor_indices] - b.centers[anchor_indices], axis=1)
        eligible = anchor_indices[spread >= _MIN_PATH_SPREAD_M]
        if not len(eligible):
            continue
        if (a.candidate_id, b.candidate_id) == ("accelerate", "brake"):
            return a, b, eligible
        spread_max = float(spread.max())
        if spread_max > best_spread:
            best, best_spread = (a, b, eligible), spread_max
    return best


def _sample_path(candidate: Candidate) -> _EgoPath:
    trajectory = candidate.trajectory
    times = np.concatenate(([0.], trajectory.times_s))
    source = np.vstack((np.zeros((1, 3)), trajectory.poses))
    yaw = np.interp(_TIMES, times, np.unwrap(source[:, 2]))
    axes = _axes(yaw)
    rear_axle = np.stack([np.interp(_TIMES, times, source[:, i]) for i in range(2)], axis=-1)
    return _EgoPath(candidate.candidate_id, rear_axle + _EGO_CENTER_X * axes[:, 0], axes)


def _actor_variants(rng, a, b, time_s, direction, side) -> tuple[ForecastObject, ForecastObject] | None:
    midpoint = .5 * (a + b)
    offset = rng.uniform(-.3, .3, size=2)
    x_near, x_far = sorted((a[0] + offset[0], b[0] + offset[0]))
    max_run = _MAX_VX_MPS * time_s
    x_min, x_max = max(2.5, x_far - max_run), min(48., x_near + max_run)
    sign_margin = _MIN_SIGN_SPEED_MPS * time_s
    if direction > 0:
        x_max = min(x_max, x_near - sign_margin)
    elif direction < 0:
        x_min = max(x_min, x_far + sign_margin)
    else:
        x_min, x_max = max(x_min, x_near + sign_margin), min(x_max, x_far - sign_margin)
    if x_max <= x_min:
        return None
    start = np.array((rng.uniform(x_min, x_max), midpoint[1] + side * rng.uniform(3.3, 7.6) * time_s))
    size = (rng.uniform(1.25, 2.2), rng.uniform(2.35, 4.5), rng.uniform(1.25, 2.1))
    yaw = -side * rng.uniform(1.05, 2.1)
    center = (*start, rng.uniform(.7, 1.4))
    velocities = tuple((target + offset - start) / time_s for target in (a, b))
    if any(abs(v[0]) > _MAX_VX_MPS or abs(v[1]) > 8. or np.linalg.norm(v) > 13. for v in velocities):
        return None
    first = ForecastObject("counterfactual-crossing", center, size, yaw, tuple(velocities[0]), .9)
    return first, replace(first, velocity_xy=tuple(velocities[1]))


def _axes(yaw):
    cosine, sine = np.cos(yaw), np.sin(yaw)
    return np.stack((np.stack((cosine, sine), axis=-1), np.stack((-sine, cosine), axis=-1)), axis=-2)


def _axis_gaps(path: _EgoPath, actor: ForecastObject) -> NDArray[np.float64]:
    # Project onto both rectangles' four axes, taking the largest signed
    # separation at each time. This evaluator never uses Fast's ranking.
    actor_axes = _axes(actor.yaw)
    axes = np.concatenate((path.axes, np.broadcast_to(actor_axes, path.axes.shape)), axis=1)
    ego_extent = np.einsum("tai,tbi->tab", axes, path.axes)
    ego_radius = np.abs(ego_extent) @ (_EGO_HALF_LENGTH, _EGO_HALF_WIDTH)
    actor_radius = np.abs(axes @ actor_axes.T) @ (actor.size_wlh[1] / 2., actor.size_wlh[0] / 2.)
    positions = np.asarray(actor.center_xyz[:2]) + _TIMES[:, None] * actor.velocity_xy
    separation = np.abs(np.einsum("tai,ti->ta", axes, positions - path.centers))
    return (separation - ego_radius - actor_radius).max(axis=1)
