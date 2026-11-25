"""Conservative two-observation association in the current ego frame."""

from dataclasses import dataclass
from math import dist, isfinite

import numpy as np


MODELS = ("geometry", "appearance")
_MIN_AGE_S, _MAX_AGE_S = .4, .6
_MAX_SPEED_MPS = {"person": 4.5, "bicycle": 10.}
_DISTANCE_ALLOWANCE_M = .55
_AMBIGUITY_GAP_M = .24
_MAX_VIEW_DISAGREEMENT_M = 1.25
_APPEARANCE_COST_M = .9
_MAX_APPEARANCE_L1 = 1.10


@dataclass(frozen=True)
class ObjectMotion:
    status: str
    previous_object_id: str | None = None
    velocity_xy_mps: tuple[float, float] = (0., 0.)
    radial_unit_xy: tuple[float, float] | None = None
    common_cameras: tuple[str, ...] = ()
    past_xy_m: tuple[float, float] | None = None


@dataclass(frozen=True)
class _Link:
    current: int
    previous: int
    cost_m: float
    motion: ObjectMotion


def associate_objects(current, previous_in_current, history_s, *, mode):
    """Return a motion or explicit abstention for every current object, in order.

    Cross-view fusion defines current positions. Speed is based on consistent
    per-camera XY changes, never on a difference between disjoint camera sets.
    Two equally plausible historical detections do not produce a track.
    """
    if mode not in MODELS:
        raise ValueError("Unknown VRU association mode")
    if not isfinite(history_s) or not _MIN_AGE_S <= history_s <= _MAX_AGE_S:
        raise ValueError("VRU tracks require an observed lookback between 0.4 and 0.6 seconds")
    for rows in (current, previous_in_current):
        ids = {o.object_id for o in rows}
        if len(ids) != len(rows) or any(o.kind not in _MAX_SPEED_MPS for o in rows):
            raise ValueError("Expected unique projected objects and supported classes")
    by_current = [[] for _ in current]
    by_past = [[] for _ in previous_in_current]
    for i, a in enumerate(current):
        for j, b in enumerate(previous_in_current):
            if a.kind != b.kind:
                continue
            candidate = _possible_link(a, b, i, j, history_s, mode)
            if candidate is not None:
                by_current[i].append(candidate)
                by_past[j].append(candidate)
    for group in (*by_current, *by_past):
        group.sort(key=lambda row: (row.cost_m, row.current, row.previous))
    results = []
    for candidates in by_current:
        if not candidates:
            results.append(ObjectMotion("no_reliable_history"))
            continue
        best = candidates[0]
        inverse = by_past[best.previous]
        if _ambiguous(candidates) or _ambiguous(inverse):
            results.append(ObjectMotion("ambiguous_history"))
        elif inverse[0] is not best:
            results.append(ObjectMotion("nonmutual_history"))
        else:
            results.append(best.motion)
    return tuple(results)


def _possible_link(a, b, i, j, age, mode):
    cap = _MAX_SPEED_MPS[a.kind]
    gap = dist(a.xy, b.xy)
    if gap > cap * age + _DISTANCE_ALLOWANCE_M:
        return None
    current = {p.camera_name: p for p in a.hits}
    previous = {p.camera_name: p for p in b.hits}
    common = sorted(current.keys() & previous.keys())
    if not common:
        return None
    offsets, weights, appearance = [], [], []
    for key in common:
        point, old = current[key], previous[key]
        offsets.append(np.subtract(point.xy, old.xy))
        quality = min(point.confidence, old.confidence)
        uncertainty = max(.15, point.uncertainty_m + old.uncertainty_m)
        weights.append(max(quality, .01) / uncertainty**2)
        appearance.append(float(np.abs(np.subtract(point.appearance, old.appearance)).sum()))
    if any(dist(left, right) > _MAX_VIEW_DISAGREEMENT_M for idx, left in enumerate(offsets) for right in offsets[:idx]):
        return None
    median_l1 = float(np.median(appearance))
    if mode == "appearance" and median_l1 > _MAX_APPEARANCE_L1:
        return None
    velocity = np.average(offsets, weights=weights, axis=0) / age
    if not np.isfinite(velocity).all() or np.linalg.norm(velocity) > cap:
        return None
    primary = common[int(np.argmax(weights))]
    point = current[primary]
    direction = np.subtract(point.xy, point.camera_xyz[:2])
    distance = np.linalg.norm(direction)
    if distance < 1e-6:
        return None
    motion = ObjectMotion("estimated", b.object_id, tuple(map(float, velocity)), tuple(map(float, direction / distance)),
                          tuple(common), b.xy)
    cost = gap + (median_l1 * _APPEARANCE_COST_M if mode == "appearance" else 0.)
    return _Link(i, j, cost, motion)


def _ambiguous(options):
    return len(options) > 1 and options[1].cost_m - options[0].cost_m < _AMBIGUITY_GAP_M
