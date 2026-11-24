"""Offline longitudinal proposals ranked on the same ego-frame box forecast.

This block is not Safety: sampled planar box contacts are not certified
collisions or evidence of road compliance. It never drops the trajectories
that touched a forecast box, and an empty forecast does not establish clearance.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
import math

import numpy as np
from numpy.typing import NDArray

from hybrid_drive.contracts._validation import finite_real, identifier, tuple_snapshot
from hybrid_drive.contracts.planning import Candidate, CandidateSet
from hybrid_drive.contracts.sensors import timestamp_us
from hybrid_drive.contracts.trajectory import Pose2D, Trajectory


@dataclass(frozen=True)
class ForecastObject:
    """One object in current ego axes, with a constant XY velocity hypothesis.

    Size is width, length, height. Yaw points along the length axis, from
    forward x toward left y. Confidence is evidence metadata, not a safety gate.
    """

    object_id: str
    center_xyz: tuple[float, float, float]
    size_wlh: tuple[float, float, float]
    yaw: float
    velocity_xy: tuple[float, float]
    confidence: float

    def __post_init__(self):
        identifier(self.object_id, "object_id")
        center = _metric_vector(self.center_xyz, 3, "center_xyz", 10_000.0)
        size = _metric_vector(self.size_wlh, 3, "size_wlh", 100.0)
        if min(size) <= 0:
            raise ValueError("size_wlh must be positive")
        yaw = math.remainder(finite_real(self.yaw, "yaw"), math.tau)
        velocity = _metric_vector(self.velocity_xy, 2, "velocity_xy", 100.0)
        confidence = finite_real(self.confidence, "confidence")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        object.__setattr__(self, "center_xyz", center)
        object.__setattr__(self, "size_wlh", size)
        object.__setattr__(self, "yaw", -math.pi if yaw == math.pi else yaw)
        object.__setattr__(self, "velocity_xy", velocity)
        object.__setattr__(self, "confidence", confidence)


@dataclass(frozen=True)
class ProposalDiagnostic:
    """The minimum signed SAT axis gap over all object/sample pairs.

    The gap is a projection margin, not Euclidean distance. Non-positive
    means touching/overlap at a sample under the footprint proxy. None means
    no forecast objects. Height, road geometry and between-sample effects
    are untested, and the earliest contact may be the initial sample (t=0).
    """

    candidate_id: str
    min_axis_gap_m: float | None
    first_overlap_s: float | None
    overlap_sample_count: int
    overlapping_object_ids: tuple[str, ...]


@dataclass(frozen=True)
class ForecastProposals:
    plan: CandidateSet
    diagnostics: tuple[ProposalDiagnostic, ...]
    forecast_object_count: int
    all_candidates_overlap: bool
    forecast_hash: str


_INTERVAL_S = 0.5
_POSE_COUNT = 5
_SCORE_SUBSTEPS = 4
_MAX_SPEED_MPS = 60.0
_PROFILES = (
    ("maintain", 0.0),
    ("ease_off", -1.2),
    ("brake", -3.0),
    ("ease_forward", 0.35),
    ("accelerate", 1.0),
)
# Diagnostic 4.5 x 1.9 m ego rectangle using the rear axle as the trajectory origin.
_EGO_REAR_M = 1.0
_EGO_FRONT_M = 3.5
_EGO_WIDTH_M = 1.9
_TOUCH_TOL_M = 1e-9
_DUPLICATE_TOL_M = 1e-5


def propose_from_forecast(
    snapshot_id: str,
    ref_time_us: int,
    reference_pose: Pose2D,
    forward_speed_mps: float,
    objects: Sequence[ForecastObject],
) -> ForecastProposals:
    """Make and rank distinct forward-only candidates on a shared 2.5 s grid.

    The speed must come from the past observations. We score rectangular
    footprints from t=0 every 0.125 s, with constant yaw/velocity for objects.
    An untouched profile keeps its nominal priority; touched ones are ordered
    by fewer affected samples, later first contact, then higher minimum gap.
    Empty forecasts retain nominal order without reporting a clearance value.
    """
    snapshot_id = identifier(snapshot_id, "snapshot_id")
    ref_time_us = timestamp_us(ref_time_us, "ref_time_us")
    if not isinstance(reference_pose, Pose2D):
        raise ValueError("reference_pose must be a Pose2D")
    speed = finite_real(forward_speed_mps, "forward_speed_mps")
    if not 0.0 <= speed <= _MAX_SPEED_MPS:
        raise ValueError(f"forward_speed_mps must be in [0, {_MAX_SPEED_MPS:g}]")
    ordered_objects = _ordered_objects(objects)
    forecast_hash = _forecast_hash(snapshot_id, ref_time_us, reference_pose, speed, ordered_objects)

    times = np.arange(_POSE_COUNT * _SCORE_SUBSTEPS + 1) * (_INTERVAL_S / _SCORE_SUBSTEPS)
    footprints = _forecast_footprints(ordered_objects, times) if ordered_objects else None
    sampled_profiles = []
    scored = []
    for priority, (name, acceleration) in enumerate(_PROFILES):
        dense_x = _profile_forward_x(times, speed, acceleration)
        sampled_x = dense_x[_SCORE_SUBSTEPS::_SCORE_SUBSTEPS]
        if any(np.max(np.abs(sampled_x - earlier)) <= _DUPLICATE_TOL_M for earlier in sampled_profiles):
            continue
        sampled_profiles.append(sampled_x)
        poses = np.zeros((_POSE_COUNT, 3), dtype=np.float64)
        poses[:, 0] = sampled_x
        candidate = Candidate(name, Trajectory(poses, _INTERVAL_S))
        diagnostic = _diagnose(name, dense_x, times, footprints, ordered_objects)
        scored.append((_preference(priority, diagnostic), candidate, diagnostic))

    scored.sort(key=lambda entry: entry[0])
    diagnostics = tuple(entry[2] for entry in scored)
    plan = CandidateSet(
        plan_id=f"metric-proposals:{forecast_hash}",
        snapshot_id=snapshot_id,
        t_ref_s=ref_time_us / 1_000_000,
        reference_pose=reference_pose,
        candidates=tuple(entry[1] for entry in scored),
    )
    return ForecastProposals(
        plan=plan,
        diagnostics=diagnostics,
        forecast_object_count=len(ordered_objects),
        all_candidates_overlap=all(item.overlap_sample_count > 0 for item in diagnostics),
        forecast_hash=forecast_hash,
    )


def _ordered_objects(objects):
    values = tuple_snapshot(objects, "objects")
    if any(not isinstance(value, ForecastObject) for value in values):
        raise ValueError("objects must contain only ForecastObject values")
    if len({obj.object_id for obj in values}) != len(values):
        raise ValueError("forecast object IDs must be unique")
    return tuple(sorted(values, key=lambda obj: obj.object_id))


def _forecast_hash(snapshot_id, ref_time_us, reference_pose, speed, objects):
    payload = {
        "schema": "metric-proposals-v1",
        "snapshot_id": snapshot_id,
        "ref_time_us": ref_time_us,
        "reference_pose": asdict(reference_pose),
        "forward_speed_mps": speed,
        "objects": [asdict(obj) for obj in objects],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _profile_forward_x(times: NDArray[np.float64], speed: float, acceleration: float) -> NDArray[np.float64]:
    if acceleration == 0.0:
        return times * speed
    end_speed = 0.0 if acceleration < 0.0 else _MAX_SPEED_MPS
    phase = np.minimum(times, (end_speed - speed) / acceleration)
    return speed * phase + 0.5 * acceleration * phase**2 + end_speed * (times - phase)


def _forecast_footprints(objects, times):
    centers = np.array([obj.center_xyz[:2] for obj in objects], dtype=np.float64)
    velocities = np.array([obj.velocity_xy for obj in objects], dtype=np.float64)
    positions = centers[:, None, :] + velocities[:, None, :] * times[None, :, None]
    yaw = np.array([obj.yaw for obj in objects])
    along = np.column_stack((np.cos(yaw), np.sin(yaw)))
    across = np.column_stack((-along[:, 1], along[:, 0]))
    ego_axes = np.broadcast_to(np.eye(2), (len(objects), 2, 2))
    axes = np.concatenate((ego_axes, along[:, None, :], across[:, None, :]), axis=1)
    ego_half_extent = np.array([(_EGO_FRONT_M + _EGO_REAR_M) / 2, _EGO_WIDTH_M / 2])
    width, length = np.array([obj.size_wlh[:2] for obj in objects]).T
    supports = np.abs(axes) @ ego_half_extent
    supports += (
        np.abs(np.einsum("nkd,nd->nk", axes, along)) * length[:, None]
        + np.abs(np.einsum("nkd,nd->nk", axes, across)) * width[:, None]
    ) / 2
    return positions, axes, supports


def _diagnose(name, dense_x, times, footprints, objects) -> ProposalDiagnostic:
    if footprints is None:
        return ProposalDiagnostic(name, None, None, 0, ())
    positions, axes, supports = footprints
    centers = np.column_stack((dense_x + (_EGO_FRONT_M - _EGO_REAR_M) / 2, np.zeros_like(dense_x)))
    offsets = positions - centers[None, :, :]
    gap = (np.abs(np.einsum("ntd,nkd->ntk", offsets, axes)) - supports[:, None, :]).max(axis=2)
    gap[np.abs(gap) <= _TOUCH_TOL_M] = 0.0
    touches = gap <= 0.0
    samples = np.flatnonzero(touches.any(axis=0))
    return ProposalDiagnostic(
        candidate_id=name,
        min_axis_gap_m=float(gap.min()),
        first_overlap_s=float(times[samples[0]]) if len(samples) else None,
        overlap_sample_count=len(samples),
        overlapping_object_ids=tuple(obj.object_id for obj, touched in zip(objects, touches.any(axis=1)) if touched),
    )


def _preference(priority: int, diagnostic: ProposalDiagnostic) -> tuple:
    if diagnostic.first_overlap_s is None:
        return (0, 0, 0, 0, priority)
    return (
        1, diagnostic.overlap_sample_count, -diagnostic.first_overlap_s, -diagnostic.min_axis_gap_m, priority,
    )


def _metric_vector(value, count: int, name: str, max_abs: float) -> tuple[float, ...]:
    values = tuple_snapshot(value, name)
    if len(values) != count:
        raise ValueError(f"{name} must contain exactly {count} coordinates")
    parsed = tuple(finite_real(x, name) for x in values)
    if max(map(abs, parsed)) > max_abs:
        raise ValueError(f"{name} must be bounded by {max_abs:g} in magnitude")
    return parsed
