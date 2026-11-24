"""Ground the Fast snapshot in current camera features and typed metric priors."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math
from typing import Literal

import numpy as np
import torch
from torch import Tensor, nn

from hybrid_drive.contracts.sensors import CameraSnapshot, SURROUND_ORDER
from hybrid_drive.fast.forecast_proposals import ForecastObject
from hybrid_drive.fast.scene_payload import FastScene
from hybrid_drive.geometry.camera import CameraCalibration
from hybrid_drive.neural.prior_tokenizer import PriorTokenizer
from hybrid_drive.neural.surround_features import SurroundFeatures


_MAX_ACTORS, _MAX_CANDIDATES, _MAX_POSES = 32, 8, 12
_ACTOR_MOTION_DT_S = 0.5
_ROAD_PROBES = (("center", 0.0), ("left", 0.95), ("right", -0.95))
_ACTOR_FEATURES = (
    "x_64m", "y_64m", "z_8m", "width_8m", "length_8m", "height_8m", "yaw_sin", "yaw_cos",
    "vx_20mps", "vy_20mps", "cv_dx_8m_after_0.5s", "cv_dy_8m_after_0.5s", "confidence",
    *(f"view_{direction.value}" for direction in SURROUND_ORDER),
)
_CORNER_SIGNS = tuple(product((-0.5, 0.5), repeat=3))
_CUBOID_EDGES = tuple((i, i ^ bit) for i in range(8) for bit in (1, 2, 4) if i < (i ^ bit))

NormalizedBox = tuple[float, float, float, float]


@dataclass(frozen=True)
class PriorRecord:
    """Provenance for one prior row, including rows masked for missing evidence."""

    kind: Literal["actor", "road", "trajectory"]
    valid: bool
    sample_time_s: float
    actor_id: str | None = None
    candidate_id: str | None = None
    sample_index: int | None = None
    camera_name: str | None = None
    camera_direction: str | None = None
    roi_xyxy: NormalizedBox | None = None
    road_probe: str | None = None
    cell_rc: tuple[int, int] | None = None


@dataclass(frozen=True, eq=False)
class BridgeTokens:
    prior_tokens: Tensor
    prior_mask: Tensor
    visual_tokens: Tensor
    visual_mask: Tensor
    records: tuple[PriorRecord, ...]
    scene_sha256: str
    snapshot_sha256: str


class SceneBridge(nn.Module):
    """One snapshot per call; parameters are trainable, not imported from FSDrive.

    Actor priors use a *current* calibrated ROI and a numeric 0.5 s constant-
    velocity hypothesis (no future RGB or occlusion estimate). Road priors
    sample the candidate rear axle and two lateral points, not the full vehicle
    footprint. Trajectory priors use exact Fast samples and preserve candidate
    rank; the bridge never resamples or rearranges the plan. Oversized scenes
    are rejected rather than silently dropping actors, candidates or timestamps.
    """

    def __init__(self, visual_dim: int = 384, prior_dim: int = 64):
        super().__init__()
        self.visual_dim = visual_dim
        self.prior_dim = prior_dim
        self.actor_tokenizer = PriorTokenizer(
            visual_dim, geometry_dim=len(_ACTOR_FEATURES), token_dim=prior_dim,
            hidden_dim=prior_dim, geometry_feature_names=_ACTOR_FEATURES,
        )
        self.trajectory_embedding = _numeric_embedding(6, prior_dim)
        self.road_embedding = _numeric_embedding(8, prior_dim)
        self.type_embeddings = nn.Parameter(torch.empty(3, prior_dim))
        nn.init.normal_(self.type_embeddings, std=0.02)

    def forward(self, features: SurroundFeatures, scene: FastScene) -> BridgeTokens:
        maps = self._validate_context(features, scene)
        actors, actor_records = self._actor_priors(features, scene)
        poses, pose_records, road, road_records = _candidate_inputs(scene)
        pose_tokens = self.trajectory_embedding(_finite_tensor(maps, poses)) + self.type_embeddings[1]
        road_tokens = self.road_embedding(_finite_tensor(maps, road)) + self.type_embeddings[2]
        records = (*actor_records, *pose_records, *road_records)
        prior_mask = torch.tensor([[record.valid for record in records]], dtype=torch.bool, device=maps.device)
        priors = torch.cat((actors + self.type_embeddings[0], pose_tokens, road_tokens), dim=0)[None]
        priors = priors.masked_fill(~prior_mask[..., None], 0)
        visual = maps.mean(dim=(-2, -1))
        if not torch.isfinite(priors).all() or not torch.isfinite(visual).all():
            raise ValueError("Bridge token output must be finite")
        return BridgeTokens(
            priors, prior_mask, visual, torch.ones(visual.shape[:2], dtype=torch.bool, device=maps.device),
            records, scene.content_sha256, scene.snapshot_sha256,
        )

    def _validate_context(self, features: SurroundFeatures, scene: FastScene) -> Tensor:
        if not isinstance(features, SurroundFeatures) or not isinstance(scene, FastScene):
            raise ValueError("SceneBridge requires SurroundFeatures and a FastScene")
        snap = features.snapshot
        if not isinstance(snap, CameraSnapshot) or (
            snap.snapshot_id, snap.content_sha256, snap.ref_time_us
        ) != (scene.snapshot_id, scene.snapshot_sha256, scene.ref_time_us):
            raise ValueError("FastScene does not match the surround feature snapshot ID, digest and reference time")
        world = snap.ego_to_world
        pose = scene.plan.reference_pose
        turn = math.remainder(pose.heading - math.atan2(world[1, 0], world[0, 0]), math.tau)
        if max(abs(pose.x - world[0, 3]), abs(pose.y - world[1, 3]), abs(turn)) > 1e-9:
            raise ValueError("FastScene plan uses a different ego reference pose")
        maps = features.feature_maps
        if (not isinstance(maps, Tensor) or maps.ndim != 5
                or maps.shape[:3] != (1, len(SURROUND_ORDER), self.visual_dim) or min(maps.shape[3:]) <= 0):
            raise ValueError(f"Expected shared surround feature maps [1,6,{self.visual_dim},H,W]")
        if (maps.dtype not in (torch.float32, torch.float64) or maps.dtype != self.type_embeddings.dtype
                or maps.device != self.type_embeddings.device or not torch.isfinite(maps).all()):
            raise ValueError("Surround maps must be finite float32/float64 on the bridge dtype and device")
        if not isinstance(features.calibrations, tuple) or len(features.calibrations) != len(snap.views) or any(
            not _same_resized_camera(calibration, view.calibration)
            for calibration, view in zip(features.calibrations, snap.views)
        ):
            raise ValueError("Feature-map calibrations must be resized cameras in snapshot order")
        if (len(scene.boxes) > _MAX_ACTORS or len(scene.plan.candidates) > _MAX_CANDIDATES
                or scene.plan.candidates[0].trajectory.num_poses > _MAX_POSES):
            raise ValueError(f"Bridge supports at most {_MAX_ACTORS} actors, {_MAX_CANDIDATES} candidates "
                             f"and {_MAX_POSES} poses per candidate")
        return maps

    def _actor_priors(self, features: SurroundFeatures, scene: FastScene) -> tuple[Tensor, tuple[PriorRecord, ...]]:
        maps = features.feature_maps
        records, rois, image_indices, geometry = [], [], [], []
        for actor in scene.boxes:
            selection = _select_camera(actor, features.calibrations)
            if selection is None:
                records.append(PriorRecord("actor", False, 0.0, actor_id=actor.object_id))
                rois.append((0., 0., 0., 0.))
                image_indices.append(0)
                geometry.append((0.,) * len(_ACTOR_FEATURES))
                continue
            index, roi = selection
            view = features.snapshot.views[index]
            records.append(PriorRecord(
                "actor", True, 0.0, actor_id=actor.object_id, camera_name=view.calibration.name,
                camera_direction=view.direction.value, roi_xyxy=roi,
            ))
            rois.append(roi)
            image_indices.append(index)
            geometry.append(_actor_geometry(actor, index))
        if not any(record.valid for record in records):
            return maps.new_zeros((len(records), self.prior_dim)), tuple(records)
        tokens = self.actor_tokenizer(
            maps, maps.new_tensor([rois]), torch.tensor([image_indices], dtype=torch.long, device=maps.device),
            maps.new_tensor([geometry]),
            torch.tensor([[record.valid for record in records]], dtype=torch.bool, device=maps.device),
        )
        return tokens.tokens[0], tuple(records)


def _numeric_embedding(input_dim: int, token_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(input_dim, token_dim), nn.GELU(), nn.Linear(token_dim, token_dim))


def _finite_tensor(like: Tensor, rows: list[tuple[float, ...]]) -> Tensor:
    values = like.new_tensor(rows)
    if not torch.isfinite(values).all():
        raise ValueError("Numeric priors must be finite in the bridge dtype")
    return values


def _same_resized_camera(actual: CameraCalibration, native: CameraCalibration) -> bool:
    if not isinstance(actual, CameraCalibration) or actual.name != native.name:
        return False
    expected = native.resized(actual.width, actual.height)
    return (actual.near_m == native.near_m and np.array_equal(actual.ego_to_camera, native.ego_to_camera)
            and np.array_equal(actual.intrinsic, expected.intrinsic))


def _actor_geometry(actor: ForecastObject, view_index: int) -> tuple[float, ...]:
    x, y, z = actor.center_xyz
    width, length, height = actor.size_wlh
    vx, vy = actor.velocity_xy
    return (
        x / 64, y / 64, z / 8, width / 8, length / 8, height / 8, math.sin(actor.yaw), math.cos(actor.yaw),
        vx / 20, vy / 20, vx * _ACTOR_MOTION_DT_S / 8, vy * _ACTOR_MOTION_DT_S / 8, actor.confidence,
        *(float(i == view_index) for i in range(len(SURROUND_ORDER))),
    )


def _candidate_inputs(scene: FastScene):
    road = scene.road
    grid, drivable, dividers, known = road.grid, road.drivable, road.lane_divider, road.known
    pose_rows, pose_records, road_rows, road_records = [], [], [], []
    for rank, candidate in enumerate(scene.plan.candidates):
        trajectory = candidate.trajectory
        rank_value = rank / (_MAX_CANDIDATES - 1)
        for index, ((x, y, heading), sample_time) in enumerate(zip(trajectory.poses, trajectory.times_s)):
            sine, cosine, time_s = math.sin(heading), math.cos(heading), float(sample_time)
            pose_rows.append((x / 64, y / 64, sine, cosine, time_s / 4, rank_value))
            pose_records.append(PriorRecord(
                "trajectory", True, time_s, candidate_id=candidate.candidate_id, sample_index=index,
            ))
            for probe, lateral_offset in _ROAD_PROBES:
                cell = grid.index(x - lateral_offset * sine, y + lateral_offset * cosine)
                valid = cell is not None and bool(known[cell])
                if valid:
                    cell_x, cell_y = grid.center(*cell)
                    row = (cell_x / 64, cell_y / 64, float(drivable[cell]), float(dividers[cell]),
                           time_s / 4, rank_value, lateral_offset, grid.step_m / 8)
                else:
                    row = (0.,) * 8
                road_rows.append(row)
                road_records.append(PriorRecord(
                    "road", valid, time_s, candidate_id=candidate.candidate_id, sample_index=index,
                    road_probe=probe, cell_rc=cell,
                ))
    return pose_rows, pose_records, road_rows, road_records


def _select_camera(actor: ForecastObject, calibrations: tuple[CameraCalibration, ...]):
    width, length, height = actor.size_wlh
    sine, cosine = math.sin(actor.yaw), math.cos(actor.yaw)
    rotation = np.array(((cosine, -sine, 0), (sine, cosine, 0), (0, 0, 1)))
    corners = (np.asarray(_CORNER_SIGNS) * (length, width, height)) @ rotation.T + actor.center_xyz
    selected, largest = None, 0.0
    for index, calibration in enumerate(calibrations):
        projection = _visible_roi(corners, calibration)
        # Pick the largest visible silhouette fraction; ties keep snapshot order.
        if projection is not None and projection[1] > largest:
            roi, largest = projection
            selected = (index, roi)
    return selected


def _visible_roi(corners: np.ndarray, calibration: CameraCalibration) -> tuple[NormalizedBox, float] | None:
    transform = calibration.ego_to_camera
    camera = corners @ transform[:3, :3].T + transform[:3, 3]
    near = calibration.near_m
    visible = camera[:, 2] >= near
    points = list(camera[visible])
    for i, j in _CUBOID_EDGES:
        if visible[i] == visible[j]:
            continue
        start, end = camera[i], camera[j]
        alpha = (near - start[2]) / (end[2] - start[2])
        crossing = (1 - alpha) * start + alpha * end
        crossing[2] = near
        points.append(crossing)
    if not points:
        return None
    points = np.asarray(points)
    intrinsic = calibration.intrinsic
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        pixels = (points[:, :2] / points[:, 2, None]) @ intrinsic[:2, :2].T + intrinsic[:2, 2]
    if not np.isfinite(pixels).all() or np.max(np.abs(pixels)) > 1e12:
        raise ValueError("Actor projection exceeds the finite 1e12-pixel clipping range")
    # Normalized image edges, not pixel centres: 0 and 1 are valid ROI borders.
    normalized = (pixels + 0.5) / (calibration.width, calibration.height)
    polygon = _convex_hull(normalized)
    for axis in (0, 1):
        polygon = _clip_polygon(polygon, axis, 0, keep_greater=True)
        polygon = _clip_polygon(polygon, axis, 1, keep_greater=False)
    if len(polygon) < 3:
        return None
    area = abs(sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(polygon, polygon[1:] + polygon[:1]))) / 2
    xy = np.clip(np.asarray(polygon), 0.0, 1.0)
    lower, upper = xy.min(axis=0), xy.max(axis=0)
    if area <= 0 or np.any(upper - lower < 1e-6):
        return None
    return (float(lower[0]), float(lower[1]), float(upper[0]), float(upper[1])), area


def _convex_hull(points: np.ndarray) -> list[tuple[float, float]]:
    """Silhouette of the near-clipped solid, including a view wholly inside it."""
    ordered = sorted({(float(x), float(y)) for x, y in points})
    if len(ordered) < 3:
        return ordered

    def chain(values):
        result = []
        for point in values:
            while len(result) >= 2:
                ax, ay = result[-2]
                bx, by = result[-1]
                if (bx - ax) * (point[1] - ay) - (by - ay) * (point[0] - ax) > 0:
                    break
                result.pop()
            result.append(point)
        return result

    return chain(ordered)[:-1] + chain(reversed(ordered))[:-1]


def _clip_polygon(polygon, axis: int, boundary: float, *, keep_greater: bool):
    if not polygon:
        return []

    def inside(point):
        return point[axis] >= boundary if keep_greater else point[axis] <= boundary

    clipped = []
    for start, end in zip(polygon[-1:] + polygon[:-1], polygon):
        first_inside, last_inside = inside(start), inside(end)
        if first_inside != last_inside:
            fraction = (boundary - start[axis]) / (end[axis] - start[axis])
            other = (1 - fraction) * start[1 - axis] + fraction * end[1 - axis]
            clipped.append((boundary, other) if axis == 0 else (other, boundary))
        if last_inside:
            clipped.append(end)
    return clipped
