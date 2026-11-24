"""Calibrated, past-only lifting of shared patch maps into a metric ego grid."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np
import torch
from torch import Tensor
from torch.nn.functional import grid_sample

from hybrid_drive.contracts.sensors import CameraSnapshot
from hybrid_drive.geometry.rigid import invert_pose
from hybrid_drive.neural.surround_features import SurroundFeatures


@dataclass(frozen=True)
class BevGrid:
    """Rows are forward x, columns are leftward y. Cell centers lie inside the bounds."""

    x_min: float = -16.0
    y_min: float = -32.0
    step_m: float = 2.0
    rows: int = 32
    cols: int = 32
    heights_m: tuple[float, ...] = (0.35, 0.85, 1.35)

    def __post_init__(self):
        if (not all(isfinite(v) for v in (self.x_min, self.y_min, self.step_m))
                or not 0.1 <= self.step_m <= 8 or not -250 <= self.x_min <= 250 or not -250 <= self.y_min <= 250):
            raise ValueError("Invalid BEV origin or resolution")
        if any(type(n) is not int or not 1 <= n <= 128 for n in (self.rows, self.cols)):
            raise ValueError("BEV shape must be bounded and positive")
        if (not isinstance(self.heights_m, tuple) or not 1 <= len(self.heights_m) <= 4
                or not all(isfinite(v) and -4 <= v <= 4 for v in self.heights_m)):
            raise ValueError("Invalid support planes")

    def xy(self) -> tuple[np.ndarray, np.ndarray]:
        x = self.x_min + (np.arange(self.rows) + .5) * self.step_m
        y = self.y_min + (np.arange(self.cols) + .5) * self.step_m
        return np.meshgrid(x, y, indexing="ij")

    def index(self, x, y):
        row, col = int(np.floor((x-self.x_min)/self.step_m)), int(np.floor((y-self.y_min)/self.step_m))
        return (row, col) if 0 <= row < self.rows and 0 <= col < self.cols else None

    def center(self, row, col):
        return self.x_min + (row+.5)*self.step_m, self.y_min + (col+.5)*self.step_m


@torch.no_grad()
def lift_features(features: SurroundFeatures, reference: CameraSnapshot, grid: BevGrid) -> tuple[Tensor, Tensor]:
    """Sample patch maps at an ego-fixed set of 3D points, masking off-camera samples."""
    snapshot = features.snapshot
    _validate_pair(snapshot, reference)
    maps = features.feature_maps
    if (maps.ndim != 5 or maps.shape[:2] != (1, 6) or len(features.calibrations) != 6
            or maps.dtype not in (torch.float32, torch.float64)):
        raise ValueError("Expected calibrated floating maps [1,6,C,H,W]")
    if not torch.isfinite(maps).all():
        raise ValueError("Expected finite surround maps")
    ego_from_reference = invert_pose(snapshot.ego_to_world) @ reference.ego_to_world
    transforms = np.stack([c.ego_to_camera @ ego_from_reference for c in features.calibrations])
    x, y = grid.xy()
    planes = np.stack([np.stack((x, y, np.full_like(x, z), np.ones_like(x)), axis=-1) for z in grid.heights_m])
    points = np.einsum("vij,hrcj->vhrci", transforms[:, :3], planes)
    k = np.stack([c.intrinsic for c in features.calibrations])
    pixels = np.einsum("vij,vhrcj->vhrci", k, points)
    near = np.array([c.near_m for c in features.calibrations])[:, None, None, None]
    z = pixels[..., 2]
    uv = pixels[..., :2] / np.maximum(z, near)[..., None]
    size = np.array([[c.width, c.height] for c in features.calibrations])[:, None, None, None]
    grid_uv = 2 * (uv+.5) / size - 1
    visible = (z >= near) & np.isfinite(grid_uv).all(axis=-1) & (np.abs(grid_uv) <= 1).all(axis=-1)
    grid_uv[~visible] = 0
    samples = torch.as_tensor(grid_uv.reshape(6, -1, grid.cols, 2), dtype=maps.dtype, device=maps.device)
    lifted = grid_sample(maps[0], samples, mode="bilinear", padding_mode="border", align_corners=False)
    lifted = lifted.reshape(6, maps.shape[2], len(grid.heights_m), grid.rows, grid.cols)
    weight = torch.as_tensor(visible[:, None], device=maps.device, dtype=maps.dtype)
    count = weight.sum(dim=(0, 2))
    pooled = (lifted * weight).sum(dim=(0, 2)) / count.clamp_min(1)
    return pooled, count / (6 * len(grid.heights_m))


@torch.no_grad()
def temporal_bev(current: SurroundFeatures, previous: SurroundFeatures, grid: BevGrid) -> tuple[Tensor, Tensor]:
    """Stack current and earlier appearance aligned by ego pose, plus map coverage and coordinates."""
    if current.encoder_id != previous.encoder_id or current.feature_maps.shape != previous.feature_maps.shape:
        raise ValueError("Temporal samples must use one encoder and patch layout")
    dt = (current.snapshot.ref_time_us - previous.snapshot.ref_time_us) / 1e6
    if not 0 < dt <= 1:
        raise ValueError("Previous observation must precede the reference by at most one second")
    now, now_mask = lift_features(current, current.snapshot, grid)
    past, past_mask = lift_features(previous, current.snapshot, grid)
    x, y = [torch.as_tensor(a, dtype=now.dtype, device=now.device) for a in grid.xy()]
    xy = torch.stack([x / 64, y / 64])
    image_evidence = torch.cat((now, past))
    geometry = torch.cat((now_mask, past_mask, xy, torch.full_like(now_mask, dt)))
    return torch.cat((image_evidence, geometry)), now_mask[0] > 0


def past_forward_speed(current: CameraSnapshot, previous: CameraSnapshot) -> float:
    """Mean forward velocity over the preceding sample interval, not CAN ground truth."""
    _validate_pair(previous, current)
    dt_s = (current.ref_time_us - previous.ref_time_us) / 1e6
    if not 0 < dt_s <= 1:
        raise ValueError("Past velocity requires a distinct recent observation")
    world_displacement = current.ego_to_world[:3, 3] - previous.ego_to_world[:3, 3]
    longitudinal = float(current.ego_to_world[:3, 0] @ world_displacement) / dt_s
    if not isfinite(longitudinal):
        raise ValueError("Past ego velocity must be finite")
    return longitudinal


def _validate_pair(observation: CameraSnapshot, reference: CameraSnapshot) -> None:
    if (observation.rig_id != reference.rig_id or observation.ref_time_us > reference.ref_time_us
            or (observation.ref_time_us == reference.ref_time_us and observation.content_sha256 != reference.content_sha256)):
        raise ValueError("Cannot lift a future/different-rig snapshot into this reference")
