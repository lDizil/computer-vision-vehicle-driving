"""A small camera-feature head for present metric boxes and next-step displacement."""

from __future__ import annotations

from math import atan2, isfinite
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from hybrid_drive.fast.metric_bev import BevGrid


_SIZE_PRIOR = (1.9, 4.5, 1.7)
_MOTION_SCALE = 8.0


@dataclass(frozen=True)
class DecodedBox:
    object_id: str
    center_xyz: tuple[float, float, float]
    size_wlh: tuple[float, float, float]
    yaw: float
    velocity_xy: tuple[float, float]
    confidence: float


class MetricForecastHead(nn.Module):
    """Per-cell objectness, offset/3D dimensions/heading, and Cartesian velocity."""

    def __init__(self, channels: int):
        super().__init__()
        if type(channels) is not int or channels <= 5:
            raise ValueError("Expected visual and geometry channels")
        self.trunk = nn.Sequential(
            nn.Conv2d(channels, 32, 1), nn.GroupNorm(4, 32), nn.SiLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.GroupNorm(4, 32), nn.SiLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.GroupNorm(4, 32), nn.SiLU(),
        )
        self.objectness = nn.Conv2d(32, 1, 1)
        self.regression = nn.Conv2d(32, 10, 1)
        nn.init.constant_(self.objectness.bias, -2)
        nn.init.zeros_(self.regression.bias)

    def forward(self, values: Tensor) -> tuple[Tensor, Tensor]:
        shared = self.trunk(values)
        return self.objectness(shared)[:, 0], self.regression(shared)


def raster_targets(frame: dict, visible: np.ndarray, grid: BevGrid):
    """Assign at most one labeled center per supported cell, never a fake velocity."""
    if visible.shape != (grid.rows, grid.cols) or visible.dtype != np.bool_:
        raise ValueError("Expected boolean grid coverage")
    positive = np.zeros_like(visible)
    motion = np.zeros_like(visible)
    regression = np.zeros((10, grid.rows, grid.cols), np.float32)
    distance = np.full_like(visible, np.inf, dtype=np.float64)
    next_dt = frame["next_dt_s"]
    if not isfinite(next_dt) or not 0 < next_dt <= 1:
        raise ValueError("Need a real next keyframe interval in (0,1] seconds")
    for box in sorted(frame["boxes"], key=lambda b: b["instance_id"]):
        x, y, z = box["center"]
        width, length, height = box["size_wlh"]
        yaw = box["yaw"]
        if (not all(isfinite(v) for v in (x,y,z,width,length,height,yaw))
                or min(width,length,height) <= 0):
            raise ValueError("Invalid ground-truth 3D box")
        cell = grid.index(x, y)
        if cell is None or not visible[cell]:
            continue
        row, col = cell
        anchor_x, anchor_y = grid.center(row, col)
        dx, dy = x-anchor_x, y-anchor_y
        d2 = dx*dx + dy*dy
        if d2 >= distance[cell]:
            continue
        distance[cell] = d2
        positive[cell] = True
        future = box["future_center"]
        velocity = np.zeros(2)
        if future is not None:
            future_x, future_y, _ = future
            velocity = (np.array([future_x, future_y]) - [x, y]) / next_dt
            if not np.isfinite(velocity).all():
                raise ValueError("Nonfinite next-step position")
        motion[cell] = future is not None
        regression[:, row, col] = (
            dx/grid.step_m, dy/grid.step_m, z/3,
            *np.log(np.array([width,length,height]) / _SIZE_PRIOR),
            np.sin(yaw), np.cos(yaw), *(velocity/_MOTION_SCALE),
        )
    return positive, motion, regression


def forecasting_loss(output, targets, visible: Tensor) -> Tensor:
    logits, prediction = output
    positive, moving, regression = targets
    # Weight sparse positives through focal classification; all unseen cells are excluded.
    labels = positive.to(logits.dtype)
    probability = torch.sigmoid(logits)
    pt = torch.where(positive, probability, 1-probability)
    weights = torch.where(positive, 0.8, 0.2)
    classified = F.binary_cross_entropy_with_logits(logits, labels, reduction="none") * (1-pt).square() * weights
    cls = (classified*visible).sum() / (positive.sum().clamp_min(1))
    pred = prediction.permute(0, 2, 3, 1)
    target = regression.permute(0, 2, 3, 1)
    position = _masked_l1(pred[..., :8], target[..., :8], positive)
    velocity = _masked_l1(pred[..., 8:], target[..., 8:], moving)
    return cls + 2*position + velocity


def _masked_l1(prediction, target, mask):
    return (F.smooth_l1_loss(prediction, target, reduction="none") * mask[..., None]).sum() / (
        mask.sum().clamp_min(1) * prediction.shape[-1]
    )


@torch.no_grad()
def decode_boxes(output, visible: Tensor, grid: BevGrid, *, threshold: float,
                 max_objects: int = 24) -> tuple[DecodedBox, ...]:
    logits, prediction = output
    if (logits.shape != (1, grid.rows, grid.cols) or prediction.shape != (1, 10, grid.rows, grid.cols)
            or visible.shape != (grid.rows, grid.cols) or visible.dtype != torch.bool
            or not isfinite(threshold) or not 0 <= threshold <= 1 or not 1 <= max_objects <= 100):
        raise ValueError("Invalid decode grid or limits")
    if not torch.isfinite(logits).all() or not torch.isfinite(prediction).all():
        raise ValueError("Nonfinite model output")
    scores = logits[0].sigmoid().masked_fill(~visible, -1)
    maxima = F.max_pool2d(scores[None, None], kernel_size=3, stride=1, padding=1)[0, 0]
    cells = torch.nonzero(visible & (scores >= threshold) & (scores == maxima)).tolist()
    cells.sort(key=lambda rc: (-float(scores[tuple(rc)]), *rc))
    decoded = []
    for row, col in cells[:max_objects]:
        x, y = grid.center(row, col)
        reg = prediction[0, :, row, col].cpu().numpy()
        center = (x+float(np.clip(reg[0], -.5, .5))*grid.step_m,
                  y+float(np.clip(reg[1], -.5, .5))*grid.step_m,
                  float(np.clip(reg[2], -2, 2))*3)
        size = tuple(float(v) for v in np.exp(np.clip(reg[3:6], -1.5, 1.5))*_SIZE_PRIOR)
        velocity = tuple(float(v) for v in np.clip(reg[8:10]*_MOTION_SCALE, -20, 20))
        decoded.append(DecodedBox(f"cell-{row}-{col}", center, size, atan2(reg[6], reg[7]),
                                  velocity, float(scores[row, col])))
    return tuple(decoded)
