"""A trainable camera-to-grid head for current pedestrian and bicycle evidence."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from hybrid_drive.experiments.visual_vru_geometry import GRID_COLS, GRID_ROWS, exclusive_corridors


VISUAL_DIM = 384
GRID_SHAPE = (GRID_ROWS, GRID_COLS)
PAIR_TIE = 1e-6


class VisualVruHead(nn.Module):
    """Predict cell occupancy evidence, not an identity, track, or safety decision."""

    def __init__(self, *, camera_input: bool):
        super().__init__()
        self.camera_input = camera_input
        width = VISUAL_DIM + 3 if camera_input else 3
        forward, lateral = torch.meshgrid(torch.linspace(-1., 1., GRID_ROWS), torch.linspace(-1., 1., GRID_COLS), indexing="ij")
        self.register_buffer("positions", torch.stack((forward, lateral))[None], persistent=False)
        self.head = nn.Sequential(
            nn.Conv2d(width, 48, 1), nn.GELU(),
            nn.Conv2d(48, 32, 3, padding=1), nn.GELU(),
            nn.Conv2d(32, 1, 1),
        )

    def forward(self, appearance, coverage):
        if (coverage.ndim != 3 or not len(coverage) or tuple(coverage.shape[1:]) != GRID_SHAPE
                or not torch.isfinite(coverage).all() or (coverage < 0).any() or (coverage > 1).any()):
            raise ValueError("Expected a nonempty finite camera coverage grid in [0,1]")
        spatial = torch.cat((self.positions.expand(len(coverage), -1, -1, -1), coverage[:, None]), dim=1)
        if self.camera_input:
            if appearance is None or tuple(appearance.shape) != (len(coverage), VISUAL_DIM, *GRID_SHAPE):
                raise ValueError("Expected the bounded current DINOv2 BEV feature grid")
            known = coverage[:, None] > 0
            if not torch.isfinite(appearance.masked_fill(~known, 0)).all():
                raise ValueError("Observed visual cells must have finite features")
            channels_last = appearance.float().masked_fill(~known, 0).movedim(1, -1)
            normalized = F.layer_norm(channels_last, (VISUAL_DIM,)).movedim(-1, 1)
            spatial = torch.cat((normalized, spatial), dim=1)
        elif appearance is not None:
            raise ValueError("The geometry control must not receive visual features")
        return self.head(spatial)[:, 0]


def candidate_masks(poses, coverage):
    """Keep only covered cells unique to each diagnostic trajectory."""
    masks = torch.as_tensor(exclusive_corridors(poses), device=coverage.device, dtype=torch.bool)
    return masks & (coverage > 0)[None]


def pair_logits(logits, masks):
    if logits.shape != GRID_SHAPE or masks.shape != (2, *GRID_SHAPE):
        raise ValueError("Expected one evidence grid and two candidate masks")
    if not masks.flatten(1).any(dim=1).all():
        return None
    return torch.stack([logits[mask].max() for mask in masks])


def flank_query(target, coverage):
    """Choose the nearest unambiguous 5 m band with one occupied side.

    This is a spatial label task, not a trajectory. The mirrored boxes come
    from a fixed bank; selecting a band can reveal that an object is present,
    but does not reveal its side. Use at most one query per observation.
    """
    known = coverage > 0
    middle = GRID_COLS // 2
    for lo in range(0, GRID_ROWS, 5):
        hi = min(lo + 5, GRID_ROWS)
        left = target[lo:hi, middle:]
        right = target[lo:hi, :middle]
        if bool(left.any()) == bool(right.any()):
            continue
        masks = torch.zeros((2, *GRID_SHAPE), dtype=torch.bool, device=coverage.device)
        masks[0, lo:hi, middle:] = known[lo:hi, middle:]
        masks[1, lo:hi, :middle] = known[lo:hi, :middle]
        side = 0 if left.any() else 1
        if masks.flatten(1).sum(1).min() >= 4 and (target & masks[side]).any():
            return masks, side, (lo, hi)
    return None


def cell_average_precision(scores, targets):
    """AP with entire equal-score groups counted together; no favorable tie order."""
    scores = np.asarray(scores)
    targets = np.asarray(targets)
    if (scores.ndim != 1 or targets.shape != scores.shape or targets.dtype != np.bool_
            or not np.isfinite(scores).all()):
        raise ValueError("AP needs aligned finite scores and boolean targets")
    positive = int(targets.sum())
    if not positive:
        return None
    order = np.argsort(-scores, kind="stable")
    ordered = scores[order]
    group_ends = np.flatnonzero(np.r_[ordered[:-1] != ordered[1:], True])
    tp = np.cumsum(targets[order])[group_ends]
    return float(np.sum(np.diff(tp, prepend=0) * (tp / (group_ends + 1))) / positive)
