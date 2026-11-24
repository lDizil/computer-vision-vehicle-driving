"""Short-horizon observed-track occupancy; no decision or controller interface."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from hybrid_drive.experiments.visual_vru_geometry import GRID_STEP_M, GRID_X_MIN, GRID_Y_MIN
from hybrid_drive.experiments.visual_vru_model import GRID_SHAPE, VISUAL_DIM


HORIZONS = (0., .5, 1.)
ARMS = ("temporal", "single_frame", "shuffled_past", "persist_now")
MIN_HISTORY_S, MAX_HISTORY_S = .4, .6
MOVING_M = .75


class TemporalVruHead(nn.Module):
    """Learn dynamics from a shared current/past encoder in the current ego frame."""

    def __init__(self):
        super().__init__()
        forward, left = torch.meshgrid(*(torch.linspace(-1., 1., n) for n in GRID_SHAPE), indexing="ij")
        self.register_buffer("positions", torch.stack((forward, left))[None], persistent=False)
        self.reduce = nn.Conv2d(VISUAL_DIM, 32, 1)
        self.predict = nn.Sequential(
            nn.Conv2d(101, 48, 1), nn.GELU(), nn.Conv2d(48, 32, 3, padding=1), nn.GELU(),
            nn.Conv2d(32, 3, 1),
        )

    def forward(self, now, past, current_coverage, past_coverage, history_s):
        n = len(now)
        if (not n or now.shape != (n, VISUAL_DIM, *GRID_SHAPE) or past.shape != now.shape
                or current_coverage.shape != (n, *GRID_SHAPE) or past_coverage.shape != current_coverage.shape
                or history_s.shape != (n,) or not torch.isfinite(history_s).all()
                or (history_s < MIN_HISTORY_S).any() or (history_s > MAX_HISTORY_S).any()):
            raise ValueError("Expected aligned camera grids and one bounded observed history interval")
        for coverage in (current_coverage, past_coverage):
            if not torch.isfinite(coverage).all() or (coverage < 0).any() or (coverage > 1).any():
                raise ValueError("Camera support must be finite in [0,1]")
        a = self._encode(now, current_coverage)
        b = self._encode(past, past_coverage)
        delta = (a - b) * ((current_coverage > 0) & (past_coverage > 0))[:, None]
        interval = (history_s[:, None, None, None] - .5).expand(-1, 1, *GRID_SHAPE) * 10
        channels = (a, b, delta, self.positions.expand(n, -1, -1, -1),
                    current_coverage[:, None], past_coverage[:, None], interval)
        return self.predict(torch.cat(channels, dim=1))

    def _encode(self, values, coverage):
        known = coverage[:, None] > 0
        visual = values.float().masked_fill(~known, 0)
        if not torch.isfinite(visual).all():
            raise ValueError("Covered grid cells must have finite camera features")
        norm = F.layer_norm(visual.movedim(1, -1), (VISUAL_DIM,)).movedim(-1, 1)
        return F.gelu(self.reduce(norm)).masked_fill(~known, 0)


def grid_cell(point):
    x, y = point
    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError("Expected a finite metric XY target")
    row, col = math.floor((x - GRID_X_MIN) / GRID_STEP_M), math.floor((y - GRID_Y_MIN) / GRID_STEP_M)
    if 0 <= row < GRID_SHAPE[0] and 0 <= col < GRID_SHAPE[1]:
        return row, col
    return None


def supervised_targets(rows, coverage):
    """Rasterize targets and collect unambiguous moving cell comparisons."""
    labels = torch.zeros((len(rows), len(HORIZONS), *GRID_SHAPE), dtype=torch.bool)
    pairs = []
    for i, row in enumerate(rows):
        tracks = [item["centers_xy_m"] for item in row["tracks"]]
        positions = [tuple(grid_cell(point) for point in tr) for tr in tracks]
        if any(cell is None for track in positions for cell in track):
            raise ValueError("Out-of-grid labels must be excluded by the observation curator")
        for tr in positions:
            for j, cell in enumerate(tr):
                labels[i, j][cell] = True
        pairs.append(_motion_pairs(tracks, positions, coverage[i]))
    return labels, pairs


def _motion_pairs(tracks, positions, coverage):
    active = []
    for idx, track in enumerate(tracks):
        anchor = positions[idx][0]
        if not coverage[anchor] > 0:
            continue
        for horizon in (1, 2):
            future = positions[idx][horizon]
            if (anchor == future or not coverage[future] > 0
                    or math.dist(track[0], track[horizon]) < MOVING_M):
                continue
            if any(((anchor in (other[0], other[horizon])) or (future in (other[0], other[horizon])))
                   for j, other in enumerate(positions) if idx != j):
                continue
            active.append((horizon, anchor, future))
    return active
