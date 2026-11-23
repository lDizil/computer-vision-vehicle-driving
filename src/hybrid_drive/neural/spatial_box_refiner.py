"""Learn local corrections to four frozen, normalized spatial boxes."""

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.ops import generalized_box_iou_loss, roi_align


_BANDS = 4
_ROI_SIZE = 2
_SMOOTH_L1_BETA = 1 / 32


class SpatialBoxRefiner(nn.Module):
    """Read current-frame patches around each box; do not change presence scores."""

    def __init__(self, visual_dim: int, *, hidden_dim: int = 64):
        super().__init__()
        if any(type(dim) is not int or dim <= 0 for dim in (visual_dim, hidden_dim)):
            raise ValueError("visual_dim and hidden_dim must be positive integers")
        self.visual_dim = visual_dim
        local_dim = max(1, hidden_dim // (_ROI_SIZE**2))
        self.local_projection = nn.Sequential(nn.LayerNorm(visual_dim), nn.Linear(visual_dim, local_dim), nn.GELU())
        self.head = nn.Sequential(
            nn.Linear(_ROI_SIZE**2 * local_dim + 4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 4),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, features: Tensor, baseline_boxes: Tensor) -> Tensor:
        _floating(features, "features", self.head[-1].weight)
        _floating(baseline_boxes, "baseline_boxes", features)
        if features.ndim != 4 or features.shape[1] != self.visual_dim or not all(features.shape):
            raise ValueError("Expected a nonempty patch grid [B,C,H,W] with C=visual_dim")
        batch, _, height, width = features.shape
        if not torch.isfinite(features).all():
            raise ValueError("Patch features must be finite")
        if baseline_boxes.shape != (batch, _BANDS, 4):
            raise ValueError("Expected baseline_boxes [B,4,4] in left-to-right band order")
        bands = torch.arange(_BANDS, device=features.device)
        _validate_boxes(baseline_boxes, bands, "Baseline boxes")

        boxes = baseline_boxes.to(torch.float64)
        lo, hi = boxes.split(2, dim=-1)
        half_size = (hi - lo) * 0.5
        center = lo + half_size
        patch = boxes.new_tensor((1 / width, 1 / height))
        context = torch.maximum(half_size * 0.5, patch * 0.5)
        padded = torch.cat(((lo - context).clamp_min(0), (hi + context).clamp_max(1)), dim=-1)
        # ROIAlign keeps the 2×2 spatial layout instead of pooling away the direction of a correction.
        pixels = (padded * boxes.new_tensor((width, height, width, height))).to(features.dtype).reshape(-1, 4)
        image = torch.arange(batch, device=features.device, dtype=features.dtype).repeat_interleave(_BANDS)
        rois = torch.cat((image[:, None], pixels), dim=1)
        local = roi_align(features, rois, output_size=_ROI_SIZE, sampling_ratio=2, aligned=True)
        tokens = self.local_projection(local.permute(0, 2, 3, 1)).reshape(batch, _BANDS, -1)
        geometry = torch.cat((center, half_size * 2), dim=-1).to(features.dtype)
        residual = self.head(torch.cat((tokens, geometry), dim=-1))
        if not torch.isfinite(residual).all():
            raise ValueError("Box residuals must be finite")
        return _refine(baseline_boxes, center, half_size, residual.to(boxes.dtype), patch, bands)


def spatial_box_refinement_loss(refined: Tensor, target_boxes: Tensor, presence: Tensor) -> Tensor | None:
    """GIoU plus 5×SmoothL1 (β=1/32) on positive queries, or None if none exist."""
    _floating(refined, "refined")
    _floating(target_boxes, "target_boxes", refined)
    if refined.ndim != 3 or refined.shape[0] == 0 or refined.shape[1:] != (_BANDS, 4):
        raise ValueError("Expected refined boxes [B,4,4]")
    if target_boxes.shape != refined.shape:
        raise ValueError("Expected target_boxes [B,4,4] matching the refined boxes")
    if (not isinstance(presence, Tensor) or presence.dtype != torch.bool
            or presence.shape != refined.shape[:2] or presence.device != refined.device):
        raise ValueError("Expected boolean presence [B,4] on the box device")
    selected = presence.nonzero(as_tuple=True)
    if selected[0].numel() == 0:
        return None
    predictions, targets, bands = refined[selected], target_boxes[selected], selected[1]
    _validate_boxes(predictions, bands, "Active refined boxes")
    _validate_boxes(targets, bands, "Active target boxes")
    giou = generalized_box_iou_loss(predictions, targets, reduction="mean")
    return giou + 5 * F.smooth_l1_loss(predictions, targets, beta=_SMOOTH_L1_BETA)


def _refine(baseline, center, half_size, residual, patch, bands):
    shift, scale = residual.tanh().split(2, dim=-1)
    intended_center = center + shift * patch
    # Reserve half the old radius at the image edges, even at the maximum contraction.
    lower = half_size * 0.5
    upper = 1 - lower
    left = bands.to(torch.float32) / _BANDS
    right = torch.nextafter(left + 1 / _BANDS, left)
    # Keep the input itself feasible even if its exact center rounds onto a band boundary.
    lower_x = torch.maximum(lower[..., 0], torch.minimum(left, center[..., 0]))
    upper_x = torch.minimum(upper[..., 0], torch.maximum(right, center[..., 0]))
    lower = torch.stack((lower_x, lower[..., 1]), dim=-1)
    upper = torch.stack((upper_x, upper[..., 1]), dim=-1)
    moved = torch.minimum(torch.maximum(intended_center, lower), upper)
    radius = half_size * (scale * math.log(2)).exp()
    radius = torch.minimum(radius, torch.minimum(moved, 1 - moved))

    # Add changes to the original corners to avoid round-tripping them at zero residual.
    offset, resize = moved - center, radius - half_size
    corners = torch.cat((offset - resize, offset + resize), dim=-1)
    adjusted = (baseline + corners).clamp(0, 1).to(baseline.dtype)
    # An extremely thin box may collapse or change bands when re-encoded in the output dtype.
    representable = (adjusted[..., 2:] > adjusted[..., :2]).all(dim=-1) & (_box_bands(adjusted) == bands)
    return torch.where(representable[..., None], adjusted, baseline)


def _floating(value: Tensor, name: str, reference: Tensor | None = None) -> None:
    if not isinstance(value, Tensor) or value.dtype not in (torch.float32, torch.float64):
        raise ValueError(f"{name} must be a float32/float64 tensor")
    if reference is not None and (value.dtype != reference.dtype or value.device != reference.device):
        raise ValueError(f"{name} must match the reference dtype and device")


def _validate_boxes(boxes: Tensor, bands: Tensor, name: str) -> None:
    if (not torch.isfinite(boxes).all() or ((boxes < 0) | (boxes > 1)).any()
            or (boxes[..., 2:] <= boxes[..., :2]).any()):
        raise ValueError(f"{name} must be finite positive-area normalized xyxy boxes")
    if (_box_bands(boxes) != bands).any():
        raise ValueError(f"{name} must keep each center in its assigned band")


def _box_bands(boxes: Tensor) -> Tensor:
    # Match the data pack's canonical FP32 midpoint (including the last-band image-edge rounding).
    boxes = boxes.to(torch.float32)
    midpoints = (boxes[..., 0] + boxes[..., 2]) * 0.5
    return (midpoints * _BANDS).floor().long().clamp_max(_BANDS - 1)
