"""Diagnostic 2D LOC on frozen front-camera patch features, not a detector."""

from dataclasses import dataclass
from math import log

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.ops import generalized_box_iou_loss

from hybrid_drive.contracts._validation import finite_real


_BANDS = 4


@dataclass(frozen=True)
class SpatialLocOutput:
    presence_logits: Tensor
    cell_logits: Tensor
    boxes_xyxy: Tensor


class SpatialLocProbe(nn.Module):
    """Predict the nearest eligible target per image band, from features alone.

    Bands are disjoint vertical quarters in left-to-right order. Cell logits
    outside a band's support are -inf. Each xyxy box is normalized to the
    full image and may extend across bands; it is not a validated detection.
    """

    def __init__(self, visual_dim: int, *, hidden_dim: int = 64):
        super().__init__()
        if any(type(dim) is not int or dim <= 0 for dim in (visual_dim, hidden_dim)):
            raise ValueError("visual_dim and hidden_dim must be positive integers")
        self.visual_dim = visual_dim
        self.trunk = nn.Sequential(
            nn.Conv2d(visual_dim + 2, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.score = nn.Conv2d(hidden_dim, 1, kernel_size=1)
        self.extent = nn.Conv2d(hidden_dim, 2, kernel_size=1)

    def forward(self, features: Tensor) -> SpatialLocOutput:
        _floating(features, "features", self.score.weight)
        if features.ndim != 4 or features.shape[0] == 0 or features.shape[1] != self.visual_dim:
            raise ValueError("Expected a nonempty patch grid [B,C,H,W] with C=visual_dim")
        batch, _, height, width = features.shape
        _validate_grid(height, width)
        if not torch.isfinite(features).all():
            raise ValueError("Patch features must be finite")

        x = (torch.arange(width, device=features.device, dtype=features.dtype) + 0.5) / width
        y = (torch.arange(height, device=features.device, dtype=features.dtype) + 0.5) / height
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        xy = torch.stack((xx, yy))
        spatial = self.trunk(torch.cat((features, xy.expand(batch, -1, -1, -1)), dim=1))
        scores = self.score(spatial).expand(-1, _BANDS, -1, -1)
        allowed = _band_columns(width, features.device)[:, None, :]
        cell_logits = scores.masked_fill(~allowed, -torch.inf)
        flat_logits = cell_logits.flatten(2)
        presence_logits = flat_logits.logsumexp(dim=-1) - log(height * (width // _BANDS))
        weights = flat_logits.softmax(dim=-1)
        eps = torch.finfo(features.dtype).eps
        centers = (weights @ xy.flatten(1).T).clamp(eps, 1 - eps)
        fractions = self.extent(spatial).sigmoid().flatten(2).transpose(1, 2)
        # Symmetric image margins keep the decoded center fixed without clipping a corner.
        margin = torch.minimum(centers, 1 - centers)
        half_sizes = ((weights @ fractions) * margin).clamp_min(eps).minimum(margin)
        boxes = torch.cat((centers - half_sizes, centers + half_sizes), dim=-1)
        return SpatialLocOutput(presence_logits, cell_logits, boxes)


@dataclass(frozen=True)
class SpatialLocLossBreakdown:
    total: Tensor
    presence: Tensor
    cell: Tensor
    box: Tensor
    giou: Tensor
    supervised_presence_count: int
    positive_count: int


class SpatialLocLoss(nn.Module):
    """Average presence over all bands, and localization only over positive bands."""

    def __init__(
        self,
        positive_weight: float,
        *,
        cell_weight: float = 1.0,
        box_weight: float = 5.0,
        giou_weight: float = 1.0,
    ):
        super().__init__()
        self.positive_weight = finite_real(positive_weight, "positive_weight")
        if self.positive_weight <= 0:
            raise ValueError("positive_weight from the training split must be strictly positive")
        for name, weight in (
            ("cell_weight", cell_weight),
            ("box_weight", box_weight),
            ("giou_weight", giou_weight),
        ):
            weight = finite_real(weight, name)
            if weight < 0:
                raise ValueError(f"{name} must be nonnegative")
            setattr(self, name, weight)

    def forward(
        self, output: SpatialLocOutput, presence: Tensor, targets_xyxy: Tensor
    ) -> SpatialLocLossBreakdown:
        if not isinstance(output, SpatialLocOutput):
            raise ValueError("Expected SpatialLocOutput")
        presence_logits, scores, boxes = output.presence_logits, output.cell_logits, output.boxes_xyxy
        _floating(presence_logits, "presence_logits")
        _floating(scores, "cell_logits", presence_logits)
        _floating(boxes, "boxes_xyxy", presence_logits)
        _floating(targets_xyxy, "targets_xyxy", presence_logits)
        if scores.ndim != 4 or scores.shape[0] == 0 or scores.shape[1] != _BANDS:
            raise ValueError("Expected a nonempty cell_logits grid [B,4,H,W]")
        batch, _, height, width = scores.shape
        _validate_grid(height, width)
        if presence_logits.shape != (batch, _BANDS) or not torch.isfinite(presence_logits).all():
            raise ValueError("Expected finite presence_logits [B,4]")
        if boxes.shape != (batch, _BANDS, 4) or targets_xyxy.shape != boxes.shape:
            raise ValueError("Expected boxes_xyxy and targets_xyxy [B,4,4]")
        if (
            not isinstance(presence, Tensor)
            or presence.dtype != torch.bool
            or presence.shape != presence_logits.shape
            or presence.device != presence_logits.device
        ):
            raise ValueError("Expected boolean presence [B,4] on the logit device")

        pos_weight = presence_logits.new_tensor(self.positive_weight)
        if not torch.isfinite(pos_weight) or pos_weight <= 0:
            raise ValueError("positive_weight must be positive and finite in the logit dtype")
        classification = F.binary_cross_entropy_with_logits(
            presence_logits, presence.to(presence_logits.dtype), pos_weight=pos_weight
        )
        positive = presence.nonzero(as_tuple=True)
        targets = targets_xyxy[positive]
        cell, box, giou = _positive_losses(scores[positive], boxes[positive], targets, positive[1])
        total = classification + self.cell_weight * cell + self.box_weight * box + self.giou_weight * giou
        return SpatialLocLossBreakdown(
            total, classification, cell, box, giou, presence.numel(), targets.shape[0]
        )


def _positive_losses(
    scores: Tensor, boxes: Tensor, targets: Tensor, bands: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    if targets.numel() == 0:
        zero_box = boxes.sum()
        return scores.sum(), zero_box, zero_box

    _normalized_boxes(targets, "Active targets")
    _normalized_boxes(boxes, "Active predictions")
    _, height, width = scores.shape
    band_width = width // _BANDS
    # The data pack assigns bands in float32 before any training dtype conversion.
    quantized = targets.to(torch.float32)
    centers = (quantized[:, :2] + quantized[:, 2:]) * 0.5
    cell_x, cell_y = (centers * quantized.new_tensor((width, height))).floor().long().unbind(dim=1)
    # The midpoint of a valid box can round to exactly 1 at the image edge.
    cell_x, cell_y = cell_x.clamp_max(width - 1), cell_y.clamp_max(height - 1)
    if (cell_x // band_width != bands).any():
        raise ValueError("Each active target center must be inside its assigned band")

    allowed = _band_columns(width, scores.device)[bands, None, :]
    if not torch.isfinite(scores.masked_select(allowed)).all():
        raise ValueError("Active cell logits inside the assigned band must be finite")
    cell = F.cross_entropy(scores.masked_fill(~allowed, -torch.inf).flatten(1), cell_y * width + cell_x)
    box = F.smooth_l1_loss(boxes, targets)
    giou = generalized_box_iou_loss(boxes, targets, reduction="mean")
    return cell, box, giou


def _band_columns(width: int, device: torch.device) -> Tensor:
    return (
        torch.arange(width, device=device) // (width // _BANDS)
        == torch.arange(_BANDS, device=device)[:, None]
    )


def _validate_grid(height: int, width: int) -> None:
    if height == 0 or width == 0 or width % _BANDS:
        raise ValueError("Expected positive grid height and positive grid width divisible by four")


def _floating(value: Tensor, name: str, reference: Tensor | None = None) -> None:
    if not isinstance(value, Tensor) or value.dtype not in (torch.float32, torch.float64):
        raise ValueError(f"{name} must be a float32/float64 tensor")
    if reference is not None and (value.dtype != reference.dtype or value.device != reference.device):
        raise ValueError(f"{name} must match the reference dtype and device")


def _normalized_boxes(boxes: Tensor, name: str) -> None:
    if (
        not torch.isfinite(boxes).all()
        or ((boxes < 0) | (boxes > 1)).any()
        or (boxes[:, 2:] <= boxes[:, :2]).any()
    ):
        raise ValueError(f"{name} must be finite positive-area normalized xyxy boxes")
