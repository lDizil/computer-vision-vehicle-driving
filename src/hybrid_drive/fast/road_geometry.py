"""Two visual BEV layers: drivable cell centers and lane-divider presence."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


LAYERS = ("drivable", "lane_divider")


class RoadGeometryHead(nn.Module):
    """Predict static road evidence from the same temporal BEV as the object head."""

    def __init__(self, channels: int):
        super().__init__()
        if type(channels) is not int or channels <= 5:
            raise ValueError("Expected visual and metric geometry channels")
        self.layers = nn.Sequential(
            nn.Conv2d(channels, 32, 1), nn.GroupNorm(4, 32), nn.SiLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.GroupNorm(4, 32), nn.SiLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.GroupNorm(4, 32), nn.SiLU(),
            nn.Conv2d(32, len(LAYERS), 1),
        )

    def forward(self, values: Tensor) -> Tensor:
        return self.layers(values)


def positive_weights(labels: Tensor, support: Tensor) -> Tensor:
    _check_target_shape(labels, support)
    yes = (labels & support[:, None]).sum(dim=(0, 2, 3))
    no = ((~labels) & support[:, None]).sum(dim=(0, 2, 3))
    if torch.any(yes == 0) or torch.any(no == 0):
        raise ValueError("Both classes of both layers must be present in supervised cells")
    return (no / yes).clamp(1, 12).view(1, 2, 1, 1)


def road_loss(logits: Tensor, labels: Tensor, support: Tensor, weights: Tensor) -> Tensor:
    _check_target_shape(labels, support)
    if logits.shape != labels.shape or tuple(weights.shape) != (1, 2, 1, 1):
        raise ValueError("Mismatched road logits or positive weights")
    selected_logits = logits.permute(0, 2, 3, 1)[support]
    selected_labels = labels.permute(0, 2, 3, 1)[support].to(logits.dtype)
    if selected_logits.numel() == 0:
        raise ValueError("Training batch has no supported labeled cells")
    bce = F.binary_cross_entropy_with_logits(
        selected_logits, selected_labels, pos_weight=weights.flatten()
    )
    probabilities = selected_logits.sigmoid()
    numerator = 2 * (probabilities * selected_labels).sum(dim=0)
    denominator = (probabilities + selected_labels).sum(dim=0)
    dice = (numerator + 1) / (denominator + 1)
    return bce + .5 * (1 - dice.mean())


def _check_target_shape(labels: Tensor, support: Tensor) -> None:
    if (labels.ndim != 4 or labels.shape[1] != len(LAYERS) or labels.dtype != torch.bool
            or support.shape != (labels.shape[0], *labels.shape[2:]) or support.dtype != torch.bool):
        raise ValueError("Expected bool labels [N,2,H,W] and support [N,H,W]")
