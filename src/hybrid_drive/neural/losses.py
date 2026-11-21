"""Project loss recipe; not an assertion of official Reason2Drive checkpoint fidelity."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.ops import generalized_box_iou_loss

from hybrid_drive.contracts._validation import finite_real
from .decoder import PerceptionOutput


@dataclass(frozen=True)
class PerceptionTargets:
    batch_indices: Tensor
    token_positions: Tensor
    event_kinds: Tensor
    validity: Tensor
    validity_mask: Tensor
    boxes_3d: Tensor
    boxes_3d_mask: Tensor
    boxes_2d: Tensor
    boxes_2d_mask: Tensor
    motion_xy: Tensor
    motion_mask: Tensor


@dataclass(frozen=True)
class LossBreakdown:
    total: Tensor
    text: Tensor
    classification: Tensor
    boxes_3d: Tensor
    giou_2d: Tensor
    motion: Tensor
    supervised_text_tokens: int
    supervised_events: int
    supervised_box_coordinates: int
    supervised_image_boxes: int
    supervised_motion_points: int


class Reason2DriveLoss(nn.Module):
    """CE + lambda_per*(BCE + box_weight*SmoothL1 + giou_weight*GIoU + motion_weight*SmoothL1).

    Text logits/targets are already aligned by the external LLM adapter; this
    module does not guess decoder-only versus encoder-decoder shifting.
    LOC=0, MOT=1. GIoU applies to the auxiliary 2D image box, never to the
    seven-dimensional oriented box. Its yaw residual takes the shortest arc.
    Every component averages its own supervised elements; absent components
    are differentiable zeros. Validity, missing labels and padding differ.
    """

    def __init__(
        self, *, lambda_per=1.0, box_weight=1.0, giou_weight=1.0, motion_weight=1.0, smooth_l1_beta=1.0
    ):
        super().__init__()
        for name, value in (
            ("lambda_per", lambda_per),
            ("box_weight", box_weight),
            ("giou_weight", giou_weight),
            ("motion_weight", motion_weight),
            ("smooth_l1_beta", smooth_l1_beta),
        ):
            value = finite_real(value, name)
            if value < 0 or (name == "smooth_l1_beta" and value == 0):
                raise ValueError(name + " must be nonnegative (beta strictly positive)")
            setattr(self, name, value)

    def forward(
        self,
        text_logits: Tensor,
        text_targets: Tensor,
        text_mask: Tensor,
        predictions: PerceptionOutput,
        targets: PerceptionTargets,
    ) -> LossBreakdown:
        if not isinstance(predictions, PerceptionOutput) or not isinstance(targets, PerceptionTargets):
            raise ValueError("Expected PerceptionOutput and PerceptionTargets")
        _float_tensor(text_logits, None, "text_logits")
        if text_logits.ndim != 3 or min(text_logits.shape) <= 0:
            raise ValueError("Text logits must be nonempty [B,L,V]")
        batch, length, vocabulary = text_logits.shape
        _integer_tensor(text_targets, (batch, length), text_logits.device, "text_targets")
        _mask(text_mask, (batch, length), text_logits.device, "text_mask")
        events = _validate_perception(predictions, targets, text_logits, batch, length)
        zero = _zero(text_logits) + sum(
            _zero(value)
            for value in (
                predictions.validity_logits,
                predictions.boxes_3d,
                predictions.boxes_2d,
                predictions.motion_xy,
            )
        )
        supervised_text = text_mask & (text_targets != -100)
        logits, labels = text_logits[supervised_text], text_targets[supervised_text]
        _finite(logits, "supervised text logits")
        if labels.numel() and ((labels < 0).any() or (labels >= vocabulary).any()):
            raise ValueError("Supervised text target is outside vocabulary")
        text = F.cross_entropy(logits, labels) if labels.numel() else zero
        validity_mask = targets.validity_mask
        validity = targets.validity[validity_mask]
        _finite(validity, "supervised validity labels")
        if ((validity != 0) & (validity != 1)).any():
            raise ValueError("Validity labels must be explicit zero or one")
        validity_logits = predictions.validity_logits[validity_mask]
        _finite(validity_logits, "supervised validity logits")
        classification = (
            F.binary_cross_entropy_with_logits(validity_logits, validity) if validity.numel() else zero
        )
        positive = validity_mask & (targets.validity == 1)
        loc = positive & (targets.event_kinds == 0)
        mot = positive & (targets.event_kinds == 1)
        box_mask = targets.boxes_3d_mask & loc[:, None]
        box_pred, box_true = predictions.boxes_3d[box_mask], targets.boxes_3d[box_mask]
        _finite(box_pred, "supervised 3D prediction")
        _finite(box_true, "supervised 3D target")
        coordinates = torch.arange(7, device=text_logits.device).expand(events, 7)[box_mask]
        dimensions = (coordinates >= 3) & (coordinates <= 5)
        if (box_true[dimensions] <= 0).any():
            raise ValueError("Supervised target width, length and height must be positive")
        delta = box_pred - box_true
        delta = torch.where(coordinates == 6, torch.atan2(torch.sin(delta), torch.cos(delta)), delta)
        boxes = (
            F.smooth_l1_loss(delta, torch.zeros_like(delta), beta=self.smooth_l1_beta)
            if delta.numel()
            else zero
        )
        image_mask = targets.boxes_2d_mask & loc
        image_pred, image_true = predictions.boxes_2d[image_mask], targets.boxes_2d[image_mask]
        _finite(image_pred, "supervised 2D prediction")
        _finite(image_true, "supervised 2D target")
        if image_true.numel():
            if ((image_true < 0) | (image_true > 1)).any() or (image_true[:, 2:] <= image_true[:, :2]).any():
                raise ValueError("2D targets must be positive-area normalized xyxy boxes")
            if (image_pred[:, 2:] < image_pred[:, :2]).any():
                raise ValueError("2D predictions must not have inverted xyxy corners")
            # Library epsilon protects degenerate predicted boxes; it is not a
            # calibrated physical tolerance or a 3D overlap approximation.
            giou = generalized_box_iou_loss(image_pred, image_true, reduction="mean", eps=1e-7)
        else:
            giou = zero
        motion_mask = targets.motion_mask & mot[:, None]
        motion_pred, motion_true = predictions.motion_xy[motion_mask], targets.motion_xy[motion_mask]
        _finite(motion_pred, "supervised motion prediction")
        _finite(motion_true, "supervised motion target")
        motion = (
            F.smooth_l1_loss(motion_pred, motion_true, beta=self.smooth_l1_beta)
            if motion_true.numel()
            else zero
        )
        perception = (
            classification + self.box_weight * boxes + self.giou_weight * giou + self.motion_weight * motion
        )
        total = text + self.lambda_per * perception
        if not torch.isfinite(total):
            raise ValueError("Computed loss is nonfinite; check input scales and weights")
        return LossBreakdown(
            total,
            text,
            classification,
            boxes,
            giou,
            motion,
            int(labels.numel()),
            int(validity.numel()),
            int(delta.numel()),
            int(image_true.shape[0]),
            int(motion_true.shape[0]),
        )


def _validate_perception(predictions, targets, reference, batch, length):
    if not isinstance(predictions.validity_logits, Tensor) or predictions.validity_logits.ndim != 1:
        raise ValueError("Validity logits must have shape [E]")
    events = predictions.validity_logits.shape[0]
    for name in ("batch_indices", "token_positions", "event_kinds"):
        value = getattr(predictions, name)
        _integer_tensor(value, (events,), reference.device, name)
        target = getattr(targets, name)
        _integer_tensor(target, (events,), reference.device, "target " + name)
        if not torch.equal(value, target):
            raise ValueError("Prediction/target event positions or kinds differ")
    if ((predictions.batch_indices < 0) | (predictions.batch_indices >= batch)).any():
        raise ValueError("Event batch index is outside text batch")
    if ((predictions.token_positions < 0) | (predictions.token_positions >= length)).any():
        raise ValueError("Event token position is outside consumed sequence")
    if ((predictions.event_kinds < 0) | (predictions.event_kinds > 1)).any():
        raise ValueError("Event kind must be LOC=0 or MOT=1")
    keys = predictions.batch_indices * length + predictions.token_positions
    if torch.unique(keys).numel() != events:
        raise ValueError("Each consumed token can own only one perception event")
    _float_tensor(predictions.motion_xy, reference, "motion_xy")
    if (
        predictions.motion_xy.ndim != 3
        or predictions.motion_xy.shape[0] != events
        or predictions.motion_xy.shape[2] != 2
    ):
        raise ValueError("Motion prediction must have shape [E,T,2]")
    steps = predictions.motion_xy.shape[1]
    if steps <= 0:
        raise ValueError("Motion horizon must be nonempty")
    for name, shape in (("validity_logits", (events,)), ("boxes_3d", (events, 7)), ("boxes_2d", (events, 4))):
        _float_shape(getattr(predictions, name), shape, reference, name)
    for name, shape in (
        ("validity", (events,)),
        ("boxes_3d", (events, 7)),
        ("boxes_2d", (events, 4)),
        ("motion_xy", (events, steps, 2)),
    ):
        _float_shape(getattr(targets, name), shape, reference, "target " + name)
    for name, shape in (
        ("validity_mask", (events,)),
        ("boxes_3d_mask", (events, 7)),
        ("boxes_2d_mask", (events,)),
        ("motion_mask", (events, steps)),
    ):
        _mask(getattr(targets, name), shape, reference.device, name)
    return events


def _float_tensor(value, reference, name):
    if not isinstance(value, Tensor) or value.dtype not in (torch.float32, torch.float64):
        raise ValueError(name + " must be a float32/float64 tensor")
    if reference is not None and (value.dtype != reference.dtype or value.device != reference.device):
        raise ValueError(name + " must share floating dtype and device")


def _float_shape(value, shape, reference, name):
    _float_tensor(value, reference, name)
    if tuple(value.shape) != shape:
        raise ValueError(name + " has an invalid shape")


def _integer_tensor(value, shape, device, name):
    if (
        not isinstance(value, Tensor)
        or value.dtype != torch.long
        or tuple(value.shape) != shape
        or value.device != device
    ):
        raise ValueError(name + " must be an aligned long tensor")


def _mask(value, shape, device, name):
    if (
        not isinstance(value, Tensor)
        or value.dtype != torch.bool
        or tuple(value.shape) != shape
        or value.device != device
    ):
        raise ValueError(name + " must be an aligned boolean mask")


def _finite(value, name):
    if not torch.isfinite(value).all():
        raise ValueError(name + " must be finite")


def _zero(value):
    # Empty selection preserves the graph without evaluating padded NaNs.
    return value.reshape(-1)[:0].sum()
