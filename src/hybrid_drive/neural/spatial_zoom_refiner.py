"""Refine observed spatial boxes from fixed local-grid features."""

import hashlib
import json

import torch
from torch import Tensor, nn

from .spatial_box_refiner import _refine, _validate_boxes, spatial_box_refinement_loss


_BANDS = 4
_GRID = 4
_PATCH_DIM = 8
_HIDDEN_DIM = 64
_SHIFT_FRACTION = 0.25
_MAX_SHIFT = 0.125
_DEFAULT_EPOCHS = 8
_DEFAULT_BATCH_SIZE = 16
_LEARNING_RATE = 1e-3
_MAX_GRADIENT_NORM = 5.0


class ZoomBoxRefiner(nn.Module):
    """One architecture for zero, global-ROI and zoom-ROI patch grids."""

    def __init__(self, visual_dim: int):
        super().__init__()
        if type(visual_dim) is not int or visual_dim <= 0:
            raise ValueError("visual_dim must be a positive integer")
        self.visual_dim = visual_dim
        cpu = {"device": "cpu", "dtype": torch.float32}
        self.projection = nn.Sequential(
            nn.LayerNorm(visual_dim, **cpu), nn.Linear(visual_dim, _PATCH_DIM, **cpu), nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(_GRID**2 * _PATCH_DIM + 8, _HIDDEN_DIM, **cpu), nn.GELU(), nn.Linear(_HIDDEN_DIM, 4, **cpu),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, boxes: Tensor, windows: Tensor, features: Tensor, active: Tensor) -> Tensor:
        """Only PRESENT input slots can move; no scores or target values enter the head."""
        _cpu_float32(self.head[-1].weight, "Model")
        _check_inputs(boxes, windows, features, active, self.visual_dim)
        selected = active.nonzero(as_tuple=True)
        if selected[0].numel() == 0:
            return boxes.clone()

        baseline = boxes[selected]
        lo, hi = baseline.to(torch.float64).split(2, dim=-1)
        win_lo, win_hi = windows[selected].to(torch.float64).split(2, dim=-1)
        span = win_hi - win_lo
        size = hi - lo
        center = lo + size * 0.5
        geometry = torch.cat(
            (center, size, ((center - win_lo) / span).clamp(0, 1), (size / span).clamp(0, 1)), dim=-1,
        ).to(torch.float32)

        # Keep the spatial order of the 4x4 grid; pooling tokens here would
        # lose the direction of the correction inside a crop.
        tokens = self.projection(features[selected].permute(0, 2, 3, 1)).flatten(start_dim=1)
        residual = self.head(torch.cat((tokens, geometry), dim=-1))
        if not torch.isfinite(residual).all():
            raise ValueError("Zoom box residuals must be finite")
        shift_budget = (span * _SHIFT_FRACTION).clamp_max(_MAX_SHIFT)
        corrected = _refine(baseline, center, size * 0.5, residual.to(torch.float64), shift_budget, selected[1])
        return boxes.index_put(selected, corrected)


def fit_zoom_refiner(
    boxes, windows, features, active, target_boxes, target_presence,
    *, seed, epochs=_DEFAULT_EPOCHS, batch_size=_DEFAULT_BATCH_SIZE,
):
    """Fit an isolated CPU head; only active slots with a present target enter the loss."""
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be a nonnegative 63-bit integer")
    if type(epochs) is not int or not 1 <= epochs <= 32:
        raise ValueError("epochs must be an integer from 1 to 32")
    if type(batch_size) is not int or not 1 <= batch_size <= 256:
        raise ValueError("batch_size must be an integer from 1 to 256")

    _cpu_float32(features, "Crop features")
    if features.ndim != 5:
        raise ValueError("Expected crop features [B,4,D,4,4]")
    visual_dim = features.shape[2]
    _check_inputs(boxes, windows, features, active, visual_dim)
    _cpu_float32(target_boxes, "Target boxes")
    if target_boxes.shape != boxes.shape:
        raise ValueError("Expected target boxes [B,4,4]")
    _check_mask(target_presence, boxes.shape[:2], "Target presence")
    supervised = active & target_presence
    labeled = supervised.nonzero(as_tuple=True)
    if labeled[0].numel() == 0:
        raise ValueError("Expected at least one active box with a present target")
    _validate_boxes(target_boxes[labeled], labeled[1], "Supervised target boxes")

    boxes, windows, features, target_boxes = (x.detach() for x in (boxes, windows, features, target_boxes))
    order_rng = torch.Generator(device="cpu").manual_seed(seed)
    # Fork the CPU generator without touching other devices' RNG state.
    with torch.random.fork_rng(devices=[]), torch.enable_grad():
        torch.random.default_generator.manual_seed(seed)
        model = ZoomBoxRefiner(visual_dim)
        params = tuple(model.parameters())
        optimizer = torch.optim.AdamW(params, lr=_LEARNING_RATE, weight_decay=0)
        trace = []
        for epoch in range(1, epochs + 1):
            order = torch.randperm(len(boxes), generator=order_rng, device="cpu")
            weighted_loss, boxes_seen, steps, skipped, largest_norm = 0.0, 0, 0, 0, 0.0
            for indices in order.split(batch_size):
                batch_positive = supervised[indices]
                count = int(batch_positive.sum())
                if count == 0:
                    skipped += 1
                    continue
                optimizer.zero_grad(set_to_none=True)
                refined = model(boxes[indices], windows[indices], features[indices], active[indices])
                loss = spatial_box_refinement_loss(refined, target_boxes[indices], batch_positive)
                if loss is None or not torch.isfinite(loss):
                    raise ValueError("A supervised zoom batch must yield a finite loss")
                loss.backward()
                if any(param.grad is None or not torch.isfinite(param.grad).all() for param in params):
                    raise ValueError("A zoom fit produced a missing or nonfinite parameter gradient")
                norm = torch.nn.utils.clip_grad_norm_(params, _MAX_GRADIENT_NORM, error_if_nonfinite=True)
                optimizer.step()
                if any(not torch.isfinite(param).all() for param in params):
                    raise ValueError("A zoom fit produced nonfinite parameters")
                weighted_loss += float(loss.detach()) * count
                boxes_seen += count
                steps += 1
                largest_norm = max(largest_norm, float(norm))
            trace.append({
                "epoch": epoch,
                "steps": steps,
                "skipped_batches": skipped,
                "supervised_boxes": boxes_seen,
                "mean_loss": weighted_loss / boxes_seen,
                "max_gradient_norm": largest_norm,
                "order_sha256": hashlib.sha256(json.dumps(order.tolist(), separators=(",", ":")).encode()).hexdigest(),
            })
        optimizer.zero_grad(set_to_none=True)
    return model.eval(), trace


def zoom_head_protocol():
    """Exact fixed defaults for paired CPU runs and their receipts."""
    return {
        "device": "cpu",
        "dtype": "float32",
        "grid": [_GRID, _GRID],
        "projected_channels": _PATCH_DIM,
        "hidden_dim": _HIDDEN_DIM,
        "geometry": "absolute and crop-relative center and size",
        "initialization": "zero last linear residual",
        "shift_fraction_of_window": _SHIFT_FRACTION,
        "max_normalized_shift": _MAX_SHIFT,
        "width_height_scale_before_border": [0.5, 2.0],
        "epochs": _DEFAULT_EPOCHS,
        "batch_size": _DEFAULT_BATCH_SIZE,
        "optimizer": "AdamW",
        "learning_rate": _LEARNING_RATE,
        "weight_decay": 0.0,
        "max_gradient_norm": _MAX_GRADIENT_NORM,
        "loss": "mean GIoU plus 5 * SmoothL1 (beta = 1/32)",
        "loss_mask": "active & target_presence",
        "mean_loss_weighting": "supervised boxes",
        "training_order": "CPU frame permutation, independent generator initialized to fit seed",
    }


def _cpu_float32(value, name):
    if not isinstance(value, Tensor) or value.dtype != torch.float32 or value.device.type != "cpu":
        raise ValueError(f"{name} must be a CPU float32 tensor")


def _check_mask(value, shape, name):
    if not isinstance(value, Tensor) or value.dtype != torch.bool or value.device.type != "cpu" or value.shape != shape:
        raise ValueError(f"{name} must be a CPU boolean mask [B,4]")


def _check_inputs(boxes, windows, features, active, visual_dim):
    _cpu_float32(boxes, "Baseline boxes")
    _cpu_float32(windows, "Crop windows")
    _cpu_float32(features, "Crop features")
    if boxes.ndim != 3 or boxes.shape[1:] != (_BANDS, 4) or len(boxes) == 0:
        raise ValueError("Expected nonempty baseline boxes [B,4,4]")
    if windows.shape != boxes.shape or visual_dim <= 0 or features.shape != (*boxes.shape[:2], visual_dim, _GRID, _GRID):
        raise ValueError("Expected windows [B,4,4] and crop features [B,4,D,4,4] aligned to boxes")
    _check_mask(active, boxes.shape[:2], "Active")
    _validate_boxes(boxes, torch.arange(_BANDS, device="cpu"), "Baseline boxes")
    if not torch.isfinite(windows).all() or not torch.isfinite(features).all():
        raise ValueError("Crop windows and features must be finite")
    visible = windows[active]
    if ((visible < 0) | (visible > 1)).any() or (visible[:, 2:] <= visible[:, :2]).any():
        raise ValueError("Active windows must be positive-area normalized xyxy boxes")
    base = boxes[active]
    tolerance = 2 * torch.finfo(torch.float32).eps
    if (visible[:, :2] > base[:, :2] + tolerance).any() or (visible[:, 2:] < base[:, 2:] - tolerance).any():
        raise ValueError("Active crop windows must contain their baseline boxes")
