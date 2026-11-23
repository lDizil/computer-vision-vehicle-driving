"""Choose local box corrections from out-of-fold zoom predictions."""

from dataclasses import dataclass

import torch
from torch import Tensor

from .spatial_box_refiner import _BANDS, _box_bands, _validate_boxes


_SCHEMA = "hybrid-drive-spatial-zoom-guard-v1"
_FEATURES = (
    "baseline_center_x", "baseline_center_y",
    "baseline_log_width", "baseline_log_height",
    "asinh_dx_in_baseline_width", "asinh_dy_in_baseline_height",
    "log_width_ratio", "log_height_ratio",
    "asinh_source_score",
)
_FRACTIONS = (0.0, 0.5, 1.0)
_SIZE_FLOOR = 1 / 4096
_SCALE_FLOOR = 0.02
_RIDGE_LAMBDA = 0.10
_GAIN_FLOOR = 0.02
_SUPPORT_MARGIN = 0.5
_MAX_SUPERVISED_BOXES = 2**63 - 1
# FP32 scores are compressed by asinh, so even their extremes fit these bounds.
_MAX_FEATURE = 100.0
_MAX_STANDARDIZED = 2 * _MAX_FEATURE / _SCALE_FLOOR
# A squared-error ridge fit of gains in [-1,1] has ||w|| <= 1/sqrt(lambda).
_MAX_WEIGHT = 4.0


@dataclass(frozen=True, slots=True)
class ZoomGuard:
    """An immutable ridge model with one gain prediction per nonzero fraction."""

    supervised_boxes: int
    feature_mean: tuple[float, ...]
    feature_scale: tuple[float, ...]
    weights: tuple[tuple[float, ...], ...]
    intercept: tuple[float, ...]
    support_min: tuple[float, ...]
    support_max: tuple[float, ...]

    def __post_init__(self):
        if type(self.supervised_boxes) is not int or not 1 <= self.supervised_boxes <= _MAX_SUPERVISED_BOXES:
            raise ValueError("Invalid number of supervised boxes")
        size = len(_FEATURES)
        if type(self.weights) not in (tuple, list) or len(self.weights) != len(_FRACTIONS) - 1:
            raise ValueError("Expected two gain weight vectors")
        weights = tuple(
            _checked_vector(row, size, -_MAX_WEIGHT, _MAX_WEIGHT, "weights") for row in self.weights
        )
        object.__setattr__(self, "weights", weights)
        for name, length, low, high in (
            ("feature_mean", size, -_MAX_FEATURE, _MAX_FEATURE),
            ("feature_scale", size, _SCALE_FLOOR, _MAX_FEATURE),
            ("intercept", len(weights), -1.0, 1.0),
            ("support_min", size, -_MAX_STANDARDIZED, _MAX_STANDARDIZED),
            ("support_max", size, -_MAX_STANDARDIZED, _MAX_STANDARDIZED),
        ):
            value = _checked_vector(getattr(self, name), length, low, high, name)
            object.__setattr__(self, name, value)
        if any(lo > hi for lo, hi in zip(self.support_min, self.support_max)):
            raise ValueError("Invalid training support interval")

    def to_dict(self) -> dict:
        return {
            "schema": _SCHEMA,
            "supervised_boxes": self.supervised_boxes,
            "feature_mean": list(self.feature_mean),
            "feature_scale": list(self.feature_scale),
            "weights": [list(row) for row in self.weights],
            "intercept": list(self.intercept),
            "support_min": list(self.support_min),
            "support_max": list(self.support_max),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ZoomGuard":
        if (
            type(data) is not dict
            or data.keys() != {
                "schema", "supervised_boxes", "feature_mean", "feature_scale",
                "weights", "intercept", "support_min", "support_max",
            }
            or type(data["schema"]) is not str
            or data["schema"] != _SCHEMA
        ):
            raise ValueError("Invalid zoom guard schema")
        return cls(**{key: value for key, value in data.items() if key != "schema"})


def zoom_guard_protocol() -> dict:
    return {
        "schema": _SCHEMA,
        "device": "cpu",
        "box_and_score_dtype": "float32",
        "fit_dtype": "float64",
        "features": list(_FEATURES),
        "geometry_size_floor": _SIZE_FLOOR,
        "population_std_floor": _SCALE_FLOOR,
        "fractions": list(_FRACTIONS),
        "target": "IoU(mixed_box, same_band_truth) - IoU(baseline_box, same_band_truth)",
        "supervision": "active & target_presence, with out-of-fold proposals supplied by the caller",
        "ridge": {"lambda": _RIDGE_LAMBDA, "normalization": "mean_squared_error", "penalize_intercept": False},
        "minimum_gain_over_baseline": _GAIN_FLOOR,
        "minimum_full_gain_over_half": _GAIN_FLOOR,
        "support": {"kind": "per_feature_training_min_max", "margin_in_standardized_units": _SUPPORT_MARGIN},
        "interpolation": "normalized_xyxy_float64_then_float32_with_unrepresentable_result_reverting_to_baseline",
    }


@torch.no_grad()
def mix_zoom_boxes(baseline: Tensor, proposal: Tensor, active: Tensor, fraction: float) -> Tensor:
    """Interpolate active boxes, keeping exact endpoints and canonical bands."""
    if type(fraction) not in (int, float) or not 0.0 <= fraction <= 1.0:
        raise ValueError("Expected a finite mix fraction in [0,1]")
    _check_box_inputs(baseline, proposal, active)
    slots = active.nonzero(as_tuple=True)
    mixed = baseline.clone()
    mixed[slots] = _mix_rows(baseline[slots], proposal[slots], slots[1], fraction)
    return mixed


@torch.no_grad()
def fit_zoom_guard(
    baseline: Tensor, proposal: Tensor, active: Tensor, target_boxes: Tensor, target_presence: Tensor, scores: Tensor,
) -> ZoomGuard:
    """Fit half/full IoU gains; ignored slots need not carry target coordinates."""
    _check_box_inputs(baseline, proposal, active)
    _cpu_float32(target_boxes, "Target boxes")
    if target_boxes.shape != baseline.shape:
        raise ValueError("Expected target boxes [B,4,4] matching the baseline")
    _check_mask(target_presence, active.shape, "Target presence")
    slots = (active & target_presence).nonzero(as_tuple=True)
    count = slots[0].numel()
    if count == 0:
        raise ValueError("Expected at least one active box with a present target")
    truth = target_boxes[slots]
    _validate_boxes(truth, slots[1], "Supervised target boxes")
    source, zoom = baseline[slots], proposal[slots]
    features = _features(source, zoom, _selected_scores(scores, active.shape, slots))
    half = _mix_rows(source, zoom, slots[1], _FRACTIONS[1])
    iou_baseline = _iou(source, truth)
    targets = torch.stack((_iou(half, truth) - iou_baseline, _iou(zoom, truth) - iou_baseline), dim=1)

    mean = features.mean(dim=0)
    centered = features - mean
    scale = centered.square().mean(dim=0).sqrt().clamp_min(_SCALE_FLOOR)
    standardized = centered / scale
    intercept = targets.mean(dim=0)
    ridge = torch.eye(len(_FEATURES), dtype=torch.float64, device="cpu") * _RIDGE_LAMBDA
    gram = standardized.T @ standardized / count + ridge
    rhs = standardized.T @ (targets - intercept) / count
    weights = torch.linalg.solve(gram, rhs).T

    return ZoomGuard(
        supervised_boxes=count,
        feature_mean=mean.tolist(),
        feature_scale=scale.tolist(),
        weights=weights.tolist(),
        intercept=intercept.tolist(),
        support_min=standardized.amin(dim=0).tolist(),
        support_max=standardized.amax(dim=0).tolist(),
    )


@torch.no_grad()
def apply_zoom_guard(
    guard: ZoomGuard, baseline: Tensor, proposal: Tensor, active: Tensor, scores: Tensor, *, require_support: bool = False,
) -> tuple[Tensor, Tensor]:
    """Return boxes and actually applied fractions; presence is not changed."""
    if not isinstance(guard, ZoomGuard):
        raise ValueError("Expected a ZoomGuard")
    if type(require_support) is not bool:
        raise ValueError("require_support must be boolean")
    _check_box_inputs(baseline, proposal, active)
    slots = active.nonzero(as_tuple=True)
    selected_scores = _selected_scores(scores, active.shape, slots)
    boxes = baseline.clone()
    fractions = baseline.new_zeros(active.shape)
    if slots[0].numel() == 0:
        return boxes, fractions

    source, zoom = baseline[slots], proposal[slots]
    features = _features(source, zoom, selected_scores)
    z = (features - features.new_tensor(guard.feature_mean)) / features.new_tensor(guard.feature_scale)
    gains = z @ z.new_tensor(guard.weights).T + z.new_tensor(guard.intercept)
    half_gain, full_gain = gains.unbind(dim=1)
    take_half = half_gain >= _GAIN_FLOOR
    take_full = (full_gain >= _GAIN_FLOOR) & (full_gain >= half_gain + _GAIN_FLOOR)
    if require_support:
        lower = z.new_tensor(guard.support_min) - _SUPPORT_MARGIN
        upper = z.new_tensor(guard.support_max) + _SUPPORT_MARGIN
        supported = ((z >= lower) & (z <= upper)).all(dim=1)
        take_half &= supported
        take_full &= supported

    half = _mix_rows(source, zoom, slots[1], _FRACTIONS[1])
    adjusted = torch.where(take_full[:, None], zoom, torch.where(take_half[:, None], half, source))
    selected_fractions = take_full.float() + (take_half & ~take_full).float() * _FRACTIONS[1]
    boxes[slots] = adjusted
    fractions[slots] = selected_fractions.masked_fill((adjusted == source).all(dim=1), 0.0)
    return boxes, fractions


def _check_box_inputs(baseline: Tensor, proposal: Tensor, active: Tensor) -> None:
    _cpu_float32(baseline, "Baseline boxes")
    _cpu_float32(proposal, "Proposal boxes")
    if baseline.ndim != 3 or baseline.shape[1:] != (_BANDS, 4) or len(baseline) == 0:
        raise ValueError("Expected nonempty baseline boxes [B,4,4]")
    if proposal.shape != baseline.shape:
        raise ValueError("Expected proposal boxes [B,4,4] matching the baseline")
    _check_mask(active, baseline.shape[:2], "Active")
    bands = torch.arange(_BANDS, device="cpu")
    _validate_boxes(baseline, bands, "Baseline boxes")
    _validate_boxes(proposal, bands, "Proposal boxes")


def _mix_rows(baseline: Tensor, proposal: Tensor, bands: Tensor, fraction: float) -> Tensor:
    if fraction == 0.0:
        return baseline
    if fraction == 1.0:
        return proposal
    mixed = torch.lerp(baseline.double(), proposal.double(), fraction).float()
    representable = (mixed[:, 2:] > mixed[:, :2]).all(dim=1) & (_box_bands(mixed) == bands)
    return torch.where(representable[:, None], mixed, baseline)


def _features(baseline: Tensor, proposal: Tensor, scores: Tensor) -> Tensor:
    lo, hi = baseline.double().split(2, dim=1)
    zoom_lo, zoom_hi = proposal.double().split(2, dim=1)
    size, zoom_size = hi - lo, zoom_hi - zoom_lo
    center, zoom_center = lo + size * 0.5, zoom_lo + zoom_size * 0.5
    safe_size = size.clamp_min(_SIZE_FLOOR)
    log_size = safe_size.log()
    return torch.cat((
        center,
        log_size,
        ((zoom_center - center) / safe_size).asinh(),
        zoom_size.clamp_min(_SIZE_FLOOR).log() - log_size,
        scores.double().asinh().unsqueeze(1),
    ), dim=1)


def _iou(left: Tensor, right: Tensor) -> Tensor:
    left_lo, left_hi = left.double().split(2, dim=1)
    right_lo, right_hi = right.double().split(2, dim=1)
    intersection = (torch.minimum(left_hi, right_hi) - torch.maximum(left_lo, right_lo)).clamp_min(0).prod(dim=1)
    union = (left_hi - left_lo).prod(dim=1) + (right_hi - right_lo).prod(dim=1) - intersection
    return (intersection / union).clamp(0, 1)


def _selected_scores(scores: Tensor, shape, slots) -> Tensor:
    _cpu_float32(scores, "Source scores")
    if scores.shape != shape:
        raise ValueError("Expected source scores [B,4] aligned with the boxes")
    values = scores[slots]
    if not torch.isfinite(values).all():
        raise ValueError("Selected source scores must be finite")
    return values


def _cpu_float32(value: Tensor, name: str):
    if (
        not isinstance(value, Tensor) or value.device.type != "cpu"
        or value.dtype != torch.float32 or value.layout != torch.strided
    ):
        raise ValueError(f"{name} must be a CPU float32 tensor")


def _check_mask(value: Tensor, shape, name: str):
    if (
        not isinstance(value, Tensor) or value.device.type != "cpu" or value.dtype != torch.bool
        or value.layout != torch.strided or value.shape != shape
    ):
        raise ValueError(f"{name} must be a CPU boolean mask [B,4]")


def _checked_vector(value, length: int, low: float, high: float, name: str) -> tuple[float, ...]:
    if (
        type(value) not in (list, tuple)
        or len(value) != length
        or any(type(number) not in (int, float) or not low <= number <= high for number in value)
    ):
        raise ValueError(f"Invalid zoom guard {name}")
    return tuple(map(float, value))
