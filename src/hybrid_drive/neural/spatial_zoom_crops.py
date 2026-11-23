"""Frozen DINO views of existing visible boxes, with no target inputs.

The zoom view resamples the prepared 224 px RGB, not the native camera
image. It gives the local region its own DINO patch lattice, not new pixels.
"""

import torch
from torch import Tensor, nn
from torch.nn import functional as F


_BANDS = 4
_IMAGE_SIDE = 224
_POOL_SIDE = 4
_CONTEXT_SCALE = 2.0
_MIN_WINDOW_SIDE = 0.18
_MAX_WINDOW_SIDE = 1.0
_MAX_FRAME_BATCH = 16
_MAX_ENCODER_BATCH = 8
_MAX_VISUAL_DIM = 1024
_MAX_PATCH_SIDE = 32


def zoom_protocol() -> dict:
    return {
        "schema": "hybrid-drive-frozen-spatial-zoom-input-v1",
        "device": "cpu",
        "dtype": "float32",
        "window": {
            "shape": "square",
            "side_multiplier_of_longer_box_extent": _CONTEXT_SCALE,
            "side_limits_normalized": [_MIN_WINDOW_SIDE, _MAX_WINDOW_SIDE],
            "placement": "center_on_baseline_then_shift_entire_window_inside_image",
        },
        "sampling": {
            "mode": "bilinear",
            "padding_mode": "border",
            "align_corners": False,
            "coordinates": "normalized_xyxy_edges_with_regular_pixel_center_sampling",
        },
        "features": {
            "global": "encode_full_frame_then_resample_its_patch_grid_to_native_grid_dimensions_in_window",
            "zoom": "resample_prepared_rgb_window_to_224x224_then_encode_with_same_frozen_model",
            "pool": "adaptive_avg_pool2d",
            "pool_size": [_POOL_SIDE, _POOL_SIDE],
        },
        "image_side": _IMAGE_SIDE,
        "present_only": True,
        "slots": "row-major_frame_then_band_inactive_exact_zero",
        "max_frame_batch": _MAX_FRAME_BATCH,
        "max_encoder_batch": _MAX_ENCODER_BATCH,
    }


def make_zoom_windows(boxes: Tensor, active: Tensor) -> Tensor:
    """Return one normalized crop window per selected slot, zero otherwise."""
    _check_boxes(boxes, active)
    selected = boxes[active].to(torch.float64)
    lower, upper = selected.split(2, dim=-1)
    extent = upper - lower
    side = (extent.amax(dim=-1, keepdim=True) * _CONTEXT_SCALE).clamp(_MIN_WINDOW_SIDE, _MAX_WINDOW_SIDE)
    centered = lower + (extent - side) * 0.5
    start = torch.minimum(centered.clamp_min(0), 1 - side)
    windows = torch.zeros_like(boxes)
    windows[active] = torch.cat((start, start + side), dim=-1).to(boxes.dtype)
    return windows


@torch.no_grad()
def encode_zoom_pair(vision: nn.Module, rgb: Tensor, boxes: Tensor, active: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Return windows and global/zoom grids [B,4,D,4,4] for the same slots.

    Only frames with a selected slot reach the full-frame encoder. Individual
    zoom inputs are ordered by frame, then band. No encoder call or dummy
    observation is made for inactive slots.
    """
    windows = make_zoom_windows(boxes, active)
    _check_rgb(rgb, len(boxes))
    dim = _check_vision(vision)
    full = boxes.new_zeros((len(boxes), _BANDS, dim, _POOL_SIDE, _POOL_SIDE))
    zoom = torch.zeros_like(full)
    slots = active.nonzero(as_tuple=False)
    if not len(slots):
        return windows, full, zoom

    frame_indices = active.any(dim=1).nonzero(as_tuple=True)[0]
    output_grids = []
    patch_shape = None
    for frame_batch in frame_indices.split(_MAX_ENCODER_BATCH):
        maps = _grid(vision, rgb[frame_batch], dim, expected_shape=patch_shape)
        patch_shape = maps.shape[-2:]
        output_grids.append(maps)
    patch_maps = torch.cat(output_grids, dim=0)
    packed_index = torch.full((len(rgb),), -1, dtype=torch.long, device=rgb.device)
    packed_index[frame_indices] = torch.arange(len(frame_indices), device=rgb.device)

    for slot_batch in slots.split(_MAX_ENCODER_BATCH):
        frames, bands = slot_batch.unbind(dim=1)
        regions = windows[frames, bands]
        # Re-sample the full-frame features at the same crop-relative lattice
        # as the zoom features before applying identical pooling.
        sampled = _sample(patch_maps[packed_index[frames]], regions, patch_shape)
        cropped = _sample(rgb[frames, 0], regions, (_IMAGE_SIDE, _IMAGE_SIDE))
        # Keep the encoder's strict [0,1] contract despite interpolation rounding.
        zoom_grid = _grid(vision, cropped.clamp_(0, 1).unsqueeze(1), dim, expected_shape=patch_shape)
        full[frames, bands] = _pool(sampled)
        zoom[frames, bands] = _pool(zoom_grid)

    _check_vision(vision)
    return windows, full, zoom


def _sample(source, regions, size):
    height, width = size
    xx = (torch.arange(width, device=source.device, dtype=source.dtype) + 0.5) / width
    yy = (torch.arange(height, device=source.device, dtype=source.dtype) + 0.5) / height
    x0, y0, x1, y1 = regions.unbind(dim=1)
    x = x0[:, None] + (x1 - x0)[:, None] * xx
    y = y0[:, None] + (y1 - y0)[:, None] * yy
    grid = torch.stack((x[:, None, :].expand(-1, height, -1), y[:, :, None].expand(-1, -1, width)), dim=-1)
    return F.grid_sample(source, 2 * grid - 1, mode="bilinear", padding_mode="border", align_corners=False)


def _pool(patch_grid):
    output = F.adaptive_avg_pool2d(patch_grid, (_POOL_SIDE, _POOL_SIDE))
    if not torch.isfinite(output).all():
        raise ValueError("Pooled zoom input features must be finite")
    return output


def _require_cpu_float32_no_grad(value, name):
    if (
        not isinstance(value, Tensor) or value.layout != torch.strided or value.dtype != torch.float32
        or value.device.type != "cpu" or value.requires_grad
    ):
        raise ValueError(f"Expected {name} as a dense CPU float32 tensor without gradients")


def _check_boxes(boxes, active):
    _require_cpu_float32_no_grad(boxes, "boxes")
    if boxes.ndim != 3 or boxes.shape[1:] != (_BANDS, 4) or not 1 <= len(boxes) <= _MAX_FRAME_BATCH:
        raise ValueError("Expected normalized xyxy boxes [B,4,4] with 1–16 frames")
    if (
        not isinstance(active, Tensor) or active.layout != torch.strided or active.dtype != torch.bool
        or active.device.type != "cpu" or active.shape != boxes.shape[:2]
    ):
        raise ValueError("Expected a dense CPU boolean active mask [B,4]")
    extents = boxes[..., 2:] - boxes[..., :2]
    if (
        not torch.isfinite(boxes).all() or (boxes < 0).any() or (boxes > 1).any()
        or (extents < 0).any() or (extents[active] <= 0).any()
    ):
        raise ValueError("Boxes must be finite normalized xyxy, with positive area in every active slot")


def _check_rgb(rgb, batch_size):
    _require_cpu_float32_no_grad(rgb, "RGB")
    if rgb.shape != (batch_size, 1, 3, _IMAGE_SIDE, _IMAGE_SIDE):
        raise ValueError("Expected prepared RGB images [B,1,3,224,224] aligned to boxes")
    if not torch.isfinite(rgb).all() or (rgb < 0).any() or (rgb > 1).any():
        raise ValueError("RGB image values must be finite and in [0,1]")


def _check_vision(vision):
    if not isinstance(vision, nn.Module) or any(module.training for module in vision.modules()):
        raise ValueError("Zoom requires a frozen vision module in eval mode")
    if any(p.requires_grad or p.grad is not None or p.device.type != "cpu" or p.dtype != torch.float32
           for p in vision.parameters()):
        raise ValueError("Vision parameters must be CPU float32, frozen and without gradients")
    if any(value.requires_grad or value.device.type != "cpu" for value in vision.buffers()):
        raise ValueError("Vision buffers must stay on CPU without gradients")
    dim = getattr(vision, "visual_dim", None)
    if type(dim) is not int or not 1 <= dim <= _MAX_VISUAL_DIM:
        raise ValueError("Vision must declare a positive bounded visual_dim")
    return dim


def _grid(vision, images, dim, *, expected_shape=None):
    grids = vision(images)
    _require_cpu_float32_no_grad(grids, "vision patch grid")
    if (
        grids.ndim != 5 or grids.shape[:3] != (len(images), 1, dim)
        or not all(_POOL_SIDE <= side <= _MAX_PATCH_SIDE for side in grids.shape[-2:])
        or (expected_shape is not None and grids.shape[-2:] != expected_shape)
        or not torch.isfinite(grids).all()
    ):
        raise ValueError("Expected finite matching vision patch grids [B,1,D,H,W]")
    return grids[:, 0]
