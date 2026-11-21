from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn
from torchvision.ops import roi_align


_DEFAULT_GEOMETRY_FEATURE_NAMES = ("x", "y", "z", "width", "length", "height", "yaw")


@dataclass(frozen=True, eq=False)
class PriorTokens:
    """Tensor references retain ordinary PyTorch mutability and autograd semantics."""

    tokens: Tensor
    valid_mask: Tensor


class PriorTokenizer(nn.Module):
    """Project-owned trainable tokenizer, not an official pretrained model.

    Each normalized ROI is in the preprocessed image coordinate system matching
    its feature map. Geometry is a separate input, never inferred from that ROI.
    The default geometry is ego-frame (x, y, z, width, length, height, yaw), in
    metres/radians; other geometry dimensions/meanings belong to the caller.

    Our architecture uses a shared learned 49-to-1 spatial projection, a
    Linear/GELU geometry embedding, and a two-linear-layer GELU fusion MLP.
    All weights are newly initialized. Floating inputs and module parameters
    must share a float32/float64 dtype and device. No input tensors are copied
    or detached to make them immutable.
    """

    def __init__(self, image_channels: int, *, geometry_dim: int = 7,
                 token_dim: int = 768, hidden_dim: int = 768,
                 geometry_feature_names: Sequence[str] | None = None):
        super().__init__()
        self.image_channels = _positive_dimension(image_channels, "image_channels")
        self.geometry_dim = _positive_dimension(geometry_dim, "geometry_dim")
        self.token_dim = _positive_dimension(token_dim, "token_dim")
        self.hidden_dim = _positive_dimension(hidden_dim, "hidden_dim")
        if geometry_feature_names is None and self.geometry_dim == 7:
            geometry_feature_names = _DEFAULT_GEOMETRY_FEATURE_NAMES
        if geometry_feature_names is not None:
            if isinstance(geometry_feature_names, str):
                raise ValueError("geometry_feature_names must be a sequence of feature names")
            geometry_feature_names = tuple(geometry_feature_names)
            if (len(geometry_feature_names) != self.geometry_dim
                    or any(not isinstance(name, str) or not name.strip() for name in geometry_feature_names)
                    or len(set(geometry_feature_names)) != self.geometry_dim):
                raise ValueError("geometry_feature_names must contain geometry_dim distinct nonempty names")
        self.geometry_feature_names = geometry_feature_names
        self.spatial_pool = nn.Linear(49, 1, bias=False)
        self.geometry_embedding = nn.Sequential(nn.Linear(self.geometry_dim, self.hidden_dim), nn.GELU())
        self.fusion = nn.Sequential(nn.Linear(self.image_channels + self.hidden_dim, self.hidden_dim),
                                    nn.GELU(), nn.Linear(self.hidden_dim, self.token_dim))

    def forward(self, feature_maps: Tensor, boxes_xyxy: Tensor, image_indices: Tensor,
                geometry: Tensor, valid_mask: Tensor, *,
                image_valid_mask: Tensor | None = None) -> PriorTokens:
        """Tokenize [B,N] priors from [B,V,C,H,W] maps; padded priors are zero.

        Active boxes must have positive area and coordinates in [0,1], and
        select an available image. Padded boxes, geometry and image indices
        are ignored, including NaNs and out-of-range indices. N may be zero.
        """
        batch_size, views, height, width = self._validate_inputs(
            feature_maps, boxes_xyxy, image_indices, geometry, valid_mask, image_valid_mask)
        batch_indices, prior_indices = valid_mask.nonzero(as_tuple=True)
        selected_images = image_indices[batch_indices, prior_indices]
        if torch.any((selected_images < 0) | (selected_images >= views)):
            raise ValueError("Active image_indices must be in [0, V)")
        if image_valid_mask is not None and torch.any(~image_valid_mask[batch_indices, selected_images]):
            raise ValueError("Active priors must not reference unavailable images")

        boxes = boxes_xyxy[batch_indices, prior_indices]
        selected_geometry = geometry[batch_indices, prior_indices]
        _validate_active_priors(boxes, selected_geometry)
        scale = boxes.new_tensor((width, height, width, height))
        source_indices = batch_indices * views + selected_images
        rois = torch.cat((source_indices[:, None].to(feature_maps.dtype), boxes * scale), dim=1)
        roi_features = roi_align(feature_maps.flatten(0, 1), rois, output_size=(7, 7),
                                 spatial_scale=1.0, sampling_ratio=2, aligned=True)
        if not torch.isfinite(roi_features).all():
            raise ValueError("Active ROI features must be finite")
        visual = self.spatial_pool(roi_features.flatten(start_dim=2)).squeeze(-1)
        embedded_geometry = self.geometry_embedding(selected_geometry)
        active_tokens = self.fusion(torch.cat((visual, embedded_geometry), dim=-1))
        tokens = feature_maps.new_zeros((batch_size, valid_mask.shape[1], self.token_dim))
        tokens = tokens.index_put((batch_indices, prior_indices), active_tokens)
        return PriorTokens(tokens, valid_mask)

    def _validate_inputs(self, feature_maps: Tensor, boxes_xyxy: Tensor, image_indices: Tensor,
                         geometry: Tensor, valid_mask: Tensor,
                         image_valid_mask: Tensor | None) -> tuple[int, int, int, int]:
        if not isinstance(feature_maps, Tensor) or feature_maps.ndim != 5:
            raise ValueError("feature_maps must have shape [B,V,C,H,W]")
        batch_size, views, channels, height, width = feature_maps.shape
        if min(feature_maps.shape) <= 0 or channels != self.image_channels:
            raise ValueError("feature_maps dimensions must be positive with C = image_channels")
        if feature_maps.dtype not in (torch.float32, torch.float64):
            raise ValueError("feature_maps must use float32 or float64")
        if not isinstance(valid_mask, Tensor) or valid_mask.ndim != 2 or valid_mask.shape[0] != batch_size:
            raise ValueError("valid_mask must have shape [B,N]")
        count = valid_mask.shape[1]
        for name, value, shape, dtype in (
            ("valid_mask", valid_mask, (batch_size, count), torch.bool),
            ("boxes_xyxy", boxes_xyxy, (batch_size, count, 4), feature_maps.dtype),
            ("image_indices", image_indices, (batch_size, count), torch.long),
            ("geometry", geometry, (batch_size, count, self.geometry_dim), feature_maps.dtype),
        ):
            _validate_tensor(value, name, shape, dtype, feature_maps.device)
        if image_valid_mask is not None:
            _validate_tensor(image_valid_mask, "image_valid_mask", (batch_size, views),
                             torch.bool, feature_maps.device)
        parameter = self.spatial_pool.weight
        if parameter.device != feature_maps.device or parameter.dtype != feature_maps.dtype:
            raise ValueError("feature_maps must match the module parameter dtype and device")
        return batch_size, views, height, width


def _positive_dimension(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_tensor(value: Tensor, name: str, shape: tuple[int, ...],
                     dtype: torch.dtype, device: torch.device) -> None:
    if not isinstance(value, Tensor) or tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if value.dtype != dtype:
        raise ValueError(f"{name} must have dtype {dtype}")
    if value.device != device:
        raise ValueError(f"{name} must be on the feature_maps device")


def _validate_active_priors(boxes: Tensor, geometry: Tensor) -> None:
    if not torch.isfinite(boxes).all() or not torch.isfinite(geometry).all():
        raise ValueError("Active boxes_xyxy and geometry must be finite")
    if torch.any((boxes < 0) | (boxes > 1)):
        raise ValueError("Active boxes_xyxy coordinates must be in [0,1]")
    if torch.any(boxes[:, 2:] <= boxes[:, :2]):
        raise ValueError("Active boxes_xyxy must have positive width and height")
