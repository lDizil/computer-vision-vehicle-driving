from __future__ import annotations

import torch
from torch import Tensor, nn
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models.vision_transformer import VisionTransformer


class ViTPatchEncoder(nn.Module):
    """Expose official torchvision ViT patch tokens as spatial feature maps.

    The public FX extractor reuses the supplied encoder's parameters; no
    weights are loaded, frozen or detached. The caller controls train/eval.
    Images must already have the encoder's size and required preprocessing:
    this wrapper does not resize, crop or normalize them. The single CLS token
    is dropped; it has no location in the returned patch grid.
    """

    def __init__(self, encoder: VisionTransformer):
        super().__init__()
        if not isinstance(encoder, VisionTransformer):
            raise ValueError("encoder must be a torchvision VisionTransformer")
        for name in ("image_size", "patch_size", "hidden_dim"):
            value = getattr(encoder, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"encoder.{name} must be a positive integer")
        if encoder.image_size % encoder.patch_size:
            raise ValueError("encoder.image_size must be divisible by encoder.patch_size")
        self.image_size = encoder.image_size
        self.hidden_dim = encoder.hidden_dim
        self.grid_size = encoder.image_size // encoder.patch_size
        self.extractor = create_feature_extractor(encoder, return_nodes={"encoder.ln": "tokens"})

    def forward(self, images: Tensor) -> Tensor:
        """Map finite [B,V,3,S,S] images to [B,V,hidden_dim,S/P,S/P]."""
        if not isinstance(images, Tensor) or images.ndim != 5:
            raise ValueError("images must have shape [B,V,3,image_size,image_size]")
        batch_size, views, channels, height, width = images.shape
        if (batch_size <= 0 or views <= 0
                or (channels, height, width) != (3, self.image_size, self.image_size)):
            raise ValueError("images must have positive B,V and shape [B,V,3,image_size,image_size]")
        if not images.is_floating_point():
            raise ValueError("images must have a floating dtype")
        parameter = next(self.extractor.parameters())
        if images.dtype != parameter.dtype or images.device != parameter.device:
            raise ValueError("images must match the encoder parameter dtype and device")
        if not torch.isfinite(images).all():
            raise ValueError("images must be finite")
        tokens = self.extractor(images.flatten(0, 1))["tokens"]
        expected_shape = (batch_size * views, self.grid_size * self.grid_size + 1, self.hidden_dim)
        if tuple(tokens.shape) != expected_shape:
            raise ValueError(f"encoder.ln tokens must have shape {expected_shape}, including one CLS token")
        patches = tokens[:, 1:, :].transpose(1, 2)
        return patches.reshape(batch_size, views, self.hidden_dim, self.grid_size, self.grid_size)
