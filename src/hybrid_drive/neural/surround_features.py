"""Encode each surround view once, then give Fast and Bridge the same maps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, TypeVar

import numpy as np
from PIL import Image
import torch
from torch import Tensor

from hybrid_drive.contracts._validation import identifier
from hybrid_drive.contracts.sensors import CameraSnapshot, SURROUND_ORDER
from hybrid_drive.geometry.camera import CameraCalibration


@dataclass(frozen=True, eq=False)
class SurroundFeatures:
    """Maps [1,6,C,H,W], in snapshot order, with preprocessed pixel calibrations.

    Consumers borrow the tensor read-only. The calibrations refer to encoder
    input pixels, not to patch cells; the support grid has its own shape.
    """
    snapshot: CameraSnapshot
    encoder_id: str
    feature_maps: Tensor
    calibrations: tuple[CameraCalibration, ...]


class SharedSurroundEncoder:
    """Frozen inference adapter for [B,1,3,H,W] -> [B,1,C,Gh,Gw] encoders.

    Treat the six source views as one batch without a second encoder pass.
    Resize the entire rectified view, with matching pinhole intrinsics.
    """

    def __init__(self, patch_encoder: Callable[[Tensor], Tensor], *, encoder_id: str,
                 resize_wh: tuple[int, int] = (224, 224)):
        if not callable(patch_encoder):
            raise ValueError("patch_encoder must be callable")
        if (not isinstance(resize_wh, tuple) or len(resize_wh) != 2
                or any(isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= 1024 for n in resize_wh)):
            raise ValueError("resize_wh must be a (width, height) pair of integers in [1, 1024]")
        self._patch_encoder = patch_encoder
        self._encoder_id = identifier(encoder_id, "encoder_id")
        self._resize_wh = resize_wh

    @torch.no_grad()
    def encode(self, snapshot: CameraSnapshot) -> SurroundFeatures:
        if not isinstance(snapshot, CameraSnapshot):
            raise ValueError("Expected a calibrated CameraSnapshot")
        rasters = np.stack([
            np.asarray(Image.fromarray(view.rgb).resize(self._resize_wh, Image.Resampling.BILINEAR))
            for view in snapshot.views
        ])
        pixels = torch.from_numpy(rasters).permute(0, 3, 1, 2).contiguous().float().div_(255).unsqueeze(1)
        output = self._patch_encoder(pixels)
        if (not isinstance(output, Tensor) or output.ndim != 5 or output.shape[:2] != (len(SURROUND_ORDER), 1)
                or min(output.shape[2:]) <= 0 or output.dtype not in (torch.float32, torch.float64)):
            raise ValueError("Encoder must return float32/float64 maps [6,1,C,Gh,Gw]")
        if not torch.isfinite(output).all():
            raise ValueError("Encoder maps must be finite")
        # Own the storage even if a backend reuses its buffers. A versioned
        # tensor lets the routing boundary catch accidental in-place writes.
        with torch.inference_mode(False):
            maps = output.transpose(0, 1).detach().clone(memory_format=torch.contiguous_format)
        calibrations = tuple(view.calibration.resized(*self._resize_wh) for view in snapshot.views)
        return SurroundFeatures(snapshot, self._encoder_id, maps, calibrations)


FastOutput = TypeVar("FastOutput")
BridgeOutput = TypeVar("BridgeOutput")


def route_surround_scene(
    snapshot: CameraSnapshot, encoder: SharedSurroundEncoder,
    fast: Callable[[SurroundFeatures], FastOutput],
    bridge: Callable[[SurroundFeatures, FastOutput], BridgeOutput],
) -> tuple[SurroundFeatures, FastOutput, BridgeOutput]:
    """Give Bridge the identical feature bundle and the exact Fast output."""
    features = encoder.encode(snapshot)
    # This guard covers ordinary PyTorch in-place ops, not unsafe foreign
    # storage aliases. A caller using an external view still must not write.
    version = features.feature_maps._version
    forecast = fast(features)
    _check_version(features.feature_maps, version, "Fast")
    bridged = bridge(features, forecast)
    _check_version(features.feature_maps, version, "Bridge")
    return features, forecast, bridged


def _check_version(maps: Tensor, expected: int, name: str) -> None:
    if maps._version != expected:
        raise ValueError(f"{name} modified shared surround features in place")
