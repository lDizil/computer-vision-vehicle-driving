"""Ground-plane BEV evidence with distinct current and past visibility."""

from __future__ import annotations

from dataclasses import replace

import torch
from torch import Tensor

from hybrid_drive.fast.metric_bev import BevGrid, lift_features
from hybrid_drive.neural.surround_features import SurroundFeatures


GROUND_HEIGHT_M = 0.0


@torch.no_grad()
def ground_temporal_bev(
    current: SurroundFeatures, previous: SurroundFeatures, grid: BevGrid,
) -> tuple[Tensor, Tensor, Tensor]:
    """Lift both images onto z=0 in the current ego frame.

    The output uses the existing temporal feature layout: current and previous
    visual channels, then their fractional coverage, scaled ego XY and dt.
    History coverage is projected into the current frame, not copied cellwise.
    The two boolean masks deliberately stay separate so observation policy can
    change without changing the visual features used to compare road heads.
    """
    now_maps, past_maps = current.feature_maps, previous.feature_maps
    if (current.encoder_id != previous.encoder_id or now_maps.shape != past_maps.shape
            or now_maps.dtype != past_maps.dtype or now_maps.device != past_maps.device):
        raise ValueError("Temporal samples must use one encoder and patch layout")
    now_frame, past_frame = current.snapshot, previous.snapshot
    if now_frame.rig_id != past_frame.rig_id or now_frame.timing != past_frame.timing:
        raise ValueError("Temporal samples must use one rig and timing basis")
    dt = (now_frame.ref_time_us - past_frame.ref_time_us) / 1e6
    if not 0 < dt <= 1:
        raise ValueError("Previous observation must precede the reference by at most one second")
    if any(view.capture_us is not None and view.capture_us > now_frame.ref_time_us for view in past_frame.views):
        raise ValueError("An earlier image was captured after the reference time")

    ground = replace(grid, heights_m=(GROUND_HEIGHT_M,))
    now, now_support = lift_features(current, now_frame, ground)
    past, past_support = lift_features(previous, now_frame, ground)
    xy = torch.stack([torch.as_tensor(a, dtype=now.dtype, device=now.device) / 64 for a in grid.xy()])
    values = torch.cat((now, past, now_support, past_support, xy, torch.full_like(now_support, dt)))
    return values, now_support[0] > 0, past_support[0] > 0
