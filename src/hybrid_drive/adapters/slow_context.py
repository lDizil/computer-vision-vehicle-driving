from __future__ import annotations

from numbers import Integral
from typing import TYPE_CHECKING

import numpy as np

from hybrid_drive.contracts.motion import EgoMotion
from hybrid_drive.contracts.slow import SlowRequest
from hybrid_drive.slow.context import CAMERA_NAMES, MAX_HEIGHT, MAX_WIDTH, CameraFrame, SlowContext, _validate_rgb

if TYPE_CHECKING:
    from navsim.common.dataclasses import AgentInput


def context_from_navsim(
    request: SlowRequest, motion: EgoMotion, agent_input: AgentInput, *, width: int = 384, height: int = 216,
) -> SlowContext:
    """Resize each current RGB view independently; no crop, stitch or projection."""
    from navsim.common.dataclasses import AgentInput
    from PIL import Image

    for name, value, maximum in (("width", width, MAX_WIDTH), ("height", height, MAX_HEIGHT)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or not 0 < value <= maximum:
            raise ValueError(f"{name} must be a positive integer no greater than {maximum}")
    if not isinstance(agent_input, AgentInput) or not agent_input.cameras:
        raise ValueError("AgentInput must contain current CAM_L0, CAM_F0 and CAM_R0 cameras")
    current = agent_input.cameras[-1]
    frames = []
    for name in CAMERA_NAMES:
        camera = getattr(current, name.lower(), None)
        rgb = getattr(camera, "image", None)
        _validate_rgb(rgb)
        image = Image.fromarray(rgb).resize((int(width), int(height)), resample=Image.Resampling.BILINEAR)
        frames.append(CameraFrame(name, np.asarray(image)))
    return SlowContext(request, motion, tuple(frames))
