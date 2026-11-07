from __future__ import annotations

import numpy as np
import torch
from navsim.agents.transfuser.transfuser_config import TransfuserConfig
from navsim.agents.transfuser.transfuser_features import TransfuserFeatureBuilder
from navsim.common.dataclasses import AgentInput, EgoStatus


class LatentCameraFeatureBuilder(TransfuserFeatureBuilder):
    """Pinned NAVSIM preprocessing without the unused LiDAR histogram.

    The latent backbone reads only lidar.shape[0] before replacing it with its
    learned lidar_latent parameter. The empty feature is a batch carrier, not
    a fabricated sensor measurement. Calibration and timestamps are not used
    by this upstream model; none are inferred or manufactured here.
    """

    def __init__(self, config: TransfuserConfig):
        if not config.latent:
            raise ValueError("Camera-only preprocessing requires latent=True")
        super().__init__(config)

    def compute_features(self, agent_input: AgentInput) -> dict[str, torch.Tensor]:
        _validate_input(agent_input)
        return super().compute_features(agent_input)

    def _get_lidar_feature(self, agent_input: AgentInput) -> torch.Tensor:
        return torch.empty(0, dtype=torch.float32)


def _validate_input(agent_input: AgentInput) -> None:
    if not isinstance(agent_input, AgentInput) or not agent_input.ego_statuses:
        raise ValueError("Expected AgentInput with current ego status")
    if not agent_input.cameras:
        raise ValueError("Current cameras CAM_L0, CAM_F0 and CAM_R0 are required")
    if len(agent_input.cameras) != len(agent_input.ego_statuses):
        raise ValueError("Camera and ego histories must have matching frame counts")
    cameras = agent_input.cameras[-1]
    heights = []
    for name in ("cam_l0", "cam_f0", "cam_r0"):
        camera = getattr(cameras, name, None)
        image = getattr(camera, "image", None)
        if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
            raise ValueError(f"{name}.image must be an RGB uint8 array")
        min_width = 0 if name == "cam_f0" else 832
        if image.ndim != 3 or image.shape[2] != 3 or image.shape[0] <= 56 or image.shape[1] <= min_width:
            raise ValueError(f"{name}.image shape must allow the official RGB camera crop")
        heights.append(image.shape[0])
    if len(set(heights)) != 1:
        raise ValueError("Current camera images must have matching heights for stitching")

    current = agent_input.ego_statuses[-1]
    if not isinstance(current, EgoStatus):
        raise ValueError("Current ego status must be a native EgoStatus")
    if current.in_global_frame is not False:
        raise ValueError("Current ego status must use the local ego frame")
    pose = _finite_vector(current.ego_pose, 3, "ego_pose")
    if not np.allclose(pose, 0, rtol=0, atol=1e-5):
        raise ValueError("Current ego pose must be the scene-local origin")
    command = _finite_vector(current.driving_command, 4, "driving_command")
    if not np.isin(command, [0, 1]).all() or command.sum() != 1:
        raise ValueError("driving_command must be a four-element one-hot vector")
    _finite_vector(current.ego_velocity, 2, "ego_velocity")
    _finite_vector(current.ego_acceleration, 2, "ego_acceleration")


def _finite_vector(value, size: int, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != (size,) or array.dtype.kind not in "iuf" or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain {size} finite real values")
    with np.errstate(over="ignore"):
        converted = array.astype(np.float32)
    if not np.isfinite(converted).all():
        raise ValueError(f"{name} exceeds the model float32 range")
    return array
