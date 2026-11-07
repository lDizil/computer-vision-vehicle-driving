from __future__ import annotations

import hashlib
from pathlib import Path
from typing import BinaryIO

import numpy as np
import timm
import torch
from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.transfuser import transfuser_backbone, transfuser_config, transfuser_features, transfuser_model
from navsim.agents.transfuser.transfuser_config import TransfuserConfig
from navsim.agents.transfuser.transfuser_model import TransfuserModel
from navsim.common.dataclasses import AgentInput, SensorConfig, Trajectory

from hybrid_drive.adapters.cameras import LatentCameraFeatureBuilder

NAVSIM_REVISION = "3e8291bfa89ff247231e0227778840cd0a036896"
CHECKPOINT_REVISION = "32d89c0ae6e7c13c311f4a034002006c250afab0"
CHECKPOINT_SHA256 = "d08e97ba6beb937412697d619171553041622859e026a06935a2e0e118e6309d"
CHECKPOINT_BYTES = 673235500
ENCODER_REVISION = "0e8757122c4c09b15fb9300c0fd09b3fe89e28bf"
ENCODER_SHA256 = "829a220f9529d2b1ffdc8719d3273463001e0daf411eaf1916865d9db62c0ed2"
ENCODER_BYTES = 87278522
ENCODER_CONFIG_SHA256 = "51c481d69e2acf61cdda5e655f42083afdcf56d6cef03992cdd687ec0acf9d8a"
ENCODER_CONFIG_BYTES = 755
# These four source files define the exact architecture and preprocessing used
# by the pinned checkpoint, including the shape-only latent LiDAR branch.
_SOURCE_HASHES = (
    (transfuser_config, "14e0e0c7a63e944223e777d0aca782812b6e2df06640bcfffc7424058c39723c"),
    (transfuser_features, "b2752e1213581c5f19eb96a4a72966b005d3a8f8e98e7b6570b3bf481fbc7f53"),
    (transfuser_backbone, "ffbbaf2cdf038174cdd71d679864a508bce65b54b43ca6da6c26da957e9565fe"),
    (transfuser_model, "3a86d9c772d3055700c7b91574870e35d4563abdcc50da1342e69c5e342e7d5f"),
)


class LatentTransfuserAgent(AbstractAgent):
    """Official NAVSIM v1.1 LTF baseline, inference-only and CPU-only.

    initialize() verifies pinned local assets before loading. It never downloads
    anything. Calls on one instance must be sequential; a failed initialization
    leaves no usable previous model. This backend is not FSDrive.
    """

    def __init__(self, checkpoint_path: str | Path, *, asset_root: str | Path | None = None):
        super().__init__()
        self._checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        self._asset_root = (Path(asset_root).expanduser().resolve() if asset_root is not None
                            else self._checkpoint_path.parent.parent)
        self._config = TransfuserConfig(latent=True)
        # timm's supported local-dir source preserves the native resnet34
        # architecture while resolving pretrained=True entirely offline.
        self._config.image_architecture = "local-dir:" + str(self._asset_root / "encoder")
        self._features = LatentCameraFeatureBuilder(self._config)
        self._model: TransfuserModel | None = None
        self._provenance: dict | None = None

    def name(self) -> str:
        return type(self).__name__

    def get_sensor_config(self) -> SensorConfig:
        return SensorConfig(cam_f0=[3], cam_l0=[3], cam_l1=False, cam_l2=False,
                            cam_r0=[3], cam_r1=False, cam_r2=False, cam_b0=False, lidar_pc=False)

    def initialize(self) -> None:
        self._model = None
        self._provenance = None
        if timm.__version__ != "1.0.30":
            raise RuntimeError("Latent TransFuser requires pinned timm==1.0.30")
        for module, digest in _SOURCE_HASHES:
            with Path(module.__file__).open("rb") as source:
                _verify(source, digest, None, module.__name__)
        encoder = self._asset_root / "encoder"
        with (encoder / "config.json").open("rb") as source:
            _verify(source, ENCODER_CONFIG_SHA256, ENCODER_CONFIG_BYTES, "ResNet configuration")
        with (encoder / "model.safetensors").open("rb") as source:
            _verify(source, ENCODER_SHA256, ENCODER_BYTES, "ResNet weights")
        with self._checkpoint_path.open("rb") as checkpoint:
            _verify(checkpoint, CHECKPOINT_SHA256, CHECKPOINT_BYTES, "LTF checkpoint")
            checkpoint.seek(0)
            # Only this verified official checkpoint is deserialized, on CPU.
            # Reusing the descriptor also prevents a path replacement between
            # the checksum and torch.load.
            payload = torch.load(checkpoint, map_location=torch.device("cpu"))
        weights = _model_weights(payload)
        del payload
        with torch.device("cpu"):
            model = TransfuserModel(self._config)
        model.load_state_dict(weights, strict=True)
        del weights
        model.cpu().requires_grad_(False).eval()
        self._model = model
        self.eval()
        self._provenance = {
            "backend": "official NAVSIM v1.1 Latent TransFuser (not FSDrive)",
            "navsim_revision": NAVSIM_REVISION,
            "navsim_source_sha256": {module.__name__: digest for module, digest in _SOURCE_HASHES},
            "checkpoint_revision": CHECKPOINT_REVISION,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "checkpoint_bytes": CHECKPOINT_BYTES,
            "checkpoint_strict": True,
            "encoder_revision": ENCODER_REVISION,
            "encoder_sha256": ENCODER_SHA256,
            "encoder_config_sha256": ENCODER_CONFIG_SHA256,
            "timm_version": timm.__version__,
            "torch_version": torch.__version__,
            "device": "cpu", "latent": True,
            "trajectory_sampling": {"num_poses": 8, "interval_s": 0.5},
            "sensors": ["CAM_L0", "CAM_F0", "CAM_R0"], "sensor_frame": 3,
            "lidar": "not read; empty batch carrier replaced by learned latent",
            "camera_preprocessing": "native crop/stitch/OpenCV resize 1024x256/ToTensor",
            "status_preprocessing": "native command[4], velocity[2], acceleration[2] float32",
            "calibration": "not used by native model", "timestamp": "not supplied by AgentInput",
        }

    def provenance(self) -> dict:
        if self._provenance is None:
            raise RuntimeError("Latent TransFuser is not initialized")
        from copy import deepcopy
        return deepcopy(self._provenance)

    def compute_trajectory(self, agent_input: AgentInput) -> Trajectory:
        if self._model is None:
            raise RuntimeError("Call initialize() with verified official weights before inference")
        self.eval()
        features = {key: value.unsqueeze(0) for key, value in self._features.compute_features(agent_input).items()}
        with torch.no_grad():
            output = self._model(features)["trajectory"]
        if output.device.type != "cpu" or output.shape != (1, 8, 3) or not torch.isfinite(output).all():
            raise RuntimeError("Latent TransFuser returned invalid CPU trajectory; expected finite (1, 8, 3)")
        poses = output[0].detach().numpy().astype(np.float32, copy=True)
        return Trajectory(poses=poses, trajectory_sampling=self._config.trajectory_sampling)


def _verify(source: BinaryIO, digest: str, size: int | None, name: str) -> None:
    hasher = hashlib.sha256()
    actual_size = 0
    while chunk := source.read(1024 * 1024):
        hasher.update(chunk)
        actual_size += len(chunk)
    if (size is not None and actual_size != size) or hasher.hexdigest() != digest:
        raise ValueError(f"{name} failed pinned size/SHA-256 verification")


def _model_weights(payload: dict) -> dict[str, torch.Tensor]:
    if not isinstance(payload, dict) or not isinstance(payload.get("state_dict"), dict):
        raise ValueError("Official LTF checkpoint must contain state_dict")
    prefix = "agent._transfuser_model."
    weights = payload["state_dict"]
    if not weights or any(not isinstance(key, str) or not key.startswith(prefix) or not isinstance(value, torch.Tensor)
                          for key, value in weights.items()):
        raise ValueError("Unexpected keys or non-tensors in official LTF checkpoint")
    return {key[len(prefix):]: value for key, value in weights.items()}
