import os
from pathlib import Path

import numpy as np
import pytest
import torch
from navsim.agents.transfuser.transfuser_features import TransfuserFeatureBuilder
from navsim.common.dataclasses import AgentInput, Camera, Cameras, EgoStatus, Lidar, Trajectory

from hybrid_drive.fast.latent_transfuser import LatentTransfuserAgent


def camera_input():
    # Values vary across pixels, channels and cameras: crop/stitch errors cannot
    # hide behind a solid colour fixture. There is deliberately no LiDAR.
    image = np.arange(900 * 960 * 3, dtype=np.uint32).reshape(900, 960, 3).astype(np.uint8)
    cameras = Cameras(cam_l0=Camera(image=image), cam_f0=Camera(image=np.roll(image, 31, axis=1)),
                      cam_r0=Camera(image=np.flip(image, axis=1).copy()), cam_l1=Camera(), cam_l2=Camera(),
                      cam_r1=Camera(), cam_r2=Camera(), cam_b0=Camera())
    status = EgoStatus(np.zeros(3), np.array([2.5, -0.2]), np.array([0.1, -0.3]), np.array([0, 1, 0, 0]))
    return AgentInput([status], [cameras], [])


def test_verified_weighted_cpu_inference_and_native_nonzero_lidar_parity():
    """Requires pinned local official weights; no download or substitute model."""
    if not os.environ.get("LTF_CHECKPOINT_PATH") or not os.environ.get("LTF_ASSET_ROOT"):
        pytest.fail("Set LTF_CHECKPOINT_PATH and LTF_ASSET_ROOT to verified local official assets")
    backend = LatentTransfuserAgent(Path(os.environ["LTF_CHECKPOINT_PATH"]),
                                   asset_root=os.environ.get("LTF_ASSET_ROOT"))
    backend.initialize()
    backend.train()  # compute_trajectory must restore inference/eval mode.
    value = camera_input()
    captured = {}
    grad_modes = []

    def capture(module, inputs, output):
        grad_modes.append(torch.is_grad_enabled())
        captured.update(output)
    handle = backend._model.register_forward_hook(capture)
    actual = backend.compute_trajectory(value)
    handle.remove()
    assert isinstance(actual, Trajectory)
    assert actual.poses.shape == (8, 3)
    assert actual.poses.dtype == np.float32
    assert np.isfinite(actual.poses).all()
    assert actual.trajectory_sampling.interval_length == 0.5
    assert actual.trajectory_sampling.num_poses == 8
    assert all(parameter.device.type == "cpu" and not parameter.requires_grad
               for parameter in backend._model.parameters())
    assert not backend.training and not backend._model.training
    assert grad_modes == [False]

    # Independent native histogram, with actual nonzero content; replacing it
    # by an empty batch carrier must leave every model output unchanged.
    value.lidars = [Lidar(lidar_pc=np.array([[1, 2, 3], [3, 4, 5], [1, 2, 3],
                                          [0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.float32))]
    native = TransfuserFeatureBuilder(backend._config).compute_features(value)
    assert torch.count_nonzero(native["lidar_feature"]) > 0
    with torch.no_grad():
        expected = backend._model({key: tensor.unsqueeze(0) for key, tensor in native.items()})
    for key in expected:
        torch.testing.assert_close(expected[key], captured[key], rtol=0, atol=0)
    np.testing.assert_array_equal(actual.poses, expected["trajectory"][0].numpy())
    provenance = backend.provenance()
    assert provenance["latent"] is True and provenance["checkpoint_strict"] is True

    # A failed reinitialization must not retain a previously valid model.
    backend._checkpoint_path = Path("/definitely/missing/ltf.ckpt")
    with pytest.raises(FileNotFoundError):
        backend.initialize()
    with pytest.raises(RuntimeError, match="initialize"):
        backend.compute_trajectory(value)
