import copy
import io

import numpy as np
import pytest
import torch
from navsim.agents.transfuser.transfuser_config import TransfuserConfig
from navsim.agents.transfuser.transfuser_features import TransfuserFeatureBuilder
from navsim.common.dataclasses import AgentInput, Camera, Cameras, EgoStatus, Lidar

from hybrid_drive.adapters.cameras import LatentCameraFeatureBuilder
from hybrid_drive.fast.latent_transfuser import LatentTransfuserAgent, _model_weights, _verify


def camera_input():
    # Values vary across pixels, channels and cameras: crop/stitch errors cannot
    # hide behind a solid colour fixture. There is deliberately no LiDAR.
    image = np.arange(900 * 960 * 3, dtype=np.uint32).reshape(900, 960, 3).astype(np.uint8)
    cameras = Cameras(cam_l0=Camera(image=image), cam_f0=Camera(image=np.roll(image, 31, axis=1)),
                      cam_r0=Camera(image=np.flip(image, axis=1).copy()), cam_l1=Camera(), cam_l2=Camera(),
                      cam_r1=Camera(), cam_r2=Camera(), cam_b0=Camera())
    status = EgoStatus(np.zeros(3), np.array([2.5, -0.2]), np.array([0.1, -0.3]), np.array([0, 1, 0, 0]))
    return AgentInput([status], [cameras], [])


def test_camera_only_features_exactly_match_native_camera_and_status_preprocessing():
    value = camera_input()
    config = TransfuserConfig(latent=True)
    native_input = copy.deepcopy(value)
    native_input.lidars = [Lidar(lidar_pc=np.array([[1, 2], [3, 4], [-1, 1], [0, 0], [0, 0], [0, 0]],
                                                dtype=np.float32))]
    native = TransfuserFeatureBuilder(config).compute_features(native_input)
    actual = LatentCameraFeatureBuilder(config).compute_features(value)
    torch.testing.assert_close(actual["camera_feature"], native["camera_feature"], rtol=0, atol=0)
    torch.testing.assert_close(actual["status_feature"], native["status_feature"], rtol=0, atol=0)
    assert actual["camera_feature"].shape == (3, 256, 1024)
    assert actual["camera_feature"].dtype == torch.float32
    assert actual["lidar_feature"].numel() == 0
    torch.testing.assert_close(actual["status_feature"], torch.tensor([0, 1, 0, 0, 2.5, -0.2, 0.1, -0.3]))


@pytest.mark.parametrize("mutation, message", [
    (lambda x: x.cameras.clear(), "cameras"),
    (lambda x: setattr(x.cameras[-1].cam_l0, "image", None), "cam_l0"),
    (lambda x: setattr(x.cameras[-1].cam_r0, "image", np.zeros((80, 832, 3), np.uint8)), "crop"),
    (lambda x: setattr(x.cameras[-1].cam_f0, "image", np.zeros((56, 960, 3), np.uint8)), "crop"),
    (lambda x: setattr(x.cameras[-1].cam_l0, "image", np.zeros((900, 960, 4), np.uint8)), "crop"),
    (lambda x: setattr(x.cameras[-1].cam_f0, "image", np.zeros((900, 0, 3), np.uint8)), "crop"),
    (lambda x: setattr(x.cameras[-1].cam_r0, "image", np.full((900, 960, 3), np.nan)), "uint8"),
    (lambda x: setattr(x.cameras[-1].cam_r0, "image", np.zeros((901, 960, 3), np.uint8)), "heights"),
    (lambda x: x.ego_statuses.append(x.ego_statuses[-1]), "frame counts"),
    (lambda x: setattr(x.ego_statuses[-1], "driving_command", np.array([1, 0, 0])), "driving_command"),
    (lambda x: setattr(x.ego_statuses[-1], "driving_command", [1, 0, np.nan, 0]), "driving_command"),
    (lambda x: setattr(x.ego_statuses[-1], "driving_command", [1, 1, 0, 0]), "one-hot"),
    (lambda x: setattr(x.ego_statuses[-1], "driving_command", [0, 0, 0, 0]), "one-hot"),
    (lambda x: setattr(x.ego_statuses[-1], "driving_command", [0.5, 0.5, 0, 0]), "one-hot"),
    (lambda x: setattr(x.ego_statuses[-1], "driving_command", [0.999999999, 0, 0, 0]), "one-hot"),
    (lambda x: setattr(x.ego_statuses[-1], "ego_velocity", [np.inf, 0]), "ego_velocity"),
    (lambda x: setattr(x.ego_statuses[-1], "ego_velocity", [True, False]), "ego_velocity"),
    (lambda x: setattr(x.ego_statuses[-1], "ego_acceleration", [1e100, 0]), "float32"),
    (lambda x: setattr(x.ego_statuses[-1], "in_global_frame", True), "local"),
])
def test_invalid_inputs_fail_before_native_preprocessing(mutation, message):
    value = camera_input()
    mutation(value)
    with pytest.raises(ValueError, match=message):
        LatentCameraFeatureBuilder(TransfuserConfig(latent=True)).compute_features(value)


def test_nonlatent_cannot_use_camera_only_adapter():
    with pytest.raises(ValueError, match="latent=True"):
        LatentCameraFeatureBuilder(TransfuserConfig())


def test_checkpoint_must_be_verified_before_deserialization(tmp_path, monkeypatch):
    import hashlib
    from hybrid_drive.fast import latent_transfuser as module

    checkpoint = tmp_path / "checkpoints" / "ltf_seed_0.ckpt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"not a checkpoint")
    encoder = tmp_path / "encoder"
    encoder.mkdir()
    for filename, prefix, payload in (("config.json", "ENCODER_CONFIG", b"test-config"),
                                      ("model.safetensors", "ENCODER", b"test-encoder")):
        (encoder / filename).write_bytes(payload)
        monkeypatch.setattr(module, prefix + "_BYTES", len(payload))
        monkeypatch.setattr(module, prefix + "_SHA256", hashlib.sha256(payload).hexdigest())
    backend = LatentTransfuserAgent(checkpoint)
    def forbidden_load(*args, **kwargs):
        pytest.fail("Unverified checkpoint reached torch.load")
    monkeypatch.setattr(torch, "load", forbidden_load)
    with pytest.raises(ValueError, match="LTF checkpoint"):
        backend.initialize()
    with pytest.raises(RuntimeError, match="not initialized"):
        backend.provenance()
    with pytest.raises(RuntimeError, match="initialize"):
        backend.compute_trajectory(camera_input())


def test_hash_verification_rejects_wrong_size_and_content():
    import hashlib
    digest = hashlib.sha256(b"abc").hexdigest()
    _verify(io.BytesIO(b"abc"), digest, 3, "test")
    for value, size in [(b"abx", 3), (b"abc", 4)]:
        with pytest.raises(ValueError, match="SHA-256"):
            _verify(io.BytesIO(value), digest, size, "test")


def test_strict_checkpoint_key_translation_does_not_silently_drop_weights():
    tensor = torch.ones(1)
    assert _model_weights({"state_dict": {"agent._transfuser_model.foo": tensor}}) == {"foo": tensor}
    for payload in [{}, {"state_dict": {}}, {"state_dict": {"other.foo": tensor}},
                    {"state_dict": {"agent._transfuser_model.foo": "not tensor"}}]:
        with pytest.raises(ValueError):
            _model_weights(payload)


