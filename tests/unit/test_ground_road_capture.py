import json
from hashlib import sha256
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from hybrid_drive.contracts.sensors import CameraSnapshot, SURROUND_ORDER, SurroundView
from hybrid_drive.experiments.ground_road_capture import _OncePerSnapshot, _mini_observations, _verify_input_binding
from hybrid_drive.fast.ground_road_projection import ground_temporal_bev
from hybrid_drive.fast.metric_bev import BevGrid, temporal_bev
from hybrid_drive.geometry.camera import CameraCalibration
from hybrid_drive.neural.surround_features import SurroundFeatures


def _features(at_us, world_x=0., *, exposure_us=None):
    ego_to_camera = np.array([[0., -1., 0., 0.], [0., 0., -1., 1.], [1., 0., 0., 0.], [0., 0., 0., 1.]])
    pose = np.eye(4)
    pose[0, 3] = world_x
    views = [
        SurroundView(direction, CameraCalibration(
            direction.value, 4, 4, [[2, 0, 1.5], [0, 2, 1.5], [0, 0, 1]], ego_to_camera,
        ), np.zeros((4, 4, 3), dtype=np.uint8), capture_us=exposure_us)
        for direction in SURROUND_ORDER
    ]
    max_skew = None if exposure_us is None else abs(exposure_us - at_us)
    snapshot = CameraSnapshot(str(at_us), at_us, pose, None, views, rig_id="ground-fixture", max_skew_us=max_skew)
    values = torch.tensor([[1., 1.], [3., 3.]]).repeat(1, 6, 1, 1, 1)
    return SurroundFeatures(snapshot, "fixture", values, tuple(view.calibration for view in views))


def test_ground_lift_changes_vertical_patch_and_keeps_temporal_channel_layout():
    now, previous = _features(1_500_000), _features(1_000_000)
    one_cell = BevGrid(x_min=1, y_min=-1, rows=1, cols=1, heights_m=(1.,))
    values, current_seen, history_seen = ground_temporal_bev(now, previous, one_cell)
    object_values, _ = temporal_bev(now, previous, one_cell)
    # Near ground, the camera at z=1 sees pixel y=2.5 (bottom patch);
    # at the object plane z=1 it sees pixel y=1.5 (between two patches).
    assert values[:, 0, 0].tolist() == pytest.approx([3, 3, 1, 1, 2/64, 0, .5])
    assert object_values[0, 0, 0].item() == pytest.approx(2.)
    assert tuple(current_seen.shape) == tuple(history_seen.shape) == (1, 1)
    assert current_seen.dtype == history_seen.dtype == torch.bool
    assert current_seen[0, 0] and history_seen[0, 0]


def test_history_observes_cell_left_behind_in_reference_ego_frame():
    now, previous = _features(1_500_000, world_x=6.), _features(1_000_000)
    one_cell = BevGrid(x_min=-3, y_min=-1, rows=1, cols=1)
    values, current, history = ground_temporal_bev(now, previous, one_cell)
    # x_ref=-2 is behind the current rig, but occurred at x_prev=4.
    assert not current[0, 0] and history[0, 0]
    assert values[:, 0, 0].tolist() == pytest.approx([0., 2.5, 0., 1., -2/64, 0., .5])


def test_projection_rejects_future_or_mixed_timestamp_bases():
    now = _features(1_500_000, exposure_us=1_500_000)
    with pytest.raises(ValueError, match="timing basis"):
        ground_temporal_bev(now, _features(1_000_000), BevGrid())
    with pytest.raises(ValueError, match="after the reference"):
        ground_temporal_bev(now, _features(1_000_000, exposure_us=1_500_001), BevGrid())
    with pytest.raises(ValueError, match="precede"):
        ground_temporal_bev(now, _features(2_000_000, exposure_us=2_000_000), BevGrid())


def test_shared_observation_is_encoded_once_when_it_is_both_current_and_history():
    calls = []

    def load(token):
        return SimpleNamespace(snapshot_id=token)

    def encode(snapshot):
        calls.append(snapshot.snapshot_id)
        return SimpleNamespace(snapshot=snapshot)

    reader = _OncePerSnapshot(SimpleNamespace(encode=encode), load, [
        {"sample_token": "t1", "previous_token": "t0"},
        {"sample_token": "t2", "previous_token": "t1"},
    ])
    shared = reader.take("t1")
    reader.take("t0")
    reader.take("t2")
    assert reader.take("t1") is shared
    assert calls == ["t1", "t0", "t2"]
    assert reader.encodes == 3


def test_mini_reuses_exact_observation_bytes_and_binds_both_snapshots(tmp_path):
    train = tmp_path / "train"
    train.mkdir()
    original = [{"sample_token": str(i), "previous_token": str(i + 100)} for i in range(56)]
    raw = (json.dumps(original, indent=1) + "\n").encode()
    path = train / "observations.json"
    path.write_bytes(raw)
    meta = {"splits": {"train": {"count": 56, "artifacts": {"observations.json": sha256(raw).hexdigest()}}}}
    rows, serialized = _mini_observations(tmp_path, meta, "train")
    assert rows == original and serialized == raw
    path.write_bytes(raw + b" ")
    with pytest.raises(ValueError, match="observations changed"):
        _mini_observations(tmp_path, meta, "train")
    now = SimpleNamespace(content_sha256="current", ref_time_us=1_500_000)
    before = SimpleNamespace(content_sha256="previous", ref_time_us=1_000_000)
    record = {"snapshot_sha256": "current", "previous_sha256": "different",
              "ref_time_us": now.ref_time_us, "earlier_ref_time_us": before.ref_time_us}
    with pytest.raises(ValueError, match="Mini cameras or reference pose changed"):
        _verify_input_binding(now, before, record)
