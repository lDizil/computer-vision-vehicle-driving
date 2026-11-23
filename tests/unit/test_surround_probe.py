import json

import numpy as np
from PIL import Image
import torch

from hybrid_drive.contracts.sensors import CameraSnapshot, SURROUND_ORDER, SurroundView
from hybrid_drive.experiments.surround_probe import run_surround_probe
from hybrid_drive.geometry.camera import CameraCalibration
from hybrid_drive.neural.surround_features import SharedSurroundEncoder


def test_probe_binds_six_calibrations_and_actual_roi_tokenizer_without_driving_claim(tmp_path):
    views = [
        SurroundView(
            direction, CameraCalibration(direction.value, 10, 8, [[7, 0, 5], [0, 7, 3], [0, 0, 1]], np.eye(4)),
            np.full((8, 10, 3), i*40, np.uint8),
        )
        for i, direction in enumerate(SURROUND_ORDER)
    ]
    snapshot = CameraSnapshot("probe-example", 123_456, np.eye(4), None, views, rig_id="test-fixture")
    calls = []

    def stage1_features(pixels):
        calls.append(tuple(pixels.shape))
        return pixels[:, :, :, ::2, ::2]

    encoder = SharedSurroundEncoder(stage1_features, encoder_id="six-flat-rasters", resize_wh=(12, 8))
    output = tmp_path / "run"
    rng_before = torch.random.get_rng_state()
    run_surround_probe(
        snapshot, encoder, output, revision="0"*40, encoder_provenance={"test": True}, source={"kind": "fixture"},
    )
    result = json.loads((output / "result.json").read_text())
    assert calls == [(6, 1, 3, 8, 12)]
    assert result["fast_input_sha256"] == result["bridge_input_sha256"]
    assert result["bridge_token_shape"] == [1, 6, 32]
    assert [v["source_camera"] for v in result["views"]] == [d.value for d in SURROUND_ORDER]
    assert len({v["feature_sha256"] for v in result["views"]}) == 6
    assert result["measured_skew_us"] is None and result["timing_basis"] == "source_association"
    assert not any(result[name] for name in ("executed_fast_planner", "executed_3d_prediction", "executed_slow_vqa"))
    assert torch.equal(rng_before, torch.random.get_rng_state())
    with Image.open(output / "cameras.png") as grid:
        assert grid.width > grid.height
