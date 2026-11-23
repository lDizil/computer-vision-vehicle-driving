import numpy as np
import pytest
import torch

from hybrid_drive.contracts.sensors import CameraSnapshot, SURROUND_ORDER, SurroundView
from hybrid_drive.geometry.camera import CameraCalibration
from hybrid_drive.neural.surround_features import SharedSurroundEncoder, route_surround_scene


def _snapshot():
    views = []
    for i, direction in enumerate(SURROUND_ORDER):
        pixels = np.zeros((20, 40, 3), np.uint8)
        pixels[..., 0] = i * 20
        pixels[:, 20:, 1] = 175
        calibration = CameraCalibration("cam-" + direction.value, 40, 20,
                                        [[27, 0, 19.5], [0, 15, 7.5], [0, 0, 1]], np.eye(4))
        views.append(SurroundView(direction, calibration, pixels))
    return CameraSnapshot("same-frame", 5000, np.eye(4), None, reversed(views), rig_id="ordering-fixture")


def _toy_features(pixels):
    return pixels[:, :, :2, ::3, ::3]


def test_same_tensor_and_forecast_reach_bridge_after_single_batched_encode():
    snapshot = _snapshot()
    events = {}
    forecast = {"candidate_ids": ["fixture-path"]}

    def encode(pixels):
        events.setdefault("encodes", []).append(tuple(pixels.shape))
        return _toy_features(pixels)

    def fast(features):
        events["fast_features"] = features
        return forecast

    def bridge(features, received_forecast):
        events["bridge_features"] = features
        assert received_forecast is forecast
        return features.feature_maps[0, 5, 0, 0, 0].item()

    features, received_plan, last_red = route_surround_scene(
        snapshot, SharedSurroundEncoder(encode, encoder_id="toy", resize_wh=(10, 15)), fast, bridge,
    )
    assert events["encodes"] == [(6, 1, 3, 15, 10)]
    assert events["fast_features"] is events["bridge_features"] is features
    assert received_plan is forecast and last_red == pytest.approx(100 / 255)
    torch.testing.assert_close(features.feature_maps[0, :, 0, 0, 0], torch.arange(6) * (20 / 255))
    assert features.snapshot is snapshot
    expected = np.array([[6.75, 0, 4.5], [0, 11.25, 5.5], [0, 0, 1]])
    np.testing.assert_allclose(features.calibrations[1].intrinsic, expected)
    np.testing.assert_array_equal(features.calibrations[1].ego_to_camera, np.eye(4))


@pytest.mark.parametrize("who", ["Fast", "Bridge"])
def test_oob_in_place_write_is_an_error_and_fast_cannot_feed_corruption_to_bridge(who):
    bridge_calls = 0

    def fast(features):
        if who == "Fast":
            features.feature_maps[0, 0].zero_()

    def bridge(features, _):
        nonlocal bridge_calls
        bridge_calls += 1
        features.feature_maps[0, 2, 1].add_(1)

    encoder = SharedSurroundEncoder(_toy_features, encoder_id="toy", resize_wh=(10, 15))
    with torch.inference_mode(), pytest.raises(ValueError, match=f"{who} modified"):
        route_surround_scene(_snapshot(), encoder, fast, bridge)
    assert bridge_calls == int(who == "Bridge")


def test_encoder_output_alias_is_not_shared_with_backend_and_checks_all_views():
    original = torch.zeros(6, 1, 8, 4, 4)
    original[5, 0] = 2
    encoder = SharedSurroundEncoder(lambda _: original, encoder_id="fixture")
    maps = encoder.encode(_snapshot()).feature_maps
    original.fill_(float("nan"))
    assert maps[0, 5, 0, 0, 0].item() == 2
    with pytest.raises(ValueError, match="finite"):
        encoder.encode(_snapshot())
    invalid = SharedSurroundEncoder(lambda x: _toy_features(x)[:5], encoder_id="only-five")
    with pytest.raises(ValueError, match=r"\[6,1"):
        invalid.encode(_snapshot())
