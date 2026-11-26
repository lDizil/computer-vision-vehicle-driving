from dataclasses import asdict
from hashlib import sha256
from types import SimpleNamespace

import numpy as np
import pytest

from hybrid_drive.experiments import vru_chain_capture as capture
from hybrid_drive.experiments import vru_chain_replay as replay
from hybrid_drive.experiments.object_vru_geometry import GroundHit, GroundObject


def test_capture_uses_one_snapshot_for_both_detectors_at_each_observed_time(tmp_path):
    seen, calls = {}, []

    def load(root, token, frame, *, rectified_wh):
        assert frame == {"token": token}
        assert root == tmp_path and rectified_wh == (48, 24)
        assert token not in seen
        shot = SimpleNamespace(content_sha256=sha256(token.encode()).hexdigest())
        seen[token] = shot
        return shot

    def detect(shot):
        calls.append(shot)
        return (), None

    anchor = {"sample_token": "a", "previous_token": "b", "older_token": None, "recording": "r"}
    index = {"frames": {token: {"token": token} for token in "ab"}}
    row = capture._observe(tmp_path, index, anchor, dict.fromkeys(("ssdlite", "faster_rcnn"), detect), (48, 24), load)
    assert len(calls) == 4 and calls[0] is calls[1] is seen["a"] and calls[2] is calls[3] is seen["b"]
    assert row["observations"]["older"] is None


def _obj(key, y, camera="front"):
    xy = (10., y)
    hit = GroundHit("hit-" + key, "person", camera, xy, (0., 0., 1.5), .5, .2, (1.,) + (0.,) * 35)
    return asdict(GroundObject(key, "person", xy, .5, (hit,)))


def _observe(ssd, faster):
    return {"sha256": "a" * 64, "objects": {"ssdlite": ssd, "faster_rcnn": faster}}


def test_replay_aligns_historical_positions_and_keeps_forecast_provenance_separate():
    present, past, old = np.eye(4), np.eye(4), np.eye(4)
    past[1, 3], old[1, 3] = .30, .40
    index = {"frames": {token: {"ref_time_us": time, "ego_to_world": pose}
                        for token, time, pose in (("a", 2_000_000, present), ("b", 1_500_000, past), ("c", 1_000_000, old))}}
    row = {"sample_token": "a", "previous_token": "b", "older_token": "c", "recording": "r",
           "observations": {
               "current": _observe([_obj("now", 0.)], [_obj("now", .02)]),
               "previous": _observe([_obj("past", -.70)], [_obj("past", -.71)]),
               "older": _observe([_obj("old", -1.20, "side")], [_obj("old", -1.22)]),
           }}
    arms = replay._window(index, row)
    baseline, = arms["ssd_primary"]["objects"]
    restored, = arms["bridge_rescue"]["objects"]
    assert arms["faster_rcnn"]["history"]["previous"][0]["xy_m"] == pytest.approx((10., -.41))
    assert arms["faster_rcnn"]["history"]["older"][0]["xy_m"] == pytest.approx((10., -.82))
    assert arms["ssd_primary"]["history"] == arms["bridge_rescue"]["history"]
    assert restored["object_id"] == baseline["object_id"] == "ssd:now"
    assert restored["xy_m"] == baseline["xy_m"] == (10., 0.)
    assert restored["motions"] == baseline["motions"]
    assert baseline["confirmation"]["status"] == "no_older_link"
    assert restored["trace"] == {
        "origin_arm": "faster_rcnn", "origin_object_id": "rcnn:now", "rescue": "same_snapshot",
        "initial": {"status": "estimated", "previous_object_id": "rcnn:past"},
        "older": {"status": "estimated", "previous_object_id": "rcnn:old"},
    }
    assert restored["forecasts"]["two_frame"] == baseline["forecasts"]["two_frame"]
    assert (restored["forecasts"]["confirmed_two_frame"]
            == arms["faster_rcnn"]["objects"][0]["forecasts"]["two_frame"])
    assert restored["forecasts"]["three_frame_trend"] == pytest.approx((-.00168, .84), abs=.0001)


def test_bad_prediction_seal_fails_before_touching_labels(tmp_path):
    predictions = tmp_path / "unsealed.json"
    predictions.write_bytes(b"{}")
    absent = tmp_path / "not-mounted"
    with pytest.raises(ValueError, match="SHA fixed"):
        replay.score_chain_replay(absent, predictions, absent, absent, tmp_path / "out", prediction_sha256="0" * 64)
    assert not (tmp_path / "out").exists()
