from hashlib import sha256
import json

import pytest

from hybrid_drive.experiments import vru_clip_control
from hybrid_drive.experiments.vru_clip_replay import DEVELOPMENT_SCORE


def test_weak_development_arm_cannot_unlock_untouched_control(tmp_path):
    dev = tmp_path / "development-score"
    raw = json.dumps({"schema": DEVELOPMENT_SCORE, "scores": {"dfine_depthpro": {"counts": {"clips": 36, "passed": 23}}}}).encode()
    dev.write_bytes(raw)
    with pytest.raises(ValueError, match="untouched control"):
        vru_clip_control.freeze_release(dev, tmp_path/"not-mounted", tmp_path/"out", dev_sha256=sha256(raw).hexdigest(),
                                        arm="dfine_depthpro", revision="a"*40)
    assert not (tmp_path/"out").exists()


def test_equivalent_looking_replacement_plan_is_rejected(monkeypatch, tmp_path):
    monkeypatch.setattr(vru_clip_control, "load_clip_inputs", lambda path: {"plan_sha256": "0"*64, "anchors": [{}]*120})
    with pytest.raises(ValueError, match="precommitted"):
        vru_clip_control.control_index(tmp_path)


def test_release_binds_the_development_variant_and_control_plan(monkeypatch, tmp_path):
    dev = tmp_path/"development.json"
    profile = {"tiling": "front", "runtime": {"torch": "2.8.0"}}
    content = json.dumps({"schema": DEVELOPMENT_SCORE, "candidate_profile": profile,
                          "input_sha256": "d"*64,
                          "scores": {"dfine_calibrated": {"counts": {"clips": 36, "passed": 24}}}}).encode()
    dev.write_bytes(content)
    monkeypatch.setattr(vru_clip_control, "control_index", lambda root: {})
    monkeypatch.setattr(vru_clip_control, "index_sha", lambda root: "e"*64)
    out = tmp_path/"sealed"
    meta = vru_clip_control.freeze_release(dev, tmp_path/"inputs", out, dev_sha256=sha256(content).hexdigest(),
                                          arm="dfine_calibrated", revision="a"*40)
    data = (out/"release.json").read_bytes()
    assert meta["release_sha256"] == sha256(data).hexdigest()
    profile_lock = json.loads(data)
    assert profile_lock["candidate_profile"] == profile
    assert profile_lock["plan_sha256"] == vru_clip_control.PLAN_SHA256
    assert profile_lock["threshold"] == {"clips": 120, "passed": 80}
