from hashlib import sha256
import json

import pytest

from hybrid_drive.experiments import vru_clip_control
from hybrid_drive.experiments.vru_clip_replay import CAPTURE_SCHEMA, DEVELOPMENT_SCORE
from hybrid_drive.experiments.vru_depthpro2 import PINNED_MANIFESTS_SHA256


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


def test_cannot_freeze_depth_release_from_old_score_without_bundle_identity(tmp_path):
    dev = tmp_path / "development.json"
    data = json.dumps({"schema": DEVELOPMENT_SCORE, "depthpro_sha256": "a"*64,
                       "scores": {"dfine_depthpro": {"counts": {"clips": 60, "passed": 40}}}}).encode()
    dev.write_bytes(data)
    with pytest.raises(ValueError, match="Depth Pro 2 execution profile"):
        vru_clip_control.freeze_release(dev, tmp_path / "unopened", tmp_path / "out",
                                        dev_sha256=sha256(data).hexdigest(), arm="dfine_depthpro", revision="a"*40)
    assert not (tmp_path / "out").exists()


def test_freeze_depth_profile_from_scored_development(monkeypatch, tmp_path):
    execution = {"worker_sha256": "a"*64, "checkpoint_manifest_sha256": sorted(PINNED_MANIFESTS_SHA256)[0],
                 "compiled_bundle_sha256": "b"*64}
    dev = tmp_path / "development.json"
    data = json.dumps({"schema": DEVELOPMENT_SCORE, "input_sha256": "c"*64, "candidate_profile": {},
                       "depthpro_sha256": "d"*64, "depthpro_profile": execution,
                       "scores": {"dfine_depthpro": {"counts": {"clips": 60, "passed": 40}}}}).encode()
    dev.write_bytes(data)
    monkeypatch.setattr(vru_clip_control, "control_index", lambda root: {})
    monkeypatch.setattr(vru_clip_control, "index_sha", lambda root: "e"*64)
    out = tmp_path / "sealed"
    vru_clip_control.freeze_release(dev, tmp_path / "inputs", out, dev_sha256=sha256(data).hexdigest(),
                                    arm="dfine_depthpro", revision="a"*40)
    assert json.loads((out / "release.json").read_bytes())["depthpro_profile"] == execution


def test_control_cannot_switch_depth_model_before_reading_labels(monkeypatch, tmp_path):
    digest = "e"*64
    plan = vru_clip_control.PLAN_SHA256
    manifest_a, manifest_b = sorted(PINNED_MANIFESTS_SHA256)
    frozen = {"worker_sha256": "a"*64, "checkpoint_manifest_sha256": manifest_a, "compiled_bundle_sha256": "b"*64}
    substituted = {**frozen, "checkpoint_manifest_sha256": manifest_b, "compiled_bundle_sha256": "f"*64}
    release = {"schema": vru_clip_control.CONTROL_RELEASE, "input_sha256": digest, "plan_sha256": plan,
               "arm": "dfine_depthpro", "candidate_profile": {}, "depthpro_profile": frozen,
               "threshold": {"clips": 120, "passed": 80}}
    baseline = {"schema": vru_clip_control.BASELINE_CAPTURE, "input_sha256": digest, "plan_sha256": plan,
                "labels_read": False, "revision": "a"*40}
    candidate = {"schema": CAPTURE_SCHEMA, "input_sha256": digest, "split": "control", "profile": {},
                 "labels_read": False, "revision": "a"*40}
    records = {"release": release, "baseline": baseline, "candidate": candidate}
    monkeypatch.setattr(vru_clip_control, "sealed_read", lambda path, seal: records[path])
    monkeypatch.setattr(vru_clip_control, "control_index", lambda root: {})
    monkeypatch.setattr(vru_clip_control, "index_sha", lambda root: digest)
    monkeypatch.setattr(vru_clip_control, "_bound_depthpro", lambda *args: ((), substituted))
    monkeypatch.setattr(vru_clip_control, "load_clip_labels", lambda *args: pytest.fail("Control labels must remain closed"))
    with pytest.raises(ValueError, match="match the frozen development profile"):
        vru_clip_control.score_control(tmp_path, "baseline", "candidate", "release", "untouched", tmp_path / "out",
                                       baseline_sha256="1"*64, candidate_sha256="2"*64, release_sha256="3"*64,
                                       depthpro_path="substituted", depthpro_sha256="4"*64)
