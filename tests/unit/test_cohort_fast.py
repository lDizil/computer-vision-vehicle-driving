import copy
import hashlib
import json
import pickle
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from hybrid_drive.adapters.real_camera import CalibratedView, OPENSCENE_NATIVE_PROFILE
from hybrid_drive.assets.cohort import load_cohort_sources
from hybrid_drive.contracts.motion import EgoMotion
from hybrid_drive.contracts.trajectory import Trajectory
from hybrid_drive.experiments import cohort_fast as capture
from hybrid_drive.experiments.protocol import content_hash, trajectory_hash, verify_manifest
from hybrid_drive.experiments.real_bundle import RealSceneBundle, load_real_bundle
from hybrid_drive.geometry.camera import CameraCalibration
from hybrid_drive.slow.context import CAMERA_NAMES, CameraFrame


def views():
    transform = np.array([[0, -1, 0, 0], [0, 0, -1, 1], [1, 0, 0, 0], [0, 0, 0, 1]], float)
    return tuple(CalibratedView(CameraFrame(name, np.full((12, 16, 3), 90, np.uint8)),
                                CameraCalibration(name, 16, 12, [[12, 0, 8], [0, 12, 4], [0, 0, 1]], transform))
                 for name in CAMERA_NAMES)


def fake_bundle(spec, current, raw, cohort_root, camera_records, manifest, *, metadata_sha256):
    provenance = {
        "source_revision": manifest["source_revision"], "navsim_revision": manifest["navsim_revision"],
        "nuplan_revision": manifest["nuplan_revision"], "metadata_sha256": metadata_sha256,
        "source_assets_manifest_hash": manifest["assets_manifest_hash"],
        "source_experiment_manifest_hash": manifest["manifest_hash"],
        "raw_candidate_hash": trajectory_hash(raw), "checkpoint_sha256": manifest["model"]["checkpoint_sha256"],
        "source_image_hashes": {name: "e" * 64 for name in CAMERA_NAMES}, "profile": OPENSCENE_NATIVE_PROFILE,
    }
    return RealSceneBundle(spec["scene_token"], spec["frame_timestamp_us"], raw, EgoMotion(2, 0, 0, 0), views(), provenance)


def setup_capture(monkeypatch, *, failing_token=None):
    sources = load_cohort_sources(capture.repository_sources("real-cohort-sources.json"))
    calls = []
    raw = Trajectory([[t, 0, 0] for t in range(1, 9)], .5)

    class Fast:
        def provenance(self): return {"checkpoint_sha256": "d" * 64}
        def get_sensor_config(self): return "current-camera-config"
        def compute_trajectory(self, token):
            calls.append(token)
            if token == failing_token:
                raise RuntimeError("real Fast failure")
            return raw

    class Loader:
        def get_agent_input_from_token(self, token): return token
        def get_scene_from_token(self, token): raise AssertionError("privileged Scene leaked into Fast capture")

    def load(sources, tokens, **kwargs):
        assert len(tokens) == 6 and kwargs["sensor_config"] == "current-camera-config"
        return Loader(), []

    initialized = []
    def initialize(root, seed):
        initialized.append(seed)
        return Fast()

    files = [{"case_id": case["case_id"], **camera, "size_bytes": 1, "sha256": "e" * 64,
              "width": 16, "height": 12, "format": "JPEG", "mode": "RGB"}
             for case in sources["cases"] for camera in case["cameras"]]
    monkeypatch.setattr(capture, "configure_environment", lambda *args, **kwargs: {})
    monkeypatch.setattr(capture, "_verified_models", lambda root: [{"path": "checkpoint", "size_bytes": 1, "sha256": "d" * 64}])
    monkeypatch.setattr("hybrid_drive.assets.cohort.verify_cohort", lambda root, spec: {"files": files})
    monkeypatch.setattr(capture, "_current_frames", lambda *args: {case["case_id"]: {} for case in sources["cases"]})
    monkeypatch.setattr(capture, "_initialize_fast", initialize)
    monkeypatch.setattr(capture, "load_cohort_inputs", load)
    monkeypatch.setattr(capture, "_make_bundle", fake_bundle)
    monkeypatch.setitem(sys.modules, "hybrid_drive.adapters.navsim", SimpleNamespace(from_navsim=lambda value: value))
    return sources, calls, initialized, raw


def test_capture_freezes_same_actual_raw_once_per_frame_without_evaluator(monkeypatch, tmp_path):
    sources, calls, initialized, raw = setup_capture(monkeypatch)
    run_dir, manifest = capture.run_cohort_fast("data", "weights", "images", tmp_path, revision="a" * 40)
    assert initialized == [0]
    assert calls == [case["scene_token"] for case in sources["cases"]]
    assert manifest["planned_cases"] == 6 and len(manifest["cases"]) == 6
    assert capture.load_fast_cohort(run_dir) == manifest
    for case in manifest["cases"]:
        bundle = load_real_bundle(run_dir / case["bundle_path"])
        assert case["raw_candidate_hash"] == trajectory_hash(raw) == trajectory_hash(bundle.trajectory)
        assert bundle.provenance["metadata_sha256"] == sources["log"]["sha256"]
        frame_manifest = json.loads((run_dir / case["case_id"] / "fast-manifest.json").read_text())
        verify_manifest(frame_manifest)
        assert frame_manifest["executed_ablations"] == []
        assert bundle.provenance["source_experiment_manifest_hash"] == frame_manifest["manifest_hash"]


def test_failed_fast_frame_retains_planned_case_and_other_frames_continue(monkeypatch, tmp_path):
    source = load_cohort_sources(capture.repository_sources("real-cohort-sources.json"))
    sources, calls, initialized, raw = setup_capture(monkeypatch, failing_token=source["cases"][2]["scene_token"])
    run_dir, manifest = capture.run_cohort_fast("data", "weights", "images", tmp_path, revision="a" * 40)
    assert len(calls) == 6 and initialized == [0]
    assert [case["status"] for case in manifest["cases"]] == ["success", "success", "failed", "success", "success", "success"]
    failed = manifest["cases"][2]
    assert failed["bundle_hash"] is None and "real Fast failure" in failed["error"]
    assert capture.load_fast_cohort(run_dir) == manifest


def test_global_capture_failure_keeps_all_six_causes(monkeypatch, tmp_path):
    setup_capture(monkeypatch)
    def fail(root): raise OSError("weights unavailable")
    monkeypatch.setattr(capture, "_verified_models", fail)
    run_dir, manifest = capture.run_cohort_fast("data", "weights", "images", tmp_path, revision="a" * 40)
    assert len(manifest["cases"]) == 6 and manifest["model"] is None
    assert all(case["status"] == "failed" and "weights unavailable" in case["error"] for case in manifest["cases"])
    assert capture.load_fast_cohort(run_dir) == manifest


@pytest.mark.parametrize("change", ["path", "duplicate_token", "missing_case", "fabricated_failed_bundle"])
def test_cohort_reader_rejects_bad_binding_even_with_recomputed_hash(monkeypatch, tmp_path, change):
    setup_capture(monkeypatch)
    run_dir, original = capture.run_cohort_fast("data", "weights", "images", tmp_path, revision="a" * 40)
    manifest = copy.deepcopy(original)
    if change == "path": manifest["cases"][0]["bundle_path"] = "../other/bundle"
    elif change == "duplicate_token": manifest["cases"][1]["scene_token"] = manifest["cases"][0]["scene_token"]
    elif change == "missing_case": manifest["cases"].pop()
    else: manifest["cases"][0].update(status="failed", error="failed")
    manifest.pop("cohort_hash")
    manifest["cohort_hash"] = content_hash(manifest)
    (run_dir / "cohort.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError): capture.load_fast_cohort(run_dir)


def test_metadata_hash_checked_before_pickle_and_current_frame_identity_is_required(monkeypatch, tmp_path):
    path = tmp_path / "metadata.pkl"
    path.write_bytes(b"unverified pickle")
    sources = {"log": {"path": "metadata.pkl", "size_bytes": path.stat().st_size, "sha256": "a" * 64}, "cases": []}
    def forbidden(*args): raise AssertionError("pickle called before verification")
    with monkeypatch.context() as patch:
        patch.setattr(capture.pickle, "loads", forbidden)
        with pytest.raises(ValueError, match="before pickle"): capture._current_frames(tmp_path, sources)
    frame = {"token": "b" * 16, "scene_token": "c" * 16, "frame_idx": 3, "timestamp": 1}
    data = pickle.dumps([{}, {}, {}, frame])
    path.write_bytes(data)
    sources["log"].update(size_bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    sources["cases"] = [{"case_id": "r00", "source_list_index": 3, "scene_token": "a" * 16,
                          "metadata_scene_token": "c" * 16, "metadata_frame_idx": 3, "frame_timestamp_us": 1}]
    with pytest.raises(ValueError, match="fixed selection"): capture._current_frames(tmp_path, sources)


@pytest.mark.parametrize("corruption", [None, "truncated", "timestamp", "frame_idx", "chunk", "duplicate_token"])
def test_current_frames_rechecks_entire_metadata_window_before_fast(tmp_path, corruption):
    frames = [{"token": f"{index:016x}", "scene_token": "c" * 16, "frame_idx": index,
               "timestamp": 1_000_000 + 500_000 * index, "annotations": {"not_model_input": True}}
              for index in range(14)]
    case = {"case_id": "r00", "source_list_index": 3, "scene_token": frames[3]["token"],
            "metadata_scene_token": "c" * 16, "metadata_frame_idx": 3, "frame_timestamp_us": frames[3]["timestamp"]}
    if corruption == "truncated": frames.pop()
    elif corruption == "timestamp": frames[8]["timestamp"] += 100_000
    elif corruption == "frame_idx": frames[8]["frame_idx"] += 1
    elif corruption == "chunk": frames[8]["scene_token"] = "d" * 16
    elif corruption == "duplicate_token": frames[8]["token"] = frames[7]["token"]
    data = pickle.dumps(frames)
    (tmp_path / "metadata.pkl").write_bytes(data)
    sources = {"log": {"path": "metadata.pkl", "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()},
               "cases": [case], "selection": {"num_history_frames": 4, "num_future_frames": 10,
                                                 "timestamp_step_us": [499000, 501000]}}
    if corruption is None:
        assert capture._current_frames(tmp_path, sources)["r00"] == frames[3]
        return
    with pytest.raises(ValueError, match="window"): capture._current_frames(tmp_path, sources)
