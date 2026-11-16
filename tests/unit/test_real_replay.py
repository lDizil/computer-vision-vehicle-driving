from dataclasses import dataclass
import json
import time

import numpy as np
import pytest

from hybrid_drive.adapters.real_camera import CalibratedView, OPENSCENE_NATIVE_PROFILE
from hybrid_drive.contracts.motion import EgoMotion
from hybrid_drive.contracts.trajectory import Trajectory
from hybrid_drive.experiments.protocol import trajectory_hash
from hybrid_drive.experiments.real_bundle import RealSceneBundle
from hybrid_drive.experiments.real_replay import context_for_bundle, run_real_replay
from hybrid_drive.geometry.camera import CameraCalibration
from hybrid_drive.slow.context import CAMERA_NAMES, CameraFrame
from hybrid_drive.slow.real_grounding import RealPathGroundingFactory, PROTOCOL_NAME


@pytest.fixture
def bundle():
    trajectory = Trajectory([[2, 0, 0], [4, 0, 0]], .5)
    transform = np.array([[0, -1, 0, 0], [0, 0, -1, 1], [1, 0, 0, 0], [0, 0, 0, 1]], float)
    intrinsic = [[12, 0, 8], [0, 12, 4], [0, 0, 1]]
    views = tuple(CalibratedView(CameraFrame(n, np.full((12, 16, 3), 90, np.uint8)),
                                CameraCalibration(n, 16, 12, intrinsic, transform)) for n in CAMERA_NAMES)
    provenance = {k: "a" * 64 for k in ("metadata_sha256", "source_assets_manifest_hash",
                  "source_experiment_manifest_hash", "checkpoint_sha256")}
    provenance.update(source_revision="b" * 40, navsim_revision="c" * 40, nuplan_revision="d" * 40,
                      raw_candidate_hash=trajectory_hash(trajectory), profile=OPENSCENE_NATIVE_PROFILE,
                      source_image_hashes={n: "e" * 64 for n in CAMERA_NAMES})
    return RealSceneBundle("0123456789abcdef", 1_600_000_000_000_000, trajectory, EgoMotion(2, 0, 0, 0), views, provenance)


@dataclass(frozen=True)
class FixtureFactory(RealPathGroundingFactory):
    kind: str = "clear"
    def __call__(self):
        return FixtureBackend(self.kind)


class FixtureBackend:
    def __init__(self, kind): self.kind = kind
    def generate(self, context):
        if self.kind == "timeout": time.sleep(5)
        if self.kind == "malformed": return "</pre><script>alert(1)</script>"
        scene = "uncertain" if self.kind == "unknown" else "visible"
        def stage(key, value): return json.dumps({key: value, "context_hash": context.content_hash})
        return json.dumps({"protocol": PROTOCOL_NAME, "visibility_raw": stage("scene", scene),
                           "path_state_raw": None if scene != "visible" else stage("path_state", self.kind)})
    def close(self): pass


def test_fresh_requests_bind_same_immutable_bundle_with_no_saved_deadlines(bundle):
    first = context_for_bundle(bundle, "model", request_id="first", started=100, deadline_s=10)
    second = context_for_bundle(bundle, "model", request_id="second", started=500, deadline_s=20)
    assert first.content_hash != second.content_hash
    assert first.request.plan.candidates_hash == second.request.plan.candidates_hash
    assert first.request.deadline_monotonic_s == 110
    assert second.request.deadline_monotonic_s == 520
    assert first.request.plan.candidates[0].trajectory is bundle.trajectory
    assert len(first.request.plan.candidates) == 1
    assert first.request.plan.snapshot_id == "bundle-" + bundle.bundle_hash
    assert first.request.plan.t_ref_s == bundle.frame_timestamp_us / 1e6
    assert any(np.any(frame.rgb != view.frame.rgb) for frame, view in zip(first.cameras, bundle.views))


@pytest.mark.parametrize("kind,status", [("blocked", "VETO"), ("clear", "NO_VETO"), ("unknown", "UNKNOWN")])
def test_actual_spawn_replay_preserves_trace_and_is_never_applied_to_plan(bundle, tmp_path, kind, status):
    before = bundle.trajectory.poses.copy()
    factory = FixtureFactory("unused", {"model_version": "fixture"}, kind=kind)
    directory, result = run_real_replay(bundle, factory, tmp_path, revision="working-tree", seeds=(0, 1))
    records = json.loads((directory / "records.json").read_text())
    manifest = json.loads((directory / "manifest.json").read_text())
    assert result["status"] == "completed" and result["completed"] == 2
    assert result["automatic_veto"] is False and result["quality"] == "unscored_no_ground_truth"
    assert [r["status"] for r in records] == [status, status]
    assert len({r["request_id"] for r in records}) == 2
    assert len({r["context_hash"] for r in records}) == 2
    assert all(json.loads(r["raw_response"])["protocol"] == PROTOCOL_NAME for r in records)
    assert manifest["camera_capture_timestamps"] == "unknown"
    assert manifest["source_revision"] == "working-tree"
    assert (directory / "CAM_F0-overlay.png").is_file()
    html = (directory / "report.html").read_text()
    assert "data:image/png;base64," in html and bundle.bundle_hash in html
    np.testing.assert_array_equal(bundle.trajectory.poses, before)


@pytest.mark.parametrize("kind,outcome", [("malformed", "invalid_output"), ("timeout", "timeout")])
def test_failed_attempt_keeps_evidence_and_returns_failed_not_success(bundle, tmp_path, kind, outcome):
    factory = FixtureFactory("unused", {"model_version": "fixture"}, kind=kind)
    directory, result = run_real_replay(bundle, factory, tmp_path, revision="a" * 40,
                                      deadline_s=.1 if kind == "timeout" else 10)
    row = json.loads((directory / "records.json").read_text())[0]
    assert result["status"] == "failed" and result["completed"] == 0
    assert row["outcome"] == outcome and row["status"] is None
    if kind == "malformed":
        assert row["raw_response"] == "</pre><script>alert(1)</script>"
        html = (directory / "report.html").read_text()
        assert "<script>" not in html and "&lt;script&gt;" in html


@pytest.mark.parametrize("kwargs", [dict(deadline_s=True), dict(deadline_s=float("nan")),
                                     dict(seeds=(0, 0)), dict(seeds=(True,)), dict(revision="main")])
def test_invalid_execution_contract_does_not_create_output(bundle, tmp_path, kwargs):
    factory = FixtureFactory("unused", {"model_version": "fixture"})
    args = {"revision": "working-tree", **kwargs}
    with pytest.raises(ValueError): run_real_replay(bundle, factory, tmp_path, **args)
    assert not list(tmp_path.iterdir())


def test_unconfirmed_cleanup_stops_series_before_next_model(bundle, tmp_path, monkeypatch):
    from types import SimpleNamespace
    from hybrid_drive.experiments import real_replay

    starts = []
    class UnconfirmedWorker:
        def __init__(self, factory, model_version, **kwargs):
            starts.append(factory.seed)
            self.factory = factory
        def submit(self, context): self.context = context
        def close(self): raise RuntimeError("Worker termination not confirmed within cleanup budget")

    def completed(worker, context):
        raw = FixtureBackend("clear").generate(context)
        return SimpleNamespace(outcome="completed", raw_response=raw,
                               result=worker.factory.parse_output(raw, context))

    monkeypatch.setattr(real_replay, "AsyncSlowWorker", UnconfirmedWorker)
    monkeypatch.setattr(real_replay, "wait_for_completion", completed)
    factory = FixtureFactory("unused", {"model_version": "fixture"})
    directory, result = run_real_replay(bundle, factory, tmp_path, revision="working-tree", seeds=(0, 1, 2))
    assert starts == [0]
    assert result["status"] == "failed" and result["planned_attempts"] == 3
    assert result["attempts"] == 1 and result["completed"] == 0
    rows = json.loads((directory / "records.json").read_text())
    assert len(rows) == 1 and rows[0]["outcome"] == "cleanup_error"
    assert rows[0]["cleanup_confirmed"] is False and rows[0]["status"] is None
    assert rows[0]["raw_response"] is not None
    assert (directory / "report.html").is_file()
