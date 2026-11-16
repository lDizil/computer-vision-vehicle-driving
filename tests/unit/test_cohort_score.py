import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from hybrid_drive.adapters.real_camera import CalibratedView, OPENSCENE_NATIVE_PROFILE
from hybrid_drive.assets.cohort import load_cohort_sources
from hybrid_drive.contracts.motion import EgoMotion
from hybrid_drive.contracts.trajectory import Trajectory
from hybrid_drive.experiments import cohort_score as scoring
from hybrid_drive.experiments.navsim import cohort_overrides, sample_arbiter
from hybrid_drive.experiments.protocol import METRICS, content_hash, trajectory_hash
from hybrid_drive.experiments.real_bundle import RealSceneBundle
from hybrid_drive.experiments.real_replay import context_for_bundle
from hybrid_drive.geometry.camera import CameraCalibration
from hybrid_drive.slow.context import CAMERA_NAMES, CameraFrame
from hybrid_drive.slow.real_grounding import PROTOCOL_NAME, parse_real_path_trace


@pytest.fixture
def bundle():
    raw = Trajectory([[t, 0, 0] for t in range(1, 9)], .5)
    transform = np.array([[0, -1, 0, 0], [0, 0, -1, 1], [1, 0, 0, 0], [0, 0, 0, 1]], float)
    views = tuple(CalibratedView(CameraFrame(name, np.full((12, 16, 3), 90, np.uint8)),
                                CameraCalibration(name, 16, 12, [[12, 0, 8], [0, 12, 4], [0, 0, 1]], transform))
                  for name in CAMERA_NAMES)
    provenance = {key: "a" * 64 for key in ("metadata_sha256", "source_assets_manifest_hash", "source_experiment_manifest_hash", "checkpoint_sha256")}
    provenance.update(source_revision="b" * 40, navsim_revision="c" * 40, nuplan_revision="d" * 40,
                      raw_candidate_hash=trajectory_hash(raw), profile=OPENSCENE_NATIVE_PROFILE,
                      source_image_hashes={name: "e" * 64 for name in CAMERA_NAMES})
    return RealSceneBundle("0123456789abcdef", 1_600_000_000_000_000, raw, EgoMotion(2, 0, 0, 0), views, provenance)


def evidence_for(bundle, *, state="clear", outcome="completed", checkpoint=102, received=101):
    context = context_for_bundle(bundle, "native-qwen", request_id="native-original", started=100, deadline_s=10)
    scene = "uncertain" if state == "unknown" else "visible"
    def stage(key, value): return json.dumps({key: value, "context_hash": context.content_hash})
    raw = json.dumps({"protocol": PROTOCOL_NAME, "visibility_raw": stage("scene", scene),
                      "path_state_raw": stage("path_state", state) if scene == "visible" else None})
    return SimpleNamespace(context=context, result=parse_real_path_trace(raw, context) if outcome == "completed" else None,
                           outcome=outcome, received_at_monotonic_s=received, decision_checkpoint_monotonic_s=checkpoint,
                           cleanup_confirmed=True, evidence_hash="f" * 64, source_revision="a" * 40)


def test_combined_uses_only_original_native_clock_and_same_raw_plan(bundle):
    evidence = evidence_for(bundle, state="blocked")
    calls = []
    actual = sample_arbiter()
    class RecordingArbiter:
        def decide(self, plan, motion, **kwargs):
            calls.append((plan, kwargs))
            return actual.decide(plan, motion, **kwargs)
    outputs, details = scoring._freeze_predictions(bundle, evidence, RecordingArbiter())
    assert outputs["baseline"] is bundle.trajectory
    assert outputs["checks_only"] is bundle.trajectory
    assert not np.array_equal(outputs["combined"].poses, bundle.trajectory.poses)
    assert len(calls) == 2 and all(plan is evidence.context.request.plan for plan, args in calls)
    assert all(args["now_monotonic_s"] == 102 and args["now_sim_s"] == evidence.context.request.plan.t_ref_s for plan, args in calls)
    assert calls[1][1]["received_at_monotonic_s"] == 101
    assert calls[1][1]["request"] is evidence.context.request
    assert details["combined"]["all_veto"] and details["combined"]["fallback_succeeded"]


def test_deadline_is_checked_at_saved_decision_time_not_only_arrival(bundle):
    outputs, details = scoring._freeze_predictions(bundle, evidence_for(bundle, state="blocked", checkpoint=111), sample_arbiter())
    assert outputs["combined"] is bundle.trajectory
    assert details["combined"]["slow_disposition"] == "deadline_exceeded"
    assert not details["combined"]["all_veto"]


@pytest.mark.parametrize("state,outcome,failed", [("unknown", "completed", False), ("clear", "backend_error", True)])
def test_unknown_abstention_and_technical_degradation_are_distinct(bundle, state, outcome, failed):
    outputs, details = scoring._freeze_predictions(bundle, evidence_for(bundle, state=state, outcome=outcome), sample_arbiter())
    assert outputs["combined"] is outputs["checks_only"] is bundle.trajectory
    assert details["combined"]["slow_failed"] is failed
    assert details["combined"]["failure"] is None
    assert details["combined"]["slow_status"] == "UNKNOWN"


def test_missing_evidence_does_not_drop_baseline_or_checks(bundle):
    outputs, details = scoring._freeze_predictions(bundle, None, sample_arbiter(), evidence_error="not_started")
    assert set(outputs) == {"baseline", "checks_only"}
    assert details["combined"]["failure"] == "not_started"
    assert details["combined"]["slow_failed"]


def test_failed_fallback_placeholder_is_retained_but_never_scored(bundle, monkeypatch, tmp_path):
    lateral = RealSceneBundle(bundle.scene_token, bundle.frame_timestamp_us, bundle.trajectory,
                              EgoMotion(2, .1, 0, 0), bundle.views, bundle.provenance)
    outputs, details = scoring._freeze_predictions(lateral, evidence_for(lateral, state="blocked"), sample_arbiter())
    assert details["combined"]["failure"] == "unsupported_motion"
    calls = []
    monkeypatch.setitem(sys.modules, "hybrid_drive.adapters.navsim", SimpleNamespace(from_navsim=lambda x: x, to_navsim=lambda x: x))
    monkeypatch.setattr(scoring, "score_hybrid_prediction", lambda value, *args: calls.append(value) or {key: .5 for key in METRICS})
    loader = SimpleNamespace(get_scene_from_token=lambda token: SimpleNamespace(get_future_trajectory=lambda **kwargs: bundle.trajectory))
    item = {"case_id": "r00", "scene_token": bundle.scene_token, "raw_candidate_hash": trajectory_hash(bundle.trajectory),
            "outputs": outputs, "details": details}
    rows, reference = scoring._score_case(item, loader, tmp_path, [], None)
    assert len(calls) == 2 and all(value is not outputs["combined"] for value in calls)
    assert rows[2]["failed"] and rows[2]["metrics"] is None and rows[2]["ade_m"] is None
    assert rows[2]["all_veto"] and rows[2]["fallback_succeeded"] is False


@pytest.mark.parametrize("failed_case", [None, 2])
def test_full_cohort_freezes_all_outputs_before_future_and_keeps_eighteen_attempts(bundle, monkeypatch, tmp_path, failed_case):
    sources = load_cohort_sources(scoring.repository_sources("real-cohort-sources.json"))
    cases = [{"case_id": case["case_id"], "scene_token": case["scene_token"], "source_list_index": case["source_list_index"],
              "bundle_path": case["case_id"] + "/bundle", "bundle_hash": bundle.bundle_hash,
              "raw_candidate_hash": trajectory_hash(bundle.trajectory), "status": "success"} for case in sources["cases"]]
    cohort = {"source_revision": "b" * 40, "sources_hash": content_hash(sources), "cases": cases, "cohort_hash": "f" * 64}
    monkeypatch.setattr(scoring, "load_fast_cohort", lambda root: cohort)
    monkeypatch.setattr("hybrid_drive.experiments.offline_evidence.load_cohort_evidence_index", lambda *args, **kwargs: {
        "evidence_index_hash": "e" * 64, "source_revision": "a" * 40, "model_version": "native-qwen",
        "model_sources_hash": "b" * 64, "protocol": PROTOCOL_NAME,
    })
    frozen = []
    def freeze(case, *args):
        frozen.append(case["case_id"])
        outputs, details = scoring._freeze_predictions(bundle, evidence_for(bundle), sample_arbiter())
        if failed_case is not None and case["case_id"] == f"r{failed_case:02d}":
            outputs = {}
            details = {mode: scoring._failed_details("Fast unavailable", slow_outcome="capture_failed") for mode in scoring.MODES}
        return {"case_id": case["case_id"], "scene_token": case["scene_token"],
                "raw_candidate_hash": trajectory_hash(bundle.trajectory), "outputs": outputs, "details": details}
    monkeypatch.setattr(scoring, "_freeze_case", freeze)
    def configure(data, run_dir, **kwargs):
        assert len(frozen) == 6
        assert len(json.loads((run_dir / "outputs.json").read_text())["cases"]) == 6
    monkeypatch.setattr(scoring, "configure_environment", configure)
    seen_future, seen_scores = [], []
    def scene(token):
        assert len(frozen) == 6
        seen_future.append(token)
        return SimpleNamespace(get_future_trajectory=lambda **kwargs: bundle.trajectory)
    monkeypatch.setattr(scoring, "load_cohort_inputs", lambda *args, **kwargs: (SimpleNamespace(get_scene_from_token=scene), []))
    monkeypatch.setattr(scoring, "cache_cohort", lambda *args: None)
    monkeypatch.setattr(scoring, "_render_case", lambda *args: None)
    monkeypatch.setitem(sys.modules, "hybrid_drive.experiments.cohort_report", SimpleNamespace(write_cohort_report=lambda *args: None))
    monkeypatch.setitem(sys.modules, "hybrid_drive.adapters.navsim", SimpleNamespace(from_navsim=lambda x: x, to_navsim=lambda x: x))
    monkeypatch.setattr(scoring, "score_hybrid_prediction", lambda value, *args: seen_scores.append(value) or {key: 0 for key in METRICS})
    run_dir, result = scoring.score_frozen_cohort("data", "fast", "evidence", tmp_path, revision="a" * 40)
    assert len(result["records"]) == result["planned_mode_attempts"] == 18
    assert len(seen_future) == (6 if failed_case is None else 5)
    assert len(seen_scores) == (18 if failed_case is None else 15)
    assert result["status"] == ("success" if failed_case is None else "failed")
    assert result["executed_modes"] == ["baseline", "checks_only", "combined"]
    assert "slow_only" in result["unavailable_modes"]
    for mode in scoring.MODES:
        summary = result["summary"][mode]
        assert summary["attempted"] == 6 and summary["score_mean_all_attempts_failure_zero"] == 0
        assert summary["failed"] == (0 if failed_case is None else 1)
    assert len((run_dir / "comparison.csv").read_text().splitlines()) == 19


def test_cohort_filter_scans_all_windows_before_exact_token_selection():
    tokens = ["a" * 16, "b" * 16]
    overrides = cohort_overrides({"log": {"path": "navsim_logs/mini/log.pkl"}}, tokens)
    assert "train_test_split.scene_filter.frame_interval=1" in overrides
    assert "train_test_split.scene_filter.max_scenes=2" in overrides
    assert "train_test_split.scene_filter.num_history_frames=4" in overrides
    with pytest.raises(ValueError): cohort_overrides({"log": {"path": "log.pkl"}}, [tokens[0], tokens[0]])


@pytest.mark.parametrize("changed", [None, "evidence_hash", "bundle_hash", "model_sources_hash", "protocol", "source_revision", "model_version"])
def test_case_evidence_is_bound_to_native_index_metadata(bundle, monkeypatch, tmp_path, changed):
    evidence = evidence_for(bundle)
    evidence.bundle_hash = bundle.bundle_hash
    evidence.protocol = PROTOCOL_NAME
    evidence.model_sources_hash = "b" * 64
    monkeypatch.setattr("hybrid_drive.experiments.offline_evidence.load_frozen_evidence", lambda *args: evidence)
    case = {"case_id": "r00", "scene_token": bundle.scene_token, "bundle_hash": bundle.bundle_hash}
    row = {**case, "status": "success", "evidence_path": "r00/evidence", "evidence_hash": evidence.evidence_hash}
    index = {"cases": [row], "model_version": evidence.context.request.model_version, "protocol": PROTOCOL_NAME,
             "source_revision": evidence.source_revision, "model_sources_hash": evidence.model_sources_hash}
    if changed in ("evidence_hash", "bundle_hash"):
        row[changed] = "0" * 64
    elif changed is not None:
        index[changed] = "different-native-input"
    if changed is None:
        assert scoring._case_evidence(bundle, case, tmp_path, index) is evidence
        return
    with pytest.raises(ValueError, match="differ"): scoring._case_evidence(bundle, case, tmp_path, index)
