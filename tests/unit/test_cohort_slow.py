from types import SimpleNamespace
import json

from test_real_replay import bundle, FixtureFactory
from hybrid_drive.experiments import cohort_slow
from hybrid_drive.experiments.offline_evidence import load_cohort_evidence_index


def test_unconfirmed_group_stops_entire_cohort_not_just_seed_series(bundle, tmp_path, monkeypatch):
    cases = [{"case_id": f"r{i:02d}", "scene_token": bundle.scene_token, "bundle_hash": bundle.bundle_hash,
              "bundle_path": f"r{i:02d}/bundle", "status": "success"} for i in range(6)]
    cohort = {"cohort_hash": "a" * 64, "cases": cases}
    monkeypatch.setattr(cohort_slow, "load_fast_cohort", lambda path: cohort)
    monkeypatch.setattr(cohort_slow, "load_real_bundle", lambda path: bundle)
    calls = []
    def replay(bundle, factory, root, **kwargs):
        calls.append(root)
        root.mkdir(parents=True)
        (root / "records.json").write_text(json.dumps([{"cleanup_confirmed": False, "latency_s": 1}]))
        return root, {"status": "failed"}
    monkeypatch.setattr(cohort_slow, "run_real_replay", replay)
    monkeypatch.setattr(cohort_slow, "write_frozen_evidence", lambda *args: SimpleNamespace(
        evidence_hash="b" * 64, outcome="cleanup_error", result=None, cleanup_confirmed=False))
    directory, result = cohort_slow.run_cohort_slow(tmp_path, FixtureFactory("unused", {"model_version": "fixture"}),
                                                  tmp_path / "output", revision="c" * 40)
    assert len(calls) == 1 and result["planned_cases"] == 6 and result["attempts"] == 1
    index = load_cohort_evidence_index(directory, cohort_hash=cohort["cohort_hash"])
    assert len(index["cases"]) == 6
    assert all(row["status"] == "not_started" for row in index["cases"][1:])
    assert result["status"] == "failed"


def test_uncertain_execution_exception_is_not_followed_by_another_model(bundle, tmp_path, monkeypatch):
    cohort = {"cohort_hash": "a" * 64, "cases": [{"case_id": f"r{i:02d}", "scene_token": bundle.scene_token,
              "bundle_hash": bundle.bundle_hash, "bundle_path": f"r{i:02d}/bundle", "status": "success"} for i in range(6)]}
    monkeypatch.setattr(cohort_slow, "load_fast_cohort", lambda path: cohort)
    monkeypatch.setattr(cohort_slow, "load_real_bundle", lambda path: bundle)
    calls = []
    def replay(*args, **kwargs):
        calls.append(1)
        raise OSError("artifact write failed")
    monkeypatch.setattr(cohort_slow, "run_real_replay", replay)
    _, result = cohort_slow.run_cohort_slow(tmp_path, FixtureFactory("unused", {"model_version": "fixture"}),
                                           tmp_path / "output", revision="c" * 40)
    assert calls == [1] and result["attempts"] == 1
