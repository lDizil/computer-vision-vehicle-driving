import json

from hybrid_drive.experiments import cohort_report
from hybrid_drive.experiments.protocol import content_hash
from hybrid_drive.experiments.offline_evidence import INDEX_SCHEMA
from hybrid_drive.slow.real_grounding import PROTOCOL_NAME


def test_invalid_evidence_index_cannot_make_report_read_outside_artifact_root(tmp_path, monkeypatch):
    evidence_root = tmp_path / "evidence"; evidence_root.mkdir()
    outside = tmp_path / "outside"; outside.mkdir()
    (outside / "CAM_F0.png").write_bytes(b"private outside artifact")
    run = tmp_path / "report"; run.mkdir(); (run / "comparison.csv").write_text("mode,score\n")
    cases = [{"case_id": f"r{i:02d}", "scene_token": f"{i:016x}", "source_list_index": i * 20 + 3} for i in range(6)]
    cohort = {"cohort_hash": "a" * 64, "cases": cases}
    monkeypatch.setattr(cohort_report, "load_fast_cohort", lambda root: cohort)
    index = {"schema": INDEX_SCHEMA, "source_revision": "b" * 40, "cohort_hash": "a" * 64,
             "model_version": "fixture-real-path-v1", "model_sources_hash": "c" * 64, "protocol": PROTOCOL_NAME,
             "cases": [{"case_id": c["case_id"], "scene_token": c["scene_token"], "bundle_hash": "d" * 64,
                        "status": "success", "evidence_path": "../outside", "evidence_hash": "e" * 64, "error": None}
                       for c in cases]}
    index["evidence_index_hash"] = content_hash(index)
    (evidence_root / "cohort-evidence.json").write_text(json.dumps(index))
    summary = {"valid": 0, "attempted": 6, "score_mean_all_attempts_failure_zero": 0,
               "metrics_mean_valid_only": {"no_at_fault_collisions": None, "ego_progress": None},
               "ade_m_valid_only": None, "fde_m_valid_only": None}
    result = {"records": [], "summary": {"combined": summary}}
    original = cohort_report.read_bounded
    def guarded(path, limit):
        assert path.is_relative_to(run), "Report attempted to read outside its verified inputs"
        return original(path, limit)
    monkeypatch.setattr(cohort_report, "read_bounded", guarded)
    cohort_report.write_cohort_report(run, result, tmp_path / "fast", evidence_root)
    html = (run / "report.html").read_text()
    assert "private outside artifact" not in html and "Нет native evidence" in html
