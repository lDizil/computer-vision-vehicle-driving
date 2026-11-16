import copy
import json

import pytest

from hybrid_drive.experiments.fallback_comparison import compare_fallback_runs


def result(curved=False):
    rows = []
    for index in range(6):
        for mode in ("baseline", "checks_only", "combined"):
            invoked = index < 2 and mode == "combined"
            failed = invoked and not curved
            row = dict(case_id=f"r{index:02d}", mode=mode, scene_token=f"{index:016x}",
                       failed=failed, metrics=None if failed else {"score":.8 if invoked else .9},
                       error="unsupported_motion" if failed else None, raw_candidate_hash="a"*64,
                       selected_trajectory_hash="c"*64 if invoked and curved else "b"*64,
                       fallback_invoked=invoked, fallback_model_report={"model_id":"curved"} if invoked and curved else None)
            if mode == "combined":
                row.update(evidence_hash="d"*64, native_source_revision="e"*40,
                           native_elapsed_to_decision_s=16.5, slow_outcome="completed",
                           slow_disposition="applied" if invoked else "unknown_result",
                           slow_status="VETO" if invoked else "UNKNOWN", slow_used=invoked)
            rows.append(row)
    policy = {"limits":{"max_acceleration_mps2":8,"max_jerk_mps3":20},
              "braking":{"deceleration_mps2":4,"jerk_mps3":4}}
    if curved: policy["fallback_model"] = "curved_snapshot_hypothesis"
    return dict(schema="hybrid-drive-real-cohort-scores-v1", execution="offline_replay_original_native_clock",
                source_revision="a"*40, fast_source_revision="b"*40, cohort_hash="c"*64, sources_hash="d"*64,
                native_evidence={"evidence_index_hash":"e"*64}, policy=policy,
                planned_cases=6, planned_mode_attempts=18, executed_modes=["baseline","checks_only","combined"],
                records=rows, status="success" if curved else "failed",
                **({"motion_observations":{"index_hash":"f"*64}} if curved else {}))


def run(tmp_path, old=None, new=None):
    for name, data in (("old",result() if old is None else old),("new",result(True) if new is None else new)):
        root=tmp_path/name;root.mkdir();(root/"result.json").write_text(json.dumps(data))
    return compare_fallback_runs(tmp_path/"old",tmp_path/"new",tmp_path/"out")


def test_retains_failed_attempts_and_separates_recovery_from_paired_quality(tmp_path):
    path, comparison = run(tmp_path)
    combined = comparison["summary"]["combined"]
    assert comparison["planned_pairs"] == len(comparison["pairs"]) == 18
    assert combined["planned_pairs"] == 6
    assert combined["control_failed"] == 2 and combined["curved_failed"] == 0
    assert combined["control_score_failed_zero_mean"] == pytest.approx(.6)
    assert combined["curved_score_failed_zero_mean"] == pytest.approx(5.2/6)
    assert combined["both_valid_pairs"] == 4
    assert combined["paired_score_delta_both_valid_mean"] == 0
    assert comparison["control_status"] == "failed"
    assert comparison["quality_claim"] is comparison["driving_safety_claim"] is False
    assert (path/"report.html").is_file()
    assert "unsupported_motion" in (path/"report.html").read_text()


@pytest.mark.parametrize("field", ["cohort_hash","sources_hash","fast_source_revision","native_evidence"])
def test_rejects_different_experimental_inputs(tmp_path, field):
    data = result(True);data[field] = "changed"
    with pytest.raises(ValueError, match="identity"):
        run(tmp_path, new=data)


def test_rejects_relaxed_braking_caps(tmp_path):
    data = result(True);data["policy"]["braking"]["jerk_mps3"] = 8
    with pytest.raises(ValueError, match="caps"):
        run(tmp_path, new=data)


@pytest.mark.parametrize("field,value", [
    ("native_elapsed_to_decision_s",1),("evidence_hash","f"*64),
    ("slow_disposition","deadline_exceeded"),("raw_candidate_hash","f"*64),
])
def test_rejects_changed_clock_evidence_or_fast_candidate(tmp_path, field, value):
    data = result(True);data["records"][2][field] = value
    with pytest.raises(ValueError): run(tmp_path, new=data)


@pytest.mark.parametrize("corruption", ["drop","duplicate","failed_metrics","nonfinite","selected"])
def test_rejects_dropped_or_corrupted_attempts(tmp_path, corruption):
    old,new=result(),result(True)
    if corruption=="drop": new["records"].pop()
    if corruption=="duplicate": new["records"][-1]=copy.deepcopy(new["records"][0])
    if corruption=="failed_metrics": old["records"][2]["metrics"]={"score":1}
    if corruption=="nonfinite": new["records"][0]["metrics"]["score"]=float("nan")
    if corruption=="selected": new["records"][0]["selected_trajectory_hash"]="f"*64
    with pytest.raises(ValueError): run(tmp_path, old, new)


def test_report_escapes_error_text(tmp_path):
    data=result();data["records"][2]["error"]="<script>bad()</script>"
    path,_=run(tmp_path, old=data)
    text=(path/"report.html").read_text()
    assert "<script>bad()" not in text and "&lt;script&gt;bad()" in text
