from dataclasses import replace
import json

import pytest

from hybrid_drive.contracts.slow import RiskStatus
from hybrid_drive.slow import replay
from scripts import run_slow_replay


@pytest.fixture(scope="module")
def report():
    return replay.run_replay()


def scenario(report, name):
    return next(value for value in report["scenarios"] if value["name"] == name)


def test_fixed_temporal_suite_is_reproducible_and_passes(report):
    assert report["passed"]
    assert len(report["scenarios"]) == 16
    assert all(all(value["checks"].values()) for value in report["scenarios"])
    assert json.dumps(report, sort_keys=True, allow_nan=False) == json.dumps(replay.run_replay(), sort_keys=True, allow_nan=False)
    assert report["counts"] == {
        "submitted": 19, "delivered": 20, "decision_calls": 18, "decisions": 17,
        "applied": 8, "degradations": 9, "deadline_misses": 2, "late_deliveries": 1,
        "fallbacks": 1, "failures": 1,
    }
    assert report["delivery_outcomes"] == {"completed": 17, "invalid_output": 3}


def test_inclusive_deadline_and_late_decision_have_distinct_evidence(report):
    exact = scenario(report, "exact_deadline_and_sim_validity")
    assert exact["decisions"][0]["slow_disposition"] == "applied"
    assert exact["response_age_s"] == [0]
    assert not exact["deliveries"][0]["after_deadline"]
    late = scenario(report, "late_answer")
    assert late["counts"]["late_deliveries"] == 1
    delayed_decision = scenario(report, "decision_after_deadline")
    assert delayed_decision["counts"]["late_deliveries"] == 0
    assert delayed_decision["counts"]["deadline_misses"] == 1
    assert delayed_decision["response_age_s"] == []


def test_invalid_outputs_do_not_become_no_veto(report):
    for name in ("invalid_json", "wrong_context_hash", "unknown_candidate"):
        value = scenario(report, name)
        assert value["deliveries"][0]["outcome"] == "invalid_output"
        assert value["decisions"][0]["slow_status"] == "UNKNOWN"
        assert value["counts"]["applied"] == 0


def test_frozen_tick_does_not_rebase_output_and_new_plan_can_use_its_own_reply(report):
    value = scenario(report, "frozen_fast_tick_then_new_plan")
    early, frozen, new = value["decisions"]
    assert early == frozen
    assert frozen["decided_at_monotonic_s"] == 10.1
    assert frozen["simulation_time_s"] == 100
    assert frozen["candidate_id"] == "first"
    assert new["plan_id"] != frozen["plan_id"]
    assert new["request_id"] != frozen["request_id"]
    assert new["candidate_id"] == "second"
    assert new["response_age_s"] == 0.25
    assert value["counts"]["decisions"] == 2
    assert value["counts"]["decision_calls"] == 3


def test_out_of_order_and_conflicting_duplicate_results_are_not_reapplied(report):
    duplicate = scenario(report, "duplicate_answers_first_completion_wins")
    assert duplicate["counts"]["delivered"] == 2
    assert duplicate["counts"]["applied"] == 1
    assert duplicate["decisions"][0]["candidate_id"] == "second"
    reordered = scenario(report, "reordered_old_plan_arrives_last")
    assert reordered["decisions"][0]["slow_status"] == "NO_VETO"
    assert reordered["decisions"][0]["request_id"] == reordered["deliveries"][0]["request_id"]
    assert reordered["decisions"][0]["request_id"] != reordered["deliveries"][1]["request_id"]
    assert scenario(report, "old_plan_only")["counts"]["applied"] == 0


def test_all_veto_uses_domain_fallback_and_preserves_failure_status(report):
    fallback = scenario(report, "all_candidates_vetoed")["decisions"][0]
    assert fallback["selection_status"] == "fallback"
    assert fallback["fallback_failure"] is None
    failure = scenario(report, "fallback_failure")["decisions"][0]
    assert failure["failed"]
    assert failure["selection_status"] == "failure"
    assert failure["fallback_failure"] == "unsupported_motion"


def test_report_makes_no_inference_process_or_resource_measurement_claim(report):
    assert report["execution"]["inference_executed"] is False
    assert report["runtime_process_tests"]["status"] == "not_run_by_replay"
    assert not {"cpu", "gpu", "queue_depth", "wall_clock_latency_s"} & report.keys()
    assert not {"cpu", "gpu", "queue_depth", "wall_clock_latency_s"} & report["counts"].keys()


def test_fixed_expectations_detect_silently_dropped_vetoes(monkeypatch):
    parse = replay.parse_response

    def discard_veto(raw, context):
        result = parse(raw, context)
        return replace(result, status=RiskStatus.NO_VETO, vetoed_candidate_ids=())

    monkeypatch.setattr(replay, "parse_response", discard_veto)
    broken = replay.run_replay()
    assert not broken["passed"]
    assert not scenario(broken, "timely_veto")["checks"]["expected_decisions"]


@pytest.mark.parametrize("passed,exit_code", [(True, 0), (False, 1)])
def test_cli_writes_machine_report_and_fails_on_contract_failure(tmp_path, monkeypatch, capsys, passed, exit_code):
    expected = {"passed": passed, "counts": {"submitted": 1}}
    monkeypatch.setattr(run_slow_replay, "run_replay", lambda: expected)
    path = tmp_path / "nested" / "report.json"
    assert run_slow_replay.main(["--output", str(path)]) == exit_code
    assert json.loads(path.read_text()) == expected
    assert json.loads(capsys.readouterr().out) == expected
