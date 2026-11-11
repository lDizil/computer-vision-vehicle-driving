from copy import deepcopy

import pytest

from hybrid_drive.contracts.slow import RiskStatus
from hybrid_drive.qualification.cases import build_suite
from hybrid_drive.qualification.metrics import summarize


@pytest.fixture(scope="module")
def suite():
    return build_suite()


def complete_records(suite, models=("small",), seeds=(0,)):
    return [dict(model_id=model, case_id=case.case_id, seed=seed, outcome="completed",
                 status=case.expected_status.value, vetoed_candidate_ids=list(case.expected_veto_ids),
                 explanation="Visible geometry.", latency_s=float(index + 1), raw_response="{}",
                 input_hash=case.input_hash, prompt_sha256=f"{index:064x}", request_id=f"{model}-{case.case_id}-{seed}")
            for model in models for index, case in enumerate(suite.cases) for seed in seeds]


def test_perfect_controlled_results_never_qualify_road_use(suite):
    summary = summarize(suite, complete_records(suite), expected_models=["small"], seeds=[0])
    value = summary["models"]["small"]
    assert summary["verdict"] == "not_road_qualified"
    assert summary["usage"] == "advisory_only" and summary["automatic_veto_allowed"] is False
    assert summary["scope"] == "controlled_suite"
    assert value["confusion"] == dict(tp=8, fp=0, tn=8, fn=0, abstained_positive=0,
                                      abstained_negative=0, failed_positive=0, failed_negative=0,
                                      total_positive=8, total_negative=8)
    assert value["diagnostic_gate"]["status"] == "pass"
    assert value["unknown_controls"]["attempts"] == value["unknown_controls"]["correct"] == 2
    assert value["counterfactuals"]["attempts"] > 0
    assert value["counterfactuals"]["both_correct_rate"] == 1


def test_all_unknown_cannot_look_like_safe_perfect_false_veto_rate(suite):
    records = complete_records(suite)
    for record in records:
        record.update(status="UNKNOWN", vetoed_candidate_ids=[])
    value = summarize(suite, records, expected_models=["small"], seeds=[0])["models"]["small"]
    counts = value["confusion"]
    assert counts["tp"] == counts["fp"] == counts["tn"] == counts["fn"] == 0
    assert counts["abstained_positive"] == counts["abstained_negative"] == 8
    assert value["rates"]["false_veto_rate"] == 0
    assert value["rates"]["risk_recall_all_attempts"] == value["rates"]["candidate_coverage"] == 0
    assert value["rates"]["risk_recall_committed_only"] is None
    assert value["unknown_controls"]["correct"] == 2
    assert value["diagnostic_gate"]["status"] == "fail"


def test_always_veto_has_full_coverage_but_false_positives_and_wrong_unknown_controls(suite):
    records = complete_records(suite)
    for case, record in zip(suite.cases, records):
        record.update(status="VETO", vetoed_candidate_ids=[candidate.candidate_id for candidate in case.plan.candidates])
    value = summarize(suite, records, expected_models=["small"], seeds=[0])["models"]["small"]
    assert value["confusion"]["tp"] == value["confusion"]["fp"] == 8
    assert value["rates"]["candidate_coverage"] == value["rates"]["risk_recall_all_attempts"] == 1
    assert value["rates"]["false_veto_rate"] == 1
    assert value["unknown_controls"]["veto"] == 2
    assert value["diagnostic_gate"]["status"] == "fail"


@pytest.mark.parametrize("outcome", ["invalid_output", "backend_error", "timeout", "process_exit", "superseded"])
def test_every_failure_keeps_candidate_denominators_and_ignores_stale_valid_status(suite, outcome):
    records = complete_records(suite)
    record = next(record for record in records if record["case_id"] == "q00")
    record["outcome"] = outcome
    value = summarize(suite, records, expected_models=["small"], seeds=[0])["models"]["small"]
    assert value["confusion"]["tp"] == value["confusion"]["tn"] == 7
    assert value["confusion"]["total_positive"] == value["confusion"]["total_negative"] == 8
    assert value["confusion"]["failed_positive"] == value["confusion"]["failed_negative"] == 1
    assert value["rates"]["risk_recall_all_attempts"] == value["rates"]["candidate_coverage"] == 7 / 8
    assert value["outcomes"][outcome] == 1 and value["outcomes"]["completed"] == 9
    assert value["latency_s"]["sample_count"] == 10
    assert value["diagnostic_gate"]["status"] == "fail"


def test_unknown_failure_is_not_a_correct_abstention_or_a_known_candidate_failure(suite):
    records = complete_records(suite)
    records[-1].update(outcome="timeout", status="UNKNOWN", vetoed_candidate_ids=[])
    value = summarize(suite, records, expected_models=["small"], seeds=[0])["models"]["small"]
    assert value["unknown_controls"]["correct"] == value["unknown_controls"]["failed"] == 1
    assert value["unknown_controls"]["correct_rate"] == 0.5
    assert value["confusion"]["failed_positive"] == value["confusion"]["failed_negative"] == 0


def test_failed_record_payload_is_not_reinterpreted_as_a_parsed_status(suite):
    records = complete_records(suite)
    records[0].update(outcome="invalid_output", status="unparsed", vetoed_candidate_ids=["foreign", "foreign"],
                      raw_response="x" * 10000, explanation=None)
    value = summarize(suite, records, expected_models=["small"], seeds=[0])["models"]["small"]
    assert value["outcomes"]["invalid_output"] == 1
    assert value["confusion"]["failed_positive"] == value["confusion"]["failed_negative"] == 1
    assert value["diagnostic_gate"]["status"] == "fail"


@pytest.mark.parametrize("change", [
    lambda rows: rows.pop(),
    lambda rows: rows.append(deepcopy(rows[0])),
    lambda rows: rows[0].update(model_id="unrequested"),
    lambda rows: rows[0].update(case_id="missing"),
    lambda rows: rows[0].update(seed=1),
    lambda rows: rows[0].update(seed=True),
    lambda rows: rows[0].pop("request_id"),
    lambda rows: rows[0].pop("prompt_sha256"),
])
def test_exact_matrix_cannot_silently_drop_missing_duplicate_or_unrequested_data(suite, change):
    rows = complete_records(suite)
    change(rows)
    with pytest.raises(ValueError):
        summarize(suite, rows, expected_models=["small"], seeds=[0])


@pytest.mark.parametrize("updates", [
    {"status": "VETO", "vetoed_candidate_ids": []},
    {"status": "NO_VETO", "vetoed_candidate_ids": ["unknown"]},
    {"status": "UNKNOWN", "vetoed_candidate_ids": None},
    {"status": "maybe"}, {"status": None}, {"explanation": None},
    {"explanation": "x" * 513}, {"explanation": "\ud800"},
    {"latency_s": -1}, {"latency_s": float("nan")}, {"latency_s": float("inf")}, {"latency_s": True},
    {"raw_response": "界" * 8192}, {"outcome": "not_run"}, {"request_id": ""},
])
def test_malformed_completed_records_fail_instead_of_becoming_quality_results(suite, updates):
    records = complete_records(suite)
    records[0].update(updates)
    with pytest.raises(ValueError):
        summarize(suite, records, expected_models=["small"], seeds=[0])


def test_duplicate_veto_ids_are_rejected_even_if_sets_match_expected(suite):
    records = complete_records(suite)
    records[0]["vetoed_candidate_ids"] *= 2
    with pytest.raises(ValueError, match="contradicts"):
        summarize(suite, records, expected_models=["small"], seeds=[0])


@pytest.mark.parametrize("models,seeds", [([], [0]), (["small", "small"], [0]), (["small"], []),
                                           (["small"], [0, 0]), (["small"], [True]),
                                           (["small"], [-1]), (["small"], [2**32])])
def test_invalid_expected_axes_do_not_define_an_empty_or_ambiguous_matrix(suite, models, seeds):
    with pytest.raises(ValueError):
        summarize(suite, [], expected_models=models, seeds=seeds)


def test_hash_mismatch_blocks_per_model_and_paired_comparisons(suite):
    records = complete_records(suite, models=("small", "large"))
    records[-1]["input_hash"] = "0" * 64
    with pytest.raises(ValueError, match="input_hash"):
        summarize(suite, records, expected_models=["small", "large"], seeds=[0])


@pytest.mark.parametrize("digest", [None, 1, [], "", "a" * 63, "a" * 65, "A" * 64, "g" * 64])
def test_prompt_sha256_must_be_a_complete_lowercase_digest_even_on_failures(suite, digest):
    records = complete_records(suite)
    records[0].update(prompt_sha256=digest, outcome="timeout")
    with pytest.raises(ValueError, match="prompt_sha256"):
        summarize(suite, records, expected_models=["small"], seeds=[0])


@pytest.mark.parametrize("outcome", ["completed", "timeout"])
def test_matching_case_hash_cannot_hide_different_actual_prompts_across_models(suite, outcome):
    records = complete_records(suite, models=("small", "large"))
    records[-1].update(prompt_sha256="f" * 64, outcome=outcome)
    assert records[-1]["input_hash"] == records[len(suite.cases) - 1]["input_hash"]
    with pytest.raises(ValueError, match="identical prompt_sha256"):
        summarize(suite, records, expected_models=["small", "large"], seeds=[0])


def test_counterfactual_requires_both_ground_truth_outputs_not_merely_different_responses(suite):
    pair = next(pair for pair in suite.pairs if pair.kind == "image_change")
    records = complete_records(suite)
    for index, record in enumerate(records):
        if record["case_id"] in (pair.first_case_id, pair.second_case_id):
            record.update(status="UNKNOWN", vetoed_candidate_ids=[], explanation=f"Different text {index}")
    value = summarize(suite, records, expected_models=["small"], seeds=[0])["models"]["small"]
    result = next(result for result in value["counterfactuals"]["results"] if result["pair_id"] == pair.pair_id)
    assert not result["first_correct"] and not result["second_correct"] and not result["both_correct"]
    assert value["diagnostic_gate"]["status"] == "fail"


def test_candidate_binding_pair_does_not_accept_reused_ids_from_the_first_image(suite):
    pair = next(pair for pair in suite.pairs if pair.kind == "candidate_change")
    records = complete_records(suite)
    first = next(record for record in records if record["case_id"] == pair.first_case_id)
    second = next(record for record in records if record["case_id"] == pair.second_case_id)
    second.update(status=first["status"], vetoed_candidate_ids=list(first["vetoed_candidate_ids"]))
    value = summarize(suite, records, expected_models=["small"], seeds=[0])["models"]["small"]
    result = next(result for result in value["counterfactuals"]["results"] if result["pair_id"] == pair.pair_id)
    assert result["first_correct"] and not result["second_correct"] and not result["both_correct"]


def test_quantiles_use_measured_attempts_and_comparisons_pair_seed_and_case(suite):
    records = complete_records(suite, models=("small", "large"), seeds=(0, 4))
    for record in records:
        record["latency_s"] = (0 if record["seed"] == 0 else 10) + (2 if record["model_id"] == "large" else 0)
    records[0].update(outcome="timeout", status=None, vetoed_candidate_ids=[])
    result = summarize(suite, records, expected_models=["small", "large"], seeds=[0, 4])
    latency = result["models"]["small"]["latency_s"]
    assert latency["sample_count"] == 20 and latency["p50"] == 5
    assert latency["p95"] == latency["p99"] == 10
    comparison = result["paired_comparisons"][0]
    assert comparison["matched_attempts"] == 20 and comparison["right_only_correct"] == 1
    assert comparison["both_correct"] == 19 and comparison["mean_latency_delta_s_right_minus_left"] == 2


def test_latency_quantiles_interpolate_raw_records_not_precomputed_quantiles(suite):
    records = complete_records(suite)
    value = summarize(suite, records, expected_models=["small"], seeds=[0])["models"]["small"]
    assert value["latency_s"]["p50"] == 5.5
    assert value["latency_s"]["p95"] == pytest.approx(9.55)
    assert value["latency_s"]["p99"] == pytest.approx(9.91)


def test_subset_is_explicitly_smoke_even_when_every_observed_response_is_correct(suite):
    subset = suite.select(["q00", "q01", "q08"])
    value = summarize(subset, complete_records(subset), expected_models=["small"], seeds=[0])
    assert value["scope"] == "smoke_subset"
    assert value["models"]["small"]["diagnostic_gate"]["status"] == "fail"
    assert value["models"]["small"]["diagnostic_gate"]["checks"]["full_controlled_suite"] is False
