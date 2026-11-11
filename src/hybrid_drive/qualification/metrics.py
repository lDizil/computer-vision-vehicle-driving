from __future__ import annotations

from collections import Counter
from itertools import combinations
import math
from numbers import Real

from hybrid_drive.contracts.slow import RiskStatus
from hybrid_drive.slow.protocol import MAX_EXPLANATION_CHARS, MAX_RESPONSE_BYTES


_OUTCOMES = ("completed", "invalid_output", "backend_error", "timeout", "process_exit", "superseded")
_FIELDS = {"model_id", "case_id", "seed", "outcome", "status", "vetoed_candidate_ids",
           "explanation", "latency_s", "raw_response", "input_hash", "prompt_sha256", "request_id"}
_CRITERIA = {
    "version": "controlled-diagnostic-exact-v1",
    "known_exact_output_rate": 1.0,
    "risk_recall_all_attempts": 1.0,
    "false_veto_rate": 0.0,
    "candidate_coverage": 1.0,
    "unknown_control_correct_rate": 1.0,
    "counterfactual_both_correct_rate": 1.0,
}
_DEFINITIONS = {
    "confusion": "TP/FP/TN/FN count only committed VETO/NO_VETO candidate decisions on known cases.",
    "risk_recall_all_attempts": "TP / all analytically risky candidate attempts, including abstentions and failures.",
    "false_veto_rate": "FP / all analytically clear candidate attempts, including abstentions and failures.",
    "candidate_coverage": "Committed candidate decisions / all known-case candidate attempts.",
    "unknown_controls": "Insufficient-evidence cases are separate; only completed UNKNOWN is correct.",
    "counterfactuals": "A pair passes only if both exact expected outputs are correct for the same model and seed.",
    "latency_s": "Measured end-to-end record durations for all outcomes; linear-interpolated sample quantiles in seconds.",
    "comparison": "Identical case inputs and actual prompt SHA256; controlled diagnostics, not road performance or model improvement.",
}


def summarize(suite, records, *, expected_models, seeds) -> dict:
    models = _unique_values(expected_models, "expected_models", _valid_text)
    seeds = _unique_values(seeds, "seeds", lambda value: type(value) is int and 0 <= value < 2**32)
    cases = {case.case_id: case for case in suite.cases}
    if not cases:
        raise ValueError("suite must contain cases")
    indexed = {}
    for record in records:
        _validate_record(record, cases, models, seeds)
        key = record["model_id"], record["case_id"], record["seed"]
        if key in indexed:
            raise ValueError(f"Duplicate qualification record: {key}")
        indexed[key] = record
    expected = {(model, case_id, seed) for model in models for case_id in cases for seed in seeds}
    missing = expected - indexed.keys()
    if missing:
        raise ValueError(f"Missing qualification records: {sorted(missing)}")
    manifest = suite.manifest()
    scope = manifest["scope"]
    if scope not in ("controlled_suite", "smoke_subset"):
        raise ValueError("suite manifest must declare controlled_suite or smoke_subset scope")
    result = {
        "schema": "hybrid-drive-qualification-summary-v1", "suite_version": suite.version,
        "scope": scope, "case_count": len(cases), "expected_models": list(models), "seeds": list(seeds),
        "verdict": "not_road_qualified", "usage": "advisory_only", "automatic_veto_allowed": False,
        "diagnostic_criteria": dict(_CRITERIA), "definitions": dict(_DEFINITIONS),
        "models": {}, "paired_comparisons": [],
    }
    for model in models:
        rows = [indexed[model, case.case_id, seed] for case in suite.cases for seed in seeds]
        result["models"][model] = _model_summary(suite, rows, indexed, model, seeds, scope)
    for left, right in combinations(models, 2):
        result["paired_comparisons"].append(_comparison(suite, indexed, seeds, left, right))
    return result


def _unique_values(values, name, predicate):
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a nonempty sequence of unique values")
    try:
        values = tuple(values)
    except TypeError as exc:
        raise ValueError(f"{name} must be a nonempty sequence of unique values") from exc
    if not values or any(not predicate(value) for value in values) or len(set(values)) != len(values):
        raise ValueError(f"{name} must be a nonempty sequence of unique valid values")
    return values


def _valid_text(value):
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        value.encode("utf-8")
    except UnicodeError:
        return False
    return True


def _validate_record(record, cases, models, seeds):
    if not isinstance(record, dict) or not _FIELDS <= record.keys():
        raise ValueError("Qualification record is missing required fields")
    model, case_id, seed = record["model_id"], record["case_id"], record["seed"]
    if not isinstance(model, str) or model not in models:
        raise ValueError("Record has an unexpected model_id")
    if not isinstance(case_id, str) or case_id not in cases:
        raise ValueError("Record has an unexpected case_id")
    if type(seed) is not int or seed not in seeds:
        raise ValueError("Record has an unexpected seed")
    if not isinstance(record["outcome"], str) or record["outcome"] not in _OUTCOMES:
        raise ValueError("Record has an unexpected outcome")
    if not _valid_text(record["request_id"]):
        raise ValueError("Record request_id must be nonempty UTF-8 text")
    if record["input_hash"] != cases[case_id].input_hash:
        raise ValueError("Record input_hash differs from the actual case input; comparisons require identical inputs")
    prompt_hash = record["prompt_sha256"]
    if (not isinstance(prompt_hash, str) or len(prompt_hash) != 64
            or any(value not in "0123456789abcdef" for value in prompt_hash)):
        raise ValueError("Record prompt_sha256 must be a lowercase SHA256 hex digest")
    latency = record["latency_s"]
    if isinstance(latency, bool) or not isinstance(latency, Real) or not math.isfinite(latency) or latency < 0:
        raise ValueError("Record latency_s must be finite and nonnegative")
    if record["status"] is not None and not isinstance(record["status"], str):
        raise ValueError("Record status must be text or None")
    vetoed = record["vetoed_candidate_ids"]
    if not isinstance(vetoed, list) or any(not isinstance(value, str) for value in vetoed):
        raise ValueError("Record vetoed_candidate_ids must be a list of strings")
    for name in ("explanation", "raw_response"):
        value = record[name]
        if value is not None and not isinstance(value, str):
            raise ValueError(f"Record {name} must be text or None")
        if value is not None:
            try:
                value.encode("utf-8")
            except UnicodeError as exc:
                raise ValueError(f"Record {name} must be valid UTF-8") from exc
    if record["outcome"] != "completed":
        return
    status = record["status"]
    if not isinstance(status, str) or status not in {value.value for value in RiskStatus}:
        raise ValueError("Completed record has an invalid status")
    known = {candidate.candidate_id for candidate in cases[case_id].plan.candidates}
    if not isinstance(vetoed, list) or any(not isinstance(value, str) or value not in known for value in vetoed):
        raise ValueError("Completed record vetoed_candidate_ids must name known candidates")
    if len(set(vetoed)) != len(vetoed) or (status == "VETO") != bool(vetoed):
        raise ValueError("Completed record status contradicts vetoed_candidate_ids")
    if not isinstance(record["explanation"], str) or len(record["explanation"]) > MAX_EXPLANATION_CHARS:
        raise ValueError("Completed record explanation exceeds the response contract")
    raw = record["raw_response"]
    if raw is not None and len(raw.encode("utf-8")) > MAX_RESPONSE_BYTES:
        raise ValueError("Completed record raw_response exceeds the response contract")


def _model_summary(suite, rows, indexed, model, seeds, scope):
    confusion = {name: 0 for name in ("tp", "fp", "tn", "fn", "abstained_positive", "abstained_negative",
                                     "failed_positive", "failed_negative", "total_positive", "total_negative")}
    known_attempts, known_correct = 0, 0
    unknown = {"attempts": 0, "correct": 0, "unknown": 0, "no_veto": 0, "veto": 0, "failed": 0}
    for case in suite.cases:
        for seed in seeds:
            row = indexed[model, case.case_id, seed]
            if case.expected_status is RiskStatus.UNKNOWN:
                unknown["attempts"] += 1
                if row["outcome"] != "completed":
                    unknown["failed"] += 1
                    continue
                unknown[row["status"].lower()] += 1
                unknown["correct"] += int(_correct(case, row))
                continue
            known_attempts += 1
            known_correct += int(_correct(case, row))
            _count_candidates(case, row, confusion)
    committed = sum(confusion[name] for name in ("tp", "fp", "tn", "fn"))
    total = confusion["total_positive"] + confusion["total_negative"]
    rates = {
        "risk_recall_all_attempts": _ratio(confusion["tp"], confusion["total_positive"]),
        "false_veto_rate": _ratio(confusion["fp"], confusion["total_negative"]),
        "candidate_coverage": _ratio(committed, total),
        "positive_coverage": _ratio(confusion["tp"] + confusion["fn"], confusion["total_positive"]),
        "negative_coverage": _ratio(confusion["fp"] + confusion["tn"], confusion["total_negative"]),
        "risk_recall_committed_only": _ratio(confusion["tp"], confusion["tp"] + confusion["fn"]),
        "false_veto_rate_committed_only": _ratio(confusion["fp"], confusion["fp"] + confusion["tn"]),
        "known_exact_output_rate": _ratio(known_correct, known_attempts),
    }
    unknown["correct_rate"] = _ratio(unknown["correct"], unknown["attempts"])
    pairs = _counterfactuals(suite, indexed, model, seeds)
    checks = {key: rates[key] == threshold for key, threshold in _CRITERIA.items() if key in rates}
    checks["unknown_control_correct_rate"] = unknown["correct_rate"] == 1.0
    checks["counterfactual_both_correct_rate"] = pairs["both_correct_rate"] == 1.0
    checks["full_controlled_suite"] = scope == "controlled_suite"
    counts = Counter(row["outcome"] for row in rows)
    return {
        "attempts": len(rows), "outcomes": {name: counts[name] for name in _OUTCOMES},
        "completion_rate": _ratio(counts["completed"], len(rows)),
        "known_attempts": known_attempts, "known_exact_correct": known_correct,
        "confusion": confusion, "rates": rates, "unknown_controls": unknown, "counterfactuals": pairs,
        "latency_s": _latencies(rows),
        "diagnostic_gate": {"status": "pass" if all(checks.values()) else "fail", "checks": checks,
                            "purpose": "controlled_diagnostic_only"},
    }


def _count_candidates(case, row, counts):
    risky = set(case.expected_veto_ids)
    for candidate in case.plan.candidates:
        positive = candidate.candidate_id in risky
        suffix = "positive" if positive else "negative"
        counts["total_" + suffix] += 1
        if row["outcome"] != "completed":
            counts["failed_" + suffix] += 1
        elif row["status"] == "UNKNOWN":
            counts["abstained_" + suffix] += 1
        else:
            veto = candidate.candidate_id in row["vetoed_candidate_ids"]
            counts[("tp" if positive else "fp") if veto else ("fn" if positive else "tn")] += 1


def _correct(case, row):
    return (row["outcome"] == "completed" and row["status"] == case.expected_status.value
            and set(row["vetoed_candidate_ids"]) == set(case.expected_veto_ids))


def _counterfactuals(suite, indexed, model, seeds):
    cases = {case.case_id: case for case in suite.cases}
    results = []
    for pair in suite.pairs:
        for seed in seeds:
            first = _correct(cases[pair.first_case_id], indexed[model, pair.first_case_id, seed])
            second = _correct(cases[pair.second_case_id], indexed[model, pair.second_case_id, seed])
            results.append({"pair_id": pair.pair_id, "kind": pair.kind, "seed": seed,
                            "first_case_id": pair.first_case_id, "second_case_id": pair.second_case_id,
                            "first_correct": first, "second_correct": second, "both_correct": first and second})
    passed = sum(row["both_correct"] for row in results)
    return {"attempts": len(results), "both_correct": passed, "both_correct_rate": _ratio(passed, len(results)),
            "results": results}


def _latencies(rows):
    values = sorted(float(row["latency_s"]) for row in rows)

    def quantile(probability):
        position = probability * (len(values) - 1)
        lower = math.floor(position)
        return values[lower] + (values[math.ceil(position)] - values[lower]) * (position - lower)

    return {"sample_count": len(values), "p50": quantile(0.5), "p95": quantile(0.95), "p99": quantile(0.99),
            "mean": math.fsum(value / len(values) for value in values),
            "interpretation": "descriptive_sample_quantiles_not_tail_confidence"}


def _comparison(suite, indexed, seeds, left, right):
    counts = {"both_correct": 0, "left_only_correct": 0, "right_only_correct": 0, "neither_correct": 0}
    differences = []
    for case in suite.cases:
        for seed in seeds:
            a, b = indexed[left, case.case_id, seed], indexed[right, case.case_id, seed]
            if a["input_hash"] != b["input_hash"]:
                raise ValueError("Paired comparison requires identical case input_hash across models")
            if a["prompt_sha256"] != b["prompt_sha256"]:
                raise ValueError("Paired comparison requires identical prompt_sha256 across models")
            good_a, good_b = _correct(case, a), _correct(case, b)
            key = ("both_correct" if good_a and good_b else "left_only_correct" if good_a
                   else "right_only_correct" if good_b else "neither_correct")
            counts[key] += 1
            differences.append(b["latency_s"] - a["latency_s"])
    return {"left_model": left, "right_model": right, "matched_attempts": len(differences), **counts,
            "mean_latency_delta_s_right_minus_left": math.fsum(value / len(differences) for value in differences),
            "interpretation": "matched_input_diagnostic_only"}


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None
