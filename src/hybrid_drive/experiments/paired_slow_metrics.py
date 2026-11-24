"""Measure whether Slow follows the controlled motion change in both worlds."""

from math import copysign, fsum, isfinite
from pathlib import Path

from hybrid_drive.experiments.bounded_io import require_hex
from hybrid_drive.experiments.fast_capture import artifact_hash, load_json
from hybrid_drive.experiments.io import write_json


CONTROL_PREDICTION_SCHEMA = "hybrid-drive-paired-slow-control-predictions-v1"
CONTROL_TARGET_SCHEMA = "hybrid-drive-paired-slow-control-targets-v1"
ARMS = ("trained", "untrained", "no_motion")
TIE_MARGIN = 1e-5


def score_controls(predictions: Path, targets: Path, output: Path, *, sealed_sha256: str) -> dict:
    require_hex(sealed_sha256, 64, "control prediction seal")
    if output.exists() or output.is_symlink():
        raise ValueError("Scoring output must be new")
    if artifact_hash(predictions) != sealed_sha256:
        raise ValueError("Control predictions differ from the recorded seal")
    estimates = load_json(predictions)
    truth = load_json(targets)
    if (estimates["schema"] != CONTROL_PREDICTION_SCHEMA or truth["schema"] != CONTROL_TARGET_SCHEMA
            or truth["control_prediction_sha256"] != sealed_sha256 or estimates["revision"] != truth["revision"]
            or estimates["heldout_targets_opened"] is not False):
        raise ValueError("Expected the matching sealed control run and geometric labels")
    result = summarize_controls(estimates["rows"], truth["rows"])
    result.update({"schema": "hybrid-drive-paired-slow-control-score-v1", "revision": estimates["revision"],
                   "prediction_sha256": sealed_sha256, "target_sha256": artifact_hash(targets)})
    output.mkdir(parents=True)
    write_json(output / "control-score.json", result)
    return result


def summarize_controls(estimated, expected):
    records, by_snapshot = [], {}
    for row, target in zip(estimated, expected, strict=True):
        base = ("sample_token", "recording", "snapshot_sha256")
        token = row["sample_token"]
        if any(row[k] != target[k] for k in base) or token in by_snapshot:
            raise ValueError("Duplicate or mismatched control snapshot identity")
        require_hex(row["snapshot_sha256"], 64, "snapshot SHA")
        seen = set()
        current = []
        for group, truth in zip(row["groups"], target["groups"], strict=True):
            item = _compare_worlds(group, truth)
            identity = tuple(group["world_sha256"])
            if any(key in seen for key in identity):
                raise ValueError("A counterfactual world occurs more than once in the snapshot")
            seen.update(identity)
            current.append(item)
            records.append((token, item))
        by_snapshot[token] = {"recording": row["recording"], "summary": _measure(current)}
    all_items = [item for _, item in records]
    hard = [(token, item) for token, item in records if item["same_vx_sign"]]
    all_summary = _measured_slice(records)
    hard_summary = _measured_slice(hard)
    gate = _gate(all_summary, hard_summary)
    return {"planned_snapshots": len(estimated), "all": all_summary, "hard_same_vx_sign": hard_summary,
            "by_snapshot": by_snapshot, "gate": gate, "outcomes": all_items}


def _compare_worlds(group, truth):
    candidates, worlds = group["candidate_ids"], group["world_sha256"]
    if (candidates != truth["candidate_ids"] or worlds != truth["world_sha256"]
            or len(candidates) != 2 or len(set(candidates)) != 2 or len(worlds) != 2 or len(set(worlds)) != 2):
        raise ValueError("Each crossover requires two identical candidate IDs in two distinct worlds")
    for digest in worlds:
        require_hex(digest, 64, "World SHA")
    labels = truth["blocked_index"]
    if (labels not in ([0, 1], [1, 0]) or any(type(value) is not int for value in labels)
            or not isinstance(group["margins"], dict) or set(group["margins"]) != set(ARMS)):
        raise ValueError("Expected one opposite blocked candidate in each world and all comparator arms")
    for pair, blocked in zip(truth["signed_axis_gaps_m"], labels, strict=True):
        first, second = _pair(pair)
        if not (pair[blocked] < 0 < pair[1 - blocked]):
            raise ValueError(f"Nonexclusive geometry labels for signed gaps {first:g}/{second:g}")
    a, b = truth["hypothetical_actors"]
    if {k: v for k, v in a.items() if k != "velocity_xy"} != {k: v for k, v in b.items() if k != "velocity_xy"}:
        raise ValueError("Counterfactual actors must differ only in velocity")
    vx_a, _ = _pair(a["velocity_xy"])
    vx_b, _ = _pair(b["velocity_xy"])
    if group["margins"]["no_motion"][0] != group["margins"]["no_motion"][1]:
        raise ValueError("Removing the only world difference must give identical outputs")
    outcomes = {}
    for arm, pair in group["margins"].items():
        gaps = []
        for scores, blocked in zip(pair, labels, strict=True):
            first, second = _pair(scores)
            gap = first - second if blocked == 0 else second - first
            if not isfinite(gap):
                raise ValueError("Comparison gap overflow")
            gaps.append(gap)
        outcomes[arm] = ["correct" if gap > TIE_MARGIN else "incorrect" if gap < -TIE_MARGIN else "tied"
                         for gap in gaps]
    return {"world_sha256": worlds, "same_vx_sign": vx_a != 0 and vx_b != 0 and copysign(1, vx_a) == copysign(1, vx_b),
            "blocked_candidate": [candidates[i] for i in labels], "outcomes": outcomes}


def _pair(value):
    if (not isinstance(value, (list, tuple)) or len(value) != 2
            or any(type(x) not in (int, float) or not isfinite(x) for x in value)):
        raise ValueError("Expected two finite numeric values")
    return value


def _measure(items):
    return {arm: {"worlds_correct": sum(item["outcomes"][arm].count("correct") for item in items),
                  "worlds_tied": sum(item["outcomes"][arm].count("tied") for item in items),
                  "both_correct": sum(item["outcomes"][arm] == ["correct", "correct"] for item in items)}
            for arm in ARMS}


def _measured_slice(items):
    by_token = {}
    for token, item in items:
        by_token.setdefault(token, []).append(item)
    aggregate = _measure([item for _, item in items])
    for arm in ARMS:
        fractions = [_measure(rows)[arm]["both_correct"] / len(rows) for rows in by_token.values()]
        aggregate[arm]["both_macro_accuracy"] = fsum(fractions) / len(fractions) if fractions else None
    return {"windows_with_groups": len(by_token), "groups": len(items), "worlds": 2 * len(items), "arms": aggregate}


def _gate(all_cases, hard):
    thresholds = {"min_windows": 8, "min_groups": 32, "min_same_sign_groups": 16,
                  "min_both_macro_accuracy": .70, "min_gain_over_untrained": .25,
                  "min_same_sign_both_macro_accuracy": .60}
    if (all_cases["windows_with_groups"] < thresholds["min_windows"] or all_cases["groups"] < thresholds["min_groups"]
            or hard["groups"] < thresholds["min_same_sign_groups"]):
        return {"status": "insufficient_support", **thresholds}
    trained = all_cases["arms"]["trained"]["both_macro_accuracy"]
    gain = trained - all_cases["arms"]["untrained"]["both_macro_accuracy"]
    hard_accuracy = hard["arms"]["trained"]["both_macro_accuracy"]
    passed = (trained >= thresholds["min_both_macro_accuracy"] and gain >= thresholds["min_gain_over_untrained"]
              and hard_accuracy >= thresholds["min_same_sign_both_macro_accuracy"]
              and all_cases["arms"]["no_motion"]["both_correct"] == 0)
    return {"status": "passed_controlled_pilot" if passed else "failed_controlled_accuracy", **thresholds,
            "both_macro_accuracy_gain": gain, "same_sign_both_macro_accuracy": hard_accuracy,
            "natural_accuracy_not_established": True}
