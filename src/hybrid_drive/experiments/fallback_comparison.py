"""Paired policy comparison without dropping failed planned attempts."""

from datetime import datetime, timezone
import html
from pathlib import Path

from .bounded_io import decode_json, read_bounded
from .io import write_json


MODES = ("baseline", "checks_only", "combined")
MAX_RESULT_BYTES = 2 * 1024 * 1024


def compare_fallback_runs(longitudinal_root, curved_root, output_root):
    legacy, curved = (_read_result(root) for root in (longitudinal_root, curved_root))
    for key in ("cohort_hash", "fast_source_revision", "sources_hash", "native_evidence"):
        if legacy[key] != curved[key] or legacy[key] is None:
            raise ValueError("Paired runs differ in frozen input identity: " + key)
    if legacy["policy"] != {key: value for key, value in curved["policy"].items() if key != "fallback_model"}:
        raise ValueError("Paired limits or braking caps differ")
    if "fallback_model" in legacy["policy"] or curved["policy"].get("fallback_model") != "curved_snapshot_hypothesis":
        raise ValueError("Expected longitudinal control and explicit curved hypothesis")
    if not isinstance(curved.get("motion_observations"), dict):
        raise ValueError("Curved run lacks observation provenance")
    old, new = _rows(legacy), _rows(curved)
    pairs = []
    for key in old:
        left, right = old[key], new[key]
        for field in ("scene_token", "raw_candidate_hash"):
            if left[field] != right[field]:
                raise ValueError("Paired case input differs: " + field)
        if key[1] == "combined":
            for field in ("evidence_hash", "native_source_revision", "native_elapsed_to_decision_s",
                          "slow_outcome", "slow_disposition", "slow_status", "slow_used"):
                if left.get(field) != right.get(field):
                    raise ValueError("Paired native Slow evidence or checkpoint differs: " + field)
        if (not left["failed"] and not right["failed"]
                and not left.get("fallback_invoked") and not right.get("fallback_invoked")
                and left.get("selected_trajectory_hash") != right.get("selected_trajectory_hash")):
            raise ValueError("Unchanged non-fallback selection differs")
        pairs.append({
            "case_id": key[0], "mode": key[1], "scene_token": left["scene_token"],
            "raw_candidate_hash": left["raw_candidate_hash"],
            "control_failed": left["failed"], "curved_failed": right["failed"],
            "control_error": left.get("error"), "curved_error": right.get("error"),
            "control_score": _score(left), "curved_score": _score(right),
            "both_valid": not left["failed"] and not right["failed"],
            "control_selected_hash": left.get("selected_trajectory_hash"),
            "curved_selected_hash": right.get("selected_trajectory_hash"),
            "curved_model_report": right.get("fallback_model_report"),
        })
    summaries = {}
    for mode in MODES:
        rows = [row for row in pairs if row["mode"] == mode]
        valid = [row for row in rows if row["both_valid"]]
        summaries[mode] = {
            "planned_pairs": len(rows), "both_valid_pairs": len(valid),
            "control_failed": sum(row["control_failed"] for row in rows),
            "curved_failed": sum(row["curved_failed"] for row in rows),
            "control_score_failed_zero_mean": sum(row["control_score"] or 0 for row in rows) / len(rows),
            "curved_score_failed_zero_mean": sum(row["curved_score"] or 0 for row in rows) / len(rows),
            "paired_score_delta_both_valid_mean": (sum(row["curved_score"]-row["control_score"] for row in valid)
                                                     / len(valid) if valid else None),
        }
    run_dir = Path(output_root) / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run_dir.mkdir(parents=True, exist_ok=False)
    result = {
        "schema": "hybrid-drive-paired-fallback-comparison-v1",
        "execution": "offline_replay_original_native_clock",
        "cohort_hash": legacy["cohort_hash"], "sources_hash": legacy["sources_hash"],
        "native_evidence": legacy["native_evidence"], "motion_observations": curved["motion_observations"],
        "control_source_revision": legacy["source_revision"], "curved_source_revision": curved["source_revision"],
        "control_status": legacy["status"], "curved_status": curved["status"],
        "planned_pairs": 18, "summary": summaries, "pairs": pairs,
        "scope": "six_fixed_windows_one_recording_debug_only",
        "quality_claim": False, "driving_safety_claim": False,
        "interpretation": "Failed-zero means include all six attempts. Valid-pair deltas exclude failed pairs explicitly; recovery is not paired driving improvement.",
    }
    write_json(run_dir / "comparison.json", result)
    (run_dir / "report.html").write_text(_report(result), encoding="utf-8")
    return run_dir, result


def _read_result(root):
    result = decode_json(read_bounded(Path(root) / "result.json", MAX_RESULT_BYTES))
    if (not isinstance(result, dict) or result.get("schema") != "hybrid-drive-real-cohort-scores-v1"
            or result.get("execution") != "offline_replay_original_native_clock"
            or result.get("planned_cases") != 6 or result.get("planned_mode_attempts") != 18
            or result.get("executed_modes") != list(MODES)):
        raise ValueError("Expected a complete fixed frozen-clock cohort result")
    return result


def _rows(result):
    rows = result["records"]
    expected = {(f"r{index:02d}", mode) for index in range(6) for mode in MODES}
    if not isinstance(rows, list) or len(rows) != 18:
        raise ValueError("All eighteen planned attempts must be retained")
    grouped = {}
    for row in rows:
        key = (row["case_id"], row["mode"])
        if key not in expected or key in grouped or type(row["failed"]) is not bool:
            raise ValueError("Invalid, duplicate or missing planned attempt")
        _score(row)
        grouped[key] = row
    return grouped


def _score(row):
    if row["failed"]:
        if row.get("metrics") is not None:
            raise ValueError("Failed attempt cannot carry measured metrics")
        return None
    metrics = row.get("metrics")
    value = None if not isinstance(metrics, dict) else metrics.get("score")
    if type(value) not in (int, float) or not 0 <= value <= 1:
        raise ValueError("Successful attempt requires a finite NAVSIM score")
    return value


def _report(result):
    def score(value): return "—" if value is None else f"{value:.6f}"
    summary = "".join(
        f"<tr><td>{html.escape(mode)}</td><td>{row['control_failed']}/6</td><td>{row['curved_failed']}/6</td>"
        f"<td>{score(row['control_score_failed_zero_mean'])}</td><td>{score(row['curved_score_failed_zero_mean'])}</td>"
        f"<td>{row['both_valid_pairs']}/6</td><td>{score(row['paired_score_delta_both_valid_mean'])}</td></tr>"
        for mode, row in result["summary"].items())
    rows = "".join(
        f"<tr><td>{html.escape(row['case_id'])}</td><td>{html.escape(row['mode'])}</td>"
        f"<td>{score(row['control_score'])}</td><td>{score(row['curved_score'])}</td>"
        f"<td>{html.escape(str(row['control_error'] or ''))}</td><td>{html.escape(str(row['curved_error'] or ''))}</td></tr>"
        for row in result["pairs"])
    return ("<!doctype html><html lang='ru'><meta charset='utf-8'><title>Сравнение fallback</title>"
            "<style>body{font:16px system-ui;max-width:1250px;margin:40px auto;padding:0 20px;background:#f7f9fc;color:#172132}"
            "table{border-collapse:collapse;background:white;width:100%;margin:20px 0}td,th{border:1px solid #dce2eb;padding:9px;text-align:left}"
            "h1,h2{color:#143a67}code{overflow-wrap:anywhere}</style>"
            "<h1>Продольный и curved fallback: одинаковые входы</h1>"
            "<p>Шесть фиксированных сцен одной записи. Сохраняем прежние Fast-траектории, ответы Qwen и исходные часы. "
            "Меняем только явно выбранную fallback-модель; лимиты не ослабляем.</p>"
            "<p>Ошибка планирования не является измеренным столкновением. Среднее с failed=0 включает все попытки; "
            "разность по валидным парам считается отдельно. Это отладочный эксперимент, не доказательство улучшения вождения.</p>"
            "<table><tr><th>Режим</th><th>Ошибки control</th><th>Ошибки curved</th><th>PDMS control, failed=0</th>"
            "<th>PDMS curved, failed=0</th><th>Обе валидны</th><th>Средняя Δ валидных пар</th></tr>" + summary + "</table>"
            "<h2>Все 18 пар</h2><table><tr><th>Сцена</th><th>Режим</th><th>Control PDMS</th><th>Curved PDMS</th>"
            "<th>Ошибка control</th><th>Ошибка curved</th></tr>" + rows + "</table>"
            "<h2>Идентичность входов</h2><p>Fast cohort: <code>" + html.escape(result["cohort_hash"]) + "</code></p>"
            "<p>Полные hashes, параметры модели и continuous bounds сохраняем в comparison.json и исходных outputs.json.</p></html>")
