import csv
import json

import numpy as np
from navsim.common.dataclasses import Trajectory

from scripts.run_neural_sample import MODES, evaluate_predictions, render_comparison, write_comparison_csv
from hybrid_drive.experiments.protocol import METRICS, summarize


def trajectory(scale=1):
    return Trajectory(np.array([[scale * t, 0, 0] for t in range(1, 9)], dtype=np.float32))


def metrics(score=0.75):
    return {key: score for key in METRICS}


def predictions():
    return {mode: trajectory(i + 1) for i, mode in enumerate(MODES)}


def test_each_mode_scores_the_actual_supplied_object_once_and_keeps_its_points():
    candidates = predictions()
    calls = []

    def score(value):
        calls.append(value)
        return metrics()

    records = evaluate_predictions(candidates, trajectory(), "real-scene", score)
    assert len(calls) == 3 and all(value is candidates[mode] for value, mode in zip(calls, MODES))
    assert [row["native_trajectory"]["poses"] for row in records] == [candidates[mode].poses.tolist() for mode in MODES]
    assert records[0]["ade_m"] == records[0]["fde_m"] == 0
    assert records[1]["ade_m"] == 4.5 and records[1]["fde_m"] == 8
    assert all(not row["failed"] for row in records)


def test_failed_placeholder_is_not_scored_but_remains_an_attempted_failed_mode():
    candidates = predictions()
    calls = []

    def score(value):
        calls.append(value)
        return metrics()

    records = evaluate_predictions(candidates, trajectory(), "scene", score,
                                   failures={"checks_only": "unsupported_motion"})
    assert len(calls) == 2 and all(value is not candidates["checks_only"] for value in calls)
    failure = records[1]
    assert failure["failed"] and "unsupported_motion" in failure["error"]
    assert failure["metrics"] is None and failure["ade_m"] is None
    summary = summarize(records, expected_tokens=["scene"], expected_modes=MODES)
    assert summary["checks_only"]["attempted"] == summary["checks_only"]["failed"] == 1


def test_scorer_failure_is_preserved_and_does_not_drop_the_other_modes():
    candidates = predictions()

    def score(value):
        if value is candidates["baseline"]:
            raise OSError("metric cache unreadable")
        return metrics()

    records = evaluate_predictions(candidates, trajectory(), "scene", score)
    assert [row["failed"] for row in records] == [True, False, False]
    assert "metric cache unreadable" in records[0]["error"]
    assert records[0]["metrics"] is None


def test_invalid_scorer_metrics_and_missing_mode_do_not_become_successes():
    candidates = predictions()
    del candidates["cv_reference"]
    records = evaluate_predictions(candidates, trajectory(), "scene", lambda _: metrics(float("nan")))
    assert len(records) == 3 and all(row["failed"] for row in records)
    assert all(row["metrics"] is None for row in records)


def test_csv_keeps_failed_mode_with_empty_measurements_instead_of_fabricated_zero(tmp_path):
    records = evaluate_predictions(predictions(), trajectory(), "scene", lambda _: metrics(),
                                   failures={"checks_only": "failed"})
    path = tmp_path / "comparison.csv"
    write_comparison_csv(path, records)
    with path.open() as source:
        rows = list(csv.DictReader(source))
    assert len(rows) == 3
    assert rows[1]["failed"] == "True" and rows[1]["score"] == rows[1]["ade_m"] == ""
    assert rows[0]["score"] == "0.75"


def test_renderer_uses_only_scored_actual_predictions_and_records_failure_label(tmp_path):
    candidates = predictions()
    records = evaluate_predictions(candidates, trajectory(), "scene", lambda _: metrics(),
                                   failures={"checks_only": "failed"})
    render_comparison(tmp_path, candidates, trajectory(), records)
    assert (tmp_path / "trajectory.png").stat().st_size > 1000
    # All result structures can be persisted as strict JSON, including failures.
    json.dumps(records, allow_nan=False)
