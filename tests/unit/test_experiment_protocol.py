import pytest

from hybrid_drive import Trajectory
from hybrid_drive.experiments.protocol import (
    METRICS, content_hash, make_manifest, require_paired, scene_result,
    summarize, trajectory_errors, trajectory_hash, validate_splits, verify_manifest,
)


def path(xs=(1, 2, 3), dt=0.5):
    return Trajectory([[x, 0, 0] for x in xs], dt)


def manifest(**changes):
    fields = dict(revision="a" * 40, sources={"navsim_revision": "b" * 40, "nuplan_revision": "c" * 40},
                  assets=assets(), model={"checkpoint_sha256": "d" * 64},
                  scene_token="scene", log_name="log", seed=0, policy={"speed": 10}, candidate=path())
    fields.update(changes)
    return make_manifest(**fields)


def assets():
    return {"sources": {"revision": "f" * 40},
            "models": [{"path": "checkpoint.ckpt", "size_bytes": 123, "sha256": "d" * 64}],
            "cameras": {"files": [{"camera": "CAM_F0", "member": "camera.jpg", "path": "camera.jpg",
                                    "size_bytes": 10, "sha256": "e" * 64, "width": 1920, "height": 1080,
                                    "format": "JPEG", "mode": "RGB"}]}}


def test_reusing_identical_assets_does_not_change_experiment_identity():
    first = assets()
    second = assets()
    first["models"][0].update(reused=False, bytes_transferred=123)
    first["bytes_transferred_this_run"] = 123
    second["models"][0].update(reused=True, bytes_transferred=0)
    second["bytes_transferred_this_run"] = 0
    require_paired(manifest(assets=first), manifest(assets=second))
    second["cameras"]["files"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="assets_manifest_hash"):
        require_paired(manifest(assets=first), manifest(assets=second))


def row(mode="baseline", token="scene", **changes):
    fields = dict(metrics={key: 0.75 for key in METRICS}, prediction=path(), reference=path())
    fields.update(changes)
    return scene_result(mode=mode, token=token, **fields)


def test_manifest_snapshots_inputs_and_hash_detects_later_tampering():
    model, policy = {"checkpoint_sha256": "d" * 64}, {"speed": 10}
    record = manifest(model=model, policy=policy)
    model["checkpoint_sha256"], policy["speed"] = "different", 99
    assert record["model"]["checkpoint_sha256"] == "d" * 64
    assert record["policy"]["speed"] == 10
    verify_manifest(record)
    record["policy"]["speed"] = 20
    with pytest.raises(ValueError, match="content changed"):
        verify_manifest(record)


@pytest.mark.parametrize("key,value", [
    ("source_revision", "e" * 40), ("navsim_revision", "e" * 40),
    ("nuplan_revision", "e" * 40), ("dataset_source_hash", "e" * 64),
    ("assets_manifest_hash", "e" * 64), ("model", {"checkpoint_sha256": "e" * 64}),
    ("scene_token", "other"), ("log_name", "other"), ("split", "test"),
    ("role", "test"), ("seed", 1), ("sampling", {"interval_s": 1}),
    ("raw_candidate_hash", "e" * 64), ("policy", {"speed": 20}),
    ("split_audit", {"validation": "different"}), ("metric_protocol", {"driving": "EPDMS"}),
])
def test_comparison_rejects_each_changed_paired_condition_even_with_a_valid_hash(key, value):
    left, right = manifest(), manifest()
    require_paired(left, right)
    right[key] = value
    right.pop("manifest_hash")
    right["manifest_hash"] = content_hash(right)
    with pytest.raises(ValueError, match=key):
        require_paired(left, right)


@pytest.mark.parametrize("seed", [True, -1, 2**32, 0.5])
def test_rejects_seeds_that_cannot_be_reproduced_by_numpy(seed):
    with pytest.raises(ValueError, match="seed"):
        manifest(seed=seed)


def test_candidate_hash_covers_geometry_heading_and_grid_not_random_plan_ids():
    assert trajectory_hash(path()) == trajectory_hash(path())
    assert trajectory_hash(path()) != trajectory_hash(path((1, 2, 4)))
    assert trajectory_hash(path()) != trajectory_hash(path(dt=1))
    assert trajectory_hash(path()) != trajectory_hash(Trajectory([[1, 0, 0.1], [2, 0, 0], [3, 0, 0]], 0.5))


def test_errors_use_xy_euclidean_distance_at_matching_future_times():
    predicted = Trajectory([[3, 4, 2], [0, 12, -1]], 0.5)
    reference = Trajectory([[0, 0, 0], [0, 0, 0]], 0.5)
    assert trajectory_errors(predicted, reference) == {"ade_m": 8.5, "fde_m": 12}


@pytest.mark.parametrize("reference", [path((1, 2)), path(dt=1)])
def test_metrics_do_not_silently_resample_or_truncate(reference):
    with pytest.raises(ValueError, match="time grid"):
        trajectory_errors(path(), reference)


def test_failure_stays_in_denominator_without_becoming_a_measured_collision():
    records = [row(), row(token="failed", failed=True, error="inference error")]
    summary = summarize(records, expected_tokens=["scene", "failed"], expected_modes=["baseline"])["baseline"]
    assert summary["attempted"] == 2 and summary["valid"] == 1 and summary["failed"] == 1
    assert summary["failure_rate"] == 0.5
    assert summary["score_mean_all_attempts_failure_zero"] == 0.375
    assert summary["score_mean_valid_only"] == 0.75
    assert summary["metrics_mean_valid_only"]["no_at_fault_collisions"] == 0.75
    assert records[1]["metrics"] is None and records[1]["ade_m"] is None


def test_all_failed_run_has_zero_penalized_score_and_unknown_measurements():
    summary = summarize([row(failed=True, error="bad camera")],
                        expected_tokens=["scene"], expected_modes=["baseline"])["baseline"]
    assert summary["score_mean_all_attempts_failure_zero"] == 0
    assert summary["score_mean_valid_only"] is None
    assert all(value is None for value in summary["metrics_mean_valid_only"].values())
    assert summary["ade_m_valid_only"] is None


@pytest.mark.parametrize("records", [[], [row(), row()], [row(token="other")], [row(mode="checks_only")]])
def test_missing_duplicate_and_unrequested_scenes_cannot_disappear_from_comparison(records):
    with pytest.raises(ValueError):
        summarize(records, expected_tokens=["scene"], expected_modes=["baseline"])


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 1.1, True])
def test_malformed_metric_does_not_turn_into_a_successful_record(value):
    metrics = {key: 1 for key in METRICS}
    metrics["score"] = value
    with pytest.raises(ValueError):
        row(metrics=metrics)


def scene(token="s1", log="log1", sources=("f1", "f2")):
    return {"scene_token": token, "log_name": log, "source_tokens": sources}


def test_split_checks_keep_neighboring_windows_together_and_smoke_separate():
    assignments = {"train": [scene(), scene("s2", sources=("f2", "f3"))],
                   "test": [scene("s3", "log2", ("f4",))],
                   "smoke": [scene("mini", "debug-log", ("debug-frame",))]}
    assert validate_splits(assignments) == {"train": 2, "test": 1, "smoke": 1}


@pytest.mark.parametrize("assignments,reason", [
    ({"train": [scene()], "test": [scene("s2", sources=("f3",))]}, "Log crosses"),
    ({"train": [scene()], "test": [scene("s2", "other", ("f2",))]}, "frame crosses"),
    ({"train": [scene()], "test": [scene("s1", "other", ("f3",))]}, "more than once"),
    ({"smoke": [scene()], "validation": [scene("s2")]}, "Log crosses"),
])
def test_split_leakage_cannot_be_hidden_by_new_scene_or_log_names(assignments, reason):
    with pytest.raises(ValueError, match=reason):
        validate_splits(assignments)
