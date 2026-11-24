from copy import deepcopy

import pytest

from hybrid_drive.experiments.paired_slow_metrics import score_controls, summarize_controls
from hybrid_drive.experiments.paired_slow_pilot import partition_observations


def fixture():
    snapshot = {"sample_token": "s", "recording": "heldout", "snapshot_sha256": "4" * 64}
    identity = {"candidate_ids": ["fast", "slow"], "world_sha256": ["1" * 64, "2" * 64]}
    predicted = {**identity, "margins": {"trained": [[2., 0.], [-2., 1.]],
                                        "untrained": [[2., 0.], [2., 0.]],
                                        "no_motion": [[1., 1.], [1., 1.]]}}
    target = {**identity, "blocked_index": [0, 1], "signed_axis_gaps_m": [[-.3, .9], [.7, -.2]],
              "hypothetical_actors": [{"name": "same", "velocity_xy": [2., -4.]},
                                      {"name": "same", "velocity_xy": [4., -4.]}]}
    return [{**snapshot, "groups": [predicted]}], [{**snapshot, "groups": [target]}]


def test_macro_support_and_same_sign_both_worlds_not_independent_counts():
    estimates, targets = fixture()
    for rows in (estimates, targets):
        rows.append({**rows[0], "sample_token": "empty", "groups": []})
    report = summarize_controls(estimates, targets)
    assert report["planned_snapshots"] == 2 and report["all"]["windows_with_groups"] == 1
    assert report["hard_same_vx_sign"]["groups"] == 1
    arms = report["all"]["arms"]
    assert arms["trained"] == {"worlds_correct": 2, "worlds_tied": 0, "both_correct": 1, "both_macro_accuracy": 1.}
    assert arms["untrained"]["worlds_correct"] == 1 and arms["untrained"]["both_correct"] == 0
    assert arms["no_motion"]["worlds_tied"] == 2
    assert report["gate"]["status"] == "insufficient_support"


@pytest.mark.parametrize("corruption", ["identity", "geometry", "non_motion_difference", "nan", "duplicate"])
def test_rejects_results_with_broken_blinding_or_world_identity(corruption):
    estimates, targets = fixture()
    estimate, target = estimates[0]["groups"][0], targets[0]["groups"][0]
    if corruption == "identity":
        target["world_sha256"] = target["world_sha256"][::-1]
    if corruption == "geometry":
        target["blocked_index"] = [1, 0]
    if corruption == "non_motion_difference":
        target["hypothetical_actors"][0]["name"] = "moved"
    if corruption == "nan":
        estimate["margins"]["trained"][0][0] = float("nan")
    if corruption == "duplicate":
        estimates[0]["groups"].append(deepcopy(estimate))
        targets[0]["groups"].append(deepcopy(target))
    with pytest.raises(ValueError):
        summarize_controls(estimates, targets)


def test_mismatched_seal_fails_before_opening_target(tmp_path):
    predictions = tmp_path / "predictions.json"
    predictions.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="seal"):
        score_controls(predictions, tmp_path / "does-not-exist.json", tmp_path / "out", sealed_sha256="0" * 64)


def _input_index():
    index = {"anchors": [], "frames": {}}
    for group in ("t", "h"):
        for i in range(20):
            frames = [f"{group}-{i}-{j}" for j in range(4)]
            index["anchors"].append({"recording": group, "history_tokens": frames[:3], "sample_token": frames[3]})
            for token in frames:
                index["frames"][token] = {"views": [{"sha256": token}]}
    return index


def test_cross_recording_partition_rejects_jpeg_reuse_in_history_even_under_another_token():
    index = _input_index()
    split = partition_observations(index)
    assert {key: len(values) for key, values in split.items()} == {
        "train": 16, "development": 4, "natural": 20, "heldout": 16,
    }
    assert {row["recording"] for row in split["train"]}.isdisjoint(row["recording"] for row in split["heldout"])
    index["frames"]["t-2-1"]["views"] = index["frames"]["h-4-2"]["views"]
    with pytest.raises(ValueError, match="JPEG crossed"):
        partition_observations(index)
