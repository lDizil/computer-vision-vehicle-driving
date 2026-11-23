import json

import pytest
import torch
from torch import nn

from hybrid_drive.experiments import spatial_accuracy as experiment
from hybrid_drive.experiments.proposal_evaluation import _Encoded
from hybrid_drive.experiments.spatial_accuracy_metrics import choose_refinement, measure_strategy
from hybrid_drive.experiments.spatial_decision_policy import ZoneDecisionPolicy


def _features(recordings):
    boxes = torch.tensor([[.04, .15, .17, .45]]) + torch.arange(4)[:, None] * torch.tensor([.25, 0, .25, 0])
    boxes = boxes[None].expand(4, -1, -1).clone()
    rng = torch.Generator().manual_seed(73)
    patches = torch.randn(4, 1, 4, 8, 8, generator=rng)
    labels = torch.eye(4, dtype=torch.bool)
    targets = boxes.clone() + torch.tensor([.03, .02, .03, .02])
    targets[~labels] = float("nan")
    logits = torch.where(labels, 2.0, -1.0)
    # A tie at a negative score exercises the strict confidence gate.
    logits[0, 2] = 1.0
    return _Encoded(patches, boxes, labels, targets, recordings, logits)


def _setup(monkeypatch, output):
    opened, restored = [], []

    class Dataset:
        def __init__(self, root, role, *, expected_manifest_hash):
            split = role if root == "original" else root
            opened.append(split)
            dataset_hash = {"original": "a", "calibration": "b", "new": "c"}[root] * 64
            assert dataset_hash == expected_manifest_hash
            self.dataset_hash = dataset_hash
            self.recording_ids = tuple(split + c for c in ("0", "0", "1", "1"))
            self.sample_ids = tuple(split + str(i) for i in range(4))
            if root == "new":
                run = next(output.iterdir())
                frozen = json.loads((run / "frozen-policy.json").read_text())
                assert frozen["data_hashes"]["evaluation"] == dataset_hash
                assert len(frozen["refiners"]) == len(list(run.glob("*.safetensors"))) == 2
                assert restored == ["refined-boxes-seed-0.safetensors", "refined-boxes-seed-1.safetensors"]

        def __len__(self):
            return 4

    class Vision(nn.Module):
        visual_dim = 4

        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1), requires_grad=False)

    class Prior:
        receipt = {"calibration": {"threshold": -1.0}, "seed": 0}

    def read_pack(reference, target, **kwargs):
        assert reference == "original" and opened == []
        if target == "new":
            assert kwargs["expected_schema"] == experiment.POLICY_EVALUATION_SOURCE_SCHEMA
            assert kwargs["previously_seen_roots"] == ("calibration",)
            return {"manifest_hash": "c" * 64}
        assert target == "calibration" and not kwargs
        return {"manifest_hash": "b" * 64}

    real_load = experiment.load_neural_weights

    def check_load(path, modules, **kwargs):
        out = real_load(path, modules, **kwargs)
        restored.append(path.name)
        return out

    monkeypatch.setattr(experiment, "SPLIT_FRAMES", {"train": 4, "dev": 4})
    monkeypatch.setattr(experiment, "SpatialAlignmentDataset", Dataset)
    monkeypatch.setattr(experiment, "read_spatial_manifest", lambda _: {"manifest_hash": "a" * 64})
    monkeypatch.setattr(experiment, "read_fresh_spatial_holdout", read_pack)
    monkeypatch.setattr(experiment, "load_alignment_vision", lambda _: (Vision().eval(), {"vision": "fixture"}))
    monkeypatch.setattr(experiment, "load_spatial_prior_source", lambda *args, **kwargs: Prior())
    monkeypatch.setattr(experiment, "encode", lambda dataset, *args, **kwargs: _features(dataset.recording_ids))
    monkeypatch.setattr(experiment, "load_neural_weights", check_load)
    monkeypatch.setattr(experiment, "write_localization_gallery", lambda path, *args: path.write_text("gallery"))
    return opened


def _run(path):
    return experiment.run_spatial_accuracy(
        "original", "calibration", "new", "models", "prior", path,
        revision="d" * 40, expected_prior_result_sha256="e" * 64,
        recipe=experiment.SpatialAccuracyRecipe(2, 2),
    )


def test_keeps_fresh_targets_sealed_until_policy_and_both_restored_states(tmp_path, monkeypatch):
    opened = _setup(monkeypatch, tmp_path)
    root, result = _run(tmp_path)
    assert result["status"] == "passed", result.get("error")
    assert opened == ["train", "dev", "calibration", "new"]
    assert result["optimizer_steps_completed"] == 8 and result["frozen_vision_unchanged"]
    assert all(row["restored_bitwise_before_final"] for row in result["refiners"])
    baseline, calibrated, selected = [result["final"][mode] for mode in ("previous", "calibrated", "selected")]
    assert baseline["decisions"]["false_present"] > calibrated["decisions"]["false_present"]
    assert baseline["geometry"] == calibrated["geometry"]
    assert calibrated["decisions"] == selected["decisions"]
    assert calibrated["geometry"]["macro_auc"] == selected["geometry"]["macro_auc"]
    assert (root / "gallery.html").exists()
    assert 'href="gallery.html"' in (root / "report.html").read_text()


def test_tampered_persisted_policy_does_not_open_final_labels(tmp_path, monkeypatch):
    opened = _setup(monkeypatch, tmp_path)
    original = experiment.write_json

    def corrupt(path, data):
        if path.name == "frozen-policy.json":
            data = {**data, "selected_refinement_seed": "tampered"}
        original(path, data)

    monkeypatch.setattr(experiment, "write_json", corrupt)
    _, result = _run(tmp_path)
    assert result["status"] == "failed" and "frozen" in result["error"]
    assert opened == ["train", "dev", "calibration"]


def test_corrupt_weights_stop_before_opening_final_labels(tmp_path, monkeypatch):
    opened = _setup(monkeypatch, tmp_path)
    original = experiment.load_neural_weights

    def tamper(path, modules, **kwargs):
        with path.open("ab") as out:
            out.write(b"unexpected byte")
        return original(path, modules, **kwargs)

    monkeypatch.setattr(experiment, "load_neural_weights", tamper)
    _, result = _run(tmp_path)
    assert result["status"] == "failed" and "expected SHA256" in result["error"]
    assert opened == ["train", "dev", "calibration"]


def test_gallery_error_does_not_mark_the_experiment_passed(tmp_path, monkeypatch):
    _setup(monkeypatch, tmp_path)

    def fail(*args):
        raise OSError("gallery unavailable")

    monkeypatch.setattr(experiment, "write_localization_gallery", fail)
    root, result = _run(tmp_path)
    assert result["status"] == "failed" and "gallery unavailable" in result["error"]
    assert json.loads((root / "result.json").read_text())["status"] == "failed"


def test_dev_selection_rejects_material_regression_on_one_recording():
    base = {"macro_positive_iou": .35, "per_recording_iou": {"A": .4, "B": .3}}
    worse_one = {"seed": 0, "dev_geometry": {"macro_positive_iou": .42, "per_recording_iou": {"A": .57, "B": .27}}}
    both = {"seed": 1, "dev_geometry": {"macro_positive_iou": .37, "per_recording_iou": {"A": .43, "B": .31}}}
    assert choose_refinement(base, [worse_one]) is None
    assert choose_refinement(base, [worse_one, both]) == 1


def test_joint_box_measure_counts_only_shown_positive_with_sufficient_overlap():
    rows = _features(("A", "A", "B", "B"))
    # Perfect positive geometry for all queries, then confidently hide two
    # correct boxes. Positive IoU remains 1, but hits and recall are halved.
    targets = rows.proposals.numpy().copy()
    cats = rows.prior_scores.numpy()
    cats[0, 0] = cats[1, 1] = 0.0
    data = _Encoded(rows.features, rows.proposals, rows.presence, torch.from_numpy(targets), rows.recordings, rows.prior_scores)
    actual = measure_strategy(data, targets, ZoneDecisionPolicy(-0.5, 0.5))
    assert actual["geometry"]["macro_positive_iou"] == 1.0
    assert actual["localized_at_iou_half"]["count"] == 2
    assert actual["localized_at_iou_half"]["recall_of_eligible"] == 0.5
    assert actual["decisions"]["uncertain_count"] == 2
