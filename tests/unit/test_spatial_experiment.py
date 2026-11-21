from types import SimpleNamespace
import json

import numpy as np
import pytest
import torch
from torch import nn

from hybrid_drive.experiments import spatial_experiment as experiment
from hybrid_drive.experiments.spatial_metrics import (
    SpatialPredictions,
    calibrate_spatial_threshold,
    shuffled_by_recording,
    spatial_metrics,
)


def _predictions(scores, labels, names):
    scores = np.array(scores, dtype=np.float32)
    labels = np.array(labels, dtype=bool)
    boxes = np.broadcast_to([0, 0, 0.5, 0.5], (*scores.shape, 4)).copy()
    truth = np.broadcast_to([0, 0, 1, 1], boxes.shape).copy().astype(np.float64)
    truth[~labels] = np.nan
    return SpatialPredictions(scores, labels, boxes, truth, tuple(names))


def test_calibration_uses_balanced_recording_scores_with_known_split():
    example = _predictions([[2, 2, -1, -2], [1, -1, -2, -5]], [[1, 1, 0, 0], [1, 0, 0, 0]], ["A", "B"])
    settings = calibrate_spatial_threshold(example)
    assert settings["threshold"] == 0
    assert settings["dev_macro_balanced_accuracy"] == 1
    metrics = spatial_metrics(example, settings["threshold"])
    assert metrics["roc_auc"] == 1
    assert metrics["macro_positive_iou"] == pytest.approx(0.25)
    assert metrics["macro_recall_recordings"] == metrics["macro_specificity_recordings"] == 2


def test_tied_scores_yield_half_auc_and_missing_classes_stay_undefined():
    data = _predictions([[0] * 4, [0] * 4], [[1, 1, 1, 1], [1, 0, 0, 0]], ["A", "B"])
    result = spatial_metrics(data, 0)
    assert result["roc_auc"] == 0.5
    assert result["per_recording"]["A"]["balanced_accuracy"] is None
    assert result["macro_balanced_accuracy"] == 0.5
    assert result["macro_recall_recordings"] == 2 and result["macro_specificity_recordings"] == 1


def test_within_recording_shuffle_is_a_derangement_and_preserves_labels():
    names = ("A", "B", "A", "B", "A", "B", "B")
    order = shuffled_by_recording(names)
    assert sorted(order.tolist()) == list(range(len(names)))
    assert all(i != j and names[i] == names[j] for i, j in enumerate(order))
    with pytest.raises(ValueError, match="singleton"):
        shuffled_by_recording(("A", "A", "B"))


class _FakeVision(nn.Module):
    visual_dim = 4

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones((1, 1, 4, 1, 1)), requires_grad=False)
        self.eval()

    def forward(self, image):
        mean = image.mean(dim=2, keepdim=True)
        return mean.expand(-1, 1, self.visual_dim, -1, -1) * self.scale


def test_all_checkpoints_and_thresholds_are_frozen_before_final_dataset(tmp_path, monkeypatch):
    expected = {"train": 4, "dev": 4, "holdout": 4}
    monkeypatch.setattr(experiment, "SPLIT_FRAMES", expected)
    opened = []

    class Dataset:
        def __init__(self, root, split):
            opened.append(split)
            self.sample_ids = tuple(f"{split}{i}" for i in range(4))
            self.recording_ids = (split + "a", split + "a", split + "b", split + "b")
            self.dataset_hash = "d" * 64
            if split == "holdout":
                directory = next(tmp_path.iterdir())
                calibration = json.loads((directory / "calibration.json").read_text())
                assert len(calibration["runs"]) == 4
                assert all(name.startswith("dev") for name in calibration["calibration_recordings"])
                assert len(list(directory.glob("*.safetensors"))) == 4

        def __len__(self):
            return len(self.sample_ids)

        def __getitem__(self, i):
            images = torch.arange(3 * 4 * 8, dtype=torch.float32).reshape(1, 3, 4, 8) / (100 * (i + 1))
            labels = torch.tensor([i % 2 == 0, i % 2 == 1, False, False])
            boxes = torch.full((4, 4), float("nan"))
            boxes[0 if labels[0] else 1] = torch.tensor(
                [0.03, 0.1, 0.2, 0.4] if labels[0] else [0.28, 0.1, 0.45, 0.4]
            )
            return self.sample_ids[i], self.recording_ids[i], images, labels, boxes

    def collate(rows):
        ids, recordings, images, labels, boxes = zip(*rows)
        return SimpleNamespace(
            sample_ids=ids,
            recording_ids=recordings,
            images=torch.stack(images),
            presence=torch.stack(labels),
            boxes_2d=torch.stack(boxes),
        )

    monkeypatch.setattr(experiment, "SpatialAlignmentDataset", Dataset)
    monkeypatch.setattr(experiment, "collate_spatial_alignment", collate)
    monkeypatch.setattr(experiment, "load_alignment_vision", lambda *args: (_FakeVision(), {"fixture": True}))
    directory, result = experiment.run_spatial_experiment(
        "data",
        "models",
        tmp_path,
        revision="a" * 40,
        recipe=experiment.SpatialRecipe(1, 2, (0, 1)),
    )
    assert result["status"] == "passed", result.get("error")
    assert opened == ["train", "dev", "holdout"]
    assert result["frozen_vision_unchanged"] and result["optimizer_steps_completed"] == 8
    assert len(result["runs"]) == 4
    for begin in (0, 2):
        first, second = result["runs"][begin : begin + 2]
        assert first["initial_weights_hash"] == second["initial_weights_hash"]
        assert first["train_order_hash"] == second["train_order_hash"]
        assert first["checkpoint_state_bitwise_equal"] and second["checkpoint_state_bitwise_equal"]
        assert first["holdout"]["frames"] == 4
        assert first["holdout_shuffled_same_recording"]["frames"] == 4
    assert len(result["paired_holdout_deltas"]) == 2
    assert "recording-weighted metrics" in (directory / "report.html").read_text()


def test_negative_only_dev_recording_influences_the_calibrated_threshold():
    data = _predictions([[2, 2, 1, 0], [1, 1, 1, 1]], [[1, 1, 1, 0], [0, 0, 0, 0]], ["A", "B"])
    # Excluding the negative-only recording would select 1 instead of 2.
    selected = calibrate_spatial_threshold(data)
    assert selected["threshold"] == 2
    assert selected["dev_macro_balanced_accuracy"] == pytest.approx(5 / 6)


def test_output_cannot_write_into_an_aliased_input_directory(tmp_path):
    source = tmp_path / "data"
    source.mkdir()
    aliased = tmp_path / "alias"
    aliased.symlink_to(source, target_is_directory=True)
    with pytest.raises(ValueError, match="outside inputs"):
        experiment.run_spatial_experiment(source, "models", aliased / "output", revision="a" * 40)
    assert not list(source.iterdir())
