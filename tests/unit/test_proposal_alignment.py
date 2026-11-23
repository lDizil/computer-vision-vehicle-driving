from types import SimpleNamespace
import json

import pytest
import torch
from torch import nn

from hybrid_drive.experiments import proposal_alignment as trial
from hybrid_drive.experiments import proposal_evaluation as evaluation


class _Vision(nn.Module):
    visual_dim = 4

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1, 1, 4, 1, 1), requires_grad=False)
        self.eval()

    def forward(self, images):
        return images.mean(dim=2, keepdim=True) * self.scale


class _Language(nn.Module):
    llm_dim = 8
    vocab_size = 8
    loc_token_id, mot_token_id, yes_token_id, no_token_id = 2, 3, 4, 5

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(self.vocab_size, self.llm_dim)
        self.head = nn.Linear(self.llm_dim, self.vocab_size)
        self.requires_grad_(False).eval()

    def prompt_ids(self, batch_size):
        return torch.tensor([0, 2, 1, 2, 1, 2, 1, 2]).repeat(batch_size, 1)

    def forward(self, prefix, ids):
        hidden = self.embedding(ids) + prefix.mean(dim=1, keepdim=True)
        return hidden, self.head(hidden)


class _Prior:
    receipt = {
        "dataset_hash": "d" * 64,
        "seed": 0,
        "condition": "image_features",
        "calibration": {"threshold": -0.75},
    }

    @staticmethod
    def predict(patches):
        left = torch.arange(4, dtype=patches.dtype) * 0.25
        offset = patches.mean(dim=(1, 2, 3, 4))[:, None] * 0.05
        x = left[None] + offset + 0.025
        return torch.stack((x, torch.full_like(x, 0.10), x + 0.12, torch.full_like(x, 0.46)), dim=-1)

    @classmethod
    def predict_spatial(cls, patches):
        boxes = cls.predict(patches)
        return SimpleNamespace(boxes_xyxy=boxes, presence_logits=boxes[..., 0] - 0.25)


def _fixture(monkeypatch, output, runs=4):
    opened = []

    class Dataset:
        def __init__(self, root, role, *, expected_manifest_hash=None):
            opened.append(role)
            self.sample_ids = tuple(f"{role}-{i}" for i in range(4))
            self.recording_ids = (role + "a", role + "a", role + "b", role + "b")
            self.dataset_hash = "d" * 64
            if role == "holdout":
                directory = next(output.iterdir())
                frozen = json.loads((directory / "calibration.json").read_text())
                assert len(frozen["runs"]) == runs
                assert all(set(row["dev_thresholds"]) == {"decoder", "text"} for row in frozen["runs"])
                assert len(list(directory.glob("*.safetensors"))) == runs
                assert all(role.startswith("dev") for role in frozen["dev_recordings"])
                if root == "fresh":
                    assert expected_manifest_hash == frozen["fresh_holdout_manifest_hash"] == "e" * 64
                    assert frozen["memory_mode"] == "roi_tokens"
                    self.dataset_hash = expected_manifest_hash
                else:
                    assert expected_manifest_hash is None

        def __len__(self):
            return len(self.sample_ids)

        def __getitem__(self, i):
            features = torch.ones(1, 3, 4, 8) * ((i + 1) * 0.17)
            labels = torch.arange(4) == i
            target = torch.full((4, 4), float("nan"))
            target[i] = torch.tensor([i * 0.25 + 0.045, 0.19, i * 0.25 + 0.20, 0.41])
            return self.sample_ids[i], self.recording_ids[i], features, labels, target

    def collate(batch):
        ids, recordings, rgb, presence, targets = zip(*batch)
        return SimpleNamespace(
            sample_ids=ids,
            recording_ids=recordings,
            images=torch.stack(rgb),
            presence=torch.stack(presence),
            boxes_2d=torch.stack(targets),
        )

    expected = {"train": 4, "dev": 4, "holdout": 4}
    monkeypatch.setattr(trial, "SPLIT_FRAMES", expected)
    monkeypatch.setattr(trial, "SpatialAlignmentDataset", Dataset)
    monkeypatch.setattr(evaluation, "collate_spatial_alignment", collate)
    monkeypatch.setattr(
        trial, "load_grounded_alignment_backbones", lambda _: (_Vision(), _Language(), {"vision": {}})
    )
    monkeypatch.setattr(trial, "load_spatial_prior_source", lambda *args, **kwargs: _Prior())
    return opened


def test_two_pairs_and_two_thresholds_frozen_before_reference_labels(tmp_path, monkeypatch):
    opened = _fixture(monkeypatch, tmp_path)
    result_root, result = trial.run_proposal_alignment(
        "data",
        "models",
        "prior",
        tmp_path,
        revision="a" * 40,
        expected_prior_result_sha256="b" * 64,
        recipe=trial.ProposalRecipe(1, 2, (0, 1)),
    )
    assert result["status"] == "passed", result.get("error")
    assert opened == ["train", "dev", "holdout"]
    assert result["optimizer_steps_completed"] == 8
    assert result["frozen_vision_and_language_unchanged"]
    assert result["mismatched_proposal_fraction"] == 1
    assert len(result["paired_holdout_deltas"]) == 2
    for start in (0, 2):
        fixed, proposals = result["runs"][start : start + 2]
        assert fixed["initial_weights_hash"] == proposals["initial_weights_hash"]
        assert fixed["train_order_hash"] == proposals["train_order_hash"]
        assert fixed["checkpoint_state_bitwise_equal"] and proposals["checkpoint_state_bitwise_equal"]
        assert len(proposals["holdout"]["decoder"]["per_recording"]) == 2
        assert "holdout_shuffled_rois_same_frame" in proposals
    trace = json.loads((result_root / "trace.json").read_text())
    ce_proofs = [
        row["ce_only_bridge_grad_norms"] for row in trace if row["ce_only_bridge_grad_norms"] is not None
    ]
    assert len(ce_proofs) == 4
    assert all(all(value > 0 for value in record.values()) for record in ce_proofs)


def test_receipt_mismatch_stops_before_start_of_optimization(tmp_path, monkeypatch):
    opened = _fixture(monkeypatch, tmp_path)

    def fail_receipt(*args, **kwargs):
        raise ValueError("Prior result SHA256 does not match the pinned receipt")

    monkeypatch.setattr(trial, "load_spatial_prior_source", fail_receipt)
    _, result = trial.run_proposal_alignment(
        "data",
        "models",
        "prior",
        tmp_path,
        revision="a" * 40,
        expected_prior_result_sha256="b" * 64,
    )
    assert result["status"] == "failed"
    assert "pinned receipt" in result["error"]
    assert result["optimizer_steps_completed"] == 0 and opened == ["train", "dev"]


def test_rejects_tampered_calibration_before_constructing_holdout(tmp_path, monkeypatch):
    opened = _fixture(monkeypatch, tmp_path)
    original = trial.write_json

    def writer(path, value):
        if path.name == "calibration.json":
            value = {**value, "schema": "changed"}
        original(path, value)

    monkeypatch.setattr(trial, "write_json", writer)
    _, result = trial.run_proposal_alignment(
        "data",
        "models",
        "prior",
        tmp_path,
        revision="a" * 40,
        expected_prior_result_sha256="b" * 64,
        recipe=trial.ProposalRecipe(1, 4, (0,)),
    )
    assert result["status"] == "failed" and "holdout gate" in result["error"]
    assert opened == ["train", "dev"]


def test_shuffling_priors_does_not_shuffle_the_source_features_or_targets():
    features = torch.arange(4 * 16, dtype=torch.float32).reshape(4, 1, 1, 4, 4) / 64
    proposals = _Prior.predict(features)
    labels = torch.eye(4, dtype=torch.bool)
    targets = proposals + 0.01
    data = evaluation._Encoded(features, proposals, labels, targets, ("A", "A", "B", "B"))
    indices = torch.arange(4)
    order = torch.tensor(evaluation.shuffled_by_recording(data.recordings))
    source, wrong, present, same_targets = data.batch(indices, "spatial_proposals", order)
    assert torch.equal(source, features)
    assert torch.equal(present, labels) and torch.equal(same_targets, targets)
    assert torch.equal(wrong, proposals[order]) and not torch.equal(wrong, proposals)


def test_output_cannot_alias_the_frozen_source_root(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(source, target_is_directory=True)
    with pytest.raises(ValueError, match="outside inputs"):
        trial.run_proposal_alignment(
            "data",
            "models",
            source,
            alias / "output",
            revision="a" * 40,
            expected_prior_result_sha256="b" * 64,
        )
    assert not list(source.iterdir())


def test_report_write_failure_does_not_leave_a_passed_receipt(tmp_path, monkeypatch):
    opened = _fixture(monkeypatch, tmp_path, runs=2)

    def fail_report(root, manifest, result):
        assert result["status"] == "passed"
        raise OSError("injected report write failure")

    monkeypatch.setattr(trial, "write_proposal_report", fail_report)
    root, result = trial.run_proposal_alignment(
        "data",
        "models",
        "prior",
        tmp_path,
        revision="a" * 40,
        expected_prior_result_sha256="b" * 64,
        recipe=trial.ProposalRecipe(1, 4, (0,)),
    )
    assert opened == ["train", "dev", "holdout"]
    assert result["status"] == "failed"
    assert "injected report write failure" in result["report_error"]
    assert result["optimizer_steps_completed"] == 2
    assert json.loads((root / "result.json").read_text())["status"] == "failed"



def test_fresh_run_reopens_the_pinned_pack_and_shows_the_declared_gallery_seed(tmp_path, monkeypatch):
    opened = _fixture(monkeypatch, tmp_path)
    evidence = {"manifest_hash": "e" * 64}
    gallery = {}

    def admit(original, candidate):
        assert (original, candidate) == ("data", "fresh") and not opened
        return evidence

    def publish(path, data_root, manifest, variants):
        assert data_root == "fresh" and manifest is evidence
        assert opened == ["train", "dev", "holdout"]
        assert len(list(path.parent.glob("*.safetensors"))) == 4
        gallery.update(variants)
        path.write_text("gallery")

    monkeypatch.setattr(trial, "read_fresh_spatial_holdout", admit)
    monkeypatch.setattr(trial, "write_localization_gallery", publish)
    root, result = trial.run_proposal_alignment(
        "data", "models", "prior", tmp_path,
        revision="a" * 40, expected_prior_result_sha256="b" * 64,
        recipe=trial.ProposalRecipe(1, 2, (1, 0)), fresh_holdout_root="fresh",
    )
    assert result["status"] == "passed", result.get("error")
    assert result["scope"] == trial.FRESH_SCOPE
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["gallery_seed"] == 1 and manifest["fresh_holdout_manifest_hash"] == "e" * 64
    direct, fixed, roi = list(gallery.values())
    assert direct.threshold == result["direct_spatial"]["dev_threshold"] == -0.75
    assert fixed.threshold == result["runs"][0]["dev_thresholds"]["decoder"]["threshold"]
    assert roi.threshold == result["runs"][1]["dev_thresholds"]["decoder"]["threshold"]
    assert direct.scores.shape == fixed.scores.shape == roi.scores.shape == (4, 4)
    for row in result["runs"]:
        assert row["checkpoint_metadata"]["memory_mode"] == "roi_tokens"
        assert row["checkpoint_metadata"]["fresh_holdout_manifest_hash"] == "e" * 64
    assert "Fresh OpenScene" in (root / "report.html").read_text()
    assert result["gallery"] == "gallery.html"


def test_rejects_bad_fresh_pack_before_training_on_old_pack(tmp_path, monkeypatch):
    opened = _fixture(monkeypatch, tmp_path)

    def reject(*args):
        raise ValueError("Fresh holdout has repeated pixels")

    monkeypatch.setattr(trial, "read_fresh_spatial_holdout", reject)
    _, result = trial.run_proposal_alignment(
        "data", "models", "prior", tmp_path,
        revision="a" * 40, expected_prior_result_sha256="b" * 64, fresh_holdout_root="fresh",
    )
    assert result["status"] == "failed" and "repeated pixels" in result["error"]
    assert result["optimizer_steps_completed"] == 0 and opened == []


def test_direct_control_shuffles_head_outputs_not_labels_or_threshold():
    features = torch.linspace(0, 1, 32).reshape(4, 1, 1, 2, 4)
    boxes = _Prior.predict(features)
    labels = torch.eye(4, dtype=torch.bool)
    targets = boxes.clone()
    scores = torch.where(labels, 2.0, 0.25)
    heldout = evaluation._Encoded(features, boxes, labels, targets, ("A", "A", "B", "B"), scores)
    reference, gallery = evaluation.evaluate_direct_spatial(heldout, threshold=1.0)
    measured, swapped = reference["holdout"], reference["holdout_shuffled_images_same_labels"]
    assert measured["macro_roc_auc"] == measured["macro_balanced_accuracy"] == measured["macro_positive_iou"] == 1.0
    assert swapped["recall"] == 0 and swapped["specificity"] == pytest.approx(2 / 3)
    assert swapped["macro_positive_iou"] < 1 and gallery.threshold == 1.0
    torch.testing.assert_close(torch.from_numpy(gallery.scores), scores)
