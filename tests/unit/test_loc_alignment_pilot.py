import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from hybrid_drive.experiments import alignment_pilot as pilot
from hybrid_drive.experiments.alignment_metrics import LocMetricAccumulator
from hybrid_drive.neural.loc_probe import LocAlignmentProbe, supervised_loc_loss
from hybrid_drive.neural.losses import Reason2DriveLoss


class _Language(nn.Module):
    llm_dim = 8
    loc_token_id = 2
    mot_token_id = 3
    yes_token_id = 4
    no_token_id = 5

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(6, 8)
        self.head = nn.Linear(8, 6)
        self.requires_grad_(False).eval()

    def prompt_ids(self, count):
        return torch.tensor([0, 1, 2]).expand(count, -1)

    def forward(self, prefix, ids):
        hidden = self.embedding(ids) + prefix.mean(1, keepdim=True)
        return hidden, self.head(hidden)


class _Vision(nn.Module):
    visual_dim = 4

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor([1.0, 2.0, 3.0, 4.0]), requires_grad=False)
        self.eval()

    def forward(self, images):
        mean = images.mean(dim=(1, 2, 3, 4))
        return (mean[:, None, None, None, None] * self.scale[None, None, :, None, None]).expand(
            -1, 1, 4, 2, 2
        )


def _objective_fixture():
    language = _Language()
    probe = LocAlignmentProbe(language, visual_dim=4)
    features = torch.linspace(-1, 1, 2 * 4 * 2 * 2).reshape(2, 1, 4, 2, 2)
    prediction = probe(features, language)
    targets = (
        torch.tensor([True, False]),
        torch.tensor([[5, 0, 0, 1, 2, 1, 0.2], [float("nan")] * 7]),
        torch.tensor([[0.1, 0.1, 0.8, 0.8], [float("nan")] * 4]),
    )
    return probe, language, prediction, targets


def test_loc_text_target_is_not_consumed_and_ce_alone_reaches_bridge():
    probe, language, prediction, targets = _objective_fixture()
    assert prediction.consumed_ids.tolist() == [[0, 1, 2], [0, 1, 2]]
    assert prediction.perception.token_positions.tolist() == [2, 2]
    assert prediction.perception.event_kinds.tolist() == [0, 0]
    loss = supervised_loc_loss(Reason2DriveLoss(), prediction, language, *targets)
    oracle = torch.nn.functional.cross_entropy(prediction.logits[:, -1], torch.tensor([4, 5]))
    torch.testing.assert_close(loss.text, oracle)
    assert (loss.supervised_box_coordinates, loss.supervised_image_boxes, loss.supervised_motion_points) == (
        7,
        1,
        0,
    )
    parameters = list(probe.projector.parameters()) + list(probe.tokenizer.parameters())
    gradients = torch.autograd.grad(loss.text, parameters)
    assert all(torch.isfinite(value).all() for value in gradients)
    assert any(gradient.abs().sum() > 0 for gradient in gradients)
    assert all(parameter.grad is None and not parameter.requires_grad for parameter in language.parameters())


def test_metrics_keep_missing_denominators_and_use_counts_not_batch_average():
    acc = LocMetricAccumulator()
    language = SimpleNamespace(yes_token_id=0, no_token_id=1)
    presence = torch.tensor([True, False])
    box3 = torch.tensor([[2.0, 0, 0, 1, 2, 1, 3.04], [float("nan")] * 7])
    box2 = torch.tensor([[0.0, 0, 0.5, 0.5], [float("nan")] * 4])
    predicted3 = torch.tensor([[5.0, 4, 0, 2, 3, 2, -3.04], [float("nan")] * 7])
    predicted2 = torch.tensor([[0.0, 0, 1.0, 1.0], [float("nan")] * 4])
    prediction = SimpleNamespace(
        logits=torch.tensor([[[1.0, 0.0]], [[1.0, 0.0]]]),
        perception=SimpleNamespace(
            validity_logits=torch.tensor([1.0, 1.0]), boxes_3d=predicted3, boxes_2d=predicted2
        ),
    )
    loss = SimpleNamespace(
        text=2.0,
        classification=1.0,
        boxes_3d=3.0,
        giou_2d=0.75,
        supervised_text_tokens=2,
        supervised_events=2,
        supervised_box_coordinates=7,
        supervised_image_boxes=1,
    )
    acc.add(prediction, loss, language, presence, box3, box2)
    s = acc.summary()
    assert s["presence_balanced_accuracy"] == 0.5
    assert s["positive_projected_box_iou"] == 0.25
    assert s["positive_center_l2_m"] == 5
    assert s["positive_mean_size_mae_m"] == 1
    assert s["positive_yaw_mae_rad"] == pytest.approx(0.203185, abs=1e-6)
    only_negative = LocMetricAccumulator()
    prediction = SimpleNamespace(
        logits=torch.tensor([[[0.0, 1.0]]]), perception=SimpleNamespace(validity_logits=torch.tensor([-1.0]))
    )
    loss = SimpleNamespace(
        text=4.0,
        classification=1.0,
        boxes_3d=0.0,
        giou_2d=0.0,
        supervised_text_tokens=1,
        supervised_events=1,
        supervised_box_coordinates=0,
        supervised_image_boxes=0,
    )
    only_negative.add(prediction, loss, language, presence[1:], box3[1:], box2[1:])
    assert only_negative.summary()["positive_center_l2_m"] is None
    assert only_negative.summary()["presence_balanced_accuracy"] is None
    acc.add(prediction, loss, language, presence[1:], box3[1:], box2[1:])
    assert acc.summary()["loss_terms"]["text"] == pytest.approx(8 / 3)


def _fake_pipeline(monkeypatch, *, shared_recording=False, bad_features=False):
    class Dataset:
        flip_holdout = False

        def __init__(self, root, split):
            self.split = split
            self.recording_id = "train" if shared_recording else split
            self.dataset_hash = "a" * 64
            self.sample_ids = tuple(f"{split}{i}" for i in range(2))

        def __len__(self):
            return len(self.sample_ids)

        def __getitem__(self, index):
            positive = index == 0
            if self.split == "holdout" and self.flip_holdout:
                positive = not positive
            rgb = torch.full((1, 3, 12, 12), 0.2 + 0.5 * index)
            if bad_features:
                rgb.fill_(float("nan"))
            return (
                self.sample_ids[index],
                rgb,
                positive,
                torch.tensor([5.0, 0, 0, 1, 2, 1, 0] if positive else [float("nan")] * 7),
                torch.tensor([0.1, 0.1, 0.8, 0.8] if positive else [float("nan")] * 4),
            )

    def collate(rows):
        ids, rgb, presence, boxes3, boxes2 = zip(*rows)
        return SimpleNamespace(
            sample_ids=ids,
            images=torch.stack(rgb),
            presence=torch.tensor(presence),
            boxes_3d=torch.stack(boxes3),
            boxes_2d=torch.stack(boxes2),
        )

    def backbones(path):
        torch.manual_seed(18)
        return _Vision(), _Language(), {"kind": "unit_fixture"}

    monkeypatch.setattr(pilot, "EXPECTED_COUNTS", {"train": 2, "holdout": 2})
    monkeypatch.setattr(pilot, "AlignmentDataset", Dataset)
    monkeypatch.setattr(pilot, "collate_alignment", collate)
    monkeypatch.setattr(pilot, "load_alignment_backbones", backbones)
    return Dataset


def test_paired_optimizers_use_same_initial_weights_and_ignore_holdout_labels(tmp_path, monkeypatch):
    dataset = _fake_pipeline(monkeypatch)
    recipe = pilot.PilotRecipe(epochs=1, batch_size=2)
    rng_before = torch.random.get_rng_state().clone()
    _, first = pilot.run_alignment_pilot("data", "assets", tmp_path, revision="a" * 40, recipe=recipe)
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert first["status"] == "passed", first.get("error")
    dataset.flip_holdout = True
    path, second = pilot.run_alignment_pilot("data", "assets", tmp_path, revision="a" * 40, recipe=recipe)
    assert second["status"] == "passed", second.get("error")
    assert second["frozen_state_unchanged"] and second["frozen_parameters_with_gradients"] == 0
    left, right = second["arms"].values()
    assert left["initial_weight_hash"] == right["initial_weight_hash"]
    assert left["final_weight_hash"] != right["final_weight_hash"]
    for arm in pilot.ARMS:
        actual = second["arms"][arm]
        assert actual["final_weight_hash"] == first["arms"][arm]["final_weight_hash"]
        assert actual["checkpoint_roundtrip_bitwise"]
        assert actual["holdout_after"]["examples"] == 2
        assert actual["steps"] == 1
        assert all(value > 0 for value in actual["text_only_bridge_gradients"].values())
    assert "Projected 2D targets" in (path / "report.html").read_text()


def test_shared_recording_fails_before_model_or_artifact_creation(tmp_path, monkeypatch):
    _fake_pipeline(monkeypatch, shared_recording=True)
    with pytest.raises(ValueError, match="distinct recordings"):
        pilot.run_alignment_pilot("data", "assets", tmp_path, revision="a" * 40)
    assert not list(tmp_path.iterdir())


def test_bad_encoder_features_are_retained_as_failed_run(tmp_path, monkeypatch):
    _fake_pipeline(monkeypatch, bad_features=True)
    path, result = pilot.run_alignment_pilot("data", "assets", tmp_path, revision="a" * 40)
    assert result["status"] == "failed"
    assert "invalid features" in result["error"]
    assert result["optimizer_steps_completed"] == 0
    assert not list(path.glob("*.safetensors"))


def test_output_cannot_escape_input_root_check_through_parent_parts(tmp_path):
    source = tmp_path / "data"
    with pytest.raises(ValueError, match="separate"):
        pilot.run_alignment_pilot(
            source, "assets", tmp_path / "sibling" / ".." / "data" / "out", revision="a" * 40
        )
    assert not source.exists()


def test_incomplete_arm_does_not_extend_to_the_end_of_the_shared_step_axis():
    from hybrid_drive.experiments.alignment_report import _loss_chart

    rows = [
        {"arm": "image_features", "loss": 4},
        {"arm": "image_features", "loss": 3},
        {"arm": "constant_zero_features", "loss": 4},
    ]
    output = _loss_chart(rows, 3)
    assert "502.50," in output and "935.00," not in output


def test_interrupted_run_has_final_diagnostic(tmp_path, monkeypatch):
    _fake_pipeline(monkeypatch)

    def interrupt(path):
        raise KeyboardInterrupt

    monkeypatch.setattr(pilot, "load_alignment_backbones", interrupt)
    root, result = pilot.run_alignment_pilot("data", "assets", tmp_path, revision="a" * 40)
    assert result["status"] == "interrupted"
    assert result["optimizer_steps_completed"] == 0
    assert "Interrupted by operator" in (root / "report.html").read_text()


def test_interrupt_during_manifest_update_preserves_hash_binding(tmp_path, monkeypatch):
    _fake_pipeline(monkeypatch)
    writer = pilot.write_json
    interrupted = False

    def interrupted_write(path, value):
        nonlocal interrupted
        if path.name == "manifest.json" and "backbones" in value and not interrupted:
            interrupted = True
            raise KeyboardInterrupt
        writer(path, value)

    monkeypatch.setattr(pilot, "write_json", interrupted_write)
    root, result = pilot.run_alignment_pilot("data", "assets", tmp_path, revision="a" * 40)
    assert interrupted and result["status"] == "interrupted"
    assert result["manifest_hash"] == pilot.content_hash(json.loads((root / "manifest.json").read_text()))
