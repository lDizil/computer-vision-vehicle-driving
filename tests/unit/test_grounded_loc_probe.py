from dataclasses import replace
from hashlib import sha256
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from hybrid_drive.neural import pretrained_alignment as pretrained
from hybrid_drive.neural.grounded_loc_probe import (
    GroundedLocProbe,
    fixed_quarter_boxes,
    grounded_presence_scores,
    supervised_grounded_loc_loss,
)
from hybrid_drive.neural.losses import Reason2DriveLoss


_QUARTER_BOXES = torch.tensor(
    [
        [0.0, 0.0, 0.25, 1.0],
        [0.25, 0.0, 0.5, 1.0],
        [0.5, 0.0, 0.75, 1.0],
        [0.75, 0.0, 1.0, 1.0],
    ]
)


class _FrozenCausalFixture(nn.Module):
    llm_dim = 8
    vocab_size = 10
    loc_token_id = 2
    mot_token_id = 3
    yes_token_id = 4
    no_token_id = 5

    def __init__(self):
        super().__init__()
        self.register_buffer("prompt", torch.tensor([6, 2, 7, 2, 8, 2, 9, 2]))
        self.embedding = nn.Embedding(self.vocab_size, self.llm_dim)
        self.head = nn.Linear(self.llm_dim, self.vocab_size)
        self.requires_grad_(False).eval()

    def prompt_ids(self, batch):
        return self.prompt.expand(batch, -1).clone()

    def forward(self, prefix, ids):
        context = torch.cat((prefix, self.embedding(ids)), dim=1)
        hidden = context.cumsum(dim=1)[:, prefix.shape[1] :]
        return hidden, self.head(hidden)


@pytest.fixture
def probe_inputs():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        language = _FrozenCausalFixture()
        probe = GroundedLocProbe(language, visual_dim=4, hidden_dim=8)
    features = torch.linspace(-1, 1, 2 * 4 * 2 * 8).reshape(2, 1, 4, 2, 8)
    return probe, language, features


def _targets():
    presence = torch.tensor([[True, False, True, False], [False, True, False, True]])
    targets = _QUARTER_BOXES.expand(2, -1, -1).clone()
    targets[~presence] = float("nan")
    return presence, targets


def test_four_quarter_priors_leave_the_global_visual_memory_unchanged(probe_inputs):
    probe, language, features = probe_inputs
    inputs = {}
    for name, module in probe.weight_modules().items():
        module.register_forward_pre_hook(lambda _, args, key=name: inputs.update({key: args}))

    boxes = fixed_quarter_boxes(features)
    torch.testing.assert_close(boxes, _QUARTER_BOXES.expand(2, -1, -1))
    prediction = probe(features, boxes, language)
    rois, images, geometry, active = inputs["tokenizer"][1:]
    torch.testing.assert_close(rois, boxes)
    expected_geometry = torch.tensor(
        [
            [0.125, 0.5, 0.25, 1],
            [0.375, 0.5, 0.25, 1],
            [0.625, 0.5, 0.25, 1],
            [0.875, 0.5, 0.25, 1],
        ]
    ).expand(2, -1, -1)
    torch.testing.assert_close(geometry, expected_geometry)
    torch.testing.assert_close(images, torch.zeros(2, 4, dtype=torch.long))
    torch.testing.assert_close(active, torch.ones(2, 4, dtype=torch.bool))
    assert inputs["projector"][2].shape == (2, 4, 768)
    torch.testing.assert_close(inputs["projector"][3], active)
    for key in ("projector", "decoder"):
        visual, visual_mask = inputs[key][:2]
        torch.testing.assert_close(visual, features[:, 0].flatten(2).transpose(1, 2))
        assert visual.shape == (2, 16, 4) and visual_mask.all()
    assert prediction.consumed_ids.tolist() == [[6, 2, 7, 2, 8, 2, 9, 2]] * 2
    assert prediction.loc_positions.tolist() == [[1, 3, 5, 7]] * 2
    assert prediction.perception.batch_indices.tolist() == [0] * 4 + [1] * 4
    assert prediction.perception.token_positions.tolist() == [1, 3, 5, 7] * 2
    assert prediction.perception.event_kinds.tolist() == [0] * 8


def test_proposals_do_not_bypass_the_latent_tokens(probe_inputs):
    probe, language, features = probe_inputs
    quarters = fixed_quarter_boxes(features)
    proposals = quarters.clone()
    proposals[..., 0] += 0.03125
    proposals[..., 1] += 0.25
    with torch.no_grad():
        fusion = probe.tokenizer.fusion[-1]
        fusion.weight.zero_()
        fusion.bias.zero_()
        reference = probe(features, quarters, language)
        changed = probe(features, proposals, language)
    torch.testing.assert_close(changed.consumed_ids, reference.consumed_ids)
    for actual, expected in zip(
        grounded_presence_scores(changed, language), grounded_presence_scores(reference, language)
    ):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_ce_uses_only_the_logits_after_each_consumed_marker(probe_inputs):
    probe, language, features = probe_inputs
    prediction = probe(features, fixed_quarter_boxes(features), language)
    truth, targets = _targets()
    after_loc = torch.tensor([1, 3, 5, 7])
    known_logits = torch.linspace(-2, 2, 80).reshape(2, 4, 10).sin()
    logits = torch.full_like(prediction.logits, float("nan"))
    logits[:, after_loc] = known_logits
    prediction = replace(prediction, logits=logits)
    loss = supervised_grounded_loc_loss(Reason2DriveLoss(), prediction, language, truth, targets)
    expected_text = F.cross_entropy(known_logits.reshape(8, 10), torch.tensor([4, 5, 4, 5, 5, 4, 5, 4]))
    torch.testing.assert_close(loss.text, expected_text)
    assert loss.supervised_text_tokens == loss.supervised_events == 8
    assert loss.supervised_image_boxes == 4
    assert loss.supervised_box_coordinates == loss.supervised_motion_points == 0

    validity, text, boxes = grounded_presence_scores(prediction, language)
    torch.testing.assert_close(text, known_logits[..., 4] - known_logits[..., 5])
    torch.testing.assert_close(validity, prediction.perception.validity_logits.reshape(2, 4))
    torch.testing.assert_close(boxes, prediction.perception.boxes_2d.reshape(2, 4, 4))


def test_text_only_gradients_reach_both_prior_and_projector_without_training_the_lm(probe_inputs):
    probe, language, features = probe_inputs
    prediction = probe(features, fixed_quarter_boxes(features), language)
    loss = supervised_grounded_loc_loss(Reason2DriveLoss(), prediction, language, *_targets())
    loss.text.backward()
    for name, parameter in (
        ("ROI", probe.tokenizer.spatial_pool.weight),
        ("geometry", probe.tokenizer.geometry_embedding[0].weight),
        ("prior attention", probe.projector.prior_projection.weight),
    ):
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all() and torch.count_nonzero(parameter.grad), name
    assert all(parameter.grad is None and not parameter.requires_grad for parameter in language.parameters())
    assert not language.training


def test_all_absent_examples_train_negative_labels_but_ignore_box_payloads(probe_inputs):
    probe, language, features = probe_inputs
    prediction = probe(features, fixed_quarter_boxes(features), language)
    truth = torch.zeros((2, 4), dtype=torch.bool)
    targets = torch.full((2, 4, 4), float("nan"))
    loss = supervised_grounded_loc_loss(Reason2DriveLoss(), prediction, language, truth, targets)

    expected = F.cross_entropy(prediction.logits[:, [1, 3, 5, 7]].reshape(8, 10), torch.full((8,), 5))
    torch.testing.assert_close(loss.text, expected)
    assert loss.supervised_text_tokens == loss.supervised_events == 8
    assert (
        loss.supervised_image_boxes == loss.supervised_box_coordinates == loss.supervised_motion_points == 0
    )
    assert loss.giou_2d == 0


def test_rejects_present_box_from_another_quarter(probe_inputs):
    probe, language, features = probe_inputs
    prediction = probe(features, fixed_quarter_boxes(features), language)
    truth, targets = _targets()
    targets[0, 0] = targets[0, 2]
    with pytest.raises(ValueError, match="assigned quarter"):
        supervised_grounded_loc_loss(Reason2DriveLoss(), prediction, language, truth, targets)


@pytest.mark.parametrize("fault", ["fewer", "extra", "early-last", "MOT", "yes", "no"])
def test_rejects_noncanonical_markers_and_answers_before_forward(probe_inputs, fault):
    probe, language, features = probe_inputs
    altered = {
        "fewer": [6, 2, 7, 8, 2, 9, 2],
        "extra": [2, 6, 2, 7, 2, 8, 2, 9, 2],
        "early-last": [6, 2, 7, 2, 8, 2, 2, 9],
        "MOT": [6, 2, 3, 2, 8, 2, 9, 2],
        "yes": [6, 2, 4, 2, 8, 2, 9, 2],
        "no": [6, 2, 5, 2, 8, 2, 9, 2],
    }
    language.prompt = torch.tensor(altered[fault])
    with pytest.raises(ValueError, match="four consumed LOC"):
        probe(features, fixed_quarter_boxes(features), language)


def test_rejects_quarter_cue_reordering_even_with_the_same_loc_positions(probe_inputs):
    probe, language, features = probe_inputs
    language.prompt[0], language.prompt[2] = language.prompt[2].clone(), language.prompt[0].clone()
    with pytest.raises(ValueError, match="fixed quarter order"):
        probe(features, fixed_quarter_boxes(features), language)


def test_rejects_forward_logits_with_a_fake_vocabulary_width(probe_inputs):
    probe, language, features = probe_inputs
    language.register_forward_hook(lambda _, __, output: (output[0], output[1][..., :-1]))
    with pytest.raises(ValueError, match="vocabulary width"):
        probe(features, fixed_quarter_boxes(features), language)


def test_rejects_wrong_decoder_order_during_forward(probe_inputs):
    probe, language, features = probe_inputs
    probe.decoder.register_forward_hook(
        lambda _, __, output: replace(output, batch_indices=output.batch_indices.flip(0))
    )
    with pytest.raises(ValueError, match="frame/token order"):
        probe(features, fixed_quarter_boxes(features), language)


def test_training_and_score_helpers_check_marker_and_decoder_identity_independently(probe_inputs):
    probe, language, features = probe_inputs
    prediction = probe(features, fixed_quarter_boxes(features), language)
    misindexed = replace(prediction, loc_positions=prediction.loc_positions.flip(1))
    misordered = replace(prediction.perception, token_positions=prediction.perception.token_positions.roll(1))
    wrong_kind = replace(
        prediction.perception, event_kinds=torch.ones_like(prediction.perception.event_kinds)
    )
    for bad, message in (
        (misindexed, "fixed consumed markers"),
        (replace(prediction, perception=misordered), "frame/token order"),
        (replace(prediction, perception=wrong_kind), "LOC kind"),
    ):
        with pytest.raises(ValueError, match=message):
            supervised_grounded_loc_loss(Reason2DriveLoss(), bad, language, *_targets())
        with pytest.raises(ValueError, match=message):
            grounded_presence_scores(bad, language)


def test_requires_four_valid_prior_boxes_in_every_frame(probe_inputs):
    probe, language, features = probe_inputs
    boxes = fixed_quarter_boxes(features).clone()
    with pytest.raises(ValueError, match="four normalized"):
        probe(features, boxes[:, :3], language)
    boxes[0, 3, 2] = boxes[0, 3, 0]
    with pytest.raises(ValueError, match="positive width"):
        probe(features, boxes, language)


class _PinnedTokenizerFixture:
    is_fast = True
    all_special_ids = [0]

    def __init__(self):
        self.grounded_ids = list(pretrained._GROUNDED_PROMPT_IDS)

    def encode(self, text, *, add_special_tokens):
        assert not add_special_tokens
        if text in pretrained._TOKEN_ALIASES:
            return [pretrained._TOKEN_ALIASES[text]]
        if text == pretrained._GROUNDED_PROMPT:
            return self.grounded_ids
        assert text == pretrained._PROMPT
        return [31, 35378]


def test_grounded_loader_preserves_the_vision_only_provenance_and_legacy_prompt(monkeypatch):
    source = {"repository": "vision-repository", "revision": "vision-revision"}
    vision_source = {"source": source, "sources_sha256": "source-checksum"}
    monkeypatch.setattr(pretrained, "verify_alignment_vision", lambda root: ("vision-path", vision_source))
    monkeypatch.setattr(
        pretrained,
        "verify_alignment_models",
        lambda root: (
            "vision-path",
            "language-path",
            {"models": {"vision": source}, "sources_sha256": "source-checksum"},
        ),
    )
    monkeypatch.setattr(pretrained, "_runtime_versions", lambda: {"torch": "2.0.1"})
    monkeypatch.setattr(pretrained, "_load_vision", lambda path: pretrained._DinoPatchGrid(nn.Linear(1, 1)))
    tokenizer = SimpleNamespace(from_pretrained=lambda path, *, local_files_only: _PinnedTokenizerFixture())
    transformers = SimpleNamespace(
        GPTNeoXConfig=object, GPTNeoXForCausalLM=nn.Linear, GPTNeoXTokenizerFast=tokenizer
    )
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setattr(
        pretrained, "_checked_config", lambda type_, path, **expected: SimpleNamespace(**expected)
    )
    monkeypatch.setattr(pretrained, "_load_model", lambda type_, path, config, *, ignored: type_(1, 1))

    _, existing = pretrained.load_alignment_vision("unused")
    vision, language, provenance = pretrained.load_grounded_alignment_backbones("unused")
    _, old_language, old_provenance = pretrained.load_alignment_backbones("unused")

    assert provenance["vision"] == existing
    assert set(existing) == {"source", "sources_sha256", "runtime", "preprocessing"}
    assert existing["runtime"] == {
        "torch": "2.0.1",
        "device": "cpu",
        "dtype": "float32",
        "attention": "eager",
    }
    assert provenance["tokenization"]["loc_positions"] == [30, 35, 40, 45]
    assert language.prompt_ids(1).tolist() == [provenance["tokenization"]["prompt_ids"]]
    assert old_language.prompt_ids(1).tolist() == [[31, 35378]]
    assert old_provenance["tokenization"]["prompt_ids"] == [31, 35378]
    for frozen in (vision, language):
        assert not frozen.training and all(not parameter.requires_grad for parameter in frozen.parameters())


def test_pinned_tokenizer_guard_rejects_reordered_quarter_cues():
    tokenizer = _PinnedTokenizerFixture()
    tokenizer.grounded_ids[27], tokenizer.grounded_ids[32] = (
        tokenizer.grounded_ids[32],
        tokenizer.grounded_ids[27],
    )
    with pytest.raises(ValueError, match="quarter order"):
        pretrained._encode_grounded_prompt(tokenizer)


def test_real_pinned_tokenizer_consumes_four_loc_markers_without_answers():
    root = os.environ.get("ALIGNMENT_ASSET_ROOT")
    if not root:
        pytest.skip("Set ALIGNMENT_ASSET_ROOT to the optional local pinned backbone assets")
    transformers = pytest.importorskip("transformers")
    path = Path(root) / "pythia-70m-deduped"
    assert sha256((path / "tokenizer.json").read_bytes()).hexdigest() == (
        "c24618a1b3e6a38167beff1c72cffd126c3a66254347304b50547d12c5f25624"
    )
    tokenizer = transformers.GPTNeoXTokenizerFast.from_pretrained(path, local_files_only=True)
    ids = pretrained._encode_grounded_prompt(tokenizer)
    assert len(ids) == 46
    assert [index for index, token in enumerate(ids) if token == 35378] == [30, 35, 40, 45]
    assert not set(ids).intersection((34794, 4754, 642))


def test_swapping_prior_quarters_is_rejected(probe_inputs):
    probe, language, features = probe_inputs
    boxes = fixed_quarter_boxes(features).flip(1)
    with pytest.raises(ValueError, match="left-to-right quarter order"):
        probe(features, boxes, language)



def test_isolated_roi_tokens_are_the_only_inputs_to_both_attention_memories(probe_inputs):
    _, language, features = probe_inputs
    probe = GroundedLocProbe(language, visual_dim=4, hidden_dim=8, memory_mode="roi_tokens")
    observed = {}
    for name in ("projector", "decoder"):
        getattr(probe, name).register_forward_pre_hook(
            lambda _, args, key=name: observed.update({key: args})
        )
    tokens = []
    probe.tokenizer.register_forward_hook(lambda _, args, out: tokens.append(out.tokens))
    prediction = probe(features, fixed_quarter_boxes(features), language)
    assert len(observed["projector"]) == 2
    for name in ("projector", "decoder"):
        bank, mask = observed[name][:2]
        assert bank is tokens[0] and bank.shape == (2, 4, 4) and mask.all()
    loss = supervised_grounded_loc_loss(Reason2DriveLoss(), prediction, language, *_targets())
    loss.text.backward()
    assert torch.count_nonzero(probe.tokenizer.spatial_pool.weight.grad)
    assert torch.count_nonzero(probe.projector.visual_projection.weight.grad)
    assert probe.projector.prior_projection.weight.grad is None
    assert all(p.grad is None for p in language.parameters())


def test_patches_cannot_bypass_the_isolated_token_bank(probe_inputs):
    _, language, features = probe_inputs
    probe = GroundedLocProbe(language, visual_dim=4, hidden_dim=8, memory_mode="roi_tokens").eval()
    boxes = fixed_quarter_boxes(features)
    with torch.no_grad():
        probe.tokenizer.fusion[-1].weight.zero_()
        probe.tokenizer.fusion[-1].bias.zero_()
        left = grounded_presence_scores(probe(features, boxes, language), language)
        right = grounded_presence_scores(probe(-features, boxes, language), language)
    for a, b in zip(left, right):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
