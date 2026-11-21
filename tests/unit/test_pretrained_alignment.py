from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from hybrid_drive.neural.pretrained_alignment import _DinoPatchGrid, _PythiaPrefix, _encode_prompt


class _PatchFixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, *, pixel_values, return_dict):
        assert return_dict
        patch_values = pixel_values[:, :, ::14, ::14].flatten(2).transpose(1, 2)
        patches = patch_values.repeat(1, 1, 128) * self.scale
        cls = patches.new_full((patches.shape[0], 1, patches.shape[-1]), 100.0)
        return SimpleNamespace(last_hidden_state=torch.cat((cls, patches), dim=1))


class _CausalFixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(8, 512)
        self.head = nn.Linear(512, 8)

    def get_input_embeddings(self):
        return self.embedding

    def gpt_neox(self, *, inputs_embeds, use_cache, return_dict):
        assert return_dict and not use_cache
        return SimpleNamespace(last_hidden_state=inputs_embeds.cumsum(dim=1))

    def get_output_embeddings(self):
        return lambda hidden: F.pad(self.head(hidden), (0, 50304 - self.head.out_features))


class _TokenizerFixture:
    is_fast = True
    all_special_ids = [0]

    def __init__(self):
        self.ids = {" LOC": [35378], " MOT": [34794], " yes": [4754], " no": [642]}
        self.prompt = [10, 11, 35378]

    def encode(self, text, *, add_special_tokens):
        assert not add_special_tokens
        if text in self.ids:
            return self.ids[text]
        assert text.endswith(" LOC")
        return self.prompt


def test_frozen_vision_normalizes_rgb_and_preserves_spatial_patch_order_for_trainable_consumers():
    vision = _DinoPatchGrid(_PatchFixture())
    images = torch.full((2, 1, 3, 224, 224), 0.5)
    images[:, :, 0, 14, 28] = torch.tensor([0.6, 0.7])[:, None]
    images.requires_grad_()
    grid = vision(images)

    assert grid.shape == (2, 1, 384, 16, 16)
    assert not grid.requires_grad
    assert not vision.training and not vision.backbone.training
    assert not vision.backbone.scale.requires_grad
    torch.testing.assert_close(grid[:, 0, 0, 1, 2], (torch.tensor([0.6, 0.7]) - 0.485) / 0.229)
    torch.testing.assert_close(grid[:, 0, 1, 1, 2], torch.full((2,), (0.5 - 0.456) / 0.224))
    torch.testing.assert_close(grid[:, 0, 2, 1, 2], torch.full((2,), (0.5 - 0.406) / 0.225))
    torch.testing.assert_close(grid[:, 0, 0, 0, 0], torch.full((2,), (0.5 - 0.485) / 0.229))

    trainable = nn.Linear(vision.visual_dim, 1)
    trainable(grid[:, 0, :, 1, 2]).square().sum().backward()
    assert trainable.weight.grad is not None and torch.count_nonzero(trainable.weight.grad)
    assert images.grad is None and vision.backbone.scale.grad is None


def test_vision_rejects_wrong_geometry_and_already_normalized_inputs():
    vision = _DinoPatchGrid(_PatchFixture())
    with pytest.raises(ValueError, match="shape"):
        vision(torch.zeros(1, 2, 3, 224, 224))
    with pytest.raises(ValueError, match="before normalization"):
        vision(torch.full((1, 1, 3, 224, 224), -0.1))


def test_frozen_language_retains_consumed_positions_and_ce_gradient_to_prefix():
    backbone = _CausalFixture()
    language = _PythiaPrefix(backbone, [1, 2, 35378])
    prefix = torch.ones((1, 2, 512), requires_grad=True)
    ids = torch.tensor([[3, 4, 5]])
    hidden, logits = language(prefix, ids)

    assert hidden.shape == (1, 3, 512)
    assert logits.shape == (1, 3, 50304)
    embedded = backbone.embedding(ids)
    torch.testing.assert_close(hidden[:, 0], 2 + embedded[:, 0])
    torch.testing.assert_close(hidden[:, 1], 2 + embedded[:, 0] + embedded[:, 1])
    torch.testing.assert_close(logits[:, 1, :8], backbone.head(hidden[:, 1]))
    assert not language.training and not backbone.training
    assert all(not parameter.requires_grad for parameter in backbone.parameters())
    F.cross_entropy(logits[:, -1], torch.tensor([6])).backward()
    assert prefix.grad is not None and torch.isfinite(prefix.grad).all() and torch.count_nonzero(prefix.grad)
    assert all(parameter.grad is None for parameter in backbone.parameters())


def test_prompt_is_fixed_to_single_native_loc_and_returned_batches_are_independent():
    language = _PythiaPrefix(_CausalFixture(), _encode_prompt(_TokenizerFixture()))
    prompts = language.prompt_ids(2)
    assert prompts.tolist() == [[10, 11, language.loc_token_id]] * 2
    prompts[0, 0] = language.yes_token_id
    assert language.prompt_ids(1).tolist() == [[10, 11, language.loc_token_id]]
    assert prompts[1, 0] == 10


def test_tokenizer_rejects_split_alias_and_prompt_without_unique_final_loc():
    tokenizer = _TokenizerFixture()
    tokenizer.ids[" yes"] = [44, 4754]
    with pytest.raises(ValueError, match="token alias"):
        _encode_prompt(tokenizer)

    tokenizer = _TokenizerFixture()
    tokenizer.prompt = [10, 35378, 11, 35378]
    with pytest.raises(ValueError, match="single consumed LOC"):
        _encode_prompt(tokenizer)


def test_language_rejects_bad_input_shape_or_out_of_vocab_id():
    language = _PythiaPrefix(_CausalFixture(), [1, 2, 35378])
    with pytest.raises(ValueError, match="prefix"):
        language(torch.zeros(1, 2, 768), torch.ones(1, 2, dtype=torch.long))
    with pytest.raises(ValueError, match="consumed_ids"):
        language(torch.zeros(1, 2, 512), torch.tensor([[0, 50304]]))
