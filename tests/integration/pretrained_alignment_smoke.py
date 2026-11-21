"""Opt-in: ALIGNMENT_ASSET_ROOT=/models python -m pytest -q tests/integration/pretrained_alignment_smoke.py."""

import os

import torch
from torch.nn.functional import cross_entropy

from hybrid_drive.neural.pretrained_alignment import load_alignment_backbones


def test_pinned_models_support_offline_vision_causal_text_and_prefix_backward():
    vision, language, _ = load_alignment_backbones(os.environ["ALIGNMENT_ASSET_ROOT"])
    grid = vision(torch.full((1, 1, 3, 224, 224), 0.5))
    assert grid.shape == (1, 1, 384, 16, 16) and torch.isfinite(grid).all()
    prefix = 0.02 * torch.randn(1, 4, language.llm_dim, generator=torch.Generator().manual_seed(0))
    prefix.requires_grad_()
    prompt = language.prompt_ids(1)
    assert prompt[0, -1] == language.loc_token_id
    hidden, logits = language(prefix, prompt)
    assert hidden.shape == (*prompt.shape, 512) and torch.isfinite(hidden).all()
    assert logits.shape == (*prompt.shape, language.vocab_size) and torch.isfinite(logits).all()

    yes = torch.tensor([[language.yes_token_id]])
    loss = cross_entropy(logits[:, -1], yes[:, 0])
    loss.backward()
    assert prefix.grad is not None and torch.isfinite(prefix.grad).all() and torch.count_nonzero(prefix.grad)
    for model in (vision, language):
        assert not model.training
        assert all(not parameter.requires_grad and parameter.grad is None for parameter in model.parameters())

    with torch.no_grad():
        extended_hidden, _ = language(prefix, torch.cat((prompt, yes), dim=1))
    torch.testing.assert_close(extended_hidden[:, :-1], hidden, rtol=1e-4, atol=1e-4)
