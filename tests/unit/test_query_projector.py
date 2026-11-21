import pytest
import torch

from hybrid_drive.neural.projector import QFormerProjector


def projector(dtype=torch.float32):
    torch.manual_seed(41)
    return QFormerProjector(3, 5, prior_dim=4, hidden_dim=8, num_queries=3,
                            num_layers=1, num_heads=2).to(dtype=dtype).eval()


def inputs(dtype=torch.float32):
    torch.manual_seed(42)
    return {
        "visual_tokens": torch.randn(2, 4, 3, dtype=dtype),
        "visual_mask": torch.tensor([[True, True, False, False], [True, False, True, False]]),
        "prior_tokens": torch.randn(2, 3, 4, dtype=dtype),
        "prior_mask": torch.tensor([[True, False, True], [False, True, False]]),
    }


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_real_query_attention_backpropagates_through_visual_priors_and_learned_queries(dtype):
    model = projector(dtype)
    values = inputs(dtype)
    for name, mask in (("visual_tokens", "visual_mask"), ("prior_tokens", "prior_mask")):
        values[name][~values[mask]] = float("nan")
        values[name].requires_grad_()
    result = model(**values)
    assert result.shape == (2, 3, 5)
    assert result.dtype == dtype
    assert result.device.type == "cpu"
    assert torch.isfinite(result).all()
    weights = torch.arange(1, result.numel() + 1, dtype=dtype).reshape_as(result)
    (result * weights).mean().backward()
    for name, mask in (("visual_tokens", "visual_mask"), ("prior_tokens", "prior_mask")):
        gradient = values[name].grad
        assert torch.isfinite(gradient).all()
        assert gradient[values[mask]].abs().sum() > 0
        assert torch.count_nonzero(gradient[~values[mask]]) == 0
    for parameter in (model.query_tokens, model.type_embeddings, model.visual_projection.weight,
                      model.prior_projection.weight, model.decoder.layers[0].self_attn.in_proj_weight):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_masked_values_and_memory_permutations_do_not_change_projection():
    model = projector()
    values = inputs()
    expected = model(**values)
    values["visual_tokens"][~values["visual_mask"]] = float("nan")
    values["prior_tokens"][~values["prior_mask"]] = float("inf")
    torch.testing.assert_close(model(**values), expected)
    visual_order = torch.tensor([3, 1, 0, 2])
    prior_order = torch.tensor([2, 0, 1])
    batch_order = torch.tensor([1, 0])
    for name in ("visual_tokens", "visual_mask"):
        values[name] = values[name][batch_order][:, visual_order]
    for name in ("prior_tokens", "prior_mask"):
        values[name] = values[name][batch_order][:, prior_order]
    torch.testing.assert_close(model(**values), expected[batch_order])


def test_empty_or_fully_masked_priors_are_equivalent_to_no_priors():
    model = projector()
    values = inputs()
    expected = model(values["visual_tokens"], values["visual_mask"])
    empty = model(values["visual_tokens"], values["visual_mask"], torch.empty(2, 0, 4),
                  torch.empty(2, 0, dtype=torch.bool))
    torch.testing.assert_close(empty, expected)
    values["prior_mask"].fill_(False)
    values["prior_tokens"].fill_(float("nan"))
    torch.testing.assert_close(model(**values), expected)


def test_visual_and_prior_type_embeddings_are_distinct_trainable_inputs():
    model = QFormerProjector(3, 5, prior_dim=3, hidden_dim=8, num_queries=2,
                             num_layers=1, num_heads=2).eval()
    with torch.no_grad():
        model.prior_projection.load_state_dict(model.visual_projection.state_dict())
    visual = torch.tensor([[[1., 2., 4.]]])
    prior = torch.tensor([[[7., 3., 5.]]])
    mask = torch.ones(1, 1, dtype=torch.bool)
    first = model(visual, mask, prior, mask)
    swapped = model(prior, mask, visual, mask)
    assert not torch.allclose(first, swapped, atol=1e-7, rtol=1e-7)
    with torch.no_grad():
        model.type_embeddings[1].copy_(model.type_embeddings[0])
    torch.testing.assert_close(model(visual, mask, prior, mask), model(prior, mask, visual, mask))


@pytest.mark.parametrize(
    "name, replacement, message",
    [("visual_tokens", torch.ones(2, 4, 2), "shape"),
     ("visual_tokens", torch.ones(2, 4, 3, dtype=torch.long), "float32 or float64"),
     ("visual_tokens", torch.ones(2, 4, 3, dtype=torch.float64), "dtype and device"),
     ("visual_tokens", torch.ones(2, 4, 3, dtype=torch.float16), "float32 or float64"),
     ("visual_tokens", torch.full((2, 4, 3), float("nan")), "finite"),
     ("visual_tokens", torch.empty(0, 4, 3), "shape"),
     ("visual_tokens", torch.empty(2, 4, 3, device="meta"), "dtype and device"),
     ("visual_mask", torch.ones(2, 4), "boolean"),
     ("visual_mask", torch.ones(2, 3, dtype=torch.bool), "shape"),
     ("visual_mask", torch.tensor([[True, False, False, False], [False, False, False, False]]),
      "Every batch row"),
     ("prior_tokens", torch.ones(1, 3, 4), "batch size"),
     ("prior_tokens", torch.full((2, 3, 4), float("inf")), "finite"),
     ("prior_mask", torch.ones(2, 3), "boolean"),
     ("prior_tokens", None, "supplied together"),
     ("prior_mask", None, "supplied together")],
)
def test_invalid_inputs_fail_before_attention(name, replacement, message):
    values = inputs()
    values[name] = replacement
    with pytest.raises(ValueError, match=message):
        projector()(**values)


def test_empty_visual_memory_is_rejected_even_with_valid_priors():
    values = inputs()
    values["visual_tokens"] = torch.empty(2, 0, 3)
    values["visual_mask"] = torch.empty(2, 0, dtype=torch.bool)
    with pytest.raises(ValueError, match="Every batch row"):
        projector()(**values)


@pytest.mark.parametrize("options", [{"hidden_dim": 7}, {"num_heads": 0}, {"num_queries": True},
                                      {"num_layers": 0}, {"prior_dim": -1}, {"dropout": float("nan")}])
def test_invalid_attention_configuration_is_rejected(options):
    with pytest.raises(ValueError):
        QFormerProjector(3, 5, **options)
