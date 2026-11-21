import math

import pytest
import torch

from hybrid_drive.neural.decoder import InstructedVisionDecoder, PerceptionOutput


FLOAT_FIELDS = ("validity_logits", "boxes_3d", "boxes_2d", "motion_xy")


def decoder(dtype=torch.float32):
    torch.manual_seed(51)
    return InstructedVisionDecoder(3, 5, 20, 30, hidden_dim=8, num_layers=1,
                                   num_heads=2, motion_steps=3, motion_interval_s=0.25).to(dtype=dtype).eval()


def inputs(dtype=torch.float32):
    torch.manual_seed(52)
    return {
        "visual_tokens": torch.randn(3, 4, 3, dtype=dtype),
        "visual_mask": torch.tensor([[True, True, False, False], [True, False, True, False],
                                      [False, True, True, False]]),
        "last_hidden_states": torch.randn(3, 6, 5, dtype=dtype),
        "consumed_token_ids": torch.tensor([[20, 8, 20, 30, 20, 9], [8, 8, 20, 30, 9, 9],
                                            [30, 8, 20, 30, 20, 9]]),
        "answer_mask": torch.tensor([[False, True, True, True, False, False],
                                      [True, True, False, False, True, True],
                                      [True, True, True, True, False, False]]),
    }


def assert_same(first, second):
    for name in PerceptionOutput.__dataclass_fields__:
        torch.testing.assert_close(getattr(first, name), getattr(second, name))


def test_events_gather_exact_consumed_h_t_in_batch_token_order_without_deduplication():
    model = decoder()
    values = inputs()
    captured = []
    hook = model.hidden_projection.register_forward_pre_hook(lambda module, args: captured.append(args[0].clone()))
    try:
        result = model(**values)
    finally:
        hook.remove()
    torch.testing.assert_close(result.batch_indices, torch.tensor([0, 0, 2, 2, 2]))
    torch.testing.assert_close(result.token_positions, torch.tensor([2, 3, 0, 2, 3]))
    torch.testing.assert_close(result.event_kinds, torch.tensor([0, 1, 1, 0, 1]))
    expected = torch.stack([values["last_hidden_states"][0, 2], values["last_hidden_states"][0, 3],
                            values["last_hidden_states"][2, 0], values["last_hidden_states"][2, 2],
                            values["last_hidden_states"][2, 3]])
    torch.testing.assert_close(captured[0], expected, rtol=0, atol=0)
    assert result.validity_logits.shape == (5,)
    assert result.boxes_3d.shape == (5, 7)
    assert result.boxes_2d.shape == (5, 4)
    assert result.motion_xy.shape == (5, 3, 2)
    assert model.motion_interval_s == 0.25
    for name in FLOAT_FIELDS:
        assert torch.isfinite(getattr(result, name)).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_decoder_and_all_heads_have_finite_gradients_only_at_selected_hidden_states(dtype):
    model = decoder(dtype)
    values = inputs(dtype)
    selected = torch.zeros(3, 6, dtype=torch.bool)
    selected[0, 2:4] = True
    selected[2, [0, 2, 3]] = True
    values["last_hidden_states"][~selected] = float("nan")
    values["last_hidden_states"].requires_grad_()
    values["visual_tokens"][~values["visual_mask"]] = float("nan")
    values["visual_tokens"].requires_grad_()
    result = model(**values)
    sum(getattr(result, name).square().mean() for name in FLOAT_FIELDS).backward()
    hidden_gradient = values["last_hidden_states"].grad
    assert torch.isfinite(hidden_gradient).all()
    assert hidden_gradient[selected].abs().sum() > 0
    assert torch.count_nonzero(hidden_gradient[~selected]) == 0
    visual_gradient = values["visual_tokens"].grad
    assert torch.isfinite(visual_gradient).all()
    assert visual_gradient[[0, 2]].abs().sum() > 0
    assert torch.count_nonzero(visual_gradient[1]) == 0
    assert torch.count_nonzero(visual_gradient[~values["visual_mask"]]) == 0
    for parameter in (model.hidden_projection[0].weight, model.visual_projection.weight,
                      model.decoder.layers[0].self_attn.in_proj_weight, model.validity_head.weight,
                      model.box_3d_head.weight, model.box_2d_head.weight, model.motion_head.weight):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    assert result.motion_xy.dtype == dtype


def test_excluded_prompt_padding_and_visual_memory_permutations_do_not_change_events():
    model = decoder()
    values = inputs()
    expected = model(**values)
    values["visual_tokens"][~values["visual_mask"]] = float("nan")
    selected = torch.zeros(3, 6, dtype=torch.bool)
    selected[expected.batch_indices, expected.token_positions] = True
    values["last_hidden_states"][~selected] = float("nan")
    values["consumed_token_ids"][~values["answer_mask"]] = 30
    order = torch.tensor([2, 0, 3, 1])
    values["visual_tokens"] = values["visual_tokens"][:, order]
    values["visual_mask"] = values["visual_mask"][:, order]
    assert_same(model(**values), expected)


def test_ragged_batch_with_an_eventless_row_matches_individual_decoding():
    model = decoder()
    values = inputs()
    combined = model(**values)
    for row in range(3):
        separate = model(**{name: value[row:row + 1] for name, value in values.items()})
        selected = combined.batch_indices == row
        torch.testing.assert_close(separate.token_positions, combined.token_positions[selected])
        torch.testing.assert_close(separate.event_kinds, combined.event_kinds[selected])
        for name in FLOAT_FIELDS:
            torch.testing.assert_close(getattr(separate, name), getattr(combined, name)[selected])


def test_future_selected_hidden_states_cannot_change_earlier_event_predictions():
    model = decoder()
    values = inputs()
    expected = model(**values)
    values["last_hidden_states"][0, 3] += torch.tensor([31., -23., 17., -11., 7.])
    values["last_hidden_states"][2, 3] += torch.tensor([-19., 29., -13., 23., -5.])
    actual = model(**values)
    earlier = torch.tensor([0, 2, 3])
    for name in FLOAT_FIELDS:
        torch.testing.assert_close(getattr(actual, name)[earlier], getattr(expected, name)[earlier])
    assert not torch.allclose(actual.boxes_3d[[1, 4]], expected.boxes_3d[[1, 4]])


def test_batched_event_predictions_match_single_events_and_reordered_batch_rows():
    model = decoder()
    values = inputs()
    combined = model(**values)
    for event, (row, position) in enumerate(zip(combined.batch_indices.tolist(), combined.token_positions.tolist())):
        single_values = {name: value[row:row + 1].clone() for name, value in values.items()}
        single_values["answer_mask"].fill_(False)
        single_values["answer_mask"][0, position] = True
        single = model(**single_values)
        torch.testing.assert_close(single.token_positions, combined.token_positions[event:event + 1])
        torch.testing.assert_close(single.event_kinds, combined.event_kinds[event:event + 1])
        for name in FLOAT_FIELDS:
            torch.testing.assert_close(getattr(single, name), getattr(combined, name)[event:event + 1])
    order = torch.tensor([2, 0, 1])
    reordered = model(**{name: value[order] for name, value in values.items()})
    for row, original_row in enumerate(order.tolist()):
        actual_events = reordered.batch_indices == row
        original_events = combined.batch_indices == original_row
        torch.testing.assert_close(reordered.token_positions[actual_events], combined.token_positions[original_events])
        torch.testing.assert_close(reordered.event_kinds[actual_events], combined.event_kinds[original_events])
        for name in FLOAT_FIELDS:
            torch.testing.assert_close(getattr(reordered, name)[actual_events], getattr(combined, name)[original_events])


def test_auxiliary_2d_boxes_have_positive_extents_without_clipping_or_changing_raw_3d_layout():
    model = decoder()
    raw_box = torch.tensor([1., 2., 3., -4., -5., -6., 7.])
    with torch.no_grad():
        model.box_2d_head.weight.zero_()
        model.box_2d_head.bias.copy_(torch.tensor([0., 0., 2., 2.]))
        model.box_3d_head.weight.zero_()
        model.box_3d_head.bias.copy_(raw_box)
    result = model(**inputs())
    extent = math.log1p(math.exp(2)) + torch.finfo(torch.float32).eps
    expected = torch.tensor([0.5 - extent / 2, 0.5 - extent / 2, 0.5 + extent / 2, 0.5 + extent / 2])
    torch.testing.assert_close(result.boxes_2d, expected.expand(5, -1))
    assert (result.boxes_2d[:, 2:] > result.boxes_2d[:, :2]).all()
    assert (result.boxes_2d[:, :2] < 0).all()
    assert (result.boxes_2d[:, 2:] > 1).all()
    torch.testing.assert_close(result.boxes_3d, raw_box.expand(5, -1))


@pytest.mark.parametrize("length", [0, 6])
def test_empty_events_keep_exact_shapes_and_backward_safe_zero_gradients(length):
    model = decoder()
    values = inputs()
    values["last_hidden_states"] = torch.full((3, length, 5), float("nan"), requires_grad=True)
    values["consumed_token_ids"] = torch.full((3, length), 20, dtype=torch.long)
    values["answer_mask"] = torch.zeros(3, length, dtype=torch.bool)
    values["visual_tokens"][~values["visual_mask"]] = float("nan")
    values["visual_tokens"].requires_grad_()
    result = model(**values)
    assert result.batch_indices.shape == result.token_positions.shape == result.event_kinds.shape == (0,)
    assert result.batch_indices.dtype == result.token_positions.dtype == result.event_kinds.dtype == torch.long
    assert result.validity_logits.shape == (0,)
    assert result.boxes_3d.shape == (0, 7)
    assert result.boxes_2d.shape == (0, 4)
    assert result.motion_xy.shape == (0, 3, 2)
    loss = sum(getattr(result, name).sum() for name in FLOAT_FIELDS)
    assert loss.item() == 0
    loss.backward()
    for name in ("last_hidden_states", "visual_tokens"):
        assert values[name].grad is not None
        assert torch.isfinite(values[name].grad).all()
        assert torch.count_nonzero(values[name].grad) == 0


@pytest.mark.parametrize(
    "name, replacement, message",
    [("last_hidden_states", torch.ones(3, 6, 4), "shape"),
     ("last_hidden_states", torch.ones(3, 6, 5, dtype=torch.long), "float32 or float64"),
     ("last_hidden_states", torch.ones(3, 6, 5, dtype=torch.float64), "dtype and device"),
     ("last_hidden_states", torch.full((3, 6, 5), float("nan")), "Selected LOC/MOT"),
     ("consumed_token_ids", torch.ones(3, 6), "long tensor"),
     ("consumed_token_ids", torch.ones(3, 5, dtype=torch.long), "shape"),
     ("consumed_token_ids", torch.full((3, 6), -1, dtype=torch.long), "nonnegative"),
     ("answer_mask", torch.ones(3, 6), "boolean"),
     ("answer_mask", torch.ones(3, 5, dtype=torch.bool), "shape"),
     ("visual_mask", torch.zeros(3, 4, dtype=torch.bool), "Every batch row"),
     ("visual_tokens", torch.full((3, 4, 3), float("inf")), "finite")],
)
def test_invalid_inputs_fail_before_decoder_attention(name, replacement, message):
    values = inputs()
    values[name] = replacement
    with pytest.raises(ValueError, match=message):
        decoder()(**values)


@pytest.mark.parametrize("loc_id, mot_id, options", [(20, 20, {}), (-1, 30, {}), (True, 30, {}),
                                                    (20, 3.5, {}), (20, 30, {"motion_steps": 0}),
                                                    (20, 30, {"motion_interval_s": float("nan")}),
                                                    (20, 30, {"motion_interval_s": 0}),
                                                    (20, 30, {"hidden_dim": 7})])
def test_invalid_event_and_attention_configuration_is_rejected(loc_id, mot_id, options):
    with pytest.raises(ValueError):
        InstructedVisionDecoder(3, 5, loc_id, mot_id, **options)
