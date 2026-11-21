from dataclasses import FrozenInstanceError

import pytest
import torch

from hybrid_drive.neural.prior_tokenizer import PriorTokenizer


def inputs(*, batch_size=1, views=2, count=2, channels=2, geometry_dim=7):
    return {
        "feature_maps": torch.arange(batch_size * views * channels * 10 * 20, dtype=torch.float32)
        .reshape(batch_size, views, channels, 10, 20) / 100,
        "boxes_xyxy": torch.tensor([0.2, 0.3, 0.7, 0.8]).expand(batch_size, count, 4).clone(),
        "image_indices": torch.zeros((batch_size, count), dtype=torch.long),
        "geometry": torch.ones((batch_size, count, geometry_dim)),
        "valid_mask": torch.ones((batch_size, count), dtype=torch.bool),
    }


def tiny_tokenizer(*, image_channels=2, geometry_dim=7):
    return PriorTokenizer(image_channels, geometry_dim=geometry_dim, token_dim=6, hidden_dim=8)


def capture_roi_features(tokenizer):
    captured = []
    handle = tokenizer.spatial_pool.register_forward_pre_hook(
        lambda _module, args: captured.append(args[0].detach().clone()))
    return captured, handle


def test_anisotropic_normalized_roi_uses_independent_width_height_and_aligned_bin_centres():
    values = inputs(views=1, count=1)
    x = torch.arange(20, dtype=torch.float32).expand(10, 20)
    y = torch.arange(10, dtype=torch.float32)[:, None].expand(10, 20)
    values["feature_maps"] = torch.stack((x, y))[None, None]
    tokenizer = tiny_tokenizer()
    captured, handle = capture_roi_features(tokenizer)
    tokenizer(**values)
    handle.remove()
    # ROI [4,3,14,8] is shifted by -0.5 in aligned ROIAlign. A linear ramp's
    # two samples per bin average exactly to that bin's centre.
    expected_x = 3.5 + (torch.arange(7) + 0.5) * 10 / 7
    expected_y = 2.5 + (torch.arange(7) + 0.5) * 5 / 7
    expected = torch.stack((expected_x.expand(7, 7), expected_y[:, None].expand(7, 7)))
    torch.testing.assert_close(captured[0].reshape(2, 7, 7), expected)


def test_batch_and_view_indices_select_the_requested_source_and_preserve_prior_order():
    values = inputs(batch_size=2, views=3, count=2, channels=1)
    values["feature_maps"] = torch.tensor([[11., 12., 13.], [21., 22., 23.]])[:, :, None, None, None]
    values["feature_maps"] = values["feature_maps"].expand(2, 3, 1, 10, 20)
    values["image_indices"] = torch.tensor([[2, 0], [1, 2]])
    tokenizer = tiny_tokenizer(image_channels=1)
    captured, handle = capture_roi_features(tokenizer)
    result = tokenizer(**values)
    handle.remove()
    expected = torch.tensor([13., 11., 22., 23.])[:, None, None].expand(4, 1, 49)
    torch.testing.assert_close(captured[0], expected)
    assert result.tokens.shape == (2, 2, 6)


def test_padding_and_unavailable_images_do_not_influence_valid_tokens():
    tokenizer = tiny_tokenizer()
    values = inputs(count=3)
    values["valid_mask"] = torch.tensor([[True, False, True]])
    expected = tokenizer(**values).tokens.detach().clone()
    values["boxes_xyxy"][0, 1] = float("nan")
    values["geometry"][0, 1] = float("nan")
    values["image_indices"][0, 1] = -900
    values["feature_maps"][:, 1] = float("nan")
    values["image_valid_mask"] = torch.tensor([[True, False]])
    result = tokenizer(**values)
    torch.testing.assert_close(result.tokens, expected)
    assert torch.count_nonzero(result.tokens[~result.valid_mask]) == 0
    assert result.valid_mask is values["valid_mask"]
    with pytest.raises(FrozenInstanceError):
        result.tokens = torch.empty(0)


def test_active_prior_cannot_reference_an_unavailable_image():
    values = inputs()
    values["image_valid_mask"] = torch.tensor([[False, True]])
    with pytest.raises(ValueError, match="unavailable images"):
        tiny_tokenizer()(**values)


def test_default_output_is_768_and_custom_geometry_names_are_owned_configuration():
    values = inputs(count=1)
    default = PriorTokenizer(2)
    assert default(**values).tokens.shape == (1, 1, 768)
    assert default.geometry_feature_names == ("x", "y", "z", "width", "length", "height", "yaw")
    names = ["speed", "turn_rate"]
    custom = PriorTokenizer(2, geometry_dim=2, token_dim=5, hidden_dim=4, geometry_feature_names=names)
    names[0] = "changed"
    assert custom.geometry_feature_names == ("speed", "turn_rate")
    assert custom(**inputs(count=1, geometry_dim=2)).tokens.shape == (1, 1, 5)
    with pytest.raises(ValueError, match="geometry must have shape"):
        custom(**values)


@pytest.mark.parametrize("count", [0, 3])
def test_empty_and_all_padded_priors_are_zero_finite_and_support_backward(count):
    values = inputs(count=count)
    values["valid_mask"].fill_(False)
    values["boxes_xyxy"].fill_(float("nan"))
    values["geometry"].fill_(float("nan"))
    values["image_indices"].fill_(-100)
    values["feature_maps"].fill_(float("nan"))
    values["feature_maps"].requires_grad_()
    values["geometry"].requires_grad_()
    values["image_valid_mask"] = torch.zeros((1, 2), dtype=torch.bool)
    tokenizer = tiny_tokenizer()
    result = tokenizer(**values)
    assert result.tokens.shape == (1, count, 6)
    assert torch.isfinite(result.tokens).all() and torch.count_nonzero(result.tokens) == 0
    result.tokens.sum().backward()
    for tensor in (values["feature_maps"], values["geometry"], *tokenizer.parameters()):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all() and torch.count_nonzero(tensor.grad) == 0


def test_backward_reaches_features_geometry_pool_embedding_and_fusion_without_padding_nan():
    values = inputs(batch_size=2, count=3)
    values["valid_mask"] = torch.tensor([[True, False, True], [False, True, False]])
    values["boxes_xyxy"][~values["valid_mask"]] = float("nan")
    values["geometry"][~values["valid_mask"]] = float("nan")
    values["image_indices"][~values["valid_mask"]] = 100
    values["feature_maps"][:, 1] = float("nan")
    values["feature_maps"].requires_grad_()
    values["geometry"].requires_grad_()
    values["image_valid_mask"] = torch.tensor([[True, False], [True, False]])
    tokenizer = tiny_tokenizer()
    with torch.no_grad():
        for parameter in tokenizer.parameters():
            parameter.fill_(0.1)
    tokenizer(**values).tokens.square().sum().backward()
    for tensor in (values["feature_maps"], values["geometry"], *tokenizer.parameters()):
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all()
        assert torch.count_nonzero(tensor.grad) > 0
    assert torch.count_nonzero(values["feature_maps"].grad[:, 1]) == 0
    assert torch.count_nonzero(values["geometry"].grad[~values["valid_mask"]]) == 0


@pytest.mark.parametrize("field", ["image_channels", "geometry_dim", "token_dim", "hidden_dim"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5, None])
def test_dimensions_must_be_positive_integers(field, value):
    kwargs = {"image_channels": 2, field: value}
    with pytest.raises(ValueError, match=field):
        PriorTokenizer(**kwargs)


@pytest.mark.parametrize("names", ["xy", ["x"], ["x", "x"], ["x", ""], ["x", 7]])
def test_custom_geometry_names_must_match_the_explicit_geometry_dimension(names):
    with pytest.raises(ValueError, match="geometry_feature_names"):
        PriorTokenizer(2, geometry_dim=2, geometry_feature_names=names)


@pytest.mark.parametrize("box", [
    [float("nan"), 0.2, 0.5, 0.8], [0.2, 0.3, float("inf"), 0.8],
    [-0.1, 0.2, 0.5, 0.8], [0.2, 0.3, 1.1, 0.8],
    [0.7, 0.2, 0.5, 0.8], [0.2, 0.8, 0.5, 0.8],
])
def test_active_boxes_are_not_clamped_reordered_or_repaired(box):
    values = inputs()
    values["boxes_xyxy"][0, 0] = torch.tensor(box)
    with pytest.raises(ValueError, match="Active"):
        tiny_tokenizer()(**values)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_active_geometry_must_be_finite(value):
    values = inputs()
    values["geometry"][0, 0, 0] = value
    with pytest.raises(ValueError, match="finite"):
        tiny_tokenizer()(**values)


@pytest.mark.parametrize("value", [-1, 2])
def test_active_image_indices_must_select_an_existing_view(value):
    values = inputs()
    values["image_indices"][0, 0] = value
    with pytest.raises(ValueError, match="image_indices"):
        tiny_tokenizer()(**values)


def test_nonfinite_selected_roi_features_are_rejected():
    values = inputs()
    values["feature_maps"][:, 0] = float("nan")
    with pytest.raises(ValueError, match="ROI features must be finite"):
        tiny_tokenizer()(**values)


@pytest.mark.parametrize("field,value", [
    ("feature_maps", torch.zeros(1, 2, 10, 20)),
    ("feature_maps", torch.zeros(1, 2, 3, 10, 20)),
    ("feature_maps", torch.zeros(1, 2, 2, 0, 20)),
    ("boxes_xyxy", torch.zeros(1, 2, 5)),
    ("image_indices", torch.zeros(1, 1, dtype=torch.long)),
    ("geometry", torch.zeros(1, 2, 4)),
    ("valid_mask", torch.ones(1, 2, 1, dtype=torch.bool)),
    ("image_valid_mask", torch.ones(1, 3, dtype=torch.bool)),
    ("geometry", None),
])
def test_shapes_are_validated_before_indexing(field, value):
    values = inputs()
    values[field] = value
    with pytest.raises(ValueError, match=field):
        tiny_tokenizer()(**values)


@pytest.mark.parametrize("field,dtype", [
    ("feature_maps", torch.int64), ("feature_maps", torch.float16),
    ("boxes_xyxy", torch.float64), ("geometry", torch.int64),
    ("image_indices", torch.int32), ("image_indices", torch.float32),
    ("valid_mask", torch.float32), ("image_valid_mask", torch.int64),
])
def test_dtypes_are_not_silently_coerced(field, dtype):
    values = inputs()
    values["image_valid_mask"] = torch.ones((1, 2), dtype=torch.bool)
    values[field] = values[field].to(dtype=dtype)
    with pytest.raises(ValueError, match=field):
        tiny_tokenizer()(**values)


@pytest.mark.parametrize("field", ["boxes_xyxy", "image_indices", "geometry", "valid_mask", "image_valid_mask"])
def test_all_input_tensors_must_share_the_feature_device(field):
    values = inputs()
    values["image_valid_mask"] = torch.ones((1, 2), dtype=torch.bool)
    values[field] = values[field].to("meta")
    with pytest.raises(ValueError, match=field):
        tiny_tokenizer()(**values)


def test_module_and_feature_device_and_dtype_must_match():
    values = inputs()
    with pytest.raises(ValueError, match="module parameter"):
        tiny_tokenizer().to("meta")(**values)
    with pytest.raises(ValueError, match="module parameter"):
        tiny_tokenizer().double()(**values)


def test_float64_module_and_inputs_preserve_dtype_and_autograd():
    values = inputs(count=1)
    for name in ("feature_maps", "boxes_xyxy", "geometry"):
        values[name] = values[name].double()
    values["feature_maps"].requires_grad_()
    result = tiny_tokenizer().double()(**values)
    assert result.tokens.dtype == torch.float64
    result.tokens.sum().backward()
    assert torch.isfinite(values["feature_maps"].grad).all()
