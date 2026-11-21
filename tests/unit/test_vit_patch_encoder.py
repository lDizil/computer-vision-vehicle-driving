import pytest
import torch
from torch import nn
from torchvision.models.vision_transformer import VisionTransformer

from hybrid_drive.neural.prior_tokenizer import PriorTokenizer
from hybrid_drive.neural.vision import ViTPatchEncoder


def tiny_encoder():
    return VisionTransformer(image_size=28, patch_size=7, num_layers=1, num_heads=2,
                             hidden_dim=16, mlp_dim=32, dropout=0, attention_dropout=0)


def images(*, batch_size=2, views=3):
    count = batch_size * views * 3 * 28 * 28
    return torch.linspace(-4, 2, count).reshape(batch_size, views, 3, 28, 28)


def official_patch_maps(encoder, values):
    captured = []
    handle = encoder.encoder.ln.register_forward_hook(
        lambda _module, _inputs, output: captured.append(output))
    try:
        encoder(values.flatten(0, 1))
    finally:
        handle.remove()
    return captured[0][:, 1:, :].transpose(1, 2).reshape(values.shape[0], values.shape[1], 16, 4, 4)


def test_patch_maps_match_official_encoder_layer_norm_without_preprocessing_or_cls_pixels():
    encoder = tiny_encoder()
    wrapper = ViTPatchEncoder(encoder)
    values = images()
    expected = official_patch_maps(encoder, values)
    actual = wrapper(values)
    assert actual.shape == (2, 3, 16, 4, 4)
    torch.testing.assert_close(actual, expected)


def test_batch_and_view_order_matches_each_independently_encoded_image():
    encoder = tiny_encoder()
    wrapper = ViTPatchEncoder(encoder)
    values = images()
    actual = wrapper(values)
    for batch in range(2):
        for view in range(3):
            expected = official_patch_maps(encoder, values[batch:batch + 1, view:view + 1])
            torch.testing.assert_close(actual[batch, view], expected[0, 0], atol=1e-6, rtol=1e-5)


def test_real_encoder_to_768_prior_tokens_preserves_end_to_end_gradients():
    encoder = tiny_encoder()
    wrapper = ViTPatchEncoder(encoder)
    values = images(batch_size=1, views=2).requires_grad_()
    tokenizer = PriorTokenizer(16)
    geometry = torch.ones((1, 2, 7), requires_grad=True)
    result = tokenizer(
        wrapper(values), torch.tensor([[[0.1, 0.2, 0.8, 0.9], [0.2, 0.1, 0.9, 0.8]]]),
        torch.tensor([[1, 0]]), geometry, torch.ones((1, 2), dtype=torch.bool))
    assert result.tokens.shape == (1, 2, 768)
    assert torch.isfinite(result.tokens).all()
    result.tokens.square().mean().backward()
    for tensor in (values, geometry, encoder.conv_proj.weight,
                   encoder.encoder.layers.encoder_layer_0.self_attention.in_proj_weight,
                   tokenizer.spatial_pool.weight, tokenizer.geometry_embedding[0].weight,
                   tokenizer.fusion[0].weight, tokenizer.fusion[2].weight):
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all()
        assert torch.count_nonzero(tensor.grad) > 0


@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("trainable", [False, True])
def test_extractor_reuses_parameters_without_overriding_caller_freeze_or_eval(training, trainable):
    encoder = tiny_encoder().train(training).requires_grad_(trainable)
    original_parameters = tuple(encoder.parameters())
    wrapper = ViTPatchEncoder(encoder)
    assert encoder.training is training
    assert wrapper.extractor.training is training
    assert all(parameter.requires_grad is trainable for parameter in wrapper.parameters())
    assert all(any(parameter is original for original in original_parameters) for parameter in wrapper.parameters())
    values = images(batch_size=1, views=1).requires_grad_()
    output = wrapper(values)
    assert output.requires_grad
    output[:, :, 0].square().sum().backward()
    assert values.grad is not None and torch.isfinite(values.grad).all()
    assert torch.count_nonzero(values.grad) > 0
    wrapper.train(not training)
    assert wrapper.extractor.training is not training


@pytest.mark.parametrize("encoder", [None, nn.Identity(), nn.Linear(3, 3)])
def test_constructor_requires_the_official_vision_transformer_type(encoder):
    with pytest.raises(ValueError, match="VisionTransformer"):
        ViTPatchEncoder(encoder)


@pytest.mark.parametrize("field", ["image_size", "patch_size", "hidden_dim"])
@pytest.mark.parametrize("value", [0, -1, True, 7.0, None])
def test_encoder_grid_configuration_must_be_positive_integers(field, value):
    encoder = tiny_encoder()
    setattr(encoder, field, value)
    with pytest.raises(ValueError, match=field):
        ViTPatchEncoder(encoder)


def test_grid_configuration_requires_divisibility_not_a_sqrt_guess():
    encoder = tiny_encoder()
    encoder.patch_size = 6
    with pytest.raises(ValueError, match="divisible"):
        ViTPatchEncoder(encoder)


@pytest.mark.parametrize("shape", [
    (2, 3, 28, 28), (2, 3, 1, 28, 28), (2, 3, 3, 27, 28),
    (2, 3, 3, 28, 29), (0, 3, 3, 28, 28), (2, 0, 3, 28, 28),
])
def test_input_shape_is_not_resized_cropped_or_reinterpreted(shape):
    with pytest.raises(ValueError, match="images.*shape"):
        ViTPatchEncoder(tiny_encoder())(torch.zeros(shape))


def test_input_must_be_a_tensor():
    with pytest.raises(ValueError, match="images.*shape"):
        ViTPatchEncoder(tiny_encoder())(None)


@pytest.mark.parametrize("dtype", [torch.int64, torch.bool, torch.complex64])
def test_input_dtype_must_be_floating(dtype):
    with pytest.raises(ValueError, match="floating dtype"):
        ViTPatchEncoder(tiny_encoder())(images().to(dtype=dtype))


def test_floating_input_dtype_must_match_encoder_parameters():
    with pytest.raises(ValueError, match="dtype and device"):
        ViTPatchEncoder(tiny_encoder())(images().double())


@pytest.mark.parametrize("encoder_device,image_device", [("cpu", "meta"), ("meta", "cpu")])
def test_input_device_must_match_encoder_parameters(encoder_device, image_device):
    wrapper = ViTPatchEncoder(tiny_encoder().to(encoder_device))
    with pytest.raises(ValueError, match="dtype and device"):
        wrapper(images().to(image_device))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_images_are_rejected_before_encoding(value):
    values = images()
    values[0, 0, 0, 0, 0] = value
    with pytest.raises(ValueError, match="finite"):
        ViTPatchEncoder(tiny_encoder())(values)


@pytest.mark.parametrize("dimension", [1, 2])
def test_extracted_token_count_and_hidden_dimension_are_verified(dimension):
    wrapper = ViTPatchEncoder(tiny_encoder())

    def truncate_tokens(_module, _inputs, output):
        tokens = output["tokens"]
        return {"tokens": tokens[:, :-1] if dimension == 1 else tokens[:, :, :-1]}

    handle = wrapper.extractor.register_forward_hook(truncate_tokens)
    try:
        with pytest.raises(ValueError, match="including one CLS token"):
            wrapper(images())
    finally:
        handle.remove()


def test_float64_encoder_preserves_dtype_and_image_gradient():
    wrapper = ViTPatchEncoder(tiny_encoder().double())
    values = images(batch_size=1, views=1).double().requires_grad_()
    result = wrapper(values)
    assert result.dtype == torch.float64
    result[:, :, 0].square().sum().backward()
    assert values.grad is not None and torch.isfinite(values.grad).all()
    assert torch.count_nonzero(values.grad) > 0
