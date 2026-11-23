import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import generalized_box_iou_loss

from hybrid_drive.neural.spatial_box_refiner import SpatialBoxRefiner, spatial_box_refinement_loss


def _boxes(dtype=torch.float32):
    return torch.tensor(
        [[
            [0.00, 0.01, 0.19, 0.18],
            [0.12, 0.16, 0.48, 0.51],
            [0.47, 0.37, 0.87, 0.61],
            [0.78, 0.75, 1.00, 0.99],
        ]],
        dtype=dtype,
    )


def _features(dtype=torch.float32, *, batch=1):
    return torch.randn(batch, 4, 8, 16, dtype=dtype, generator=torch.Generator().manual_seed(123))


def _centers(boxes):
    boxes = boxes.to(torch.float64)
    return (boxes[..., :2] + boxes[..., 2:]) * 0.5


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_zero_residual_is_identity_at_boundaries_and_edges(dtype):
    device_zero = torch.tensor(0.0, dtype=dtype)
    quarter_edge = torch.nextafter(torch.tensor(0.5, dtype=torch.float32), torch.tensor(0.0)).item()
    image_edge = torch.nextafter(torch.tensor(1.0, dtype=dtype), device_zero).item()
    baseline = torch.tensor(
        [[
            [0.0, 0.0, quarter_edge, 0.25],
            [0.0, 0.75, 0.50, 1.0],
            [0.125, 0.25, 0.875, 0.75],
            [image_edge, image_edge, 1.0, 1.0],
        ]],
        dtype=dtype,
    )
    with torch.no_grad():
        output = SpatialBoxRefiner(4).to(dtype)(_features(dtype), baseline)
    torch.testing.assert_close(output, baseline, rtol=0, atol=torch.finfo(dtype).eps)


@pytest.mark.parametrize("sign", [-1, 1])
def test_extreme_correction_stays_inside_image_and_band(sign):
    model, baseline = SpatialBoxRefiner(4), _boxes()
    with torch.no_grad():
        model.head[-1].bias.copy_(torch.tensor((80, 80, 80, -80)) * sign)
        refined = model(_features(), baseline)
    assert ((refined >= 0) & (refined <= 1)).all()
    centers = (refined[..., 0] + refined[..., 2]) * 0.5
    assert torch.equal((centers * 4).floor().long(), torch.arange(4).unsqueeze(0))
    assert (torch.abs(_centers(refined) - _centers(baseline)) <= torch.tensor([1 / 16, 1 / 8]) + 1e-7).all()
    ratios = (refined[..., 2:] - refined[..., :2]) / (baseline[..., 2:] - baseline[..., :2])
    assert ((ratios >= 0.5 - 1e-6) & (ratios <= 2 + 1e-6)).all()


def test_batch_does_not_pool_a_box_from_another_frame():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(31)
        model = SpatialBoxRefiner(4)
        nn.init.normal_(model.head[-1].weight, std=0.20)
    features, boxes = _features(batch=2), _boxes().repeat(2, 1, 1)
    with torch.no_grad():
        both = model(features, boxes)
        singles = torch.cat([model(f, b) for f, b in zip(features.split(1), boxes.split(1))])
    torch.testing.assert_close(both, singles)
    assert not torch.allclose(both[0], both[1])


def test_positive_loss_and_gradients_refine_boxes_without_using_inactive_targets():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        model = SpatialBoxRefiner(4)
    features, baseline = _features(), _boxes()
    presence = torch.tensor([[False, True, True, False]])
    targets = torch.full_like(baseline, torch.nan)
    targets[presence] = baseline[presence] + torch.tensor([0.021, 0.028, 0.021, 0.028])
    optimizer = torch.optim.SGD(model.parameters(), lr=0.02)
    refined = model(features, baseline)
    initial = spatial_box_refinement_loss(refined, targets, presence)
    expected = generalized_box_iou_loss(refined[presence], targets[presence], reduction="mean")
    expected += 5 * F.smooth_l1_loss(refined[presence], targets[presence], beta=1 / 32)
    torch.testing.assert_close(initial, expected)
    alternative = targets.clone()
    alternative[~presence] = torch.inf
    torch.testing.assert_close(initial, spatial_box_refinement_loss(refined, alternative, presence))
    for _ in range(5):
        optimizer.zero_grad()
        loss = spatial_box_refinement_loss(model(features, baseline), targets, presence)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        assert model.head[-1].bias.grad.abs().sum() > 0
        optimizer.step()
    remaining = spatial_box_refinement_loss(model(features, baseline), targets, presence)
    assert remaining.item() < initial.item()


def test_empty_positives_skip_regression_without_reading_nan_targets():
    refined = SpatialBoxRefiner(4)(_features(), _boxes())
    empty = torch.zeros((1, 4), dtype=torch.bool)
    assert spatial_box_refinement_loss(refined, torch.full_like(refined, torch.nan), empty) is None


@pytest.mark.parametrize(
    ("features", "boxes", "match"),
    [
        (_features()[:, :3], _boxes(), "patch grid"),
        (_features() * torch.nan, _boxes(), "features must be finite"),
        (_features(), _boxes()[:, :3], "Expected baseline_boxes"),
        (_features(), _boxes().double(), "must match"),
        (_features(), _boxes() * torch.nan, "positive-area"),
        (_features(), _boxes().flip(-1), "positive-area"),
        (_features(), _boxes().roll(1, dims=1), "assigned band"),
    ],
)
def test_refiner_rejects_invalid_features_or_box_assignment(features, boxes, match):
    with pytest.raises(ValueError, match=match):
        SpatialBoxRefiner(4)(features, boxes)


def test_loss_rejects_invalid_active_targets_instead_of_dropping_them():
    boxes = _boxes()
    positive = torch.tensor([[True, False, False, False]])
    wrong_band = boxes.roll(1, dims=1)
    with pytest.raises(ValueError, match="assigned band"):
        spatial_box_refinement_loss(boxes, wrong_band, positive)
    wrong_band[positive] = torch.nan
    with pytest.raises(ValueError, match="positive-area"):
        spatial_box_refinement_loss(boxes, wrong_band, positive)
