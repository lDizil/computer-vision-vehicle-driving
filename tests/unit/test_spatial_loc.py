import math

import pytest
import torch

from hybrid_drive.neural.spatial_loc import SpatialLocLoss, SpatialLocOutput, SpatialLocProbe


def _output(*, height=4, width=8, dtype=torch.float64):
    return SpatialLocOutput(
        torch.zeros(1, 4, dtype=dtype),
        torch.zeros(1, 4, height, width, dtype=dtype),
        torch.tensor([0.25, 0.25, 0.75, 0.75], dtype=dtype).repeat(1, 4, 1),
    )


def test_uniform_patch_scores_use_band_local_centers_and_normalized_presence():
    probe = SpatialLocProbe(2, hidden_dim=4).double()
    with torch.no_grad():
        for parameter in probe.parameters():
            parameter.zero_()
        probe.score.bias.fill_(0.75)

    output = probe(torch.ones(1, 2, 3, 8, dtype=torch.float64))
    support = torch.tensor(
        [
            [1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1],
        ],
        dtype=torch.bool,
    )[None, :, None, :].expand(1, 4, 3, 8)
    assert torch.equal(torch.isfinite(output.cell_logits), support)
    assert torch.isneginf(output.cell_logits[~support]).all()
    torch.testing.assert_close(output.presence_logits, torch.full((1, 4), 0.75, dtype=torch.float64))
    torch.testing.assert_close(
        output.boxes_xyxy,
        torch.tensor(
            [
                [
                    [0.0625, 0.25, 0.1875, 0.75],
                    [0.1875, 0.25, 0.5625, 0.75],
                    [0.4375, 0.25, 0.8125, 0.75],
                    [0.8125, 0.25, 0.9375, 0.75],
                ]
            ],
            dtype=torch.float64,
        ),
    )


def test_spatial_objective_matches_hand_calculated_bce_cell_box_and_giou():
    output = _output()
    output.presence_logits[0] = torch.tensor([-1.0, -1.0, 1.0, -1.0], dtype=torch.float64)
    output.cell_logits[0, 2].fill_(100.0)
    output.cell_logits[0, 2, :, 4:6] = 0
    output.cell_logits[0, 2, 2, 5] = math.log(2)
    output.boxes_xyxy[0, 2] = torch.tensor([9, 5, 11, 7], dtype=torch.float64) / 16
    presence = torch.tensor([[False, False, True, False]])
    targets = torch.full_like(output.boxes_xyxy, torch.nan)
    targets[0, 2] = torch.tensor([9, 9, 11, 11], dtype=torch.float64) / 16

    unit = SpatialLocLoss(1.0)(output, presence, targets)
    weighted = SpatialLocLoss(3.0)(output, presence, targets)
    assert (weighted.supervised_presence_count, weighted.positive_count) == (4, 1)
    bce = math.log1p(math.exp(-1))
    assert unit.presence.item() == pytest.approx(bce)
    assert weighted.presence.item() == pytest.approx((3 + 3) * bce / 4)
    # Eight supported cells, target at (column=5, row=2) with double probability mass.
    assert weighted.cell.item() == pytest.approx(math.log(9 / 2))
    assert weighted.box.item() == pytest.approx(1 / 64)
    # Disjoint equal boxes; their combined area is 2/3 of the enclosing rectangle.
    assert weighted.giou.item() == pytest.approx(4 / 3)
    assert weighted.total.item() == pytest.approx(1.5 * bce + math.log(9 / 2) + 5 / 64 + 4 / 3)


@pytest.mark.parametrize(
    ("center", "band", "column"),
    [
        (0.25 - 1 / 512, 0, 2),
        (0.25, 1, 3),
        (0.50 - 1 / 512, 1, 5),
        (0.50, 2, 6),
        (0.75 - 1 / 512, 2, 8),
        (0.75, 3, 9),
    ],
)
def test_target_at_band_boundary_selects_the_half_open_band(center, band, column):
    output = _output(height=1, width=12)
    output.cell_logits[0, band, 0, column] = math.log(4)
    presence = torch.zeros(1, 4, dtype=torch.bool)
    presence[0, band] = True
    targets = torch.full_like(output.boxes_xyxy, torch.nan)
    targets[0, band] = torch.tensor([center - 0.125, 0.25, center + 0.125, 0.75], dtype=torch.float64)

    loss = SpatialLocLoss(1.0)(output, presence, targets)
    assert loss.cell.item() == pytest.approx(math.log(6 / 4))


def test_target_midpoint_rounded_to_image_edge_still_uses_last_cell():
    output = _output(height=2, width=8, dtype=torch.float32)
    output.cell_logits[0, 3, 1, 7] = math.log(3)
    presence = torch.tensor([[False, False, False, True]])
    targets = torch.full_like(output.boxes_xyxy, torch.nan)
    below_one = torch.nextafter(torch.tensor(1.0), torch.tensor(0.0))
    targets[0, 3, :2] = below_one
    targets[0, 3, 2:] = 1

    loss = SpatialLocLoss(1.0)(output, presence, targets)
    assert loss.cell.item() == pytest.approx(math.log(6 / 3))


def test_negative_batch_connects_finite_zero_localization_gradients():
    probe = SpatialLocProbe(2, hidden_dim=4)
    output = probe(torch.ones(2, 2, 2, 8))
    loss = SpatialLocLoss(4.0)(
        output, torch.zeros(2, 4, dtype=torch.bool), torch.full_like(output.boxes_xyxy, torch.nan)
    )
    assert (loss.supervised_presence_count, loss.positive_count) == (8, 0)
    for value in (loss.cell, loss.box, loss.giou):
        assert value.item() == 0 and value.requires_grad
    torch.testing.assert_close(loss.total, loss.presence)

    loss.total.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in probe.parameters())
    assert probe.score.bias.grad.item() > 0
    assert torch.count_nonzero(probe.extent.weight.grad) == 0


@pytest.mark.parametrize(
    ("invalid_box", "error"),
    [
        ([0.125, 0.25, 0.375, 0.75], "assigned band"),
        ([0.05, 0.25, 0.2, float("nan")], "finite positive-area normalized xyxy"),
        ([0.1, 0.25, 0.1, 0.75], "finite positive-area normalized xyxy"),
        ([-0.01, 0.25, 0.2, 0.75], "finite positive-area normalized xyxy"),
    ],
)
def test_loss_rejects_invalid_active_boxes(invalid_box, error):
    output = _output()
    presence = torch.tensor([[True, False, False, False]])
    targets = torch.full_like(output.boxes_xyxy, torch.nan)
    targets[0, 0] = torch.tensor(invalid_box, dtype=targets.dtype)

    with pytest.raises(ValueError, match=error):
        SpatialLocLoss(1.0)(output, presence, targets)


def test_loss_validates_label_types_and_weights():
    output = _output()
    presence = torch.zeros(1, 4, dtype=torch.bool)
    targets = output.boxes_xyxy
    with pytest.raises(ValueError, match="targets_xyxy must match"):
        SpatialLocLoss(1.0)(output, presence, targets.float())
    with pytest.raises(ValueError, match="boolean presence"):
        SpatialLocLoss(1.0)(output, presence.float(), targets)
    with pytest.raises(ValueError, match="positive_weight"):
        SpatialLocLoss(0.0)
    with pytest.raises(ValueError, match="box_weight"):
        SpatialLocLoss(1.0, box_weight=-1.0)
    with pytest.raises(ValueError, match="giou_weight"):
        SpatialLocLoss(1.0, giou_weight=math.nan)


@pytest.mark.parametrize(
    ("features", "error"),
    [
        (torch.ones(1, 2, 3, 6), "grid width divisible by four"),
        (torch.empty(1, 2, 0, 8), "positive grid height"),
        (torch.ones(1, 2, 3, 8, dtype=torch.int64), "float32/float64"),
    ],
)
def test_probe_rejects_invalid_patch_grids(features, error):
    with pytest.raises(ValueError, match=error):
        SpatialLocProbe(2)(features)


def test_positive_cell_and_box_losses_update_the_probe():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        probe = SpatialLocProbe(2, hidden_dim=8)
    features = torch.zeros(2, 2, 3, 8)
    features[0, 0, 1, 1] = 2
    features[1, 1, 2, 4] = 3
    presence = torch.tensor([[True, False, False, False], [False, False, True, False]])
    targets = torch.full((2, 4, 4), torch.nan)
    targets[presence] = torch.tensor([[0.125, 0.375, 0.25, 0.625], [0.5, 0.75, 0.625, 1.0]])
    criterion = SpatialLocLoss(3.0)
    optimizer = torch.optim.SGD(probe.parameters(), lr=0.01)

    output = probe(features)
    assert not torch.equal(output.cell_logits, probe(features.flip(dims=(-1,))).cell_logits)
    before = criterion(output, presence, targets)
    (cell_grad,) = torch.autograd.grad(before.cell, probe.score.weight, retain_graph=True)
    (extent_grad,) = torch.autograd.grad(before.box + before.giou, probe.extent.weight, retain_graph=True)
    for grad in (cell_grad, extent_grad):
        assert torch.isfinite(grad).all() and torch.count_nonzero(grad) > 0
    before.total.backward()
    optimizer.step()

    after = criterion(probe(features), presence, targets)
    assert after.total.item() < before.total.item()


def test_upcast_boxes_keep_canonical_float32_band_assignment():
    output = _output(height=2, width=16, dtype=torch.float64)
    mask = torch.tensor([[False, True, False, False]])
    targets = torch.full_like(output.boxes_xyxy, torch.nan)
    xmin = torch.nextafter(torch.tensor(0.125, dtype=torch.float32), torch.tensor(0.0))
    targets[0, 1] = torch.tensor([xmin, 0.1, 0.375, 0.3], dtype=torch.float64)
    assert (targets[0, 1, 0] + targets[0, 1, 2]) * 0.5 < 0.25
    loss = SpatialLocLoss(1)(output, mask, targets)
    assert loss.positive_count == 1
    assert torch.isfinite(loss.total)
