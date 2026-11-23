import json

import pytest
import torch

from hybrid_drive.neural.spatial_zoom_refiner import ZoomBoxRefiner, fit_zoom_refiner, zoom_head_protocol


def _inputs(batch=1):
    boxes = torch.tensor(
        [[[0.00, 0.01, 0.19, 0.18], [0.30, 0.35, 0.43, 0.50],
          [0.47, 0.37, 0.87, 0.61], [0.78, 0.75, 1.00, 0.99]]],
    ).repeat(batch, 1, 1)
    windows = boxes + torch.tensor((-0.1, -0.1, 0.1, 0.1))
    windows = windows.clamp(0, 1)
    features = torch.randn(batch, 4, 4, 4, 4, generator=torch.Generator().manual_seed(72))
    mask = torch.tensor([[True, True, False, True]]).repeat(batch, 1)
    return boxes, windows, features, mask


@pytest.mark.parametrize("zero_features", [False, True])
def test_initial_head_is_exact_identity_on_edges(zero_features):
    boxes, windows, features, active = _inputs()
    if zero_features:
        features.zero_()
    model = ZoomBoxRefiner(4)
    with torch.no_grad():
        actual = model(boxes, windows, features, active)
    assert torch.equal(actual, boxes)
    assert actual.data_ptr() != boxes.data_ptr()


def test_empty_mask_preserves_boxes_without_using_empty_crop_windows():
    boxes, windows, feats, active = _inputs()
    with torch.no_grad():
        output = ZoomBoxRefiner(4)(boxes, torch.zeros_like(windows), feats, torch.zeros_like(active))
    assert torch.equal(boxes, output) and output.data_ptr() != boxes.data_ptr()


@pytest.mark.parametrize("sign", [-1, 1])
def test_extreme_shifts_obey_bounds_and_keep_canonical_band(sign):
    boxes, windows, features, active = _inputs()
    model = ZoomBoxRefiner(4)
    with torch.no_grad():
        model.head[-1].bias.copy_(torch.tensor((80, 80, 80, -80)) * sign)
        corrected = model(boxes, windows, features, active)
    assert ((corrected >= 0) & (corrected <= 1)).all()
    assert (corrected[..., 2:] > corrected[..., :2]).all()
    centers = (corrected[..., :2] + corrected[..., 2:]) / 2
    assert torch.equal((centers[..., 0] * 4).floor().long(), torch.arange(4)[None])
    source_center = (boxes[..., :2] + boxes[..., 2:]) / 2
    limit = ((windows[..., 2:] - windows[..., :2]) * 0.25).clamp_max(0.125)
    assert (torch.abs(centers - source_center) <= limit + 1e-7).all()
    assert torch.equal(corrected[~active], boxes[~active])
    assert not torch.equal(corrected[active], boxes[active])


def test_frame_and_slot_outputs_do_not_use_each_others_features():
    boxes, windows, features, active = _inputs(batch=2)
    model = ZoomBoxRefiner(4)
    with torch.no_grad():
        model.head[-1].weight.normal_(std=0.1, generator=torch.Generator().manual_seed(63))
        combined = model(boxes, windows, features, active)
        by_frame = torch.cat([model(*(v[i:i + 1] for v in (boxes, windows, features, active))) for i in range(2)])
        modified = features.clone()
        modified[1, 1] = -modified[1, 1]
        modified[~active] = 400
        changed = model(boxes, windows, modified, active)
    torch.testing.assert_close(combined, by_frame)
    assert torch.equal(changed[0], combined[0])
    assert torch.equal(changed[1, [0, 2, 3]], combined[1, [0, 2, 3]])
    assert not torch.equal(changed[1, 1], combined[1, 1])


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda b, w, f, a: (b.double(), w, f, a), "CPU float32"),
        (lambda b, w, f, a: (b, w, f[:, :, :3], a), "aligned to boxes"),
        (lambda b, w, f, a: (b, w, f, a.long()), "boolean mask"),
        (lambda b, w, f, a: (b.roll(1, dims=1), w, f, a), "assigned band"),
        (lambda b, w, f, a: (b, w, f * torch.nan, a), "must be finite"),
        (lambda b, w, f, a: (b, torch.zeros_like(w), f, a), "positive-area"),
        (lambda b, w, f, a: (b, torch.tensor([0.0, 0.0, 0.2, 0.2]).expand_as(w), f, a), "contain"),
    ],
)
def test_rejects_invalid_input_instead_of_silently_fitting_another_view(mutation, match):
    with pytest.raises(ValueError, match=match):
        ZoomBoxRefiner(4)(*mutation(*_inputs()))


def _supervised_example(batch=24):
    boxes, windows, features, _ = _inputs(batch)
    active = torch.zeros((batch, 4), dtype=torch.bool)
    active[:, 1] = True
    active[:, 2] = True
    present = torch.zeros_like(active)
    present[:, 1] = True
    present[:, 0] = True  # This label has no visible prediction.
    direction = torch.where(torch.arange(batch) % 2 == 0, -1.0, 1.0)
    features[:, 1] = 0
    features[:, 1, 0] = direction[:, None, None]
    targets = torch.full_like(boxes, torch.nan)
    targets[:, 1] = boxes[:, 1] + direction[:, None] * torch.tensor([0.035, 0.0, 0.035, 0.0])
    return boxes, windows, features, active, targets, present


def test_fit_learns_opposite_offsets_and_is_deterministic_without_mutating_inputs():
    inputs = _supervised_example()
    for x in (inputs[0], inputs[1], inputs[2], inputs[4]):
        x.requires_grad_(True)
    saved = tuple(t.detach().clone() for t in inputs)
    prior_rng = torch.random.get_rng_state().clone()
    with torch.no_grad():
        learned, trace = fit_zoom_refiner(*inputs, seed=117, epochs=12, batch_size=8)
    second, second_trace = fit_zoom_refiner(*inputs, seed=117, epochs=12, batch_size=8)
    assert not learned.training
    assert trace == second_trace
    json.dumps(trace, allow_nan=False)
    assert len(trace) == 12 and all(t["steps"] == 3 and t["supervised_boxes"] == 24 for t in trace)
    assert len({t["order_sha256"] for t in trace}) == len(trace)
    assert trace[-1]["mean_loss"] < trace[0]["mean_loss"]
    assert torch.equal(prior_rng, torch.random.get_rng_state())
    for a, b in zip(learned.state_dict().values(), second.state_dict().values()):
        assert torch.equal(a, b)
    for original, current in zip(saved, inputs):
        torch.testing.assert_close(original, current, equal_nan=True, rtol=0, atol=0)
        assert current.grad is None

    boxes, windows, features, active, targets, _ = saved
    with torch.no_grad():
        corrected = learned(boxes, windows, features, active)
    source = (boxes[:, 1, 0] + boxes[:, 1, 2]) * 0.5
    wanted = (targets[:, 1, 0] + targets[:, 1, 2]) * 0.5
    actual = (corrected[:, 1, 0] + corrected[:, 1, 2]) * 0.5
    assert ((actual - source) * (wanted - source) > 0).all()
    assert (actual - wanted).abs().mean() < 0.75 * (source - wanted).abs().mean()


def test_both_controls_share_order_and_unused_labels_do_not_affect_training():
    boxes, windows, features, active, labels, present = _supervised_example(batch=8)
    other_labels = labels.clone()
    other_labels[~(active & present)] = torch.inf
    other_present = present.clone()
    other_present[~active] = ~other_present[~active]
    learned, trace = fit_zoom_refiner(boxes, windows, features, active, labels, present, seed=0, epochs=3, batch_size=4)
    check, other_trace = fit_zoom_refiner(boxes, windows, features, active, other_labels, other_present,
                                         seed=0, epochs=3, batch_size=4)
    assert trace == other_trace
    assert all(torch.equal(a, b) for a, b in zip(learned.state_dict().values(), check.state_dict().values()))
    zero, zero_trace = fit_zoom_refiner(boxes, windows, torch.zeros_like(features), active, labels, present,
                                        seed=0, epochs=3, batch_size=4)
    assert tuple(learned.state_dict()) == tuple(zero.state_dict())
    assert [row["order_sha256"] for row in trace] == [row["order_sha256"] for row in zero_trace]
    assert [row["supervised_boxes"] for row in trace] == [row["supervised_boxes"] for row in zero_trace]


def test_empty_supervision_and_bad_visible_labels_do_not_silently_pass():
    boxes, windows, features, active, targets, present = _supervised_example(batch=2)
    with pytest.raises(ValueError, match="at least one active"):
        fit_zoom_refiner(boxes, windows, features, torch.zeros_like(active), targets, present, seed=1)
    with pytest.raises(ValueError, match="Supervised target"):
        fit_zoom_refiner(boxes, windows, features, active, torch.full_like(targets, torch.nan), present, seed=1)


def test_receipt_declares_fixed_defaults():
    protocol = zoom_head_protocol()
    json.dumps(protocol, allow_nan=False)
    assert protocol["device"] == "cpu"
    assert protocol["epochs"] == 8 and protocol["batch_size"] == 16
    assert protocol["grid"] == [4, 4] and protocol["loss_mask"] == "active & target_presence"
