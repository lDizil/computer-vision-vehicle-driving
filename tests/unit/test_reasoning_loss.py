from dataclasses import replace
import math

import pytest
import torch

from hybrid_drive.neural.decoder import PerceptionOutput
from hybrid_drive.neural.losses import PerceptionTargets, Reason2DriveLoss


def example(dtype=torch.float32):
    def floating(value, grad=False):
        return torch.tensor(value, dtype=dtype, requires_grad=grad)

    indices = torch.tensor([0, 0])
    positions = torch.tensor([1, 2])
    kinds = torch.tensor([0, 1])
    prediction = PerceptionOutput(
        indices,
        positions,
        kinds,
        floating([0, 0], True),
        floating([[1, 0, 0, 1, 1, 1, math.pi - 0.1], [float("nan")] * 7], True),
        floating([[0, 0, 2, 2], [float("nan")] * 4], True),
        floating([[[float("nan")] * 2] * 2, [[2, 4], [float("nan")] * 2]], True),
    )
    target = PerceptionTargets(
        indices.clone(),
        positions.clone(),
        kinds.clone(),
        floating([1, 1]),
        torch.ones(2, dtype=torch.bool),
        floating([[0, 0, 0, 1, 1, 1, -math.pi + 0.1], [float("nan")] * 7]),
        torch.tensor([[True, False, False, False, False, False, True], [True] * 7]),
        floating([[0, 0, 1, 1], [float("nan")] * 4]),
        torch.tensor([True, True]),
        floating([[[float("nan")] * 2] * 2, [[1, 2], [float("nan")] * 2]]),
        torch.tensor([[True, True], [True, False]]),
    )
    logits = floating([[[float("nan")] * 2, [0, 0], [0, 0]]], True)
    labels = torch.tensor([[-999, 0, 1]])
    mask = torch.tensor([[False, True, True]])
    return logits, labels, mask, prediction, target


def test_hand_calculated_components_reductions_yaw_wrap_and_gradients():
    logits, labels, mask, pred, target = example()
    result = Reason2DriveLoss()(logits, labels, mask, pred, target)
    assert result.text.item() == pytest.approx(math.log(2))
    assert result.classification.item() == pytest.approx(math.log(2))
    assert result.boxes_3d.item() == pytest.approx(0.26, abs=1e-6)
    assert result.giou_2d.item() == pytest.approx(0.75, abs=1e-6)
    assert result.motion.item() == pytest.approx(1)
    assert result.total.item() == pytest.approx(2 * math.log(2) + 0.26 + 0.75 + 1, abs=1e-6)
    assert (
        result.supervised_text_tokens,
        result.supervised_events,
        result.supervised_box_coordinates,
        result.supervised_image_boxes,
        result.supervised_motion_points,
    ) == (2, 2, 2, 1, 1)
    result.total.backward()
    assert torch.count_nonzero(logits.grad[0, 0]) == 0
    torch.testing.assert_close(pred.validity_logits.grad, torch.tensor([-0.25, -0.25]))
    assert pred.boxes_3d.grad[0, 0].item() == pytest.approx(0.5)
    assert pred.boxes_3d.grad[0, 6].item() == pytest.approx(-0.1, abs=1e-6)
    for value in (logits, pred.validity_logits, pred.boxes_3d, pred.boxes_2d, pred.motion_xy):
        assert value.grad is not None and torch.isfinite(value.grad).all()
    assert torch.count_nonzero(pred.motion_xy.grad[1, 1]) == 0
    assert torch.count_nonzero(pred.boxes_3d.grad[1]) == 0


def test_negative_and_unsupervised_validity_never_supervise_regression():
    logits, labels, mask, pred, target = example()
    target = replace(
        target, validity=torch.tensor([0.0, float("nan")]), validity_mask=torch.tensor([True, False])
    )
    pred = replace(
        pred,
        validity_logits=torch.tensor([0.0, float("nan")], requires_grad=True),
        boxes_3d=torch.full((2, 7), float("nan"), requires_grad=True),
        boxes_2d=torch.full((2, 4), float("nan"), requires_grad=True),
        motion_xy=torch.full((2, 2, 2), float("nan"), requires_grad=True),
    )
    result = Reason2DriveLoss()(logits, labels, mask, pred, target)
    assert result.total.item() == pytest.approx(2 * math.log(2))
    assert result.supervised_events == 1 and result.supervised_box_coordinates == 0
    result.total.backward()
    assert pred.validity_logits.grad[0] > 0
    for value in (pred.boxes_3d, pred.boxes_2d, pred.motion_xy):
        assert torch.isfinite(value.grad).all() and torch.count_nonzero(value.grad) == 0


def test_zero_perception_weight_is_a_real_text_only_ablation():
    args = example()
    result = Reason2DriveLoss(lambda_per=0)(*args)
    assert result.total.item() == pytest.approx(math.log(2))
    result.total.backward()
    for value in (args[3].validity_logits, args[3].boxes_3d, args[3].boxes_2d, args[3].motion_xy):
        assert torch.count_nonzero(value.grad) == 0


def test_empty_events_and_all_ignored_text_are_differentiable_zero_not_nan():
    logits = torch.full((1, 3, 2), float("nan"), requires_grad=True)
    index = torch.empty(0, dtype=torch.long)

    def empty(shape):
        return torch.empty(shape, requires_grad=True)

    pred = PerceptionOutput(index, index, index, empty((0,)), empty((0, 7)), empty((0, 4)), empty((0, 2, 2)))
    target = PerceptionTargets(
        index,
        index,
        index,
        torch.empty(0),
        torch.empty(0, dtype=torch.bool),
        torch.empty(0, 7),
        torch.empty(0, 7, dtype=torch.bool),
        torch.empty(0, 4),
        torch.empty(0, dtype=torch.bool),
        torch.empty(0, 2, 2),
        torch.empty(0, 2, dtype=torch.bool),
    )
    result = Reason2DriveLoss()(
        logits, torch.full((1, 3), -100, dtype=torch.long), torch.ones(1, 3, dtype=torch.bool), pred, target
    )
    assert result.total.item() == 0
    result.total.backward()
    for value in (logits, pred.validity_logits, pred.boxes_3d, pred.boxes_2d, pred.motion_xy):
        assert value.grad is not None and torch.isfinite(value.grad).all()
        assert torch.count_nonzero(value.grad) == 0


@pytest.mark.parametrize(
    "name,value",
    [
        ("batch_indices", torch.tensor([0, 1])),
        ("token_positions", torch.tensor([0, 2])),
        ("event_kinds", torch.tensor([1, 1])),
    ],
)
def test_targets_are_bound_to_event_identity_not_only_shape(name, value):
    logits, labels, mask, pred, target = example()
    with pytest.raises(ValueError, match="positions or kinds"):
        Reason2DriveLoss()(logits, labels, mask, pred, replace(target, **{name: value}))


@pytest.mark.parametrize(
    "corruption",
    ["text_target", "text_nan", "validity", "box_nan", "dimension", "inverted", "target_roi", "motion_nan"],
)
def test_active_bad_supervision_fails_instead_of_becoming_zero_loss(corruption):
    logits, labels, mask, pred, target = example()
    with torch.no_grad():
        if corruption == "text_target":
            labels[0, 1] = 2
        if corruption == "text_nan":
            logits[0, 1, 0] = float("nan")
        if corruption == "validity":
            target.validity[0] = 0.5
        if corruption == "box_nan":
            pred.boxes_3d[0, 0] = float("nan")
        if corruption == "dimension":
            target.boxes_3d[0, 3] = -1
            target.boxes_3d_mask[0, 3] = True
        if corruption == "inverted":
            pred.boxes_2d[0] = torch.tensor([2, 0, 1, 1])
        if corruption == "target_roi":
            target.boxes_2d[0, 3] = 1.1
        if corruption == "motion_nan":
            pred.motion_xy[1, 0, 0] = float("nan")
    with pytest.raises(ValueError):
        Reason2DriveLoss()(logits, labels, mask, pred, target)


@pytest.mark.parametrize("field", ["validity_mask", "boxes_3d_mask", "boxes_2d_mask", "motion_mask"])
def test_supervision_masks_are_not_coerced_from_numbers(field):
    logits, labels, mask, pred, target = example()
    with pytest.raises(ValueError, match="boolean mask"):
        Reason2DriveLoss()(
            logits, labels, mask, pred, replace(target, **{field: getattr(target, field).float()})
        )


def test_float64_is_preserved_and_unknown_text_labels_are_ignored():
    logits, labels, mask, pred, target = example(torch.float64)
    result = Reason2DriveLoss()(logits, labels, mask, pred, target)
    assert result.total.dtype == torch.float64
    assert result.boxes_3d.item() == pytest.approx(0.26, abs=1e-12)
    result.total.backward()
    assert torch.isfinite(logits.grad).all()


@pytest.mark.parametrize(
    "corruption", ["duplicate", "out_of_batch", "position", "kind", "motion_shape", "dtype"]
)
def test_misaligned_prediction_contract_is_rejected(corruption):
    logits, labels, mask, pred, target = example()
    if corruption == "duplicate":
        pred = replace(pred, token_positions=torch.tensor([1, 1]))
        target = replace(target, token_positions=pred.token_positions)
    if corruption == "out_of_batch":
        pred = replace(pred, batch_indices=torch.tensor([0, 1]))
        target = replace(target, batch_indices=pred.batch_indices)
    if corruption == "position":
        pred = replace(pred, token_positions=torch.tensor([1, 3]))
        target = replace(target, token_positions=pred.token_positions)
    if corruption == "kind":
        pred = replace(pred, event_kinds=torch.tensor([0, 2]))
        target = replace(target, event_kinds=pred.event_kinds)
    if corruption == "motion_shape":
        pred = replace(pred, motion_xy=torch.zeros(2, 2, 3))
    if corruption == "dtype":
        pred = replace(pred, validity_logits=pred.validity_logits.double())
    with pytest.raises(ValueError):
        Reason2DriveLoss()(logits, labels, mask, pred, target)
