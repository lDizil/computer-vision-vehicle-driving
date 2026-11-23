import json

import pytest
import torch

from hybrid_drive.neural.spatial_zoom_guard import (
    ZoomGuard, apply_zoom_guard, fit_zoom_guard, mix_zoom_boxes, zoom_guard_protocol,
)


def _boxes(batch=1):
    baseline = torch.tensor(
        [[[0.00, 0.10, 0.20, 0.30], [0.27, 0.25, 0.39, 0.52],
          [0.51, 0.40, 0.72, 0.80], [0.78, 0.60, 1.00, 1.00]]],
        dtype=torch.float32,
    ).repeat(batch, 1, 1)
    proposal = baseline + torch.tensor(
        [[[0.06, -0.10, 0.00, 0.05], [0.08, 0.00, 0.08, 0.00],
          [0.02, 0.05, 0.02, 0.05], [0.00, -0.15, -0.04, 0.00]]],
        dtype=torch.float32,
    )
    return baseline, proposal


def _supervision():
    baseline, proposal = _boxes(batch=18)
    active = torch.tensor([[False, True, True, False]]).repeat(18, 1)
    present = torch.tensor([[True, True, False, False]]).repeat(18, 1)
    targets = torch.full_like(baseline, torch.nan)
    targets[:6, 1] = baseline[:6, 1]
    targets[6:12, 1] = ((baseline[6:12, 1].double() + proposal[6:12, 1].double()) / 2).float()
    targets[12:, 1] = proposal[12:, 1]
    scores = torch.zeros((18, 4), dtype=torch.float32)
    scores[:, 1] = torch.tensor([-3.0, 0, 3.0]).repeat_interleave(6)
    return baseline, proposal, active, targets, present, scores


def test_mix_uses_exact_endpoints_and_only_moves_active_slots():
    baseline, proposal = _boxes()
    active = torch.tensor([[True, False, True, False]])
    start = mix_zoom_boxes(baseline, proposal, active, 0)
    middle = mix_zoom_boxes(baseline, proposal, active, 0.5)
    end = mix_zoom_boxes(baseline, proposal, active, 1)
    expected_middle = torch.tensor([[0.030, 0.050, 0.20, 0.325], [0.52, 0.425, 0.73, 0.825]])
    assert torch.equal(start, baseline)
    assert start.data_ptr() != baseline.data_ptr()
    assert torch.equal(end[active], proposal[active])
    assert torch.equal(middle[~active], baseline[~active])
    assert torch.equal(end[~active], baseline[~active])
    torch.testing.assert_close(middle[active], expected_middle, rtol=0, atol=6e-8)
    centers_x = (middle[..., 0] + middle[..., 2]) / 2
    assert torch.equal((centers_x * 4).floor().long().clamp_max(3), torch.arange(4)[None])
    assert (middle[..., 2:] > middle[..., :2]).all()
    assert ((middle >= 0) & (middle <= 1)).all()


@pytest.mark.parametrize("fraction", [True, -0.001, 1.001, float("nan"), float("inf")])
def test_mix_rejects_invalid_fractions(fraction):
    baseline, proposal = _boxes()
    with pytest.raises(ValueError, match="mix fraction"):
        mix_zoom_boxes(baseline, proposal, torch.ones((1, 4), dtype=torch.bool), fraction)


def test_half_step_falls_back_for_unrepresentable_geometry():
    baseline, proposal = _boxes()
    middle = torch.tensor(0.5, dtype=torch.float32)
    baseline[0, 0, 1] = torch.nextafter(middle, torch.zeros_like(middle))
    baseline[0, 0, 3] = middle
    proposal[0, 0, 1] = middle
    proposal[0, 0, 3] = torch.nextafter(middle, torch.ones_like(middle))
    proposal[0, 0, (0, 2)] = baseline[0, 0, (0, 2)]
    active = torch.tensor([[True, False, False, False]])
    scores = torch.zeros((1, 4), dtype=torch.float32)
    assert torch.equal(mix_zoom_boxes(baseline, proposal, active, 0.5), baseline)
    # The positive but extremely thin source and target boxes still have a well-defined IoU.
    guard = fit_zoom_guard(baseline, proposal, active, proposal, active, scores)
    assert guard.intercept == (0.0, 1.0)
    full, fractions = apply_zoom_guard(guard, baseline, proposal, active, scores)
    assert fractions.tolist() == [[1.0, 0.0, 0.0, 0.0]]
    assert torch.equal(full[active], proposal[active])

    artifact = guard.to_dict()
    artifact["intercept"] = [0.5, 0.0]
    corrected, fractions = apply_zoom_guard(ZoomGuard.from_dict(artifact), baseline, proposal, active, scores)
    assert torch.equal(corrected, baseline)
    assert not fractions.any()


def test_guard_learns_baseline_half_and_full_from_immutable_inputs():
    baseline, proposal, active, targets, present, scores = inputs = _supervision()
    for original in (baseline, proposal, targets, scores):
        original.requires_grad_(True)
    saved = tuple(original.detach().clone() for original in inputs)
    rng = torch.random.get_rng_state().clone()
    guard = fit_zoom_guard(*inputs)
    repeated = fit_zoom_guard(*inputs)
    updated, fractions = apply_zoom_guard(guard, baseline, proposal, active, scores, require_support=True)
    assert guard.to_dict() == repeated.to_dict()
    assert torch.equal(rng, torch.random.get_rng_state())
    assert guard.supervised_boxes == 18
    assert fractions.dtype == torch.float32 and fractions.shape == active.shape
    assert fractions[:, 1].tolist() == [0.0] * 6 + [0.5] * 6 + [1.0] * 6
    assert torch.equal(updated[:, 1], targets[:, 1])
    assert torch.equal(updated[~active], baseline[~active])
    assert not fractions[~active].any()
    assert not updated.requires_grad and not fractions.requires_grad
    for original, version in zip(inputs, saved):
        torch.testing.assert_close(original, version, rtol=0, atol=0, equal_nan=True)
        assert original.grad is None


def test_masked_targets_and_scores_cannot_affect_the_fit():
    baseline, proposal, active, targets, present, scores = inputs = _supervision()
    ignored = ~(active & present)
    changed_targets = targets.clone()
    changed_targets[ignored] = torch.inf
    changed_presence = present ^ ~active
    changed_scores = scores.clone()
    changed_scores[ignored] = torch.nan
    changed_proposal = proposal.clone()
    changed_proposal[ignored] = baseline[ignored]
    assert fit_zoom_guard(*inputs) == fit_zoom_guard(
        baseline, changed_proposal, active, changed_targets, changed_presence, changed_scores,
    )


@pytest.mark.parametrize(
    ("half_gain", "full_gain", "expected"),
    [(0.04, 0.055, 0.5), (0.019, 0.03, 0.0), (0.02, 0.04, 1.0), (-0.2, 0.02, 1.0), (-0.1, 0.019, 0.0)],
)
def test_full_step_must_clear_both_gain_floors(half_gain, full_gain, expected):
    baseline, proposal, active, _, _, scores = inputs = _supervision()
    payload = fit_zoom_guard(*inputs).to_dict()
    payload["weights"] = [[0.0] * len(row) for row in payload["weights"]]
    payload["intercept"] = [half_gain, full_gain]
    _, fractions = apply_zoom_guard(ZoomGuard.from_dict(payload), baseline, proposal, active, scores)
    assert fractions[active].tolist() == [expected] * int(active.sum())


@pytest.mark.parametrize("novel_feature", ["score", "geometry"])
def test_strict_support_abstains_even_when_the_predicted_gain_is_positive(novel_feature):
    baseline, proposal, active, _, _, scores = inputs = _supervision()
    guard = fit_zoom_guard(*inputs)
    baseline, proposal, active, scores = (tensor[-1:].clone() for tensor in (baseline, proposal, active, scores))
    _, in_support = apply_zoom_guard(guard, baseline, proposal, active, scores, require_support=True)
    assert in_support[0, 1] == 1
    if novel_feature == "score":
        scores[0, 1] = 20
    else:
        for boxes in (baseline, proposal):
            boxes[0, 1, (1, 3)] += 0.22
    loose, without_support_check = apply_zoom_guard(guard, baseline, proposal, active, scores)
    strict, with_support_check = apply_zoom_guard(guard, baseline, proposal, active, scores, require_support=True)
    assert without_support_check[0, 1] == 1
    assert torch.equal(loose[0, 1], proposal[0, 1])
    assert with_support_check[0, 1] == 0
    assert torch.equal(strict[0, 1], baseline[0, 1])


def test_no_active_slots_keep_boxes_even_without_a_finite_score():
    baseline, proposal, active, _, _, scores = inputs = _supervision()
    guard = fit_zoom_guard(*inputs)
    updated, fractions = apply_zoom_guard(
        guard, baseline, proposal, torch.zeros_like(active), torch.full_like(scores, torch.nan),
        require_support=True,
    )
    assert torch.equal(updated, baseline)
    assert updated.data_ptr() != baseline.data_ptr()
    assert not fractions.any()


def test_artifact_roundtrip_does_not_share_mutable_lists():
    baseline, proposal, active, _, _, scores = inputs = _supervision()
    guard = fit_zoom_guard(*inputs)
    original = guard.to_dict()
    decoded = json.loads(json.dumps(original, allow_nan=False))
    restored = ZoomGuard.from_dict(decoded)
    assert restored == guard
    expected = apply_zoom_guard(guard, baseline, proposal, active, scores, require_support=True)
    actual = apply_zoom_guard(restored, baseline, proposal, active, scores, require_support=True)
    assert all(torch.equal(x, y) for x, y in zip(expected, actual))
    decoded["weights"][0][0] = 999
    assert restored.to_dict() == original
    zoom_guard_protocol()["features"].clear()
    assert len(zoom_guard_protocol()["features"]) == len(original["feature_mean"])


@pytest.mark.parametrize(
    ("change", "match"),
    [
        (lambda d: d.update(extra=True), "schema"),
        (lambda d: d.pop("support_max"), "schema"),
        (lambda d: d.update(schema="not-a-guard"), "schema"),
        (lambda d: d.update(supervised_boxes=0), "number of supervised"),
        (lambda d: d.update(supervised_boxes=True), "number of supervised"),
        (lambda d: d.update(weights=d["weights"][:1]), "two gain weight"),
        (lambda d: d["weights"][0].pop(), "weights"),
        (lambda d: d["weights"][0].__setitem__(0, torch.inf), "weights"),
        (lambda d: d["weights"][0].__setitem__(0, 10.0), "weights"),
        (lambda d: d["feature_mean"].__setitem__(0, float("nan")), "feature_mean"),
        (lambda d: d["feature_scale"].__setitem__(0, 0.0), "feature_scale"),
        (lambda d: d["intercept"].__setitem__(0, False), "intercept"),
        (lambda d: d["support_min"].__setitem__(0, 10.0), "training support interval"),
    ],
)
def test_rejects_malformed_artifacts(change, match):
    data = fit_zoom_guard(*_supervision()).to_dict()
    change(data)
    with pytest.raises(ValueError, match=match):
        ZoomGuard.from_dict(data)


def test_rejects_invalid_selected_evidence():
    baseline, proposal, active, targets, present, scores = _supervision()
    with pytest.raises(ValueError, match="CPU float32"):
        mix_zoom_boxes(baseline.double(), proposal, active, 0.5)
    with pytest.raises(ValueError, match="boolean mask"):
        mix_zoom_boxes(baseline, proposal, active.float(), 0.5)
    with pytest.raises(ValueError, match="assigned band"):
        mix_zoom_boxes(baseline, proposal.roll(1, dims=1), active, 0.5)
    with pytest.raises(ValueError, match="at least one active"):
        fit_zoom_guard(baseline, proposal, active, targets, torch.zeros_like(present), scores)
    targets[0, 1] = baseline[0, 0]
    with pytest.raises(ValueError, match="assigned band"):
        fit_zoom_guard(baseline, proposal, active, targets, present, scores)
    targets[0, 1] = baseline[0, 1]
    scores[0, 1] = torch.nan
    with pytest.raises(ValueError, match="must be finite"):
        fit_zoom_guard(baseline, proposal, active, targets, present, scores)
