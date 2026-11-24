from pathlib import Path

import pytest
import torch

from hybrid_drive.experiments.temporal_vru_eval import score_temporal_vru
from hybrid_drive.experiments.temporal_vru_model import TemporalVruHead, grid_cell, supervised_targets
from hybrid_drive.experiments.temporal_vru_training import camera_batch, forecast_loss, fresh_head
from hybrid_drive.experiments.visual_vru_model import GRID_SHAPE, VISUAL_DIM


def _inputs(n=2):
    now = torch.zeros(n, VISUAL_DIM, *GRID_SHAPE)
    past = now.clone()
    support = torch.ones(n, *GRID_SHAPE)
    return {"current": now, "past_in_current": past, "current_coverage": support,
            "past_coverage": support.clone(), "history_s": torch.full((n,), .5)}


def test_supported_past_features_affect_forecast_and_do_not_replace_current_images():
    grids = _inputs(1)
    model = fresh_head().eval()
    with torch.no_grad():
        a = model(*camera_batch(grids, [0], "temporal"))
        grids["past_in_current"][0, 19, 8, 4] = 8.
        b = model(*camera_batch(grids, [0], "temporal"))
    assert a.shape == b.shape == (1, 3, *GRID_SHAPE)
    assert not torch.equal(a, b) and grids["current"].count_nonzero() == 0


def test_past_unknown_cells_cannot_create_evidence_through_gelu_convolution():
    grids = _inputs(1)
    grids["past_coverage"][0, 9, 5] = 0
    model = fresh_head().eval()
    with torch.no_grad():
        first = model(*camera_batch(grids, [0], "temporal"))
        grids["past_in_current"][0, :, 9, 5] = float("nan")
        second = model(*camera_batch(grids, [0], "temporal"))
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_single_camera_training_receives_no_historical_material():
    grids = _inputs()
    grids["past_in_current"].fill_(99.)
    grids["past_coverage"].zero_()
    now, past, nc, pc, time = camera_batch(grids, [0], "single_frame")
    assert now is past and nc is pc
    assert now.count_nonzero() == 0 and nc.all() and time.tolist() == [.5]


def test_shuffled_history_changes_only_mutually_supported_historical_cells():
    grids = _inputs()
    grids["past_in_current"][0].fill_(3.)
    grids["past_in_current"][1].fill_(7.)
    grids["past_coverage"][1, 0, 1] = 0
    grids["past_coverage"][0, 0, 2] = 0
    now, past, nc, pc, _ = camera_batch(grids, [0], "shuffled_past", history_ids=[1])
    assert now.count_nonzero() == 0 and nc.all()
    assert past[0, 0, 0, :3].tolist() == [7., 3., 3.]
    torch.testing.assert_close(pc, grids["past_coverage"][[0]], rtol=0, atol=0)


def test_history_outside_declared_past_window_is_not_inferred():
    model = TemporalVruHead()
    grids = _inputs(1)
    for value in (.2, .7, float("nan")):
        grids["history_s"].fill_(value)
        with pytest.raises(ValueError, match="history interval"):
            model(*camera_batch(grids, [0], "temporal"))


def test_future_optimization_requires_unambiguous_covered_change():
    coverage = torch.ones(1, *GRID_SHAPE)
    one = [[6.2, .2], [6.6, .4], [7.4, .3]]
    samples = [{"tracks":[{"centers_xy_m":one}]}]
    target, moves = supervised_targets(samples, coverage)
    origin, future = grid_cell(one[0]), grid_cell(one[-1])
    assert target.sum() == 3 and moves == [[(2, origin, future)]]
    another = [[9.8, 3.], [8., 2.], [6.2, .2]]
    labels, moves = supervised_targets([{"tracks":[{"centers_xy_m":one}, {"centers_xy_m":another}]}], coverage)
    assert labels.sum() == 6 and (2,origin,future) not in moves[0]
    coverage[0][future] = 0
    assert supervised_targets(samples, coverage)[1] == [[]]


def test_displacement_loss_pushes_forward_occupancy_without_suppressing_original_label():
    values = torch.zeros(1, 3, *GRID_SHAPE, requires_grad=True)
    known = torch.ones(1, *GRID_SHAPE, dtype=torch.bool)
    label = torch.zeros_like(values, dtype=torch.bool)
    old, new = (4,9), (5,9)
    label[0,0][old] = label[0,1][old] = label[0,2][new] = True
    forecast_loss(values,label,known,[[(2,old,new)]]).backward()
    assert values.grad[0,0][old] < 0
    assert values.grad[0,2][new] < 0 < values.grad[0,2][old]


def test_broken_forecast_seal_is_rejected_before_targets_or_index(tmp_path: Path):
    (tmp_path/"predictions.json").write_bytes(b"{}")
    missing = tmp_path/"no-file"
    with pytest.raises(ValueError, match="recorded seal"):
        score_temporal_vru(missing, missing, tmp_path, tmp_path/"out", sealed_sha256="0"*64)
