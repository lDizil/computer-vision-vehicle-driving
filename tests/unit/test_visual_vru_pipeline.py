import numpy as np
import pytest
import torch

from hybrid_drive.experiments.visual_vru_eval import donor_indices, pair_summary, score_vru, summarize
from hybrid_drive.experiments.visual_vru_model import (
    GRID_SHAPE, VISUAL_DIM, VisualVruHead, cell_average_precision, flank_query, pair_logits,
)
from hybrid_drive.experiments.visual_vru_training import _training_loss, initialized_head


def test_invalid_or_off_camera_values_do_not_create_evidence_in_visible_cells():
    head = initialized_head("visual").eval()
    features = torch.zeros(1, VISUAL_DIM, *GRID_SHAPE, dtype=torch.float16)
    support = torch.ones(1, *GRID_SHAPE)
    support[:, 0, 0] = 0.
    corrupt = features.clone()
    corrupt[:, :, 0, 0] = float("nan")
    with torch.no_grad():
        torch.testing.assert_close(head(features, support), head(corrupt, support), rtol=0, atol=0)
    corrupt[:, :, 0, 1] = float("nan")
    with pytest.raises(ValueError, match="finite features"):
        head(corrupt, support)


def test_geometry_control_will_not_silently_consume_images():
    head = VisualVruHead(camera_input=False)
    coverage = torch.ones(2, *GRID_SHAPE)
    with pytest.raises(ValueError, match="must not receive"):
        head(torch.zeros(2, VISUAL_DIM, *GRID_SHAPE), coverage)
    assert head(None, coverage).shape == coverage.shape


def test_unobserved_candidate_abstains_and_does_not_zero_fill_unknown_as_safe():
    candidates = torch.zeros(2, *GRID_SHAPE, dtype=torch.bool)
    logits = torch.full(GRID_SHAPE, 2.)
    candidates[0, 1, 3] = True
    assert pair_logits(logits, candidates) is None
    candidates[1, 1, 7] = True
    logits[1, 7] = -3.
    torch.testing.assert_close(pair_logits(logits, candidates), torch.tensor([2., -3.]))


def test_paired_loss_rewards_opposite_candidate_when_only_truth_is_on_that_side():
    mask = torch.zeros(2, *GRID_SHAPE, dtype=torch.bool)
    mask[0, 1, 0] = mask[1, 1, 2] = True
    target = torch.zeros(1, *GRID_SHAPE, dtype=torch.bool)
    target[0, 1, 2] = True
    actual = torch.zeros(1, *GRID_SHAPE, requires_grad=True)
    known = torch.ones(1, *GRID_SHAPE, dtype=torch.bool)
    loss = _training_loss(actual, known, target, [(mask, 1)])
    loss.backward()
    assert actual.grad[0, 1, 2] < 0 < actual.grad[0, 1, 0]
    assert actual.grad[0, 0, 0] > 0


def test_sealed_optical_shuffle_uses_only_other_recordings():
    anchors = [{'recording': r} for r in ('left','left','right','right','mid')]
    idx = donor_indices(anchors)
    assert idx == [4, 4, 0, 1, 2]
    assert all(a['recording'] != anchors[j]['recording'] for a,j in zip(anchors,idx))
    with pytest.raises(ValueError, match="separate recording"):
        donor_indices(anchors[:2])


def test_ap_groups_ties_and_excludes_unknown_without_inventing_perfect_negative_ap():
    assert cell_average_precision(np.array([3., 3., 2.]), np.array([True, False, False])) == .5
    assert cell_average_precision(np.array([3., 3., 2.]), np.array([False, True, False])) == .5
    assert cell_average_precision(np.array([3., 2., 1.]), np.array([True, True, False])) == 1.
    assert cell_average_precision(np.array([3., 2.]), np.array([False, False])) is None


def test_pair_benchmark_reports_balanced_sides_and_does_not_credit_unobserved():
    rows = [dict(recording='a', current=0, future=1, margins={'visual':[1.,0.], 'geometry':[1.,0.], 'shuffled_camera':None}),
            dict(recording='a', current=0, future=None, margins={'visual':[1.,0.], 'geometry':[1.,0.], 'shuffled_camera':[1.,0.]}),
            dict(recording='b', current=1, future=1, margins={'visual':[0.,1.], 'geometry':[1.,0.], 'shuffled_camera':[0.,1.]})]
    stat = pair_summary(rows, "current")
    assert stat['arms']['visual']['balanced_accuracy'] == 1.
    assert stat['arms']['geometry']['balanced_accuracy'] == .5
    assert stat['arms']['shuffled_camera']['unobserved'] == 1
    assert pair_summary(rows, "future")['pairs'] == 2
    assert pair_summary(rows, "future")['arms']['visual']['correct'] == 1


def test_flank_query_uses_fixed_mirrored_bands_and_skips_mixed_or_unobserved_targets():
    target = torch.zeros(*GRID_SHAPE, dtype=torch.bool)
    coverage = torch.ones(*GRID_SHAPE)
    target[2,1] = target[3,10] = True
    target[7,10] = True
    masks, side, band = flank_query(target,coverage)
    assert side == 0 and band == (5,10)
    assert torch.equal(masks[0], masks[1].flip(-1))
    coverage[7,10] = 0
    assert flank_query(target,coverage) is None


def test_unknown_annotated_cells_are_reported_but_not_counted_as_false_negatives():
    anchors = [dict(recording='test', sample_token='case', probe_poses=None)]
    truth = [dict(visible_cells=[[1,1],[1,2]], blocked_index=None, future_blocked_index=None)]
    coverage = torch.ones(1, *GRID_SHAPE)
    coverage[0,1,2] = 0.
    scores = torch.zeros_like(coverage)
    scores[0,1,1] = 1.
    result = summarize(anchors,truth,{'visual':scores, 'geometry':scores, 'shuffled_camera':scores},coverage)
    assert result['counts']['annotated_cells'] == 2 and result['counts']['covered_annotated_cells'] == 1
    assert result['cell_metrics']['visual']['micro_ap'] == 1.
    assert result['cell_gate']['status'] == 'insufficient_support'
    assert result['current_pair_gate']['status'] == 'insufficient_support'


def test_bad_prediction_seal_fails_before_opening_heldout_labels(tmp_path):
    (tmp_path/'predictions.json').write_bytes(b"{}")
    missing = tmp_path/'does-not-exist'
    with pytest.raises(ValueError, match="recorded seal"):
        score_vru(missing, missing, tmp_path, tmp_path/'out', sealed_sha256='0'*64)
