from dataclasses import replace
import math

import numpy as np
import pytest
import torch
from torch import nn

from hybrid_drive.contracts.planning import Candidate, CandidateSet
from hybrid_drive.contracts.sensors import CameraSnapshot, SURROUND_ORDER, SurroundView
from hybrid_drive.contracts.trajectory import Pose2D, Trajectory
from hybrid_drive.experiments.paired_slow_training import LabeledControl, fit_paired, initialized_model
from hybrid_drive.fast.forecast_proposals import ForecastObject
from hybrid_drive.fast.scene_payload import FastScene, RoadEvidence, SceneGrid
from hybrid_drive.geometry.camera import CameraCalibration
from hybrid_drive.neural.paired_scene_risk import RiskRequest, relative_actor_samples
from hybrid_drive.neural.surround_features import SurroundFeatures


class FrozenReadout(nn.Module):
    def __init__(self):
        super().__init__()
        self.output = nn.Linear(12, 2, bias=False)
        self.eval().requires_grad_(False)

    def binary_answer_logits(self, prefix):
        return self.output(prefix.mean(dim=1))


def observation(key="current", *, hidden=False):
    rgb = np.zeros((24, 32, 3), np.uint8)
    camera_axes = ((0, -1, 0, 0), (0, 0, -1, 0), (1, 0, 0, 0), (0, 0, 0, 1))
    native = tuple(CameraCalibration("cam-" + side.value, 32, 24,
                    ((20, 0, 15.5), (0, 18, 11.5), (0, 0, 1)), camera_axes) for side in SURROUND_ORDER)
    snapshot = CameraSnapshot(key, 5_000_000, np.eye(4), None,
                              tuple(SurroundView(side, calibration, rgb)
                                    for side, calibration in zip(SURROUND_ORDER, native)), rig_id="test-rig")
    features = SurroundFeatures(snapshot, "test", torch.arange(288).float().reshape(1, 6, 3, 4, 4) / 300, native)
    trajectories = (Candidate("first", Trajectory(((2, 0, 0), (5, .2, .1)), .5)),
                    Candidate("second", Trajectory(((1, 0, 0), (3, 0, 0)), .5)))
    plan = CandidateSet("plan", key, 5., Pose2D(0, 0, 0), trajectories)
    grid = SceneGrid(0, -2, 1, 6, 4)
    road = RoadEvidence(grid, np.full((6, 4), .8), np.full((6, 4), .15), np.ones((6, 4), dtype=bool))
    actor = ForecastObject("actor", (-9. if hidden else 9., 1., .8), (1.9, 4.5, 1.6), .4, (-2., -1.), .8)
    scene = FastScene(key, snapshot.content_sha256, snapshot.ref_time_us, (actor,), plan, road)
    return RiskRequest(features, scene, ("first", "second"))


def test_relative_tokens_propagate_motion_and_rotate_at_exact_candidate_times():
    actor = ForecastObject("actor", (2, 4, 1), (2, 4, 2), math.pi, (3, -2), 1)
    candidate = Trajectory(((1, 2, math.pi / 2), (3, 1, 0)), .5)
    result = relative_actor_samples((actor,), candidate)
    assert result.shape == (2, 11)
    np.testing.assert_allclose(result[0, :7], (1 / 8, -2.5 / 4, 1, 0, .5, .5, .125), atol=1e-6)
    np.testing.assert_allclose(result[1, :7], (2 / 8, 1 / 4, 0, -1, .5, .5, .25), atol=1e-6)


def test_batch_members_and_request_order_do_not_change_candidate_margins():
    request = observation()
    other = observation("other", hidden=True)
    model, readout = initialized_model(visual_dim=3, prior_dim=12, llm_dim=12).eval(), FrozenReadout()
    swapped = replace(request, candidate_ids=("second", "first"))
    with torch.no_grad():
        single, = model([request], readout)
        second, first, = model([other, swapped], readout)
        alone, = model([other], readout)
    torch.testing.assert_close(first.flip(0), single)
    torch.testing.assert_close(second, alone)


def test_hidden_actor_adds_no_motion_information_even_when_very_fast():
    request = observation(hidden=True)
    actor, = request.scene.boxes
    different = replace(request, scene=replace(request.scene, boxes=(replace(actor, velocity_xy=(20., 7.)),)))
    model, readout = initialized_model(visual_dim=3, prior_dim=12, llm_dim=12).eval(), FrozenReadout()
    with torch.no_grad():
        a, b = model([request, different], readout)
    torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_backprop_reaches_relative_projection_bridge_and_qformer_but_not_frozen_language():
    request = observation()
    model, readout = initialized_model(visual_dim=3, prior_dim=12, llm_dim=12), FrozenReadout()
    margins, = model([request], readout)
    (margins[0] - margins[1]).backward()
    for parameters in (model.relative.parameters(), model.bridge.actor_tokenizer.parameters(), model.slow.parameters()):
        assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0 for p in parameters)
    assert all(p.grad is None and not p.requires_grad for p in readout.parameters())


def test_other_candidate_geometry_and_query_do_not_leak_into_selected_candidate():
    request = observation()
    scene = request.scene
    selected, _ = scene.plan.candidates
    different = Candidate("second", Trajectory(((0, -1, -.2), (4, -1, -.8)), .5))
    changed = replace(request, scene=replace(scene, plan=replace(scene.plan, candidates=(selected, different))))
    model, readout = initialized_model(visual_dim=3, prior_dim=12, llm_dim=12).eval(), FrozenReadout()
    with torch.no_grad():
        first, second = model([request, changed], readout)
    torch.testing.assert_close(first[0], second[0], atol=1e-7, rtol=0)


@pytest.mark.parametrize("keys", [(), ("first", "first"), ("unknown",)])
def test_requests_require_unique_owned_candidate_ids(keys):
    model = initialized_model(visual_dim=3, prior_dim=12, llm_dim=12)
    with pytest.raises(ValueError, match="unique candidate IDs"):
        model([replace(observation(), candidate_ids=keys)], FrozenReadout())


def test_fit_rejects_development_input_seen_by_optimizer():
    request = observation()
    group = LabeledControl((request, request), (0, 1))
    model = initialized_model(visual_dim=3, prior_dim=12, llm_dim=12)
    with pytest.raises(ValueError, match="separate from optimizer"):
        fit_paired(model, FrozenReadout(), [group], [group], epochs=1)


def test_unfrozen_language_is_not_silently_added_to_training_graph():
    model = initialized_model(visual_dim=3, prior_dim=12, llm_dim=12)
    language = FrozenReadout().requires_grad_(True)
    with pytest.raises(ValueError, match="frozen"):
        model([observation()], language)
