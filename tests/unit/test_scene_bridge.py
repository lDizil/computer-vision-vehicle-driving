from dataclasses import asdict, replace
import math

import numpy as np
import pytest
import torch

from hybrid_drive.contracts.planning import Candidate, CandidateSet
from hybrid_drive.contracts.sensors import CameraSnapshot, SURROUND_ORDER, SurroundView
from hybrid_drive.contracts.trajectory import Pose2D, Trajectory
from hybrid_drive.fast.forecast_proposals import ForecastObject
from hybrid_drive.fast.scene_payload import FastScene, RoadEvidence, SceneGrid
from hybrid_drive.geometry.camera import CameraCalibration
from hybrid_drive.neural.scene_bridge import SceneBridge
from hybrid_drive.neural.surround_features import SurroundFeatures


_EGO_TO_FRONT = np.array(((0, -1, 0, 0), (0, 0, -1, 0), (1, 0, 0, 0), (0, 0, 0, 1)))
_EGO_TO_REAR = _EGO_TO_FRONT @ np.diag((-1, -1, 1, 1))
_GRID = SceneGrid(0., -2., 1., 6, 4)


def _features(*, focal_scales=(1.,), dtype=torch.float32, rgb_value=0):
    rgb = np.full((80, 100, 3), rgb_value, np.uint8)
    views = []
    for index, direction in enumerate(SURROUND_ORDER):
        is_front = index < len(focal_scales)
        f = focal_scales[index] if is_front else 1.
        views.append(SurroundView(direction, CameraCalibration(
            "cam-" + direction.value, 100, 80,
            ((50 * f, 0, 49.5), (0, 40 * f, 39.5), (0, 0, 1)),
            _EGO_TO_FRONT if is_front else _EGO_TO_REAR,
        ), rgb))
    snapshot = CameraSnapshot("capture", 7_250_000, np.eye(4), None, views, rig_id="calibrated-rig")
    maps = torch.arange(6 * 4 * 6 * 8, dtype=dtype).reshape(1, 6, 4, 6, 8) / 1000
    return SurroundFeatures(snapshot, "bridge-fixture", maps,
                            tuple(view.calibration.resized(50, 40) for view in snapshot.views))


def _actor(object_id="actor", *, xyz=(10., 0., 0.), wlh=(2., 4., 2.), yaw=0., velocity=(3., -2.)):
    return ForecastObject(object_id, xyz, wlh, yaw, velocity, .8)


def _candidate(candidate_id="keep-course", poses=((1.4, .1, 0.), (2.15, .15, math.pi / 2))):
    return Candidate(candidate_id, Trajectory(poses, .5))


def _scene(features, *, actors=(), candidates=None, known=None, p_drivable=None, p_lane=None):
    shape = (_GRID.rows, _GRID.cols)
    road = RoadEvidence(
        _GRID, np.full(shape, .7) if p_drivable is None else p_drivable,
        np.full(shape, .2) if p_lane is None else p_lane,
        np.ones(shape, np.bool_) if known is None else known,
    )
    snapshot = features.snapshot
    plan = CandidateSet("prediction", snapshot.snapshot_id, snapshot.ref_time_us / 1e6,
                        Pose2D(0, 0, 0), candidates if candidates is not None else (_candidate(),))
    return FastScene(snapshot.snapshot_id, snapshot.content_sha256, snapshot.ref_time_us, actors, plan, road)


def _bridge():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(47)
        return SceneBridge(visual_dim=4, prior_dim=12)


def test_metric_cuboid_xyxy_uses_length_axis_yaw_and_resized_pixel_edges():
    features = _features()
    scene = _scene(features, actors=(_actor("straight"), _actor("rotated", yaw=math.pi / 2)))
    tokens = _bridge()(features, scene)

    straight, rotated = tokens.records[:2]
    assert straight.actor_id == "straight" and rotated.actor_id == "rotated"
    assert straight.roi_xyxy == pytest.approx((.4375, .4375, .5625, .5625))
    assert rotated.roi_xyxy == pytest.approx((.5 - 1 / 9, .5 - .5 / 9, .5 + 1 / 9, .5 + .5 / 9))
    assert (straight.camera_name, straight.camera_direction) == ("cam-front", "front")
    assert straight.kind == "actor" and straight.sample_time_s == 0.
    assert tokens.snapshot_sha256 == features.snapshot.content_sha256
    assert tokens.scene_sha256 == scene.content_sha256
    assert tokens.prior_tokens.shape == (1, 10, 12)
    assert tokens.prior_mask.dtype == torch.bool and tokens.prior_mask.all()
    assert tokens.visual_mask.dtype == torch.bool and tokens.visual_mask.shape == (1, 6)
    assert tokens.visual_mask.all()
    torch.testing.assert_close(tokens.visual_tokens, features.feature_maps.mean(dim=(-2, -1)))


def test_near_clip_keeps_full_view_silhouette_and_image_edge_cuts_partial_box():
    features = _features()
    scene = _scene(features, actors=(
        _actor("near", xyz=(.12, 0., 0.), wlh=(2., .4, 2.)),
        _actor("partial", xyz=(5., -5.5, 0.), wlh=(2., 2., 2.)),
        _actor("out-of-fov", xyz=(5., -8., 0.), wlh=(2., 2., 2.)),
        _actor("below-near", xyz=(.05, 0., 0.), wlh=(.02, .02, .02)),
    ))
    tokens = _bridge()(features, scene)
    near, partial, outside, too_near = tokens.records[:4]

    assert near.valid and near.roi_xyxy == (0., 0., 1., 1.)
    assert partial.valid and partial.roi_xyxy[::2] == pytest.approx((.875, 1.))
    for record in (outside, too_near):
        assert record.kind == "actor" and not record.valid
        assert record.camera_name is None and record.roi_xyxy is None
    assert tokens.prior_mask[0, :4].tolist() == [True, True, False, False]
    assert tokens.prior_tokens[0, 2:4].count_nonzero() == 0


@pytest.mark.parametrize("focal_scales,expected", [((1., 1.), 0), ((1., 1.6), 1)])
def test_camera_choice_largest_visible_silhouette_with_snapshot_order_tie_break(focal_scales, expected):
    features = _features(focal_scales=focal_scales)
    scene = _scene(features, actors=(_actor(),))
    bridge = _bridge()
    baseline = bridge(features, scene)
    assert baseline.records[0].camera_direction == SURROUND_ORDER[expected].value

    # Both interventions change global visual pooling; only the actual ROI camera changes this prior.
    for index in (0, 1):
        changed = features.feature_maps.clone()
        changed[0, index].add_(2.)
        other = bridge(replace(features, feature_maps=changed), scene)
        assert (not torch.allclose(other.prior_tokens[0, 0], baseline.prior_tokens[0, 0])) == (index == expected)
        torch.testing.assert_close(other.prior_tokens[0, 1:], baseline.prior_tokens[0, 1:])


def test_actor_motion_changes_its_numeric_prior_without_moving_current_pixel_roi():
    features, model = _features(), _bridge()
    moving = model(features, _scene(features, actors=(_actor(velocity=(8., 3.)),)))
    parked = model(features, _scene(features, actors=(_actor(velocity=(0., 0.)),)))

    assert moving.records == parked.records
    assert not torch.allclose(moving.prior_tokens[0, 0], parked.prior_tokens[0, 0])
    torch.testing.assert_close(moving.prior_tokens[0, 1:], parked.prior_tokens[0, 1:])
    torch.testing.assert_close(moving.visual_tokens, parked.visual_tokens)


def test_road_probes_are_candidate_and_time_keyed_respect_heading_and_mask_unknown():
    features = _features()
    known = np.zeros((_GRID.rows, _GRID.cols), dtype=bool)
    known[1, 2] = known[1, 1] = known[3, 2] = True
    candidates = (_candidate(), _candidate("outside", ((-1., 0., 0.), (-1., 0., 0.))))
    scene = _scene(features, candidates=candidates, known=known)
    result = _bridge()(features, scene)
    assert all(record.valid == mask for record, mask in zip(result.records, result.prior_mask[0].tolist()))

    pose = [rec for rec in result.records if rec.kind == "trajectory"]
    assert [(rec.candidate_id, rec.sample_time_s, rec.sample_index) for rec in pose] == [
        ("keep-course", .5, 0), ("keep-course", 1., 1), ("outside", .5, 0), ("outside", 1., 1),
    ]
    probes = [rec for rec in result.records if rec.kind == "road" and rec.candidate_id == "keep-course"]
    assert [(rec.sample_time_s, rec.road_probe, rec.cell_rc, rec.valid) for rec in probes] == [
        (.5, "center", (1, 2), True), (.5, "left", (1, 3), False), (.5, "right", (1, 1), True),
        (1., "center", (2, 2), False), (1., "left", (1, 2), True), (1., "right", (3, 2), True),
    ]
    outside = [rec for rec in result.records if rec.kind == "road" and rec.candidate_id == "outside"]
    assert len(outside) == 6 and all(rec.cell_rc is None and not rec.valid for rec in outside)
    assert result.prior_tokens[~result.prior_mask].count_nonzero() == 0
    assert asdict(probes[2])["candidate_id"] == "keep-course"


def test_only_matching_road_priors_see_known_probability_changes():
    features, model = _features(), _bridge()
    known = np.zeros((_GRID.rows, _GRID.cols), dtype=bool)
    known[1, 2] = True
    probabilities = np.full(known.shape, .25)
    original = model(features, _scene(features, known=known, p_lane=probabilities))
    probabilities[~known] = .99
    unknown_edit = model(features, _scene(features, known=known, p_lane=probabilities))
    torch.testing.assert_close(original.prior_tokens, unknown_edit.prior_tokens, rtol=0, atol=0)

    probabilities[1, 2] = .90
    known_edit = model(features, _scene(features, known=known, p_lane=probabilities))
    changed = (~torch.isclose(original.prior_tokens, known_edit.prior_tokens)).any(dim=-1)[0]
    assert changed.tolist() == [record.kind == "road" and record.cell_rc == (1, 2) for record in original.records]
    torch.testing.assert_close(original.prior_mask, known_edit.prior_mask)


def test_one_changed_candidate_pose_changes_its_token_without_changing_actor_or_other_poses():
    features, model = _features(), _bridge()
    actors = (_actor(),)
    other = _candidate("brake", ((.1, 0., 0.), (.15, 0., 0.)))
    baseline = model(features, _scene(features, actors=actors, candidates=(_candidate(), other)))
    moved = _candidate(poses=((1.65, .1, 0.), (2.15, .15, math.pi / 2)))
    changed = model(features, _scene(features, actors=actors, candidates=(moved, other)))
    assert baseline.records == changed.records   # The changed point still samples the same road cells.
    assert baseline.scene_sha256 != changed.scene_sha256
    index = [i for i, record in enumerate(baseline.records)
             if record.kind == "trajectory" and record.candidate_id == "keep-course" and record.sample_index == 0]
    difference = (~torch.isclose(baseline.prior_tokens, changed.prior_tokens)).any(dim=-1)[0]
    assert difference.nonzero().flatten().tolist() == index


@pytest.mark.parametrize("mismatch", ["content", "id", "time", "calibration"])
def test_reject_separate_capture_or_incorrect_feature_calibration(mismatch):
    features = _features()
    scene = _scene(features)
    if mismatch == "content":
        features = _features(rgb_value=1)  # Same ID/time and tensor values, different snapshot digest.
    elif mismatch == "id":
        scene = replace(scene, snapshot_id="wrong", plan=replace(scene.plan, snapshot_id="wrong"))
    elif mismatch == "time":
        scene = replace(scene, ref_time_us=8_250_000, plan=replace(scene.plan, t_ref_s=8.25))
    else:
        features = replace(features, calibrations=tuple(reversed(features.calibrations)))
    with pytest.raises(ValueError, match="snapshot ID|calibrations"):
        _bridge()(features, scene)


@pytest.mark.parametrize("oversized", ["actors", "candidates", "poses"])
def test_reject_oversized_scenes_without_partial_prior_output(oversized):
    features = _features()
    overrides = {
        "actors": {"actors": tuple(_actor(str(i)) for i in range(33))},
        "candidates": {"candidates": tuple(_candidate(str(i)) for i in range(9))},
        "poses": {"candidates": (_candidate(poses=[(i, 0., 0.) for i in range(13)]),)},
    }
    with pytest.raises(ValueError, match="at most 32 actors, 8 candidates and 12 poses"):
        _bridge()(features, _scene(features, **overrides[oversized]))


def test_float64_gradients_reach_each_typed_encoder_and_only_the_selected_roi_view():
    features = _features(dtype=torch.float64)
    features.feature_maps.requires_grad_()
    scene = _scene(features, actors=(_actor(),))
    bridge = _bridge().double()
    tokens = bridge(features, scene)
    assert tokens.prior_tokens.dtype == tokens.visual_tokens.dtype == torch.float64
    tokens.prior_tokens[tokens.prior_mask].square().sum().backward()

    for name, parameter in bridge.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all() and parameter.grad.count_nonzero(), name
    gradient = features.feature_maps.grad
    assert gradient[0, 0].count_nonzero() > 0
    assert gradient[0, 1:].count_nonzero() == 0



def test_correct_snapshot_identity_still_rejects_wrong_plan_ego_origin():
    features = _features()
    scene = _scene(features)
    mismatched = replace(scene, plan=replace(scene.plan, reference_pose=Pose2D(.1,0.,0.)))
    with pytest.raises(ValueError,match="different ego reference pose"):
        _bridge()(features,mismatched)
