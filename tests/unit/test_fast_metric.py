from dataclasses import asdict

import numpy as np
import pytest
import torch

from hybrid_drive.contracts.sensors import CameraSnapshot, SURROUND_ORDER, SurroundView
from hybrid_drive.experiments.fast_capture import artifact_hash
from hybrid_drive.experiments.fast_metrics import match_centers
from hybrid_drive.experiments.fast_trial import _tensor, fit_fast, predict_fast, score_fast
from hybrid_drive.experiments.io import write_json
from hybrid_drive.fast.metric_bev import BevGrid, lift_features, past_forward_speed, temporal_bev
from hybrid_drive.fast.metric_forecast import decode_boxes, forecasting_loss, raster_targets
from hybrid_drive.geometry.camera import CameraCalibration
from hybrid_drive.neural.surround_features import SurroundFeatures


def _features(now_us, translation, *, all_back=False):
    local_to_camera = np.eye(4)
    local_to_camera[:3, :3] = ((0,-1,0),(0,0,-1),(1,0,0))
    back_to_front = np.diag([-1,-1,1,1])
    views = []
    for i, direction in enumerate(SURROUND_ORDER):
        transform = local_to_camera @ back_to_front if i or all_back else local_to_camera
        c = CameraCalibration(direction.value, 4, 4, [[2,0,1.5],[0,2,1.5],[0,0,1]], transform)
        views.append(SurroundView(direction, c, np.zeros((4,4,3), np.uint8)))
    pose = np.eye(4)
    pose[:3, 3] = translation
    snapshot = CameraSnapshot(str(now_us), now_us, pose, None, views, rig_id="fixture-rig")
    values = torch.tensor([[[[[1.,3.],[1.,3.]]]]]).repeat(1,6,1,1,1)
    values[:,1:] = 99
    return SurroundFeatures(snapshot, "fixture-encoder", values, tuple(v.calibration for v in views))


def test_lift_excludes_behind_camera_views_and_aligns_the_past_pose():
    grid = BevGrid(x_min=1,y_min=-1,rows=1,cols=1,heights_m=(0.,))
    cur = _features(1_500_000, (0.6,1.,0.))
    past = _features(1_000_000, (0,0,0))
    values, observed = temporal_bev(cur, past, grid)
    # Point (2,0) projects to pixel x=1.5 in current, so center of the 1/3 feature ramp.
    # In previous ego it is (2.6,1) and x=1.5-2/2.6. align_corners=False
    # maps that pixel to a feature value of 2-2/2.6 on the linear ramp.
    assert values[0,0,0].item() == pytest.approx(2)
    assert values[1,0,0].item() == pytest.approx(2-2/2.6)
    assert values[2,0,0].item() == pytest.approx(1/6)
    assert observed[0,0]
    assert past_forward_speed(cur.snapshot, past.snapshot) == pytest.approx(1.2)


def test_no_camera_support_and_future_inputs_are_distinct_from_negatives():
    grid = BevGrid(x_min=1,y_min=-1,rows=1,cols=1,heights_m=(0.,))
    cur = _features(1_500_000, (0,0,0), all_back=True)
    earlier = _features(1_000_000, (0,0,0))
    pooled, visibility = lift_features(cur,cur.snapshot,grid)
    assert not torch.count_nonzero(pooled) and not torch.count_nonzero(visibility)
    with pytest.raises(ValueError,match="future"):
        lift_features(cur,earlier.snapshot,grid)
    with pytest.raises(ValueError,match="precede"):
        temporal_bev(cur,cur,grid)


def _box(name, center=(1.,1.,1.), future=(1.5,1.,1.)):
    return {"instance_id":name,"center":list(center),"size_wlh":[1.9,4.5,1.7],
            "yaw":0.,"future_center":None if future is None else list(future)}


def test_targets_duplicate_cell_chooses_nearer_and_missing_motion_has_no_gradient():
    grid = BevGrid(x_min=0,y_min=0,rows=2,cols=2)
    visibility = np.array([[True,True],[False,True]])
    frame = {"next_dt_s":0.5,"boxes":[
        _box("early", (0.2,1.,0.), future=(2,1.,0.)), _box("near",future=None),
        _box("seen", (1.,3.,0.), future=(3.,3.,0.)), _box("not_seen", (3.,1.,0.)),
    ]}
    target = raster_targets(frame,visibility,grid)
    assert target[0].tolist() == [[True,True],[False,False]]
    assert target[1].tolist() == [[False,True],[False,False]]
    assert target[2][8,0,1] == pytest.approx(.5)  # (3-1)/0.5 / 8
    scores = torch.zeros(1,2,2,requires_grad=True)
    attrs = torch.zeros(1,10,2,2,requires_grad=True)
    loss = forecasting_loss((scores,attrs), tuple(torch.from_numpy(a)[None] for a in target),
                            torch.from_numpy(visibility)[None])
    loss.backward()
    assert attrs.grad[0,8,0,0].item() == 0
    assert attrs.grad[0,8,0,1].item() < 0
    assert scores.grad[0,1,0].item() == 0


def test_decode_boxes_keeps_only_supported_high_confidence_local_maxima():
    grid = BevGrid(x_min=0,y_min=0,rows=3,cols=3)
    score = torch.tensor([[[6.,5.,-1.],[4.,1.,-1.],[-1.,-1.,6.]]])
    attrs = torch.zeros((1,10,3,3))
    attrs[0,0,0,0] = .25
    attrs[0,7,0,0] = 1.
    attrs[0,8,0,0] = .25
    visible = torch.ones(3,3,dtype=torch.bool)
    visible[-1,-1] = False
    decoded, = decode_boxes((score,attrs),visible,grid,threshold=.7)
    assert decoded.center_xyz == pytest.approx((1.5,1.,0.))
    assert decoded.size_wlh == pytest.approx((1.9,4.5,1.7))
    assert decoded.velocity_xy == pytest.approx((2.,0.))


def test_control_keeps_geometry_only_and_xy_matching_is_one_to_one():
    values = np.ones((2,7,2,2),np.float16)
    control = _tensor(values,"geometry_only")
    assert not control[:,:2].any() and control[:,2:].all() and values.all()
    preds = [{"center_xyz":[0.,.2,1.]},{"center_xyz":[0.,.1,1.]}]
    targets = [{"center":[0.,0.,1.]},{"center":[4.,0.,1.]}]
    assert match_centers(preds,targets) == [(1,0,.1)]


def test_observation_only_prediction_precedes_test_targets(tmp_path):
    capture = tmp_path/"capture"
    grid = BevGrid(x_min=0,y_min=0,rows=2,cols=2)
    splits = {}
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for split in ("train","dev","test"):
            out = capture/split
            out.mkdir(parents=True)
            values = np.ones((1,7,2,2),np.float16)
            values[0,0,0,0] = 2
            np.save(out/"features.npy",values,allow_pickle=False)
            np.save(out/"coverage.npy",np.ones((1,2,2),np.bool_),allow_pickle=False)
            write_json(out/"observations.json",[{
                "sample_token":split,"snapshot_sha256":"0"*64,"forward_speed_mps_from_past":2.,
                "ref_time_us":1_000_000, "ego_world_xy_yaw":[0.,0.,0.],"scene":split,
            }])
            splits[split] = {"count":1,"shape":list(values.shape),
                             "artifacts":{f.name:artifact_hash(f) for f in out.iterdir()}}
        write_json(capture/"capture.json", {"schema":"hybrid-drive-fast-capture-v1","grid":asdict(grid),
                   "targets_opened":False,"revision":"a"*40,"splits":splits})
        for split in ("train","dev"):
            write_json(tmp_path/(split+".json"),{split:{"next_dt_s":.5,"boxes":[_box(split)]}})
        rev="a"*40
        fit_fast(capture,tmp_path/"train.json",tmp_path/"dev.json",tmp_path/"fit",revision=rev,epochs=1)
        assert not (tmp_path/"test.json").exists()
        predict_fast(capture,tmp_path/"fit",tmp_path/"blind",revision=rev)
        prediction = tmp_path/"blind"/"predictions.json"
        sealed = artifact_hash(prediction)
        write_json(tmp_path/"test.json",{"test":{"next_dt_s":.5,"boxes":[_box("test")]}})
        with pytest.raises(ValueError,match="differ from the digest"):
            score_fast(capture,prediction,tmp_path/"test.json",tmp_path/"wrong",sealed_sha256="0"*64)
        result = score_fast(capture,prediction,tmp_path/"test.json",tmp_path/"results",sealed_sha256=sealed)
        assert result["frames"] == result["planned_frames"] == result["scored_targets"] == 1
        assert result["sealed_predictions_sha256"] == artifact_hash(prediction)
        weights=tmp_path/"fit"/"vision.safetensors"
        with weights.open("ab") as h:
            h.write(b"bad")
        with pytest.raises(ValueError,match="weights changed"):
            predict_fast(capture,tmp_path/"fit",tmp_path/"changed",revision=rev)
    finally:
        torch.set_num_threads(old_threads)
