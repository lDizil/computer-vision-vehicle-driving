from types import SimpleNamespace

import numpy as np
import pytest
import torch

from hybrid_drive.experiments.fast_capture import artifact_hash
from hybrid_drive.experiments.io import write_json
from hybrid_drive.experiments.road_scores import binary_metrics, check_quality, load_road_targets, select_cut
from hybrid_drive.experiments.road_trial import _inputs, _route, _score_routing, score_roads
from hybrid_drive.fast.metric_bev import BevGrid
from hybrid_drive.fast.road_geometry import positive_weights, road_loss


def test_unknown_area_has_no_gradient_for_both_layers_and_loss_is_finite():
    t = torch.tensor([[[[True, False, True]], [[False, True, True]]]])
    valid = torch.tensor([[[True, True, False]]])
    values = torch.zeros(1,2,1,3, requires_grad=True)
    road_loss(values, t, valid, positive_weights(t, valid)).backward()
    assert tuple(values.grad[0,:,0,-1].tolist()) == (0.,0.)
    assert values.grad[0,0,0,0].item() < 0 and values.grad[0,1,0,0].item() > 0
    with pytest.raises(ValueError,match="no supported"):
        road_loss(values, t, torch.zeros_like(valid), positive_weights(t, valid))


def test_metric_excludes_unsupported_cells_and_uses_dev_cut():
    p = np.array([.83,.30,.12,.91])
    t = np.array([True,True,False,False])
    known = np.array([True,True,True,False])
    initial = binary_metrics(p,t,known,.5)
    assert (initial["tp"],initial["fp"],initial["fn"],initial["tn"]) == (1,0,1,1)
    assert initial["iou"]==.5 and initial["f1"]==pytest.approx(2/3)
    cut,score = select_cut(p,t,known,"lane_divider")
    assert cut == .25 and score["f1"]==1
    with pytest.raises(ValueError, match="positive and negative"):
        select_cut(p,np.zeros(4,dtype=np.bool_),known,"lane_divider")


def test_quality_gates_are_independent_between_road_and_divider():
    vision = {"drivable":{"iou":.75},"lane_divider":{"f1":.45}}
    control = {"drivable":{"iou":.68},"lane_divider":{"f1":.44}}
    gates = check_quality(vision,control)
    assert gates["drivable"]["passed_pilot"] and not gates["lane_divider"]["passed_pilot"]
    vision["lane_divider"]["f1"] = None
    assert not check_quality(vision,control)["lane_divider"]["passed_pilot"]


def test_geometry_only_isolates_visual_signal():
    source = np.ones((1,8,2,2), dtype=np.float16)
    masked = _inputs(source,"geometry_only")
    assert masked.dtype==torch.float32
    assert not masked[0,:3].any() and masked[0,3:].all() and source.all()


def test_labels_reject_observation_order_change(tmp_path):
    p = tmp_path/"capture"
    train = p/"train"
    train.mkdir(parents=True)
    write_json(train/"observations.json",[{"sample_token":"t0"},{"sample_token":"t1"}])
    grid = {"x_min":-2,"y_min":-2,"step_m":2,"rows":2,"cols":2,"heights_m":[.85]}
    write_json(p/"capture.json",{"grid":grid})
    target = tmp_path/"target"
    target.mkdir()
    mask = np.ones((2,2,2),dtype=np.bool_)
    np.savez(target/"targets.npz",labels=np.ones((2,2,2,2),dtype=np.bool_),known=mask)
    m={"schema":"hybrid-drive-road-map-targets-v1","observation_sha256":artifact_hash(train/"observations.json"),
       "target_sha256":artifact_hash(target/"targets.npz"),"tokens":["t0","t1"],"locations":["Boston","Boston"],
       "capture_sha256":artifact_hash(p/"capture.json"),"grid":grid,"channels":["drivable_area","lane_divider"],
       "map_sha256":"2"*64}
    info=SimpleNamespace(coverage=mask,observations=[{"sample_token":"t0"},{"sample_token":"t1"}])
    write_json(target/"targets.json",m)
    assert load_road_targets(target,p,"train",info).map_known.all()
    m["grid"]={**grid,"step_m":1}
    write_json(target/"targets.json",m)
    with pytest.raises(ValueError,match="not bound"):
        load_road_targets(target,p,"train",info)
    m["grid"]=grid
    m["tokens"].reverse()
    write_json(target/"targets.json",m)
    with pytest.raises(ValueError,match="not bound"):
        load_road_targets(target,p,"train",info)


def test_forecast_checked_before_opening_inputs_or_holdout_labels(tmp_path):
    observed=tmp_path/"capture"
    labels=tmp_path/"map"
    payload=tmp_path/"predictions.json"
    write_json(payload,{})
    with pytest.raises(ValueError,match="differ from the seal"):
        score_roads(observed,payload,labels,tmp_path/"boxes",tmp_path/"out",sealed_sha256="0"*64)
    assert not observed.exists() and not labels.exists() and not (tmp_path/"out").exists()


def test_road_ranking_changes_only_order_and_offroad_count_on_frozen_poses():
    grid = BevGrid(x_min=-4,y_min=-8,rows=12,cols=12)
    _,y=grid.xy()
    road=np.abs(y)<=2
    known=np.ones((12,12),dtype=np.bool_)
    original={"status":"proposed_not_safety_checked","interval_s":.5,"candidates":[
        {"candidate_id":"lateral","poses":[[2,4,0],[4,6,0],[6,8,0]],"diagnostic":{"first_overlap_s":None}},
        {"candidate_id":"keep","poses":[[2,0,0],[4,0,0],[6,0,0]],"diagnostic":{"first_overlap_s":None}},
    ]}
    anchor={"sample_token":"t0","snapshot_sha256":"1"*64}
    route = _route(anchor,road.astype(np.float32),known,original,grid,.5)
    assert route["base_order"]==("lateral","keep") and route["new_order"]==("keep","lateral")
    tg=SimpleNamespace(labels=np.stack((road,road))[None],map_known=known[None])
    capture=SimpleNamespace(observations=[anchor],grid=grid)
    result = _score_routing({"routing":[route]},[{"proposals":original}],tg,capture)
    assert result["first_changed"]==1 and result["before_offroad_frames"]==1 and result["after_offroad_frames"]==0
