from dataclasses import asdict

import numpy as np
import pytest

from hybrid_drive.contracts.trajectory import Pose2D
from hybrid_drive.experiments.ground_road_scores import ARMS, quality_gate, score_layers, score_routes
from hybrid_drive.experiments.ground_road_trial import GroundRole, _rank_frame, score_ground_road
from hybrid_drive.experiments.io import write_json
from hybrid_drive.fast.forecast_proposals import ForecastObject, propose_from_forecast
from hybrid_drive.fast.metric_bev import BevGrid


def test_common_support_does_not_give_accuracy_credit_for_newly_visible_cells():
    one = np.array([[[True, False, True, True]]])
    masks = {"object_current":one, "ground_current":np.array([[[True, True, False, True]]]),
             "ground_history":np.ones_like(one)}
    labels = np.array([[[[True, True, False, True]],[[False,True,True,False]]]])
    cuts = {a:{"drivable":.5,"lane_divider":.5} for a in ARMS}
    value = np.array([[[[.9, .1, .9, .9]],[[.1,.9,.9,.1]]]],dtype=np.float32)
    guesses = {a:value for a in ARMS}
    modes,common,gain,extra=score_layers(guesses,masks,labels,np.ones_like(one),cuts)
    assert common==2 and gain==1
    assert all(summary["common"]["drivable"]["iou"]==1 for summary in modes.values())
    assert modes["ground_history"]["supported_cells"]==4
    assert modes["ground_history"]["own"]["drivable"]["iou"]==.5
    assert extra["drivable"]["fp"]==1


def test_more_covered_cells_by_itself_does_not_pass_quality_threshold():
    def score(iou,f1,n):
        return {"common":{"drivable":{"iou":iou},"lane_divider":{"f1":f1}},
                "supported_cells":n}
    report={"object_current":score(.59,.34,10),"ground_current":score(.61,.35,11),
            "ground_history":score(.68,.33,12)}
    assert quality_gate(report)["passes_quality_pilot"]
    report["ground_history"]=score(.62,.33,100)
    assert not quality_gate(report)["passes_quality_pilot"]
    report["ground_history"]=score(.68,.26,100)
    assert not quality_gate(report)["passes_quality_pilot"]


def test_wrong_prediction_receipt_is_rejected_without_loading_targets(tmp_path):
    pred=tmp_path/"pred"
    pred.mkdir()
    write_json(pred/"predictions.json",{})
    with pytest.raises(ValueError,match="before opening external"):
        score_ground_road(tmp_path/"capture",pred,tmp_path/"labels",tmp_path/"out",sealed_sha256="0"*64)
    assert not (tmp_path/"labels").exists() and not (tmp_path/"out").exists()


def test_history_support_changes_diagnostic_order_only_on_same_geometry():
    grid=BevGrid(x_min=-4,y_min=-6,step_m=1,rows=26,cols=12)
    x,_=grid.xy()
    support=np.ones((1,26,12),dtype=np.bool_)
    current=support.copy()
    current[:,3,:]=False  # Missing at the shared t=0 rear footprint.
    past=~current
    road=(x<6).astype(np.float32)
    probabilities={name:np.stack((road,road))[None] for name in ARMS}
    cuts={a:{"drivable":.5,"lane_divider":.5} for a in ARMS}
    obs={"sample_token":"t0","snapshot_sha256":"0"*64,"recording":"log","ego_world_xy_yaw":[0.,0.,0.],
         "ref_time_us":2_000_000,"forward_speed_mps_from_past":2.5}
    obj=ForecastObject("vehicle",(20.,4.,.6),(2.,4.,1.5),0.,(0.,0.),.8)
    base=propose_from_forecast("t0",2_000_000,Pose2D(0.,0.,0.),2.5,(obj,))
    first=[c.candidate_id for c in base.plan.candidates]
    frozen={"boxes":[asdict(obj)],"routing":{"frozen_learned":{"ordered":first,"all_overlap":base.all_candidates_overlap}}}
    test=GroundRole([obs],np.empty(0),current,past,old_seen=current)
    route=_rank_frame(obs,frozen,test,0,probabilities,cuts,grid)
    for arm in ARMS:
        route["rankings"][arm]["ordered"]=list(route["rankings"][arm]["ordered"])
    assert route["rankings"]["ground_current"]["ordered"]==first
    assert route["rankings"]["ground_history"]["ordered"]!=first
    labels=np.stack((road.astype(bool),road.astype(bool)))[None]
    metrics=score_routes([route],labels,support,grid)
    assert metrics["ground_current"]["comparable_frames"]==0
    assert metrics["ground_history"]["comparable_frames"]==1
    assert metrics["ground_history"]["original_map_offroad"]==1
    assert metrics["ground_history"]["ranked_map_offroad"]==0
