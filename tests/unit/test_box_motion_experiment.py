from dataclasses import asdict

import pytest

from hybrid_drive.experiments.box_motion_inputs import load_motion_inputs, load_motion_snapshot
from hybrid_drive.experiments.box_motion_metrics import score_box_motion, summarize
from hybrid_drive.experiments.fast_capture import artifact_hash
from hybrid_drive.experiments.io import write_json
from hybrid_drive.fast.metric_bev import BevGrid


def _synthetic_cohort(*, recording_count=2, moving_each=6, quiet_speed=0.):
    grid = BevGrid(x_min=0,y_min=0,rows=2,cols=2)
    rows, targets = [], {}
    for i in range(recording_count):
        rec = f"recording-{i}"
        for j in range(moving_each+1):
            moving=j < moving_each
            token=f"{rec}-{j}"
            truth={"instance_id":"known","center":[1.,1.,.8],"size_wlh":[1.8,4.6,1.6],
                   "yaw":0.,"future_center":[2. if moving else 1.,1.,.8]}
            predicted={"object_id":"reused-cell","center_xyz":[1.4,1.,.8],"size_wlh":[1.8,4.6,1.6],
                       "yaw":0.,"confidence":.9,"velocity_xy":[4.,0.]}
            rows.append({"recording":rec,"sample_token":token,"boxes":[predicted],
                         "track_velocity_xy":[[2. if moving else quiet_speed,0.]],
                         "track_evidence":[{"object_id":"reused-cell","status":"estimated" if moving else "quiet"}],
                         "supported_cells":[0],"routing":None})
            targets[token]={"next_dt_s":.5,"boxes":[truth]}
    return {"schema":"hybrid-drive-box-motion-predictions-v1","revision":"1"*40,
            "heldout_targets_opened":False,"grid":asdict(grid),"rows":rows}, targets


def test_three_motion_modes_use_identical_target_matches_but_separate_position_and_displacement():
    pred,labels=_synthetic_cohort()
    r=summarize(pred,labels)
    group=r["overall"]["moving"]
    assert r["counts"]["matches"] == 14 and group["pairs"]==12
    assert group["stationary"]["displacement_error_m"] == pytest.approx(1.)
    assert group["frozen_learned"]["displacement_error_m"] == pytest.approx(1.)
    assert group["associated"]["displacement_error_m"] == 0.
    assert group["associated"]["next_center_error_m"] == pytest.approx(.4)
    assert group["stationary"]["next_center_error_m"] == pytest.approx(.6)
    assert r["gate"]["status"]=="passed_pilot"


@pytest.mark.parametrize(("nr","n","quiet","state"),[
    (1,12,0.,"insufficient_support"),(2,5,0.,"insufficient_support"),(2,6,.7,"failed_accuracy"),
])
def test_quality_gate_needs_two_recordings_enough_moving_matches_and_no_quiet_regression(nr,n,quiet,state):
    pred,labels=_synthetic_cohort(recording_count=nr,moving_each=n,quiet_speed=quiet)
    r=summarize(pred,labels)
    assert r["gate"]["status"]==state


def test_sealed_score_rejects_hash_before_reading_targets(tmp_path):
    output=tmp_path/"out"
    pred,labels=_synthetic_cohort()
    predictions,targets=tmp_path/"pred.json",tmp_path/"targets.json"
    write_json(predictions,pred)
    with pytest.raises(ValueError,match="differ from the seal"):
        score_box_motion(predictions,targets,output,sealed_sha256="0"*64)
    assert not targets.exists() and not output.exists()
    earlier_seal=artifact_hash(predictions)
    write_json(targets,labels)
    score=score_box_motion(predictions,targets,output,sealed_sha256=earlier_seal)
    assert score["prediction_seal"]==earlier_seal and score["targets_sha256"]==artifact_hash(targets)


def test_target_with_no_temporal_label_is_not_cost_free_motion_and_unobserved_targets_are_excluded():
    pred,labels=_synthetic_cohort()
    token=pred["rows"][0]["sample_token"]
    labels[token]["boxes"][0]["future_center"]=None
    labels[token]["boxes"].append({"instance_id":"hidden","center":[3.,3.,.8],"future_center":[3.,4.,.8]})
    score=summarize(pred,labels)
    assert score["counts"]["matches"]==14 and score["counts"]["targets"]==14
    assert score["counts"]["unsupported_targets"]==score["counts"]["moving_without_coverage"]==1
    assert score["counts"]["in_scope_annotated_targets"]==15
    assert score["overall"]["all"]["pairs"]==13
    assert score["gate"]["status"]=="insufficient_support"


def _index(positions=(1_000_000,1_500_000,2_000_000,2_500_000)):
    return {"schema":"hybrid-drive-box-motion-inputs-v1","source":{},
            "frames":{str(i):{"recording":"one","ref_time_us":t} for i,t in enumerate(positions)},
            "anchors":[{"recording":"one","sample_token":"3","history_tokens":["0","1","2"]}]}


def test_observation_windows_fail_closed_on_clock_or_recording_changes(tmp_path):
    f=tmp_path/"index.json"
    write_json(f,_index())
    assert load_motion_inputs(tmp_path)["anchors"][0]["sample_token"]=="3"
    bad=_index()
    bad["frames"]["1"]["recording"]="another"
    write_json(f,bad)
    with pytest.raises(ValueError,match="mix recording"):
        load_motion_inputs(tmp_path)
    write_json(f,_index((1_000_000,2_700_000,2_000_000,2_500_000)))
    with pytest.raises(ValueError,match="ordered, nearby"):
        load_motion_inputs(tmp_path)


def test_publisher_camera_order_checked_before_opening_image(tmp_path):
    with pytest.raises(ValueError,match="camera order"):
        load_motion_snapshot(tmp_path,"example",{"views":[{"camera":"CAM_R0"}]})
