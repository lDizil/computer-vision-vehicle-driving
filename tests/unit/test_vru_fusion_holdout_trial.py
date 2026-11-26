from types import SimpleNamespace

import numpy as np
import pytest

from hybrid_drive.experiments.object_vru_geometry import GroundHit, GroundObject
from hybrid_drive.experiments import vru_fusion_holdout_trial as trial


def ground(key, x, y):
    xy = (x, y)
    hit = GroundHit("hit", "person", "front", xy, (0.,0.,1.5), .8, .2, (1.,)+(0.,)*35)
    return GroundObject(key, "person", xy, .8, (hit,))


def test_all_three_arms_use_common_frame_and_only_move_after_ego_alignment(monkeypatch):
    base = {"sample_token":"now", "previous_token":"past", "older_token":"older", "recording":"drive"}
    stamps = {"now":(1.,1_000_000,0.), "past":(.5,500_000,-.5), "older":(0.,0,-1.)}
    images = {}
    for key, (world_x, stamp, y) in stamps.items():
        pose = np.eye(4)
        pose[0,3] = world_x
        images[key] = SimpleNamespace(snapshot_id=key, ego_to_world=pose, ref_time_us=stamp,
                                      content_sha256={"now":"1", "past":"2", "older":"3"}[key]*64)
    monkeypatch.setattr(trial,"load_motion_snapshot",lambda root,key,entry,rectified_wh:images[key])

    def detector(offset):
        def detect(image):
            assert image is images[image.snapshot_id]
            world_x,_,y = stamps[image.snapshot_id]
            return (ground("d0",11.-world_x+offset,y),), {}
        return detect

    rows = trial._capture_window(None,{"frames":{key:{} for key in stamps}},base,detector(0.),detector(3.))
    assert [len(rows[arm]["objects"]) for arm in trial.ARMS] == [1,1,2]
    assert [rows[arm]["older_history_s"] for arm in trial.ARMS]==[.5,.5,.5]
    for row in rows.values():
        assert row["history_s"]==.5 and row["current_sha256"]=="1"*64
        assert all(x["confirmation"]["status"]=="confirmed" for x in row["objects"])
        assert all(x["forecasts"]["three_frame_trend"]==pytest.approx((0.,1.)) for x in row["objects"])


def test_missing_third_frame_keeps_current_object_and_distinguishes_unproven_speed():
    current = (ground("n",10.,0.),)
    history = (ground("p",10.,-.5),)
    row, = trial._track(current,history,None,.5,None)
    assert row["xy_m"]==(10.,0.) and row["motions"]["appearance"]["status"]=="estimated"
    assert row["confirmation"]["status"]=="no_older_observation"
    assert row["forecasts"]["two_frame"]==(0.,1.) and row["forecasts"]["three_frame_trend"]==(0.,0.)


def test_wrong_seal_is_rejected_before_labels_or_inputs_can_be_read(tmp_path):
    p=tmp_path/"capture.json"
    p.write_bytes(b"{}")
    with pytest.raises(ValueError, match="seal fixed before"):
        trial.score_fusion_holdout(tmp_path/"missing-inputs",p,tmp_path/"missing-labels",
                                   tmp_path/"score",sealed_sha256="0"*64)
