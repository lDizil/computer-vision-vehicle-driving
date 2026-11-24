from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from hybrid_drive.contracts.planning import Candidate, CandidateSet
from hybrid_drive.contracts.trajectory import Pose2D, Trajectory
from hybrid_drive.experiments.io import write_json
from hybrid_drive.experiments.paired_road_replay import _preserves_obstacle_blocks, score_paired_replay
from hybrid_drive.fast.forecast_proposals import ForecastObject
from hybrid_drive.fast.scene_payload import FastScene, RoadEvidence, SceneGrid
from hybrid_drive.neural.scene_slow_prefix import SceneSlowPrefix


def _scene():
    shape=(2,2)
    road=RoadEvidence(SceneGrid(-1,-1,1,*shape),np.full(shape,.7),np.zeros(shape),np.ones(shape,dtype=bool))
    plan=CandidateSet("plan-1","snapshot-1",1.,Pose2D(0,0,0),(
        Candidate("c1",Trajectory([[1,0,0],[2,0,0]],.5)),
        Candidate("c2",Trajectory([[1,.4,0],[2,.4,0]],.5)),
    ))
    target=ForecastObject("x",(2,0,1),(2,3,1),0,(0,0),1)
    return FastScene("snapshot-1","6"*64,1_000_000,(target,),plan,road)


def test_scene_identity_binds_immutable_road_pose_and_box_geometry():
    x = _scene()
    p=x.road.drivable.copy()
    q=p.copy()
    r=np.ones_like(p,dtype=np.bool_)
    episode=replace(x,road=RoadEvidence(x.road.grid,q,q,r))
    before=episode.content_sha256
    q[:]=0
    r[:]=False
    assert episode.content_sha256==before and episode.road.known.all()
    assert np.array_equal(episode.road.drivable,p)
    with pytest.raises(ValueError,match="read-only"):
        episode.road.drivable[0,0]=0
    assert replace(x,boxes=(replace(x.boxes[0],velocity_xy=(1,0)),)).content_sha256!=x.content_sha256
    assert replace(x,road=episode.road).content_sha256!=x.content_sha256
    with pytest.raises(ValueError,match="another snapshot/time"):
        replace(x,ref_time_us=1_000_001)
    with pytest.raises(ValueError,match="Duplicate object"):
        replace(x,boxes=x.boxes+x.boxes)


def test_scene_grid_is_half_open_even_at_float_neighbour():
    grid=SceneGrid(-1,-1,1,2,2)
    assert grid.index(1.,0.) is None
    assert grid.index(np.nextafter(1.,-np.inf),np.nextafter(1.,-np.inf))==(1,1)
    assert grid.index(-1.,-1.)==(0,0)
    assert grid.index(np.nan,0.) is None


def _tokens(bank):
    return SimpleNamespace(
        prior_tokens=bank, prior_mask=torch.tensor([[True,True,True,True,False]]),
        visual_tokens=torch.zeros((1,4,8)),visual_mask=torch.ones((1,4),dtype=torch.bool),
        records=tuple(SimpleNamespace(kind=kind,candidate_id=cid) for kind,cid in (
            ("actor",None),("trajectory","left"),("trajectory","right"),("road","right"),("road","left")
        )),scene_sha256="1"*64,
    )


def test_candidate_query_keeps_shared_actors_and_masks_other_candidate_pose_and_road():
    probe=SceneSlowPrefix(visual_dim=8,prior_dim=8,llm_dim=16).eval()
    values=torch.arange(40,dtype=torch.float32).reshape(1,5,8)/40
    _,mask=probe.project(_tokens(values),"left")
    assert mask.tolist()==[[True,True,False,False,False]]
    assert probe.project(_tokens(values),"right")[1].tolist()==[[True,False,True,True,False]]
    predicted=probe.project(_tokens(values),"left")[0]
    changed=values.clone()
    changed[0,2:]+=4
    torch.testing.assert_close(probe.project(_tokens(changed),"left")[0],predicted,rtol=0,atol=0)
    changed[0,1,0]+=5
    assert not torch.equal(probe.project(_tokens(changed),"left")[0],predicted)
    with pytest.raises(ValueError,match="No valid pose"):
        probe.project(_tokens(values),"missing")
    invalid=_tokens(values)
    invalid.records=(*invalid.records[:3],SimpleNamespace(kind="road",candidate_id=None),invalid.records[-1])
    with pytest.raises(ValueError,match="Invalid binding"):
        probe.project(invalid,"left")


def test_score_rejects_wrong_seal_before_opening_external_map(tmp_path):
    replay=tmp_path/"replay.json"
    write_json(replay,{})
    truth=tmp_path/"not-present"
    with pytest.raises(ValueError,match="differs from receipt"):
        score_paired_replay(truth,truth,replay,truth,tmp_path/"out",sealed_sha256="1"*64)
    assert not truth.exists() and not (tmp_path/"out").exists()



def test_score_guard_cannot_swap_across_an_obstacle_or_permute_conflict_slots():
    start=["free-a","free-b","blocked-a","blocked-b","free-c"]
    permitted={key:key.startswith("free") for key in start}
    assert _preserves_obstacle_blocks(start,["free-b","free-a",*start[2:]],permitted)
    assert not _preserves_obstacle_blocks(start,["free-c","free-b","blocked-a","blocked-b","free-a"],permitted)
    assert not _preserves_obstacle_blocks(start,[*start[:2],"blocked-b","blocked-a","free-c"],permitted)
