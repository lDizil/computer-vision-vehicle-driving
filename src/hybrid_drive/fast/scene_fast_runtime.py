"""Experimental Fast composition; not the stock agent and not a control API."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from math import atan2
from pathlib import Path

import torch
from safetensors.torch import load_file

from hybrid_drive.contracts.trajectory import Pose2D
from hybrid_drive.fast.forecast_proposals import ForecastObject, ForecastProposals, propose_from_forecast
from hybrid_drive.fast.ground_road_projection import ground_temporal_bev
from hybrid_drive.fast.metric_bev import BevGrid, past_forward_speed, temporal_bev
from hybrid_drive.fast.metric_forecast import MetricForecastHead, decode_boxes
from hybrid_drive.fast.pairwise_road_rank import PairwiseRoadRanking, rank_paired_road_candidates
from hybrid_drive.fast.road_geometry import RoadGeometryHead
from hybrid_drive.fast.scene_payload import FastScene, RoadEvidence
from hybrid_drive.neural.surround_features import SurroundFeatures


OBJECT_FIT_SHA = "870aae0d1afe69c0bcc740d9c9e7b2763c920a9b2962d486e25be62354c919c1"
OBJECT_WEIGHTS_SHA = "bb2e0946286a9c12cc67fdec943db76f8854c04e35a38d85f92429639abcff7d"
ROAD_FIT_SHA = "b737b753132b17521d02e15399895f8786e3fa6c62f54222c878d992f1894d2a"
ROAD_WEIGHTS_SHA = "a37d5bd5230026aebc566c1f204db25277a6cc21d707bdf3a3390ca02a445a14"
ROAD_MARGIN = .10


@dataclass(frozen=True)
class FastSceneResult:
    scene: FastScene
    base: ForecastProposals
    paired_road: PairwiseRoadRanking


class FrozenSceneFast:
    """Run the previously measured object and road heads on one camera bundle."""

    def __init__(self,box_fit: Path,road_fit: Path):
        _fit_record(box_fit, OBJECT_FIT_SHA)
        road_record=_fit_record(road_fit, ROAD_FIT_SHA)
        self.road_threshold=road_record["arms"]["ground_history"]["thresholds"]["drivable"]
        self.box_head=_load_head(MetricForecastHead(773),box_fit/"vision.safetensors",OBJECT_WEIGHTS_SHA)
        self.road_head=_load_head(RoadGeometryHead(773),road_fit/"ground_history.safetensors",ROAD_WEIGHTS_SHA)
        self.grid=BevGrid()

    @torch.no_grad()
    def __call__(self,current: SurroundFeatures,previous: SurroundFeatures) -> FastSceneResult:
        value,visible=temporal_bev(current,previous,self.grid)
        decoded=decode_boxes(self.box_head(_head_input(value)),visible,self.grid,threshold=.2)
        boxes=tuple(ForecastObject(**asdict(box)) for box in decoded)
        ground,current_support,prior_support=ground_temporal_bev(current,previous,self.grid)
        p=self.road_head(_head_input(ground)).sigmoid()[0].cpu().numpy()
        road=RoadEvidence(self.grid,p[0],p[1],(current_support|prior_support).cpu().numpy())
        snapshot=current.snapshot
        raw_speed=past_forward_speed(snapshot,previous.snapshot)
        if not -.10 <= raw_speed <= 60:
            raise ValueError("The diagnostic planner does not model reverse or out-of-envelope speeds")
        pose=snapshot.ego_to_world
        base=propose_from_forecast(snapshot.snapshot_id,snapshot.ref_time_us,
                                   Pose2D(float(pose[0,3]),float(pose[1,3]),atan2(pose[1,0],pose[0,0])),
                                   max(0,raw_speed),boxes)
        scene=FastScene(snapshot.snapshot_id,snapshot.content_sha256,snapshot.ref_time_us,boxes,base.plan,road)
        rank=rank_paired_road_candidates(
            scene.plan.candidates,{d.candidate_id:d.first_overlap_s is None for d in base.diagnostics},
            road.drivable,road.known,road.grid,offroad_threshold=self.road_threshold-ROAD_MARGIN,
            onroad_threshold=self.road_threshold+ROAD_MARGIN,
        )
        return FastSceneResult(scene,base,rank)


def _head_input(value):
    # Existing frozen fits were trained from the float16 capture, not raw float32.
    return value.to(torch.float16).to(torch.float32).unsqueeze(0)


def _fit_record(root,digest):
    raw=_checked(root/"fit.json",digest,256*1024)
    return json.loads(raw)


def _load_head(head,path,digest):
    _checked(path,digest,8*1024*1024)
    head.load_state_dict(load_file(path),strict=True)
    return head.eval().requires_grad_(False)


def _checked(path,digest,cap):
    if path.is_symlink() or not path.is_file() or not 0<path.stat().st_size<=cap:
        raise ValueError("Expected a bounded, regular frozen model artifact")
    data=path.read_bytes()
    if len(data)>cap or sha256(data).hexdigest()!=digest:
        raise ValueError(f"Frozen model artifact changed: {path.name}")
    return data
