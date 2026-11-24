"""Replay #45 scenes without map labels, then compare against a separate map oracle."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from itertools import groupby
from pathlib import Path

import numpy as np

from hybrid_drive.contracts.planning import Candidate
from hybrid_drive.contracts.trajectory import Trajectory
from hybrid_drive.experiments.bounded_io import require_hex
from hybrid_drive.experiments.fast_capture import artifact_hash, load_json
from hybrid_drive.experiments.io import write_json
from hybrid_drive.fast.pairwise_road_rank import rank_paired_road_candidates
from hybrid_drive.fast.road_candidate_rank import assess_road
from hybrid_drive.fast.scene_payload import SceneGrid
from hybrid_drive.fast.scene_fast_runtime import ROAD_MARGIN


SOURCE_ROAD_PRED_SHA="e8f0b1ac9764e41524f857a9a8bbe49d5af62b6392f1d7eadba20b5f2f00c244"
SOURCE_ROAD_CAPTURE_SHA="f53cd694fb13c31904d7912985f867dd9157c416320de1ac3aaed166c1d0310f"
SOURCE_EXTERNAL_MAP_MANIFEST_SHA="9a6a3cbe231f5ad8ed7eccd82b5ae3878821cd6b9742b775e22dfb3e87aeb977"
SOURCE_EXTERNAL_MAP_DATA_SHA="febac3b6f9389d083ed06441188e86784cc205753f5189ac1f5305abe716b589"


@dataclass(frozen=True)
class RoadReplay:
    grid: SceneGrid
    rows: list
    probabilities: np.ndarray
    coverage: np.ndarray
    threshold: float


def load_frozen_road_replay(ground_predictions: Path, ground_capture: Path) -> RoadReplay:
    p, c = ground_predictions/"predictions.json", ground_capture/"capture.json"
    _check_sha(p,SOURCE_ROAD_PRED_SHA)
    _check_sha(c,SOURCE_ROAD_CAPTURE_SHA)
    src,meta=load_json(p),load_json(c)
    if src["target_labels_opened"] is not False or src["capture_sha256"]!=SOURCE_ROAD_CAPTURE_SHA:
        raise ValueError("Unexpected #45 road proof")
    values=ground_predictions/"ground_history.npy"
    arm=src["arms"]["ground_history"]
    _check_sha(values,arm["probability_sha256"])
    masks=[]
    for name in ("current_coverage.npy","history_coverage.npy"):
        pth=ground_capture/"test"/name
        _check_sha(pth,meta["roles"]["test"]["artifacts"][name])
        masks.append(np.load(pth,allow_pickle=False))
    grid=SceneGrid(**{k:meta["grid"][k] for k in ("x_min","y_min","step_m","rows","cols")})
    probabilities=np.load(values,allow_pickle=False)
    shape=(len(src["routes"]),grid.rows,grid.cols)
    if (any(v.shape!=shape or v.dtype!=np.bool_ for v in masks)
            or probabilities.shape!=(shape[0],2,*shape[1:]) or probabilities.dtype!=np.float32):
        raise ValueError("Unexpected #45 shapes")
    return RoadReplay(grid,src["routes"],probabilities,masks[0]|masks[1],arm["thresholds"]["drivable"])


def _check_sha(path,digest):
    if path.is_symlink() or not path.is_file() or not 0<path.stat().st_size<=12*1024*1024:
        raise ValueError("Invalid bounded replay artifact")
    if artifact_hash(path)!=digest:
        raise ValueError(f"Frozen replay changed: {path.name}")


def _plans(row):
    values=tuple(Candidate(key,Trajectory(data["poses"],row["interval_s"]))
                 for key,data in row["candidates"].items())
    return values,{key:doc["obstacle_free"] for key,doc in row["candidates"].items()}


def predict_paired_replay(ground_predictions,ground_capture,output,*,revision):
    require_hex(revision,40,"revision")
    if output.exists():
        raise ValueError("Output directory must be new")
    frozen=load_frozen_road_replay(ground_predictions,ground_capture)
    rows=[]
    t=frozen.threshold
    for i,row in enumerate(frozen.rows):
        candidates,obstacles=_plans(row)
        result=rank_paired_road_candidates(candidates,obstacles,frozen.probabilities[i,0],frozen.coverage[i],
                                           frozen.grid,offroad_threshold=t-ROAD_MARGIN,onroad_threshold=t+ROAD_MARGIN)
        rows.append({
            "sample_token":row["sample_token"],"snapshot_sha256":row["snapshot_sha256"],
            "original":[c.candidate_id for c in candidates],"ordered":list(result.ordered),
            "initial_unknown":result.initial_unknown,"initial_offroad":result.initial_offroad,
            "cycle_detected":result.cycle_detected,"comparisons":[asdict(value) for value in result.comparisons],
        })
    output.mkdir(parents=True)
    write_json(output/"replay.json",{
        "schema":"hybrid-drive-paired-road-replay-v1","revision":revision,
        "targets_opened":False,"inputs_sha256":SOURCE_ROAD_PRED_SHA,"rows":rows,
    })


def score_paired_replay(ground_predictions,ground_capture,replay,targets,output,*,sealed_sha256):
    require_hex(sealed_sha256,64,"paired replay seal")
    if output.exists():
        raise ValueError("Output directory must be new")
    if artifact_hash(replay)!=sealed_sha256:
        raise ValueError("Paired replay differs from receipt fixed before label access")
    result=load_json(replay)
    if (result["schema"]!="hybrid-drive-paired-road-replay-v1" or result["targets_opened"] is not False
            or result["inputs_sha256"]!=SOURCE_ROAD_PRED_SHA):
        raise ValueError("Not a sealed label-free replay of #45")
    frozen=load_frozen_road_replay(ground_predictions,ground_capture)
    map_targets=targets/"targets.npz"
    _check_sha(targets/"targets.json",SOURCE_EXTERNAL_MAP_MANIFEST_SHA)
    _check_sha(map_targets,SOURCE_EXTERNAL_MAP_DATA_SHA)
    info=load_json(targets/"targets.json")
    if [row["sample_token"] for row in frozen.rows]!=info["tokens"]:
        raise ValueError("Map labels differ from frozen replay anchors")
    with np.load(map_targets,allow_pickle=False) as archive:
        truth,known=archive["labels"],archive["known"]
    if (truth.shape!=frozen.probabilities.shape or truth.dtype!=np.bool_
            or known.shape!=frozen.coverage.shape or known.dtype!=np.bool_):
        raise ValueError("Map label shape mismatch")
    stats=_compare_replay(result["rows"],frozen,truth,known)
    payload={
        "schema":"hybrid-drive-paired-road-score-v1","revision":result["revision"],
        "sealed_replay_sha256":sealed_sha256,"map_sha256":SOURCE_EXTERNAL_MAP_DATA_SHA,
        "scope":"retrospective_offline_not_heldout_or_safety",
        **stats,
    }
    output.mkdir(parents=True)
    write_json(output/"score.json",payload)
    return payload


def _compare_replay(rows,frozen,truth,known):
    counts=Counter(frames=0,common_initial_unknown=0,learned_comparable_frames=0,learned_directional_frames=0,
                   learned_first_changed=0,old_first_footprint_outside=0,paired_first_footprint_outside=0,
                   oracle_comparable_frames=0,oracle_directional_frames=0,oracle_first_changed=0,
                   oracle_first_footprint_outside=0)
    for i,(row,src) in enumerate(zip(rows,frozen.rows,strict=True)):
        if any(row[k]!=src[k] for k in ("sample_token","snapshot_sha256")):
            raise ValueError("Replay snapshot order differs from frozen #45")
        candidates,obstacle_free=_plans(src)
        ids=[c.candidate_id for c in candidates]
        by_id={c.candidate_id:c for c in candidates}
        if row["original"]!=ids or sorted(row["ordered"])!=sorted(ids):
            raise ValueError("Replay did not preserve the candidate set")
        if not _preserves_obstacle_blocks(ids,row["ordered"],obstacle_free):
            raise ValueError("Replay crossed an obstacle barrier or moved a conflict slot")
        counts["frames"]+=1
        counts["common_initial_unknown"]+=row["initial_unknown"]
        counts["learned_comparable_frames"]+=any(c["comparable_samples"]>0 for c in row["comparisons"])
        counts["learned_directional_frames"]+=any(c["winner_id"] is not None for c in row["comparisons"])
        counts["learned_first_changed"]+=row["ordered"][0]!=ids[0]
        drive=truth[i,0]
        # The oracle is diagnostic only: retain the exact same optical support.
        oracle=rank_paired_road_candidates(candidates,obstacle_free,drive,frozen.coverage[i]&known[i],frozen.grid,
                                           offroad_threshold=.25,onroad_threshold=.75)
        counts["oracle_comparable_frames"]+=any(c.comparable_samples>0 for c in oracle.comparisons)
        counts["oracle_directional_frames"]+=any(c.winner_id is not None for c in oracle.comparisons)
        counts["oracle_first_changed"]+=oracle.ordered[0]!=ids[0]
        for label,key in (("old",ids[0]),("paired",row["ordered"][0]),("oracle",oracle.ordered[0])):
            check=assess_road(by_id[key].trajectory,drive,known[i],frozen.grid,
                              offroad_threshold=.25,onroad_threshold=.75)
            counts[label+"_first_footprint_outside"]+=check.offroad_samples>0
    return dict(counts)



def _preserves_obstacle_blocks(original, ordered, obstacle_free):
    for free,group in groupby(enumerate(original),key=lambda item: obstacle_free[item[1]]):
        slots=tuple(group)
        before=[key for _,key in slots]
        after=[ordered[i] for i,_ in slots]
        if (set(before) != set(after) if free else before != after):
            return False
    return True
