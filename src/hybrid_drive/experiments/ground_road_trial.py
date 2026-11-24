"""Train and compare ground-plane/temporal road evidence on an external cohort."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from hybrid_drive.contracts.trajectory import Pose2D
from hybrid_drive.experiments.bounded_io import require_hex
from hybrid_drive.experiments.fast_capture import artifact_hash, load_capture, load_json
from hybrid_drive.experiments.ground_road_scores import ARMS, load_external_targets, quality_gate, score_layers, score_routes
from hybrid_drive.experiments.io import write_json
from hybrid_drive.experiments.road_scores import load_road_targets, select_cut
from hybrid_drive.experiments.road_trial import fit_captured_road_head, predict_captured_road_head
from hybrid_drive.fast.forecast_proposals import ForecastObject, propose_from_forecast
from hybrid_drive.fast.metric_bev import BevGrid
from hybrid_drive.fast.road_candidate_rank import rank_road_candidates
from hybrid_drive.fast.road_geometry import LAYERS, RoadGeometryHead, positive_weights


BASE_FIT_SHA = "076587b8ba5ab05918c774f390ca7c63a89f53893d20863fbf84978e6c2b0c75"
BASE_WEIGHTS_SHA = "34ae017f24f18650f3259598212b86cda3e8182c84c58a0b58217164c53c4129"
BASE_MOTION_SHA = "6d9c321a719bdce30686ed8d4c64658f179984fee1c067333b0e0cb6e2ca5017"
NEW_ARMS = ARMS[1:]
ROAD_CONFIDENCE_MARGIN = .10


@dataclass(frozen=True)
class GroundRole:
    observations: list[dict]
    values: np.ndarray
    current_seen: np.ndarray
    past_seen: np.ndarray
    old_values: np.ndarray | None = None
    old_seen: np.ndarray | None = None

    def coverage(self, arm):
        if arm == "object_current" and self.old_seen is not None:
            return self.old_seen
        if arm == "ground_history":
            return self.current_seen | self.past_seen
        if arm == "ground_current":
            return self.current_seen
        raise ValueError("Unknown or unavailable projection arm")


@dataclass(frozen=True)
class Features:
    values: np.ndarray
    observations: list[dict]


def capture_header(root):
    meta = load_json(root / "capture.json")
    if (meta["schema"] != "hybrid-drive-ground-road-capture-v1" or meta["test_targets_opened"] is not False
            or meta["ground_plane_m"] != 0):
        raise ValueError("Unknown or future-conditioned ground capture")
    return meta


def capture_role(root: Path, header: dict, role: str) -> GroundRole:
    folder = root / role
    spec = header["roles"][role]
    paths = ["features_ground.npy","current_coverage.npy","history_coverage.npy","observations.json"]
    if role=="test":
        paths += ["features_object.npy","object_coverage.npy"]
    for name in paths:
        item=folder/name
        if item.is_symlink() or not 0 < item.stat().st_size <= 512*1024*1024:
            raise ValueError(f"Ground capture {role}/{name} is not a bounded regular artifact")
        if artifact_hash(item)!=spec["artifacts"][name]:
            raise ValueError(f"Ground capture {role}/{name} changed")
    def read(name):
        return np.load(folder/name,mmap_mode="r",allow_pickle=False)
    obs=load_json(folder/"observations.json")
    values,current,prior=(read(k) for k in paths[:3])
    shape=(spec["count"],header["grid"]["rows"],header["grid"]["cols"])
    if (len(obs)!=shape[0] or not len(obs) or values.shape!=(shape[0],773,*shape[1:])
            or values.dtype!=np.float16 or not _valid_masks(shape,current,prior)):
        raise ValueError("Wrong ground feature capture shapes")
    if role=="test":
        objects,old=read(paths[-2]),read(paths[-1])
        if objects.shape!=values.shape or objects.dtype!=values.dtype or not _valid_masks(shape,old):
            raise ValueError("Wrong frozen object feature shapes")
        return GroundRole(obs,values,current,prior,objects,old)
    return GroundRole(obs,values,current,prior)


def _valid_masks(shape,*masks):
    return all(m.shape==shape and m.dtype==np.bool_ for m in masks)


def fit_ground_road(capture, mini_capture, train_targets, dev_targets, output, *, revision):
    require_hex(revision,40,"revision")
    if output.exists():
        raise ValueError("Output must be a new directory")
    header=capture_header(capture)
    if header["revision"]!=revision or header["source_mini_capture_sha256"]!=artifact_hash(mini_capture/"capture.json"):
        raise ValueError("Training ground capture and Mini receipts do not match")
    train,dev=(capture_role(capture,header,role) for role in ("train","dev"))
    old_train,old_dev=(load_capture(mini_capture,role) for role in ("train","dev"))
    for role,a,b in (("train",train,old_train),("dev",dev,old_dev)):
        if a.observations!=b.observations:
            raise ValueError(f"Mini {role} anchors changed")
    tr=load_road_targets(train_targets,mini_capture,"train",old_train)
    dv=load_road_targets(dev_targets,mini_capture,"dev",old_dev)
    if tr.map_sha256!=dv.map_sha256:
        raise ValueError("Different Mini map releases in train and dev")
    output.mkdir(parents=True)
    fitted={}
    labels=torch.from_numpy(tr.labels)
    for name in NEW_ARMS:
        train_support=torch.from_numpy(train.coverage(name)&tr.map_known)
        pos=positive_weights(labels,train_support)
        model,_=fit_captured_road_head(train,labels,train_support,pos,"vision",18)
        probabilities=predict_captured_road_head(model,dev,"vision")
        mask=dev.coverage(name)&dv.map_known
        selection={}
        for i,layer in enumerate(LAYERS):
            cut,stats=select_cut(probabilities[:,i],dv.labels[:,i],mask,layer)
            selection[layer]={"threshold":cut,"scores":stats}
        path=output/(name+".safetensors")
        save_file(model.state_dict(),path)
        fitted[name]={"thresholds":{layer:s["threshold"] for layer,s in selection.items()},
                      "dev":selection,"weights_sha256":artifact_hash(path),
                      "train_supported_cells":int(train_support.sum())}
        print(f"{name}: dev IoU={selection['drivable']['scores']['iou']:.4f}; "
              f"divider F1={selection['lane_divider']['scores']['f1']:.4f}",flush=True)
    write_json(output/"fit.json", {"schema":"hybrid-drive-ground-road-fit-v1","revision":revision,
                                  "capture_sha256":artifact_hash(capture/"capture.json"),
                                  "original_map_sha256":tr.map_sha256,"external_targets_opened":False,
                                  "train_labels_sha256":artifact_hash(train_targets/"targets.json"),
                                  "dev_labels_sha256":artifact_hash(dev_targets/"targets.json"),
                                  "epochs":18,"arms":fitted})


def predict_ground_road(capture, fitted, baseline, objects, output, *, revision):
    require_hex(revision,40,"revision")
    if output.exists():
        raise ValueError("Output must be a new directory")
    header=capture_header(capture)
    learned=load_json(fitted/"fit.json")
    if (learned["schema"]!="hybrid-drive-ground-road-fit-v1" or learned["revision"]!=revision
            or header["revision"]!=revision or learned["external_targets_opened"] is not False
            or learned["capture_sha256"]!=artifact_hash(capture/"capture.json")):
        raise ValueError("Wrong frozen ground fits or capture")
    baseline_spec=_baseline(baseline)
    test=capture_role(capture,header,"test")
    rows=_same_motion_geometry(objects,test.observations)
    grid=_grid(header)
    output.mkdir(parents=True)
    all_probabilities,cuts,meta={},{},{}
    for name in ARMS:
        if name=="object_current":
            model=_load(baseline/"vision.safetensors",BASE_WEIGHTS_SHA)
            data=Features(test.old_values,test.observations)
            cuts[name]=baseline_spec["thresholds"]
        else:
            arm=learned["arms"][name]
            model=_load(fitted/(name+".safetensors"),arm["weights_sha256"])
            data=Features(test.values,test.observations)
            cuts[name]=arm["thresholds"]
        value=predict_captured_road_head(model,data,"vision")
        all_probabilities[name]=value
        path=output/(name+".npy")
        np.save(path,value,allow_pickle=False)
        meta[name]={"thresholds":cuts[name],"probability_sha256":artifact_hash(path)}
    routes=[]
    for i,(anchor,old) in enumerate(zip(test.observations,rows,strict=True)):
        routes.append(_rank_frame(anchor,old,test,i,all_probabilities,cuts,grid))
    write_json(output/"predictions.json", {
        "schema":"hybrid-drive-ground-road-predictions-v1","revision":revision,
        "target_labels_opened":False,"capture_sha256":learned["capture_sha256"],
        "fit_sha256":artifact_hash(fitted/"fit.json"),"baseline_fit_sha256":BASE_FIT_SHA,
        "source_index_sha256":header["source_test_index_sha256"],"object_forecast_sha256":BASE_MOTION_SHA,
        "arms":meta,"routes":routes,
    })


def _baseline(root):
    if artifact_hash(root/"fit.json")!=BASE_FIT_SHA:
        raise ValueError("Expected the exact frozen #44 road baseline")
    check=load_json(root/"fit.json")
    return check["arms"]["vision"]


def _same_motion_geometry(path,observations):
    if artifact_hash(path)!=BASE_MOTION_SHA:
        raise ValueError("Expected the exact sealed #43 box predictions")
    payload=load_json(path)
    if payload["heldout_targets_opened"] is not False:
        raise ValueError("Box forecasts were not frozen before target access")
    rows=payload["rows"]
    for box, obs in zip(rows,observations,strict=True):
        if any(box[key]!=obs[key] for key in ("sample_token","snapshot_sha256","recording","ref_time_us")):
            raise ValueError("Box forecast is not on the same external camera observation")
    return rows


def _load(path,digest):
    if artifact_hash(path)!=digest:
        raise ValueError("Road head weights have changed")
    model=RoadGeometryHead(773)
    model.load_state_dict(load_file(path),strict=True)
    return model.eval()


def _rank_frame(obs,compiled,test,index,probabilities,cuts,grid):
    base={"sample_token":obs["sample_token"],"snapshot_sha256":obs["snapshot_sha256"],"recording":obs["recording"]}
    known=compiled["routing"]
    speed=obs["forward_speed_mps_from_past"]
    if known is None:
        if -.10 <= speed <= 60:
            raise ValueError("A frozen candidate set is missing at an eligible speed")
        return {**base,"candidates":{},"rankings":{}}
    box=tuple(ForecastObject(**x) for x in compiled["boxes"])
    plan=propose_from_forecast(obs["sample_token"],obs["ref_time_us"],Pose2D(*obs["ego_world_xy_yaw"]),max(0,speed),box)
    old_order=list(c.candidate_id for c in plan.plan.candidates)
    if (old_order!=known["frozen_learned"]["ordered"]
            or plan.all_candidates_overlap!=known["frozen_learned"]["all_overlap"]):
        raise ValueError("Cannot reproduce the frozen #43 candidates from old object geometry")
    evidence={}
    for c,d in zip(plan.plan.candidates,plan.diagnostics,strict=True):
        evidence[c.candidate_id]={"poses":c.trajectory.poses.tolist(),"obstacle_free":d.first_overlap_s is None}
    ranks={}
    for arm in ARMS:
        threshold=cuts[arm]["drivable"]
        r=rank_road_candidates(plan.plan.candidates,{k:v["obstacle_free"] for k,v in evidence.items()},
                               probabilities[arm][index,0],test.coverage(arm)[index],grid,
                               offroad_threshold=threshold-ROAD_CONFIDENCE_MARGIN,
                               onroad_threshold=threshold+ROAD_CONFIDENCE_MARGIN)
        ranks[arm]={"ordered":r.ordered,"assessments":{k:asdict(assessment) for k,assessment in r.assessments.items()}}
    return {**base,"interval_s":plan.plan.candidates[0].trajectory.interval_s,"candidates":evidence,"rankings":ranks}


def score_ground_road(capture: Path, predictions: Path, targets: Path, output: Path, *, sealed_sha256: str) -> dict:
    require_hex(sealed_sha256,64,"prediction seal")
    if output.exists():
        raise ValueError("Output must be a new directory")
    receipt=predictions/"predictions.json"
    if artifact_hash(receipt)!=sealed_sha256:
        raise ValueError("Prediction seal mismatch before opening external map labels")
    pred=load_json(receipt)
    header=capture_header(capture)
    if (pred["schema"]!="hybrid-drive-ground-road-predictions-v1" or pred["target_labels_opened"] is not False
            or pred["revision"]!=header["revision"] or pred["capture_sha256"]!=artifact_hash(capture/"capture.json")
            or pred["source_index_sha256"]!=header["source_test_index_sha256"]):
        raise ValueError("Unknown external road prediction provenance")
    test=capture_role(capture,header,"test")
    for p,o in zip(pred["routes"],test.observations,strict=True):
        if any(p[k]!=o[k] for k in ("sample_token","snapshot_sha256","recording")):
            raise ValueError("Prediction route order does not match the external camera capture")
    probs={}
    for name in ARMS:
        path=predictions/(name+".npy")
        if artifact_hash(path)!=pred["arms"][name]["probability_sha256"]:
            raise ValueError("Road probability artifact changed before score")
        a=np.load(path,allow_pickle=False)
        if a.shape!=(test.values.shape[0],2,*test.values.shape[2:]) or a.dtype!=np.float32:
            raise ValueError("Wrong external road probability layout")
        probs[name]=a
    labels,known,target_manifest=load_external_targets(targets,header,test)
    masks={a:test.coverage(a) for a in ARMS}
    cuts={a:pred["arms"][a]["thresholds"] for a in ARMS}
    m,count,extra,extra_metrics=score_layers(probs,masks,labels,known,cuts)
    result={
        "schema":"hybrid-drive-ground-road-score-v1","revision":pred["revision"],"prediction_seal":sealed_sha256,
        "frames":len(test.observations),"recordings":sorted({r["recording"] for r in test.observations}),
        "locations":sorted(set(target_manifest["locations"])),
        "target_manifest_sha256":artifact_hash(targets/"targets.json"),
        "common_supported_cells":count,"added_history_cells":extra,"added_history_scores":extra_metrics,
        "arm_metrics":m,"gate":quality_gate(m),"route_metrics":score_routes(pred["routes"],labels,known,_grid(header)),
    }
    output.mkdir(parents=True)
    write_json(output/"score.json",result)
    return result


def _grid(header):
    config=dict(header["grid"])
    config["heights_m"]=tuple(config["heights_m"])
    return BevGrid(**config)
