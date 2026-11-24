"""Rebuild a measured Fast scene and route its exact camera features to a real frozen LM."""

from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np
import torch

from hybrid_drive.experiments.bounded_io import require_hex
from hybrid_drive.experiments.box_motion_inputs import load_motion_inputs, load_motion_snapshot
from hybrid_drive.experiments.fast_capture import artifact_hash, load_json
from hybrid_drive.experiments.io import write_json
from hybrid_drive.experiments.paired_road_replay import load_frozen_road_replay
from hybrid_drive.fast.scene_fast_runtime import FrozenSceneFast
from hybrid_drive.fast.scene_payload import RoadEvidence
from hybrid_drive.neural.pretrained_alignment import load_scene_bridge_backbones
from hybrid_drive.neural.scene_bridge import SceneBridge
from hybrid_drive.neural.scene_slow_prefix import SceneSlowPrefix
from hybrid_drive.neural.surround_features import SharedSurroundEncoder, route_surround_scene


SOURCE_INPUTS_SHA="e04c01e8394d675d7a3fd1412658d714b4a8cdeb10e22ab093c5e64fcc704d21"
SOURCE_OBJECTS_SHA="6d9c321a719bdce30686ed8d4c64658f179984fee1c067333b0e0cb6e2ca5017"
ANCHOR_INDEXES=(0,6,13,19,20,26,33,39)
TOKEN_SEED=611


def run_handoff(input_root: Path, model_root: Path, box_fit: Path, road_fit: Path, old_objects: Path,
                ground_predictions: Path, ground_capture: Path, output: Path, *, revision: str):
    require_hex(revision,40,"revision")
    if output.exists():
        raise ValueError("Output directory must be new")
    if artifact_hash(input_root/"index.json")!=SOURCE_INPUTS_SHA or artifact_hash(old_objects)!=SOURCE_OBJECTS_SHA:
        raise ValueError("Expected exact #43 images and sealed object forecasts")
    index=load_motion_inputs(input_root)
    past_objects=load_json(old_objects)
    frozen=load_frozen_road_replay(ground_predictions,ground_capture)
    if (len(index["anchors"])!=40 or past_objects["heldout_targets_opened"] is not False
            or len(past_objects["rows"])!=40 or len(frozen.rows)!=40):
        raise ValueError("Unexpected frozen scene cohort")
    vision,language,orig=load_scene_bridge_backbones(model_root)
    model_id=orig["models"]["vision"]
    encoder=SharedSurroundEncoder(vision,encoder_id=f'{model_id["repository"]}@{model_id["revision"]}')
    runtime=FrozenSceneFast(box_fit,road_fit)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(TOKEN_SEED)
        bridge=SceneBridge().eval()
        slow=SceneSlowPrefix().eval()
    rows,prefixes=[],[]
    probe_source=None
    for i in ANCHOR_INDEXES:
        anchor=index["anchors"][i]
        now,ago=anchor["sample_token"],anchor["history_tokens"][-1]
        previous=encoder.encode(load_motion_snapshot(input_root,ago,index["frames"][ago]))
        snapshot=load_motion_snapshot(input_root,now,index["frames"][now])
        exchanged={}

        def fast(shared):
            exchanged["maps"]=shared.feature_maps
            return runtime(shared,previous)

        def tokenize(shared, combined):
            if shared.feature_maps is not exchanged["maps"]:
                raise ValueError("Bridge and Fast used different camera maps")
            with torch.no_grad():
                return bridge(shared,combined.scene)

        shared,result,tokens=route_surround_scene(snapshot,encoder,fast,tokenize)
        _match_prior_results(result, past_objects["rows"][i], frozen, i)
        if tokens.scene_sha256!=result.scene.content_sha256 or tokens.snapshot_sha256!=snapshot.content_sha256:
            raise ValueError("Bridge output lost the source scene identity")
        allowed=list(result.paired_road.ordered)
        # Two choices are intentionally queried; no language value changes their order.
        selected=(allowed[0],allowed[-1])
        outputs=[]
        for key in selected:
            with torch.no_grad():
                p=slow(tokens,key,language)
            outputs.append({"candidate_id":key, "prefix_index":len(prefixes),
                            "selected_prior_count":p.selected_prior_count,
                            "yes_no_probe_logits":p.answer_logits.tolist()})
            prefixes.append(p.prefix.detach().cpu().numpy()[0])
        active=tokens.prior_mask[0].tolist()
        if probe_source is None and any(r.kind=="actor" and v for r,v in zip(tokens.records,active,strict=True)):
            probe_source=(shared,result.scene,selected[0])
        rows.append({
            "sample_token":now,"recording":anchor["recording"],"snapshot_sha256":snapshot.content_sha256,
            "scene_sha256":result.scene.content_sha256,"plan_sha256":result.scene.plan.candidates_hash,
            "initial_footprint_unknown":result.paired_road.initial_unknown,"paired_order":allowed,
            "same_camera_map_object":shared.feature_maps is exchanged["maps"],
            "object_and_road_outputs_match_frozen":True,"records":[asdict(r) for r in tokens.records],
            "valid_mask":active,"language_probes":outputs,
        })
        print(f"Bound real handoff {len(rows)}/{len(ANCHOR_INDEXES)}: {len(active)} prior slots",flush=True)
    if probe_source is None:
        raise ValueError("Need a visible actor for the controlled geometry-only perturbation")
    perturbation=_geometry_probe(bridge,slow,language,*probe_source)
    gradient=_gradient_probe(bridge,slow,language,*probe_source)
    output.mkdir(parents=True)
    path=output/"prefixes.npy"
    np.save(path,np.stack(prefixes),allow_pickle=False)
    write_json(output/"handoff.json",{
        "schema":"hybrid-drive-scene-bridge-handoff-v1","revision":revision,
        "status":"wiring_only_not_a_calibrated_veto","training_performed":False,"labels_opened":False,
        "language_backbone":orig["models"]["language"],"prompt":orig["tokenization"]["prompt"],
        "prefix_sha256":artifact_hash(path),"prefix_shape":list(np.stack(prefixes).shape),
        "geometry_perturbation":perturbation,"frozen_language_gradient":gradient,"rows":rows,
    })


def _match_prior_results(result,old,frozen,i):
    scene=result.scene
    row=frozen.rows[i]
    if any(x["sample_token"]!=scene.snapshot_id or x["snapshot_sha256"]!=scene.snapshot_sha256 for x in (row,old)):
        raise ValueError("Fast/Slow replay does not bind to the exact old scene")
    if json.loads(json.dumps([asdict(box) for box in scene.boxes]))!=old["boxes"]:
        raise ValueError("Joint Fast object head did not reproduce frozen #43 boxes")
    if not (np.array_equal(scene.road.drivable,frozen.probabilities[i,0])
            and np.array_equal(scene.road.lane_divider,frozen.probabilities[i,1])
            and np.array_equal(scene.road.known,frozen.coverage[i])):
        raise ValueError("Joint Fast road head did not reproduce frozen #45 road evidence")
    actual={c.candidate_id:c.trajectory.poses.tolist() for c in scene.plan.candidates}
    prior={key:value["poses"] for key,value in row["candidates"].items()}
    if actual!=prior or list(actual)!=list(prior):
        raise ValueError("Joint Fast candidate geometry does not reproduce #45")


def _geometry_probe(bridge,slow,language,features,scene,candidate):
    with torch.no_grad():
        tokens=bridge(features,scene)
        baseline=slow(tokens,candidate,language)
        actor=next(r.actor_id for r,valid in zip(tokens.records,tokens.prior_mask[0].tolist(),strict=True)
                   if r.kind=="actor" and valid)
        moved=tuple(replace(b,velocity_xy=(b.velocity_xy[0]+3,b.velocity_xy[1]))
                    if b.object_id==actor else b for b in scene.boxes)
        shifted=replace(scene,boxes=moved)
        changes=bridge(features,shifted)
        if tokens.records!=changes.records:
            raise ValueError("Changing only actor velocity changed the observation geometry/ROI metadata")
        answer=slow(changes,candidate,language)
        motion={
            "actor_id":actor,
            "camera_maps_unchanged":True,
            "prefix_max_abs_delta":float((answer.prefix-baseline.prefix).abs().max()),
            "logit_max_abs_delta":float((answer.answer_logits-baseline.answer_logits).abs().max()),
        }
        selected=next((r for r,v in zip(tokens.records,tokens.prior_mask[0].tolist(),strict=True)
                       if r.kind=="road" and r.candidate_id==candidate and v),None)
        road_delta=None
        if selected is not None:
            q=scene.road.drivable.copy()
            q[selected.cell_rc]=.1 if q[selected.cell_rc]>.5 else .9
            changed=replace(scene,road=RoadEvidence(scene.road.grid,q,scene.road.lane_divider,scene.road.known))
            other=slow(bridge(features,changed),candidate,language)
            road_delta={"cell_rc":selected.cell_rc,"prefix_max_abs_delta":float((other.prefix-baseline.prefix).abs().max())}
    if (motion["prefix_max_abs_delta"]<=1e-7 or motion["logit_max_abs_delta"]<=1e-7
            or (road_delta is not None and road_delta["prefix_max_abs_delta"]<=1e-7)):
        raise ValueError("The handoff lost a controlled numeric geometry change")
    return {"motion":motion,"road":road_delta,"candidate_id":candidate,
            "snapshot_sha256":scene.snapshot_sha256,"scope":"counterfactual_wiring_not_accuracy"}


def _gradient_probe(bridge,slow,language,features,scene,candidate):
    if any(p.requires_grad for p in language.parameters()):
        raise ValueError("Expected the language head to remain frozen")
    probe=slow(bridge(features,scene),candidate,language)
    bridge_params=tuple(p for p in bridge.parameters() if p.requires_grad)
    slow_params=tuple(p for p in slow.parameters() if p.requires_grad)
    grads=torch.autograd.grad(probe.answer_logits[0]-probe.answer_logits[1],bridge_params+slow_params,allow_unused=True)
    if any(g is not None and not torch.isfinite(g).all() for g in grads):
        raise ValueError("Nonfinite language-to-bridge derivative")
    b=sum(float(g.abs().sum()) for g in grads[:len(bridge_params)] if g is not None)
    s=sum(float(g.abs().sum()) for g in grads[len(bridge_params):] if g is not None)
    if not min(b,s)>0:
        raise ValueError("The frozen language backbone blocked gradients to the bridge")
    return {"bridge_l1":b,"projector_l1":s,"language_parameters_frozen":True}
