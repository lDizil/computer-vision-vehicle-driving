"""Common-support comparison and routing diagnostics for an external road holdout."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from hybrid_drive.contracts.trajectory import Trajectory
from hybrid_drive.experiments.fast_capture import artifact_hash, load_json
from hybrid_drive.experiments.road_scores import layer_scores
from hybrid_drive.fast.road_candidate_rank import assess_road


ARMS = ("object_current", "ground_current", "ground_history")
GROUND_MIN_ROAD_IOU = .65
GROUND_MIN_DIVIDER_F1 = .25
MIN_ROAD_DELTA = .03
MAX_DIVIDER_DROP = .02
XY_GRID = ("x_min", "y_min", "step_m", "rows", "cols")


def load_external_targets(root: Path, header, test) -> tuple[np.ndarray, np.ndarray, dict]:
    manifest = load_json(root / "targets.json")
    data = root / "targets.npz"
    if data.is_symlink() or not 0 < data.stat().st_size <= 4 * 1024 * 1024:
        raise ValueError("Expected a bounded, regular external map target archive")
    if (manifest["schema"] != "hybrid-drive-ground-external-targets-v1"
            or manifest["observation_sha256"] != header["source_test_index_sha256"]
            or manifest["tokens"] != [row["sample_token"] for row in test.observations]
            or manifest["target_sha256"] != artifact_hash(data)
            or manifest["channels"] != ["drivable_area", "lane_divider"]
            or {k: manifest["grid"][k] for k in XY_GRID} != {k: header["grid"][k] for k in XY_GRID}):
        raise ValueError("External map labels do not match tested observations/grid")
    with np.load(data, allow_pickle=False) as archive:
        if set(archive.files) != {"labels", "known"}:
            raise ValueError("External map has unexpected arrays")
        labels, known = archive["labels"], archive["known"]
    shape = test.current_seen.shape
    if (labels.shape != (shape[0], 2, *shape[1:]) or labels.dtype != np.bool_
            or known.shape != shape or known.dtype != np.bool_ or (labels & ~known[:, None]).any()):
        raise ValueError("Inconsistent external map target shape or mask")
    if manifest["locations"] != ["us-nv-las-vegas-strip"] * shape[0]:
        raise ValueError("External target city differs from the predeclared holdout")
    return labels, known, manifest


def score_layers(probabilities, masks, labels, map_known, cuts):
    common = map_known.copy()
    for name in ARMS:
        common &= masks[name]
    by_arm = {
        name: {
            "common": layer_scores(probabilities[name], labels, common, cuts[name]),
            "own": layer_scores(probabilities[name], labels, map_known & masks[name], cuts[name]),
            "supported_cells": int((map_known & masks[name]).sum()),
        }
        for name in ARMS
    }
    added = map_known & masks["ground_history"] & ~masks["ground_current"]
    added_metrics = layer_scores(probabilities["ground_history"], labels, added, cuts["ground_history"])
    return by_arm, int(common.sum()), int(added.sum()), added_metrics


def quality_gate(metrics):
    base, fresh = (metrics[arm]["common"] for arm in ("object_current", "ground_history"))
    old_iou = base["drivable"]["iou"]
    new_iou = fresh["drivable"]["iou"]
    old_div = base["lane_divider"]["f1"]
    new_div = fresh["lane_divider"]["f1"]
    gain = None if old_iou is None or new_iou is None else new_iou - old_iou
    loss = None if old_div is None or new_div is None else old_div - new_div
    return {
        "min_external_road_iou": GROUND_MIN_ROAD_IOU,
        "min_external_divider_f1": GROUND_MIN_DIVIDER_F1,
        "min_road_delta_common": MIN_ROAD_DELTA,
        "max_divider_loss_common": MAX_DIVIDER_DROP,
        "road_delta_common":gain, "divider_loss_common":loss,
        "coverage_gain":metrics["ground_history"]["supported_cells"]-metrics["ground_current"]["supported_cells"],
        "passes_quality_pilot": (
            gain is not None and loss is not None and new_iou >= GROUND_MIN_ROAD_IOU
            and new_div >= GROUND_MIN_DIVIDER_F1 and gain >= MIN_ROAD_DELTA and loss <= MAX_DIVIDER_DROP
            and metrics["ground_history"]["supported_cells"] > metrics["ground_current"]["supported_cells"]
        ),
    }


def score_routes(routes, labels, known, grid):
    totals = {name: Counter(frames=0, completely_observed_candidates=0, comparable_frames=0,
                            changed_first=0, changed_any_order=0, predicted_offroad_frames=0,
                            original_map_offroad=0, ranked_map_offroad=0,
                            original_map_unknown=0, ranked_map_unknown=0)
              for name in ARMS}
    for i,row in enumerate(routes):
        base=row["candidates"]
        if not base:
            continue
        original=list(base)
        for name in ARMS:
            rank=row["rankings"][name]
            order=rank["ordered"]
            graph=rank["assessments"]
            if sorted(order)!=sorted(original) or set(graph)!=set(base):
                raise ValueError("Ranking discarded or added a candidate")
            if any(base[earlier]["obstacle_free"] and not base[later]["obstacle_free"]
                   for earlier,later in zip(original,order,strict=True)):
                raise ValueError("Road ranking promoted a predicted obstacle conflict")
            c=totals[name]
            c["frames"]+=1
            n=sum(v["unknown_samples"]==0 and base[k]["obstacle_free"] for k,v in graph.items())
            c["completely_observed_candidates"]+=n
            c["comparable_frames"]+=n>=2
            c["changed_first"]+=order[0]!=original[0]
            c["changed_any_order"]+=order!=original
            c["predicted_offroad_frames"]+=any(v["offroad_samples"]>0 for v in graph.values())
            for prefix,key in (("original",original[0]),("ranked",order[0])):
                t=Trajectory(base[key]["poses"],row["interval_s"])
                truth=assess_road(t,labels[i,0],known[i],grid,offroad_threshold=.25,onroad_threshold=.75)
                c[prefix+"_map_offroad"]+=truth.offroad_samples>0
                c[prefix+"_map_unknown"]+=truth.unknown_samples>0
    return {k:dict(v) for k,v in totals.items()}
