"""Paired motion scoring with the detector, matched current boxes and labels held fixed."""

from __future__ import annotations

from collections import Counter
from math import fsum, hypot, isfinite
from pathlib import Path

import numpy as np

from hybrid_drive.experiments.bounded_io import require_hex
from hybrid_drive.experiments.fast_capture import artifact_hash, load_json
from hybrid_drive.experiments.fast_metrics import match_centers, supported_targets
from hybrid_drive.experiments.io import write_json
from hybrid_drive.fast.metric_bev import BevGrid


ARMS = ("stationary","frozen_learned","associated")
MOVING_M = .50
MIN_MOVING_PAIRS = 12
REQUIRED_MOTION_GAIN_M = .10
MAX_QUIET_PENALTY_M = .05


def score_box_motion(predictions: Path, targets: Path, output: Path, *, sealed_sha256: str) -> dict:
    require_hex(sealed_sha256,64,"prediction seal")
    if output.exists():
        raise ValueError("Scoring output must be new")
    if artifact_hash(predictions) != sealed_sha256:
        raise ValueError("Predictions differ from the seal fixed before accessing labels")
    predicted = load_json(predictions)
    if (predicted["schema"] != "hybrid-drive-box-motion-predictions-v1"
            or predicted["heldout_targets_opened"] is not False):
        raise ValueError("Expected frozen target-free predictions")
    result = summarize(predicted,load_json(targets))
    result.update({
        "schema":"hybrid-drive-box-motion-score-v1","revision":predicted["revision"],
        "prediction_seal":sealed_sha256,"targets_sha256":artifact_hash(targets),
    })
    output.mkdir(parents=True)
    write_json(output/"score.json",result)
    return result


def summarize(predicted, labels):
    rows = predicted["rows"]
    if len(rows) != len(labels) or set(labels) != {row["sample_token"] for row in rows}:
        raise ValueError("Target and prediction identities differ")
    spec = dict(predicted["grid"])
    spec["heights_m"] = tuple(spec["heights_m"])
    grid = BevGrid(**spec)
    by_recording = {}
    all_pairs = []
    counts = Counter()
    routes = Counter()
    tracks = Counter()
    for row in rows:
        label = labels[row["sample_token"]]
        dt = label["next_dt_s"]
        if not isfinite(dt) or not .25 <= dt <= 1.0:
            raise ValueError("Expected an actual next-frame time interval")
        boxes, velocities, evidence = row["boxes"], row["track_velocity_xy"], row["track_evidence"]
        if any(len(v)!=len(boxes) for v in (velocities,evidence)) or any(
            ev["object_id"]!=box["object_id"] for ev,box in zip(evidence,boxes)
        ):
            raise ValueError("Motion outputs are not aligned with detector boxes")
        valid = np.zeros(grid.rows*grid.cols,dtype=np.bool_)
        cells = row["supported_cells"]
        if (any(type(x) is not int or not 0<=x<len(valid) for x in cells) or len(cells)!=len(set(cells))):
            raise ValueError("Invalid support cells in capture")
        valid[cells] = True
        current = supported_targets(label,valid.reshape(grid.rows,grid.cols),grid)
        ids = match_centers(boxes,current)
        count = by_recording.setdefault(row["recording"],{"counts":Counter(),"pairs":[]})
        totals = Counter(frames=1, boxes=len(boxes), targets=len(current), matches=len(ids),
                         in_scope_annotated_targets=len(label["boxes"]), unsupported_targets=len(label["boxes"])-len(current))
        totals["scorable_moving_targets"] = sum(_is_moving(t) for t in current if t["future_center"] is not None)
        annotated_moving = sum(_is_moving(t) for t in label["boxes"] if t["future_center"] is not None)
        totals["moving_without_coverage"] = annotated_moving-totals["scorable_moving_targets"]
        count["counts"].update(totals)
        counts.update(totals)
        tracks.update(ev["status"] for ev in evidence)
        for i,j,_ in ids:
            gt = current[j]
            if gt["future_center"] is None:
                continue
            point = _evaluate_pair(boxes[i], velocities[i], gt, dt, evidence[i]["status"])
            count["pairs"].append(point)
            all_pairs.append(point)
        route = row["routing"]
        if route is not None:
            routes["compared_frames"] += 1
            ranked = [tuple(route[arm]["ordered"]) for arm in ARMS]
            routes["first_changed_vs_stationary"] += ranked[2][0] != ranked[0][0]
            routes["first_changed_vs_learned"] += ranked[2][0] != ranked[1][0]
            routes["any_order_changed_vs_stationary"] += ranked[2] != ranked[0]
    overall = _reduce(all_pairs)
    recording_summary = {}
    for key, values in sorted(by_recording.items()):
        recording_summary[key] = {"counts":dict(values["counts"]),**_reduce(values["pairs"])}
    supported_recordings = sum(item["moving"]["pairs"]>0 for item in recording_summary.values())
    return {
        "counts":dict(counts),"overall":overall,"by_recording":recording_summary,
        "association_statuses":dict(tracks),"routing":dict(routes),
        "gate":_gate(overall,supported_recordings),
    }


def _is_moving(gt):
    a,b=gt["center"],gt["future_center"]
    return hypot(a[0]-b[0],a[1]-b[1]) >= MOVING_M


def _evaluate_pair(box, tracked, target, dt, status):
    now, future = target["center"],target["future_center"]
    dxdy = (future[0]-now[0],future[1]-now[1])
    location = box["center_xyz"]
    result = {}
    for arm, velocity in (("stationary",(0,0)),("frozen_learned",box["velocity_xy"]),("associated",tracked)):
        ux,uy=(dt*float(x) for x in velocity)
        if not isfinite(ux) or not isfinite(uy):
            raise ValueError("Velocity must be finite")
        displ = hypot(dxdy[0]-ux,dxdy[1]-uy)
        future_xy = hypot(future[0]-location[0]-ux,future[1]-location[1]-uy)
        result[arm]=(displ,future_xy)
    return (_is_moving(target),status,result)


def _reduce(pairs):
    result = {}
    for name, rows in (("all",pairs),("moving",[p for p in pairs if p[0]]),
                       ("quiet",[p for p in pairs if not p[0]])):
        result[name] = {
            "pairs":len(rows),"associated_nonzero":sum(p[1]=="estimated" for p in rows),
            **{arm:{
                "displacement_error_m":_mean(row[2][arm][0] for row in rows),
                "next_center_error_m":_mean(row[2][arm][1] for row in rows),
            } for arm in ARMS},
        }
    return result


def _mean(items):
    values = tuple(items)
    return fsum(values)/len(values) if values else None


def _gate(totals,supported_recordings):
    moving,quiet,overall=(totals[x] for x in ("moving","quiet","all"))
    gate = {
        "min_moving_pairs":MIN_MOVING_PAIRS,"min_recordings":2,
        "moving_displacement_gain_required_m":REQUIRED_MOTION_GAIN_M,
        "quiet_displacement_penalty_max_m":MAX_QUIET_PENALTY_M,
        "requires_all_next_centers_not_worse":True,
        "recordings_with_moving_matches":supported_recordings,
    }
    if moving["pairs"] < MIN_MOVING_PAIRS or supported_recordings < 2 or quiet["pairs"] == 0:
        return {**gate,"status":"insufficient_support"}
    gain = moving["stationary"]["displacement_error_m"]-moving["associated"]["displacement_error_m"]
    quiet_penalty = quiet["associated"]["displacement_error_m"]-quiet["stationary"]["displacement_error_m"]
    center_penalty = overall["associated"]["next_center_error_m"]-overall["stationary"]["next_center_error_m"]
    passed = gain >= REQUIRED_MOTION_GAIN_M and quiet_penalty <= MAX_QUIET_PENALTY_M and center_penalty <= 0
    return {**gate,"status":"passed_pilot" if passed else "failed_accuracy",
            "moving_displacement_gain_m":gain,"quiet_displacement_penalty_m":quiet_penalty,
            "next_center_penalty_m":center_penalty}
