"""Compare a replay-fitted constant plane with unchanged accepted camera hits."""

from collections import defaultdict
from dataclasses import replace
from hashlib import sha256
from math import isfinite
from statistics import median

from hybrid_drive.experiments.bounded_io import read_bounded, require_hex
from hybrid_drive.experiments.io import write_json
from hybrid_drive.experiments.object_vru_geometry import fuse_ground_hits
from hybrid_drive.experiments.vru_chain_capture import (
    ensure_new_output, index_sha, read_ground_objects, save_sealed, sealed_read, validate_anchor,
)
from hybrid_drive.experiments.vru_detector_coverage_metrics import summarize_detector_coverage
from hybrid_drive.experiments.vru_fusion_holdout_data import (
    CHAIN_CONTROL_SOURCE_SHA256, LABEL_LIMIT, SOURCE_SHA256, load_fusion_inputs, load_fusion_labels,
)
from hybrid_drive.experiments.vru_observation_fusion import fuse_vru_observations
from hybrid_drive.experiments.vru_projection_audit import CAPTURE_SCHEMA
from hybrid_drive.experiments.vru_projection_truth import load_projection_truth


FIT = "hybrid-drive-vru-plane-fit-v1"
PREDICTIONS = "hybrid-drive-vru-plane-predictions-v1"
SCORE = "hybrid-drive-vru-plane-score-v1"
FAR_X_M = 16.
MAX_PLANE_HEIGHT_M = .50
FIT_MIN_DISTINCT_RECORDINGS = 4
FIT_MIN_PER_RECORDING = 3
GATE = {"min_extra_matched": 3, "min_improved_recordings": 2, "min_near_matched_change": 0,
        "min_common_matches": 12, "max_common_error_increase_m": .10}


def fit_plane(input_root, labels_path, truth_path, output):
    ensure_new_output(output)
    index = _inputs(input_root, SOURCE_SHA256)
    truth = load_projection_truth(input_root, labels_path, truth_path)
    groups = defaultdict(list)
    for anchor, row in zip(index["anchors"], truth["rows"], strict=True):
        for target in row["targets"]:
            x, _, z, _, _, dz, _ = target["box_ego"]
            if x >= FAR_X_M:
                groups[anchor["recording"]].append(z - dz / 2.)
    height, used = _fit_height(groups)
    evidence = {"schema": FIT, "development_sha256": truth["input_sha256"], "labels_sha256": truth["labels_sha256"],
                "truth_sha256": sha256(read_bounded(truth_path, LABEL_LIMIT)).hexdigest(),
                "fitted_plane_z_m": height, "min_development_x_m": FAR_X_M, "recordings": used}
    seal = save_sealed(output, "fit.json", evidence)
    return {"plane_z_m": height, "recordings": len(used), "fit_sha256": seal}


def _fit_height(groups):
    support = {name: {"targets": len(values), "median_bottom_z_m": median(values)}
               for name, values in sorted(groups.items()) if len(values) >= FIT_MIN_PER_RECORDING}
    if len(support) < FIT_MIN_DISTINCT_RECORDINGS:
        raise ValueError("Not enough independent development recordings to estimate a plane")
    z = round(median(row["median_bottom_z_m"] for row in support.values()), 3)
    _check_height(z)
    return z, support


def forecast_plane(input_root, trace_path, fit_path, output, *, trace_sha256, fit_sha256, revision):
    ensure_new_output(output)
    require_hex(revision, 40, "plane forecast revision")
    trace = sealed_read(trace_path, trace_sha256)
    fit = sealed_read(fit_path, fit_sha256)
    index, input_sha = _inputs(input_root, CHAIN_CONTROL_SOURCE_SHA256), index_sha(input_root)
    if (trace["schema"] != CAPTURE_SCHEMA or trace["input_sha256"] != input_sha or trace["labels_mounted"] is not False
            or fit["schema"] != FIT or fit["development_sha256"] == input_sha
            or set(fit["recordings"]) & {anchor["recording"] for anchor in index["anchors"]}):
        raise ValueError("Expected separate development fit and input-only control trace")
    height = fit["fitted_plane_z_m"]
    _check_height(height)
    before, after = [], []
    for row, anchor in zip(trace["rows"], index["anchors"], strict=True):
        validate_anchor(anchor, row)
        a, b = _forecast_window(row, height)
        before.append(a)
        after.append(b)
    seal = save_sealed(output, "predictions.json", {
        "schema": PREDICTIONS, "revision": revision, "input_sha256": input_sha, "labels_mounted": False,
        "trace_sha256": trace_sha256, "fit_sha256": fit_sha256, "fitted_plane_z_m": height,
        "baseline": before, "candidate": after, "gate": GATE,
    })
    return {"windows": len(before), "prediction_sha256": seal}


def _forecast_window(row, height):
    single = {name: read_ground_objects(row["profiles"][name]["objects"]) for name in ("ssdlite", "faster_rcnn")}
    original = fuse_vru_observations(single["ssdlite"], single["faster_rcnn"])
    corrected = {name: fuse_ground_hits(tuple(_move_hit(hit, height) for obj in objects for hit in obj.hits))
                 for name, objects in single.items()}
    moved = fuse_vru_observations(corrected["ssdlite"], corrected["faster_rcnn"])
    head = {key: value for key, value in row.items() if key != "profiles"}
    return tuple({**head, "objects": [_position(obj) for obj in arm]} for arm in (original, moved))


def _move_hit(hit, height):
    x, y, z = hit.camera_xyz
    if height == 0.:
        return hit
    if z <= 0. or height >= z:
        raise ValueError("Expected the experimental plane to lie below the observed camera centre")
    scale = (z - height) / z
    px, py = hit.xy
    return replace(hit, xy=(x + (px - x) * scale, y + (py - y) * scale),
                   uncertainty_m=max(.15, hit.uncertainty_m * scale))


def _position(obj):
    return {"object_id": obj.object_id, "xy_m": obj.xy, "confidence": obj.confidence, "kind": obj.kind}


def score_plane(input_root, prediction_path, labels_path, output, *, sealed_sha256):
    ensure_new_output(output)
    pred = sealed_read(prediction_path, sealed_sha256)
    index = _inputs(input_root, CHAIN_CONTROL_SOURCE_SHA256)
    if (pred["schema"] != PREDICTIONS or pred["input_sha256"] != index_sha(input_root)
            or pred["labels_mounted"] is not False or pred["gate"] != GATE):
        raise ValueError("Expected the predeclared input-only plane forecast")
    for name in ("baseline", "candidate"):
        for anchor, row in zip(index["anchors"], pred[name], strict=True):
            validate_anchor(anchor, row)
    labels = load_fusion_labels(input_root, labels_path)["rows"]
    result = summarize_detector_coverage(pred["baseline"], pred["candidate"], labels)
    result["plane_gate"] = _gate(result, labels)
    output.mkdir(parents=True)
    write_json(output / "score.json", {
        "schema": SCORE, "revision": pred["revision"], "prediction_sha256": sealed_sha256,
        "input_sha256": pred["input_sha256"], "fit_sha256": pred["fit_sha256"], "fitted_plane_z_m": pred["fitted_plane_z_m"],
        "labels_sha256": sha256(read_bounded(labels_path, LABEL_LIMIT)).hexdigest(), "summary": result,
    })
    return result["plane_gate"]


def _gate(summary, labels):
    comparison = summary["gate"]
    common_change = comparison["common_mean_error_change_m"]
    near = {side: 0 for side in ("reference", "candidate")}
    for window, label in zip(summary["rows"], labels, strict=True):
        for pair in window["pairs"]:
            if label["tracks"][pair["target_index"]]["centers_xy_m"][0][0] < FAR_X_M:
                for side in near:
                    near[side] += pair[side] is not None
    checks = {
        "additional_match_count": comparison["additional_matched"] >= GATE["min_extra_matched"],
        "multiple_improved_recordings": comparison["improved_recordings"] >= GATE["min_improved_recordings"],
        "near_coverage_preserved": near["candidate"] - near["reference"] >= GATE["min_near_matched_change"],
        "common_center_bound": common_change is not None and common_change <= GATE["max_common_error_increase_m"],
    }
    status = "signal" if all(checks.values()) else "gain_not_established"
    if comparison["common_matches"] < GATE["min_common_matches"]:
        status = "insufficient_common_support"
    return {"status": status, "checks": checks, "matched_under_16m": near, "thresholds": GATE}


def _inputs(input_root, source_sha):
    inputs = load_fusion_inputs(input_root)
    if inputs["source_sha256"] != source_sha:
        raise ValueError("Unexpected cohort for this fixed development or control operation")
    return inputs


def _check_height(z):
    if type(z) not in (int, float) or not isfinite(z) or abs(z) > MAX_PLANE_HEIGHT_M:
        raise ValueError("Fitted plane height outside the bounded geometric perturbation")
