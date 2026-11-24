"""Per-layer road metrics and diagnostics for fixed trajectory alternatives."""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass

import numpy as np

from hybrid_drive.experiments.bounded_io import require_hex
from hybrid_drive.experiments.fast_capture import artifact_hash, load_json
from hybrid_drive.fast.road_geometry import LAYERS


THRESHOLDS = (.25, .35, .45, .50, .55, .65, .75)
ROAD_MIN_IOU = .65
ROAD_MIN_VISUAL_GAIN = .03
DIVIDER_MIN_F1 = .25
DIVIDER_MIN_VISUAL_GAIN = .03


@dataclass(frozen=True)
class RoadTargets:
    labels: np.ndarray
    map_known: np.ndarray
    locations: tuple[str, ...]
    map_sha256: str


def load_road_targets(root: Path, capture_root: Path, split: str, captured) -> RoadTargets:
    manifest = load_json(root / "targets.json")
    archive = root / "targets.npz"
    capture_meta_path = capture_root / "capture.json"
    capture_header = load_json(capture_meta_path)
    if archive.is_symlink() or not 0 < archive.stat().st_size < 4 * 1024 * 1024:
        raise ValueError("Expected bounded mask archive")
    if (manifest["schema"] != "hybrid-drive-road-map-targets-v1"
            or manifest["capture_sha256"] != artifact_hash(capture_meta_path)
            or manifest["grid"] != capture_header["grid"]
            or manifest["channels"] != ["drivable_area", "lane_divider"]
            or manifest["observation_sha256"] != artifact_hash(capture_root / split / "observations.json")
            or manifest["tokens"] != [row["sample_token"] for row in captured.observations]
            or manifest["target_sha256"] != artifact_hash(archive)):
        raise ValueError("Road labels are not bound to those captured observations")
    with np.load(archive, allow_pickle=False) as raw:
        if set(raw.files) != {"labels", "known"}:
            raise ValueError("Expected road labels and known map cells")
        labels = raw["labels"]
        known = raw["known"]
    shape = captured.coverage.shape
    if (labels.shape != (shape[0], 2, *shape[1:]) or labels.dtype != np.bool_
            or known.shape != shape or known.dtype != np.bool_
            or np.any(labels & ~known[:, None])):
        raise ValueError("Wrong or contradictory map target layout")
    locations = manifest["locations"]
    if len(locations) != shape[0] or any(not isinstance(x, str) or not x for x in locations):
        raise ValueError("Expected one map location per anchor")
    map_sha = manifest["map_sha256"]
    require_hex(map_sha, 64, "map SHA256")
    return RoadTargets(labels, known, tuple(locations), map_sha)


def binary_metrics(probabilities: np.ndarray, labels: np.ndarray, support: np.ndarray, threshold: float) -> dict:
    if not (0 <= threshold <= 1 and probabilities.shape == labels.shape == support.shape
            and support.dtype == labels.dtype == np.bool_
            and probabilities.dtype.kind == "f" and np.isfinite(probabilities).all()
            and ((0 <= probabilities) & (probabilities <= 1)).all()):
        raise ValueError("Invalid road probabilities, target shape or threshold")
    predicted = probabilities >= threshold
    tp = int((predicted & labels & support).sum())
    fp = int((predicted & ~labels & support).sum())
    fn = int((~predicted & labels & support).sum())
    tn = int((~predicted & ~labels & support).sum())
    return {
        "cells": tp + fp + fn + tn, "positives": tp + fn, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "iou": ratio(tp, tp+fp+fn), "f1": ratio(2*tp, 2*tp+fp+fn),
        "precision": ratio(tp, tp+fp), "recall": ratio(tp, tp+fn),
    }


def ratio(num, den):
    return num/den if den else None


def select_cut(probabilities, labels, support, layer):
    if not np.any(labels & support) or not np.any((~labels) & support):
        raise ValueError(f"Need positive and negative dev supervision for {layer}")
    metric = "iou" if layer == "drivable" else "f1"
    scores = {t: binary_metrics(probabilities, labels, support, t) for t in THRESHOLDS}
    valid = [t for t in THRESHOLDS if scores[t][metric] is not None]
    if not valid:
        raise ValueError(f"No positive dev supervision for {layer}")
    best = max(valid, key=lambda t:(scores[t][metric], -abs(t-.5), t))
    return best, scores[best]


def layer_scores(probabilities, labels, support, thresholds):
    return {
        name: binary_metrics(probabilities[:, i], labels[:, i], support, thresholds[name])
        for i, name in enumerate(LAYERS)
    }


def check_quality(vision, geometry_only):
    gates = {}
    for name, metric, minimum, gain in (
        ("drivable", "iou", ROAD_MIN_IOU, ROAD_MIN_VISUAL_GAIN),
        ("lane_divider", "f1", DIVIDER_MIN_F1, DIVIDER_MIN_VISUAL_GAIN),
    ):
        v, g = vision[name][metric], geometry_only[name][metric]
        delta = None if v is None or g is None else v-g
        gates[name] = {
            "metric":metric, "minimum":minimum, "gain_required":gain, "gain":delta,
            "passed_pilot": delta is not None and v >= minimum and delta >= gain,
        }
    return gates
