"""Measure decisions and box geometry separately on the same eligible queries."""

import numpy as np

from .spatial_decision_policy import PRESENT, measure_zone_policy
from .spatial_metrics import SpatialPredictions, spatial_metrics


def measure_geometry(data, boxes):
    predictions = SpatialPredictions(
        data.prior_scores.numpy(), data.presence.numpy(), boxes,
        data.target_boxes.numpy(), data.recordings,
    )
    # ROC AUC and positive-box IoU do not use the presence cutoff.
    metrics = spatial_metrics(predictions, threshold=0.0)
    return {
        "macro_auc": metrics["macro_roc_auc"],
        "macro_positive_iou": metrics["macro_positive_iou"],
        "per_recording_iou": {
            key: row["positive_iou"] for key, row in metrics["per_recording"].items()
        },
    }


def measure_strategy(data, candidate_boxes, policy):
    scores, labels = data.prior_scores.numpy(), data.presence.numpy()
    decisions = policy.decide(scores)
    shown = decisions == PRESENT
    boxes = candidate_boxes.astype(np.float64)
    targets = data.target_boxes.numpy()[labels].astype(np.float64)
    predicted = boxes[labels]
    interior = np.maximum(
        np.minimum(predicted[:, 2:], targets[:, 2:]) - np.maximum(predicted[:, :2], targets[:, :2]), 0,
    ).prod(axis=1)
    union = (
        (predicted[:, 2:] - predicted[:, :2]).prod(axis=1)
        + (targets[:, 2:] - targets[:, :2]).prod(axis=1)
        - interior
    )
    localized = int((shown[labels] & (interior / union >= 0.5)).sum())
    all_shown, all_positives = int(shown.sum()), int(labels.sum())
    return {
        "decisions": measure_zone_policy(scores, labels, data.recordings, policy),
        "geometry": measure_geometry(data, candidate_boxes),
        "localized_at_iou_half": {
            "count": localized,
            "precision_of_shown": localized / all_shown if all_shown else None,
            "recall_of_eligible": localized / all_positives if all_positives else None,
        },
    }


def choose_refinement(baseline, candidates, *, min_macro_gain=0.01, max_recording_loss=0.02):
    """Return a dev-eligible seed or None to retain the original geometry."""
    old = baseline["macro_positive_iou"]
    if old is None:
        raise ValueError("A refinement comparison needs positive dev boxes")
    by_recording = {
        name: value for name, value in baseline["per_recording_iou"].items() if value is not None
    }
    if len(by_recording) < 2:
        raise ValueError("A refinement decision needs positive boxes on at least two dev recordings")
    eligible = []
    for candidate in candidates:
        geometry = candidate["dev_geometry"]
        if geometry["per_recording_iou"].keys() != baseline["per_recording_iou"].keys():
            raise ValueError("Candidate geometry must use the same dev recordings")
        macro = geometry["macro_positive_iou"]
        if macro is None or macro - old < min_macro_gain:
            continue
        if any(
            geometry["per_recording_iou"][name] is None
            or value - geometry["per_recording_iou"][name] > max_recording_loss
            for name, value in by_recording.items()
        ):
            continue
        eligible.append((-macro, candidate["seed"]))
    return min(eligible)[1] if eligible else None
