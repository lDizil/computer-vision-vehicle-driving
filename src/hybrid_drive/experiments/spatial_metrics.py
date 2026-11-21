"""Presence and localization metrics for a fixed four-query image task."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SpatialPredictions:
    scores: np.ndarray
    present: np.ndarray
    boxes: np.ndarray
    target_boxes: np.ndarray
    recording_ids: tuple[str, ...]

    def __post_init__(self):
        shape = (len(self.recording_ids), 4)
        if not shape[0] or self.scores.shape != shape or self.present.shape != shape:
            raise ValueError("Each frame must retain its four spatial queries")
        if self.present.dtype != np.bool_ or not np.isfinite(self.scores).all():
            raise ValueError("Presence scores/labels are not valid")
        if (
            self.boxes.shape != (*shape, 4)
            or self.target_boxes.shape != (*shape, 4)
            or not np.isfinite(self.boxes).all()
            or not np.isfinite(self.target_boxes[self.present]).all()
        ):
            raise ValueError("Invalid spatial regression output")
        if np.any(self.boxes[..., 2:] <= self.boxes[..., :2]):
            raise ValueError("Predicted boxes must have positive extent")
        targets = self.target_boxes[self.present]
        if np.any(targets[:, 2:] <= targets[:, :2]):
            raise ValueError("Positive targets must have positive extent")


def calibrate_spatial_threshold(predictions):
    """Use dev examples only. Keep tie-breaking explicit and deterministic."""
    scores = np.asarray(predictions.scores, dtype=np.float64)
    if not np.any(predictions.present) or np.all(predictions.present):
        raise ValueError("Dev calibration requires examples of both classes")
    groups = tuple(_recording_groups(predictions.recording_ids).values())
    candidates = np.unique(np.r_[0.0, scores.ravel(), np.nextafter(scores.max(), np.inf)])
    best = None
    for threshold in candidates:
        flags = scores >= threshold
        presence = [_presence(flags[idx], predictions.present[idx]) for idx in groups]
        quality = _macro_presence(presence)["macro_balanced_accuracy"]
        candidate = (-quality, abs(float(threshold)), float(threshold))
        if best is None or candidate < best:
            best = candidate
    return {
        "threshold": best[2],
        "dev_macro_balanced_accuracy": -best[0],
        "candidates": len(candidates),
        "recordings": len(groups),
        "weighting": "classwise_macro_by_recording",
        "tie_break": "closest_to_zero_then_lower_threshold",
    }


def spatial_metrics(predictions, threshold):
    if not np.isfinite(threshold):
        raise ValueError("Expected finite dev threshold")
    scores = predictions.scores.astype(np.float64)
    result = _metrics(scores, predictions.present, predictions.boxes, predictions.target_boxes, threshold)
    per_recording = {
        name: _metrics(
            scores[idx],
            predictions.present[idx],
            predictions.boxes[idx],
            predictions.target_boxes[idx],
            threshold,
        )
        for name, idx in _recording_groups(predictions.recording_ids).items()
    }
    result["per_recording"] = per_recording
    for key in ("positive_iou", "roc_auc"):
        values = [metrics[key] for metrics in per_recording.values() if metrics[key] is not None]
        result["macro_" + key] = float(np.mean(values)) if values else None
        result["macro_" + key + "_recordings"] = len(values)
    result.update(_macro_presence(per_recording.values()))
    return result


def shuffled_by_recording(recording_ids):
    """Pair each input with a different frame from the same recording."""
    indices = np.arange(len(recording_ids))
    for rows in _recording_groups(recording_ids).values():
        if len(rows) < 2:
            raise ValueError("Cannot shuffle a singleton recording without crossing recordings")
        indices[rows] = np.roll(rows, len(rows) // 2)
    return indices


def _recording_groups(recording_ids):
    groups = {}
    for idx, recording_id in enumerate(recording_ids):
        groups.setdefault(recording_id, []).append(idx)
    return groups


def _metrics(scores, present, boxes, target_boxes, threshold):
    result = _presence(scores >= threshold, present)
    positives, negatives = scores[present], scores[~present]
    if len(positives) and len(negatives):
        differences = positives[:, None] - negatives
        result["roc_auc"] = float(np.mean((differences > 0) + 0.5 * (differences == 0)))
    else:
        result["roc_auc"] = None
    box_a, box_b = boxes[present].astype(np.float64), target_boxes[present].astype(np.float64)
    result.update(frames=len(scores), positive_iou=None, positive_center_l2_image_fraction=None)
    if len(positives):
        intersection = np.maximum(
            np.minimum(box_a[:, 2:], box_b[:, 2:]) - np.maximum(box_a[:, :2], box_b[:, :2]), 0.0
        ).prod(axis=1)
        union = (box_a[:, 2:] - box_a[:, :2]).prod(axis=1) + (box_b[:, 2:] - box_b[:, :2]).prod(axis=1)
        result["positive_iou"] = float(np.mean(intersection / (union - intersection)))
        center = (box_a[:, :2] + box_a[:, 2:] - box_b[:, :2] - box_b[:, 2:]) / 2
        result["positive_center_l2_image_fraction"] = float(np.linalg.norm(center, axis=1).mean())
    return result


def _macro_presence(recordings):
    rows = tuple(recordings)
    recalls = [row["recall"] for row in rows if row["recall"] is not None]
    specs = [row["specificity"] for row in rows if row["specificity"] is not None]
    recall = float(np.mean(recalls)) if recalls else None
    specificity = float(np.mean(specs)) if specs else None
    return {
        "macro_recall": recall,
        "macro_specificity": specificity,
        "macro_recall_recordings": len(recalls),
        "macro_specificity_recordings": len(specs),
        "macro_balanced_accuracy": (recall + specificity) / 2 if recalls and specs else None,
    }


def _presence(flags, present):
    positives, negatives = int(present.sum()), int((~present).sum())
    true_positive = int((flags & present).sum())
    true_negative = int((~flags & ~present).sum())
    recall = true_positive / positives if positives else None
    specificity = true_negative / negatives if negatives else None
    return {
        "positives": positives,
        "negatives": negatives,
        "recall": recall,
        "specificity": specificity,
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": negatives - true_negative,
        "false_negative": positives - true_positive,
        "balanced_accuracy": (recall + specificity) / 2 if positives and negatives else None,
    }
