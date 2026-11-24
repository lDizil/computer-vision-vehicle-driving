"""Small Mini pilot metrics: one-to-one XY center matching, with honest denominators."""

from __future__ import annotations

from math import atan2, cos, hypot, sin
import numpy as np


def supported_targets(frame, coverage, grid):
    truth = []
    for box in frame["boxes"]:
        cell = grid.index(*box["center"][:2])
        if cell is not None and coverage[cell]:
            truth.append(box)
    return truth


def match_centers(predictions, targets, radius_m=2.0):
    edges = []
    for i, box in enumerate(predictions):
        for j, target in enumerate(targets):
            error = xy_distance(box["center_xyz"], target["center"])
            if error <= radius_m:
                edges.append((error, i, j))
    pairs, used_pred, used_target = [], set(), set()
    for error, i, j in sorted(edges):
        if i in used_pred or j in used_target:
            continue
        used_pred.add(i)
        used_target.add(j)
        pairs.append((i, j, error))
    return pairs


def xy_distance(a, b):
    return hypot(a[0]-b[0], a[1]-b[1])


def assess_frames(frames):
    """frames contains (predictions, in_scope_targets, next_dt_s); never estimate safety."""
    n_predictions = n_targets = n_matches = 0
    errors, height, size, heading, future_error, hold_error = [], [], [], [], [], []
    for predicted, targets, next_dt_s in frames:
        n_predictions += len(predicted)
        n_targets += len(targets)
        for i, j, error in match_centers(predicted, targets):
            pred, target = predicted[i], targets[j]
            n_matches += 1
            errors.append(error)
            height.append(abs(pred["center_xyz"][2]-target["center"][2]))
            size.append(sum(abs(a-b) for a,b in zip(pred["size_wlh"],target["size_wlh"]))/3)
            delta = pred["yaw"]-target["yaw"]
            heading.append(abs(atan2(sin(delta),cos(delta))))
            future = target["future_center"]
            if future is not None:
                now = pred["center_xyz"]
                rollout = [now[axis]+next_dt_s*pred["velocity_xy"][axis] for axis in (0,1)]
                future_error.append(xy_distance(rollout, future))
                hold_error.append(xy_distance(now, future))
    precision = n_matches/n_predictions if n_predictions else 0
    recall = n_matches/n_targets if n_targets else 0
    return {
        "predictions": n_predictions, "targets": n_targets, "matches": n_matches,
        "precision": precision, "recall": recall,
        "f1": 2*precision*recall/(precision+recall) if precision+recall else 0,
        "mean_xy_error_m": _mean(errors), "mean_z_error_m": _mean(height),
        "mean_size_error_m": _mean(size), "mean_wrapped_heading_error_rad": _mean(heading),
        "motion_pairs": len(future_error), "mean_next_xy_error_m": _mean(future_error),
        "mean_next_static_baseline_error_m": _mean(hold_error),
    }


def _mean(values):
    return float(np.mean(values, dtype=np.float64)) if values else None
