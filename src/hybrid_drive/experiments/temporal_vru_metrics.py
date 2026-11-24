"""Evaluate offline, anchor-visible VRU positions on the current camera grid.

A camera-covered cell without a matched label is a negative only for this
curated in-scope set. These metrics are not clearance, collision, or calibrated
future-occupancy scores. No ground truth enters model prediction through here.
"""

from collections import Counter, defaultdict
from collections.abc import Mapping
from math import fsum

import numpy as np

from hybrid_drive.experiments.visual_vru_geometry import (
    GRID_COLS,
    GRID_ROWS,
    GRID_STEP_M,
    GRID_X_MIN,
    GRID_Y_MIN,
)


ARMS = ("temporal", "single_frame", "shuffled_past", "persist_now")
HORIZONS = (("now", 0.), ("0.5s", .5), ("1.0s", 1.))
_GRID_SHAPE = (GRID_ROWS, GRID_COLS)
_GRID_ORIGIN = np.array([GRID_X_MIN, GRID_Y_MIN])
_GRID_EDGES = tuple(origin + GRID_STEP_M * np.arange(n + 1) for origin, n in zip(_GRID_ORIGIN, _GRID_SHAPE))
_MAX_TARGET_LAG_S = .1
_MOVING_M = .75
_QUIET_M = .25
_RADIUS_M = 1.
_SCORE_TIE = 1e-6
_NUMERIC_EPS = 1e-12
_MIN_PAIRS = 40
_MIN_RECORDINGS = 4
_MIN_CHOICE_GAIN = .10
_MAX_AP_LOSS = .02
_EXCLUSIONS = ("quiet", "intermediate", "same_cell", "uncovered", "cross_track")


def summarize_motion(anchors, labels, margins, coverage):
    """Summarize sealed [window,horizon,x,y] grids against separately read labels."""
    anchors, labels = tuple(anchors), tuple(labels)
    recordings, centers, timing = _validate_labels(anchors, labels)
    grids, known = _validate_predictions(margins, coverage, len(anchors))
    cell_ids = tuple(_grid_indices(points) for points in centers)
    support = _group_indexes(recordings)
    result = {"counts": {"windows": len(anchors), "recordings": len(support),
                         "tracks": sum(len(points) for points in centers)}, "horizons": {}}
    for step, (key, seconds) in enumerate(HORIZONS):
        positions = tuple(points[:, step] for points in centers)
        cells = tuple(indexes[:, step] for indexes in cell_ids)
        truth, covered_positions = _occupancy(positions, cells, known)
        by_arm = {arm: grids[arm][:, step] for arm in ARMS}
        metrics = {
            "seconds": seconds,
            "counts": {"annotated_tracks": sum(map(len, positions)),
                       "covered_tracks": sum(map(len, covered_positions)),
                       "annotated_cells": int(truth.sum()),
                       "covered_annotated_cells": int((truth & known).sum()),
                       "covered_cells": int(known.sum())},
            "cell_metrics": {arm: _cell_metrics(scores, truth, known, support) for arm, scores in by_arm.items()},
            "localization_1m": {arm: _localization(scores, known, covered_positions, support)
                                for arm, scores in by_arm.items()},
        }
        if step:
            elapsed = timing[:, step - 1]
            metrics["observed_target_dt_s"] = {"minimum": float(elapsed.min()), "maximum": float(elapsed.max())}
            metrics["displacement"] = _displacement(recordings, centers, cell_ids, known, by_arm, step)
        result["horizons"][key] = metrics
    result["temporal_gate"] = _temporal_gate(result["horizons"]["1.0s"])
    return result


def _validate_labels(anchors, labels):
    if not anchors or len(anchors) != len(labels):
        raise ValueError("A nonempty equal count of observation and label rows is required")
    seen, recordings, centers, timing = set(), [], [], []
    for anchor, label in zip(anchors, labels):
        now, previous, recording = (anchor[key] for key in ("sample_token", "previous_token", "recording"))
        if (not all(isinstance(v, str) and v for v in (now, previous, recording)) or now == previous
                or now in seen or label["sample_token"] != now):
            raise ValueError("Observation and target identities must be aligned and unambiguous")
        seen.add(now)
        recordings.append(recording)
        elapsed = np.asarray(label["target_dt_s"])
        if (elapsed.shape != (2,) or elapsed.dtype.kind not in "fiu" or not np.isfinite(elapsed).all()
                or (np.abs(elapsed - [.5, 1.]) > _MAX_TARGET_LAG_S + _NUMERIC_EPS).any()):
            raise ValueError("Future labels must align to the 0.5 s and 1.0 s sample horizons")
        timing.append(elapsed)
        tracks = label["tracks"]
        if not isinstance(tracks, list):
            raise ValueError("Each target row must contain a list of matched tracks")
        xy = np.asarray([track["centers_xy_m"] for track in tracks]) if tracks else np.empty((0, 3, 2))
        if xy.shape != (len(tracks), len(HORIZONS), 2) or xy.dtype.kind not in "fiu" or not np.isfinite(xy).all():
            raise ValueError("Persisted tracks require three finite XY centers")
        if (xy < _GRID_ORIGIN).any() or (xy >= _GRID_ORIGIN + GRID_STEP_M * np.array(_GRID_SHAPE)).any():
            raise ValueError("A labeled track is outside the declared grid")
        centers.append(xy.astype(np.float64, copy=False))
    return recordings, centers, np.array(timing)


def _validate_predictions(margins, coverage, n):
    if not isinstance(margins, Mapping) or set(margins) != set(ARMS):
        raise ValueError("Expected exactly the four declared prediction arms")
    known = np.asarray(coverage)
    if (known.shape != (n, *_GRID_SHAPE) or known.dtype.kind != "f" or not np.isfinite(known).all()
            or (known < 0).any() or (known > 1).any()):
        raise ValueError("Coverage must be a finite [N,x,y] float grid in [0,1]")
    grids = {arm: np.asarray(margins[arm]) for arm in ARMS}
    for scores in grids.values():
        if scores.shape != (n, len(HORIZONS), *_GRID_SHAPE) or scores.dtype.kind != "f" or not np.isfinite(scores).all():
            raise ValueError("Each arm must have finite float prediction grids [N,3,x,y]")
    return grids, known > 0


def _grid_indices(xy):
    # Search absolute edges so rounding an offset against the upper edge
    # cannot turn an inside point into an out-of-range index.
    return np.stack([np.searchsorted(edges, xy[..., axis], side="right") - 1
                     for axis, edges in enumerate(_GRID_EDGES)], axis=-1)


def _group_indexes(recordings):
    groups = defaultdict(list)
    for i, key in enumerate(recordings):
        groups[key].append(i)
    return dict(sorted(groups.items()))


def _occupancy(positions, indexes, known):
    truth = np.zeros_like(known)
    covered = []
    for row, (xy, cells) in enumerate(zip(positions, indexes)):
        ix, iy = cells.T
        truth[row, ix, iy] = True
        covered.append(xy[known[row, ix, iy]])
    return truth, covered


def _cell_metrics(scores, truth, known, groups):
    by_recording = {}
    for key, rows in groups.items():
        mask = known[rows]
        by_recording[key] = _average_precision(scores[rows][mask], truth[rows][mask])
    return {"micro_ap": _average_precision(scores[known], truth[known]),
            "recording_macro_ap": _mean(by_recording.values()), "by_recording": by_recording}


def _average_precision(scores, actual):
    # Whole equal-score blocks count together: tie order is not a ranking signal.
    total = int(actual.sum())
    if not total:
        return None
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    ends = np.flatnonzero(np.r_[sorted_scores[:-1] != sorted_scores[1:], True])
    positives = np.cumsum(actual[order])[ends]
    return float(np.sum(np.diff(positives, prepend=0) * positives / (ends + 1)) / total)


def _mean(values):
    observed = tuple(x for x in values if x is not None)
    return fsum(observed) / len(observed) if observed else None


def _localization(scores, known, actual, groups):
    """Top-K with K fixed to covered ground truth, not a detection threshold."""
    hits = [_localization_hits(grid, mask, xy) for grid, mask, xy in zip(scores, known, actual)]
    by_recording = {}
    for key, rows in groups.items():
        by_recording[key] = _recall(sum(hits[i] for i in rows), sum(len(actual[i]) for i in rows))
    return {**_recall(sum(hits), sum(map(len, actual))),
            "recording_macro_recall": _mean(row["recall"] for row in by_recording.values()),
            "by_recording": by_recording}


def _recall(hits, total):
    return {"hits": hits, "covered_tracks": total, "recall": hits / total if total else None}


def _localization_hits(scores, known, actual):
    if not len(actual):
        return 0
    cells = np.argwhere(known)
    ranking = np.argsort(-scores[known], kind="stable")
    selected = []
    # Deterministic row order for equal scores. Suppress peaks <=1 m apart.
    for index in ranking:
        point = _GRID_ORIGIN + GRID_STEP_M * (cells[index] + .5)
        if all(np.linalg.norm(point - prior) > _RADIUS_M for prior in selected):
            selected.append(point)
            if len(selected) == len(actual):
                break
    error = np.linalg.norm(np.asarray(selected)[:, None] - actual[None], axis=2)
    candidates = sorted((error[i, j], i, j) for i, j in zip(*np.nonzero(error <= _RADIUS_M)))
    used_pred, used_truth = set(), set()
    for _, i, j in candidates:
        if i not in used_pred and j not in used_truth:
            used_pred.add(i)
            used_truth.add(j)
    return len(used_truth)


def _displacement(recordings, centers, cell_ids, known, grids, step):
    eligible, seen_windows = [], set()
    exclusions = Counter({reason: 0 for reason in _EXCLUSIONS})
    for row, (points, indexes) in enumerate(zip(centers, cell_ids)):
        source, destination = indexes[:, 0], indexes[:, step]
        occupancy = Counter(map(tuple, np.concatenate((source, destination))))
        for distance, old, new in zip(np.linalg.norm(points[:, step] - points[:, 0], axis=1), source, destination):
            old, new = tuple(old), tuple(new)
            reason = _exclusion(distance, old, new, known[row], occupancy)
            if reason is None:
                eligible.append((row, old, new))
                seen_windows.add(row)
            else:
                exclusions[reason] += 1
    groups = _group_indexes([recordings[row] for row, _, _ in eligible])
    arms = {}
    for arm, scores in grids.items():
        orders = [_compare(float(scores[row][new]), float(scores[row][old])) for row, old, new in eligible]
        by_recording = {key: _choice_counts([orders[i] for i in rows]) for key, rows in groups.items()}
        arms[arm] = {**_choice_counts(orders),
                     "recording_macro_accuracy": _mean(stat["accuracy"] for stat in by_recording.values()),
                     "by_recording": by_recording}
    return {"pairs": len(eligible), "windows": len(seen_windows), "recordings": len(groups),
            "moving_threshold_m": _MOVING_M, "quiet_threshold_m": _QUIET_M,
            "exclusions": dict(exclusions), "arms": arms}


def _exclusion(distance, old, new, known, occupancy):
    if distance <= _QUIET_M:
        return "quiet"
    if distance < _MOVING_M:
        return "intermediate"
    if old == new:
        return "same_cell"
    if not (known[old] and known[new]):
        return "uncovered"
    # A competing track at either endpoint, in either of the two times, can
    # make a correct occupancy score look like right/wrong movement.
    if occupancy[old] > 1 or occupancy[new] > 1:
        return "cross_track"
    return None


def _compare(destination, source):
    delta = destination - source
    if abs(delta) <= _SCORE_TIE:
        return "tied"
    return "correct" if delta > 0 else "incorrect"


def _choice_counts(orders):
    counts = Counter(orders)
    return {"pairs": len(orders), "correct": counts["correct"], "incorrect": counts["incorrect"],
            "tied": counts["tied"], "accuracy": counts["correct"] / len(orders) if orders else None}


def _temporal_gate(horizon):
    motion = horizon["displacement"]
    policy = {"horizon_seconds": horizon["seconds"], "minimum_pairs": _MIN_PAIRS,
              "minimum_recordings": _MIN_RECORDINGS, "minimum_gain_over_temporal_controls": _MIN_CHOICE_GAIN,
              "maximum_ap_loss_vs_single_frame": _MAX_AP_LOSS}
    if motion["pairs"] < _MIN_PAIRS or motion["recordings"] < _MIN_RECORDINGS:
        return {"status": "insufficient_support", **policy}
    arms = motion["arms"]
    control = ("single_frame", "shuffled_past")
    primary = arms["temporal"]["recording_macro_accuracy"]
    gains = {f"gain_vs_{arm}": primary - arms[arm]["recording_macro_accuracy"] for arm in ARMS[1:]}
    by_ap = horizon["cell_metrics"]
    ap_delta = by_ap["temporal"]["recording_macro_ap"] - by_ap["single_frame"]["recording_macro_ap"]
    wins = sum(row["accuracy"] > max(arms[c]["by_recording"][name]["accuracy"] for c in control)
               for name, row in arms["temporal"]["by_recording"].items())
    status = "temporal_signal"
    if min(gains[f"gain_vs_{arm}"] for arm in control) + _NUMERIC_EPS < _MIN_CHOICE_GAIN or 2 * wins <= motion["recordings"]:
        status = "temporal_gain_not_established"
    elif gains["gain_vs_persist_now"] < -_NUMERIC_EPS or ap_delta < -_MAX_AP_LOSS - _NUMERIC_EPS:
        status = "conflicting_metrics"
    return {"status": status, **policy, **gains, "ap_delta_vs_single_frame": ap_delta,
            "recordings_beating_both_temporal_controls": wins}
