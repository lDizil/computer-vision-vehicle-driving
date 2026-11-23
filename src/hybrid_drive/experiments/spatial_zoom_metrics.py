"""Offline box-geometry measurements with frame identities and decisions held fixed."""

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, replace
from math import fsum
from numbers import Integral, Real

import numpy as np

from hybrid_drive.assets.spatial_manifest import BAND_COUNT
from .protocol import content_hash
from .spatial_decision_policy import PRESENT
from .spatial_track_metrics import TrackingTarget, _iou, evaluate_tracking, match_tracking_frame
from .spatial_vehicle_tracks import TRACK_RECIPES, replay_vehicle_tracks
from .temporal_spatial_types import FrameDecision


_MIN_IOU = 0.5
_SMALL_AREA = 0.01
_AREA_BUCKETS = ("small_lt_0p01", "large_ge_0p01")
_SEEDS = (0, 1)
_FAMILIES = ("global", "zoom")
_FIXED_COUNTS = ("frames", "labeled_boxes", "output_boxes")
_MATCHED = "detections_matched"


def apply_zoom_boxes(decisions, refined) -> tuple[FrameDecision, ...]:
    """Replace geometry only in original PRESENT slots, with canonical FP32 boxes.

    A CPU tensor or nested numeric array may have rounded unchanged values to
    FP32. Inactive boxes must agree at that precision; their original
    coordinates (and the original observation) are carried through exactly.
    """
    source = _decisions(decisions)
    expected = (len(source), BAND_COUNT, 4)
    candidates = _box_array(refined, expected).astype(np.float32)
    result = []
    for decision, frame in zip(source, candidates):
        before = np.asarray(decision.boxes, dtype=np.float32)
        boxes = []
        for index, (state, current, proposed) in enumerate(zip(decision.states, before, frame)):
            unchanged = np.array_equal(current, proposed)
            if state != PRESENT and not unchanged:
                raise ValueError("Refinement must not change inactive boxes")
            boxes.append(decision.boxes[index] if unchanged else tuple(map(float, proposed)))
        updated = tuple(boxes)
        result.append(decision if updated == decision.boxes else replace(decision, boxes=updated))
    return tuple(result)


def _box_array(value, shape):
    try:
        boxes = np.asarray(value)
    except (TypeError, ValueError, RuntimeError, OverflowError) as exc:
        raise ValueError("Refined boxes must be a CPU numeric array [B,4,4]") from exc
    if boxes.shape != shape or boxes.dtype.kind not in "fiu":
        raise ValueError("Refined boxes must be a CPU numeric array [B,4,4]")
    # A mixed Python sequence can coerce booleans to integers/floats before
    # the dtype check. They are not coordinates in the FrameDecision schema.
    if isinstance(value, (tuple, list)) and any(
        not isinstance(x, Real) or isinstance(x, (bool, np.bool_))
        for x in np.asarray(value, dtype=object).flat
    ):
        raise ValueError("Refined coordinates must be real numbers, not booleans")
    if (
        not np.isfinite(boxes).all()
        or (boxes < 0).any()
        or (boxes > 1).any()
        or (boxes[..., :2] >= boxes[..., 2:]).any()
    ):
        raise ValueError("Refined boxes must have finite, normalized positive-area coordinates")
    return boxes


def evaluate_zoom_boxes(decisions, targets, *, include_tracking=False) -> dict:
    """Measure fixed-threshold frame matching before any identity association.

    Each label is small if its projected xyxy area is strictly below 0.01 of
    the normalized image, otherwise large. Area counts belong to the target,
    not the size of the prediction. Mean IoU uses only the assigned matches.
    """
    rows = _decisions(decisions)
    if not isinstance(targets, Mapping) or set(targets) != {d.observation.sample_id for d in rows}:
        raise ValueError("Expected exactly one tracking target per evaluated sample ID")
    if any(not isinstance(target, TrackingTarget) for target in targets.values()):
        raise ValueError("Expected TrackingTarget values by sample ID")

    total = _Counts()
    per_recording = {}
    provenance = []
    for decision in rows:
        source = decision.observation
        target = targets[source.sample_id]
        pairs = match_tracking_frame(decision, target, min_iou=_MIN_IOU)
        label_buckets = {i: _area_bucket(box) for i, box in enumerate(target.boxes) if box is not None}
        hits = tuple(
            (label_buckets[j], _iou(decision.boxes[i], target.boxes[j])) for i, j in pairs
        )
        output = decision.states.count(PRESENT)
        group = per_recording.setdefault(source.recording_id, _Counts())
        total.add(output, label_buckets.values(), hits)
        group.add(output, label_buckets.values(), hits)
        provenance.append([
            asdict(source), decision.states, decision.scores, decision.reasons,
            decision.reset_reason, asdict(target),
        ])

    report = total.summary()
    report["per_recording"] = {name: group.summary() for name, group in per_recording.items()}
    # Bind ordered, unchanged predictions and exact labels, not just aggregate
    # counts. Both current-frame geometry and association are allowed to vary.
    report["evaluation_identity"] = content_hash([
        "hybrid-drive-spatial-zoom-metrics-v1", _MIN_IOU, _SMALL_AREA, provenance,
    ])
    if include_tracking:
        report["tracking"] = {
            name: evaluate_tracking(replay_vehicle_tracks(rows, TRACK_RECIPES[name]), targets, min_iou=_MIN_IOU)
            for name in ("zone", "motion_strict")
        }
    return report


def _decisions(values):
    try:
        rows = tuple(values)
    except TypeError as exc:
        raise ValueError("Expected an ordered sequence of FrameDecision values") from exc
    if not rows or any(not isinstance(row, FrameDecision) for row in rows):
        raise ValueError("Expected a nonempty sequence of FrameDecision values")
    if len({row.observation.sample_id for row in rows}) != len(rows):
        raise ValueError("Duplicate evaluated sample ID")
    return rows


def _area_bucket(box):
    return int((box[2] - box[0]) * (box[3] - box[1]) >= _SMALL_AREA)


@dataclass
class _Counts:
    frames: int = 0
    output_boxes: int = 0
    labels_by_area: list[int] = field(default_factory=lambda: [0, 0])
    hits_by_area: list[int] = field(default_factory=lambda: [0, 0])
    overlaps: list[float] = field(default_factory=list)

    def add(self, output, label_buckets, hits):
        self.frames += 1
        self.output_boxes += output
        for bucket in label_buckets:
            self.labels_by_area[bucket] += 1
        for bucket, overlap in hits:
            self.hits_by_area[bucket] += 1
            self.overlaps.append(overlap)

    def summary(self):
        labels, hits = sum(self.labels_by_area), len(self.overlaps)
        return {
            "frames": self.frames,
            "labeled_boxes": labels,
            "output_boxes": self.output_boxes,
            _MATCHED: hits,
            "recall": _ratio(hits, labels),
            "precision": _ratio(hits, self.output_boxes),
            "mean_matched_iou": _ratio(fsum(self.overlaps), hits),
            "idf1_ceiling": _ratio(2 * hits, labels + self.output_boxes),
            "projected_area": {
                name: {
                    "labeled_boxes": self.labels_by_area[i],
                    _MATCHED: self.hits_by_area[i],
                    "recall": _ratio(self.hits_by_area[i], self.labels_by_area[i]),
                }
                for i, name in enumerate(_AREA_BUCKETS)
            },
        }


def _ratio(number, total):
    return number / total if total else None


def select_zoom_candidate(dev_results, *, min_gain=2, max_recording_drop=1) -> dict:
    """Choose only after a same-seed improvement in both development repeats."""
    if not _valid_count(min_gain) or min_gain < 1:
        raise ValueError("Minimum matched gain must be a positive integer")
    if not _valid_count(max_recording_drop):
        raise ValueError("Maximum recording drop must be a nonnegative integer")
    if not isinstance(dev_results, Mapping) or "baseline" not in dev_results:
        raise ValueError("Development metrics must include the frozen baseline")
    baseline = dev_results["baseline"]
    fixed = _fixed_summary(baseline)
    if fixed is None:
        raise ValueError("Invalid frozen baseline metrics")

    names = [f"{family}_{seed}" for family in ("geometry", *_FAMILIES) for seed in _SEEDS]
    invalid = {name: _incompatibility(dev_results.get(name), baseline["evaluation_identity"], fixed)
               for name in names}
    candidates, eligible = {}, []
    for family in _FAMILIES:
        for seed in _SEEDS:
            name = f"{family}_{seed}"
            comparisons = ["baseline", f"geometry_{seed}"]
            if family == "zoom":
                comparisons.append(f"global_{seed}")
            candidates[name] = _assess(
                name, comparisons, dev_results, invalid, min_gain, max_recording_drop,
            )
        if all(candidates[f"{family}_{seed}"]["eligible"] for seed in _SEEDS):
            eligible.append(family)

    chosen = f"{eligible[-1]}_0" if eligible else "baseline"
    return {
        "name": chosen,
        "baseline_name": "baseline",
        "eligible_families": eligible,
        "summary": {
            "scope": "development_only",
            "minimum_matched_gain": int(min_gain),
            "maximum_recording_drop": int(max_recording_drop),
            "seeds_required": list(_SEEDS),
            "chosen_seed": 0 if eligible else None,
            "preference": ["zoom", "global", "baseline"],
        },
        "candidates": candidates,
    }


def _assess(name, comparisons, results, invalid, min_gain, max_drop):
    reasons, changes, drops = [], {}, {}
    if invalid[name]:
        reasons.append(f"{name}:{invalid[name]}")
    for other in comparisons:
        if invalid.get(other):
            reasons.append(f"{other}:{invalid[other]}")
        elif not invalid[name]:
            gain = int(results[name][_MATCHED] - results[other][_MATCHED])
            changes[other] = gain
            if gain < min_gain:
                reasons.append(f"insufficient_gain_vs_{other}")
    if not invalid[name]:
        baseline = results["baseline"]["per_recording"]
        for rec, score in sorted(results[name]["per_recording"].items()):
            amount = int(baseline[rec][_MATCHED] - score[_MATCHED])
            if amount > 0:
                drops[rec] = amount
            if amount > max_drop:
                reasons.append(f"recording_drop:{rec}")
    return {"eligible": not reasons, "reasons": reasons, "matched_changes": changes, "recording_drops": drops}


def _incompatibility(candidate, identity, fixed):
    if candidate is None:
        return "missing_metrics"
    if not isinstance(candidate, Mapping):
        return "invalid_metrics"
    if candidate.get("evaluation_identity") != identity:
        return "different_frozen_inputs_or_labels"
    signature = _fixed_summary(candidate)
    if signature is None:
        return "invalid_metrics"
    return None if signature == fixed else "different_fixed_counts"


def _fixed_summary(result):
    if (
        not _valid_row(result)
        or not isinstance(result.get("evaluation_identity"), str)
        or not result["evaluation_identity"].strip()
    ):
        return None
    by_recording = result.get("per_recording")
    if (
        not isinstance(by_recording, Mapping) or not by_recording
        or any(not isinstance(name, str) or not name.strip() or not _valid_row(row)
               for name, row in by_recording.items())
        or any(sum(row[field] for row in by_recording.values()) != result[field]
               for field in (*_FIXED_COUNTS, _MATCHED))
    ):
        return None
    return (
        tuple(result[field] for field in _FIXED_COUNTS),
        tuple((name, *(row[field] for field in _FIXED_COUNTS)) for name, row in sorted(by_recording.items())),
    )


def _valid_row(row):
    return (
        isinstance(row, Mapping)
        and all(_valid_count(row.get(field)) for field in (*_FIXED_COUNTS, _MATCHED))
        and row["frames"] > 0
        and max(row["labeled_boxes"], row["output_boxes"]) <= BAND_COUNT * row["frames"]
        and row[_MATCHED] <= min(row["labeled_boxes"], row["output_boxes"])
    )


def _valid_count(value):
    return isinstance(value, Integral) and not isinstance(value, (bool, np.bool_)) and value >= 0
