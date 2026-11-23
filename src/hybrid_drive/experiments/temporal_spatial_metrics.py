"""Offline, label-aligned measures of per-zone decisions across source frames."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from itertools import groupby
import json
from math import hypot, isfinite
from numbers import Integral, Real
from typing import Mapping

from .spatial_decision_policy import ABSENT, PRESENT, UNCERTAIN


_BANDS = 4
_MAX_RECALL_LOSS = 0.02
_MAX_EXTRA_MEDIAN_DELAY_US = 500_000
_MIN_JITTER_GAIN = 1e-9
_MAX_IOU_LOSS = 0.01
_MAX_LOCALIZED_LOSS = 1
Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class TemporalTarget:
    present: tuple[bool, bool, bool, bool]
    boxes: tuple[Box | None, Box | None, Box | None, Box | None]

    def __post_init__(self):
        labels = _four(self.present, "Target presence")
        boxes = _four(self.boxes, "Target boxes")
        if any(type(value) is not bool for value in labels):
            raise ValueError("Target presence must be boolean")
        if any((box is not None) != present for present, box in zip(labels, boxes)):
            raise ValueError("Only positive target zones must have boxes")
        object.__setattr__(self, "present", labels)
        object.__setattr__(self, "boxes", tuple(_normalized_box(box) if box is not None else None for box in boxes))


def evaluate_temporal(decisions, targets: Mapping[str, TemporalTarget], *, baseline=None, max_gap_us=800_000) -> dict:
    """Score a full ordered trace, or a stress trace with its exact surviving labels.

    A temporal edge requires adjacent source indices and timestamps within the
    limit, in the same clip. Delay is measured from an observed absent-to-present
    label edge until the first PRESENT in that observed positive run. A run cut
    before detection counts as undetected on the observed interval. A positive
    at a clip start is not an observed appearance.

    Jitter measures changes in the prediction's center error, not its speed.
    Adjacent positives in the same zone are only a geometry proxy, not proof
    that the annotations refer to the same object. Paired jitter uses only
    edges where both methods show a box at both ends.
    """
    if not isinstance(max_gap_us, Integral) or isinstance(max_gap_us, bool) or max_gap_us <= 0:
        raise ValueError("The temporal gap limit must be a positive integer in microseconds")
    frames = _checked_frames(decisions)
    if not isinstance(targets, Mapping) or set(targets) != {row.sample_id for row in frames}:
        raise ValueError("Expected exactly one target per evaluated sample ID")
    if any(not isinstance(target, TemporalTarget) for target in targets.values()):
        raise ValueError("Expected TemporalTarget values by sample ID")
    baseline_frames = _checked_frames(baseline) if baseline is not None else (None,) * len(frames)
    if baseline is not None and [row.identity for row in baseline_frames] != [row.identity for row in frames]:
        raise ValueError("Baseline must match the ordered source identities and timestamps")

    total = _Counts()
    per_recording = {}
    pairs = zip(frames, baseline_frames)
    for recording, rows in groupby(pairs, key=lambda pair: pair[0].recording_id):
        measured = _measure_recording(rows, targets, max_gap_us)
        total.add(measured)
        per_recording[recording] = measured.summary()
    summary = total.summary()
    ious = [entry["positive_iou"] for entry in per_recording.values() if entry["positive_iou"] is not None]
    summary["macro_positive_iou"] = _ratio(sum(ious), len(ious))
    summary["per_recording"] = per_recording
    # Keep alternatives accountable to the same observations, labels, and
    # temporal-adjacency rule even after saving their summaries.
    signature = [
        [row.identity, targets[row.sample_id].present, targets[row.sample_id].boxes]
        for row in frames
    ]
    summary["evaluation_identity"] = _identity_hash([int(max_gap_us), signature])
    summary["prediction_identity"] = _prediction_identity(frames)
    summary["paired_baseline_identity"] = _prediction_identity(baseline_frames) if baseline is not None else None
    return summary


def _identity_hash(content):
    return sha256(json.dumps(content, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _prediction_identity(frames):
    return _identity_hash([[row.identity, row.states, row.boxes] for row in frames])


@dataclass(frozen=True)
class _Frame:
    recording_id: str
    clip_id: str
    sample_id: str
    source_index: int
    timestamp_us: int
    states: tuple[int, ...]
    boxes: tuple[Box, ...]

    @property
    def identity(self):
        return self.recording_id, self.clip_id, self.sample_id, self.source_index, self.timestamp_us


def _checked_frames(decisions):
    frames = []
    sample_ids, clip_ids, recordings = set(), set(), set()
    previous = None
    for decision in decisions:
        row = _frame(decision)
        if row.sample_id in sample_ids:
            raise ValueError("Duplicate evaluated sample ID")
        if previous is None or row.recording_id != previous.recording_id:
            if row.recording_id in recordings:
                raise ValueError("An evaluated recording must occupy one contiguous block")
            recordings.add(row.recording_id)
        elif row.source_index <= previous.source_index or row.timestamp_us <= previous.timestamp_us:
            raise ValueError("Source indices and timestamps must increase within a recording")
        clip = (row.recording_id, row.clip_id)
        if previous is None or clip != (previous.recording_id, previous.clip_id):
            if clip in clip_ids:
                raise ValueError("An evaluated clip must occupy one contiguous block")
            clip_ids.add(clip)
        previous = row
        sample_ids.add(row.sample_id)
        frames.append(row)
    if not frames:
        raise ValueError("Expected at least one evaluated frame")
    return frames


def _frame(decision):
    try:
        obs = decision.observation
        named = (obs.recording_id, obs.clip_id, obs.sample_id)
        indexed = (obs.source_index, obs.timestamp_us)
        states = _four(decision.states, "Decision states")
        boxes = _four(decision.boxes, "Predicted boxes")
    except AttributeError as exc:
        raise ValueError("Expected frame decisions with source identities, states, and boxes") from exc
    if any(not isinstance(value, str) or not value.strip() for value in named):
        raise ValueError("Recording, clip, and sample IDs must be nonempty")
    if any(not isinstance(value, Integral) or isinstance(value, bool) or not 0 <= value < 2**63 for value in indexed):
        raise ValueError("Source index and timestamp must be bounded nonnegative integers")
    if any(
        not isinstance(value, Integral) or isinstance(value, bool) or value not in (ABSENT, UNCERTAIN, PRESENT)
        for value in states
    ):
        raise ValueError("Every zone must have an ABSENT, UNCERTAIN, or PRESENT decision")
    return _Frame(*named, *map(int, indexed), tuple(map(int, states)), tuple(map(_normalized_box, boxes)))


def _four(values, title):
    try:
        values = tuple(values)
    except TypeError as exc:
        raise ValueError(f"{title} must have four values") from exc
    if len(values) != _BANDS:
        raise ValueError(f"{title} must have four values")
    return values


def _normalized_box(box):
    points = _four(box, "A normalized box")
    if any(
        not isinstance(value, Real) or isinstance(value, bool) or not isfinite(value) or not 0 <= value <= 1
        for value in points
    ):
        raise ValueError("Expected finite normalized box coordinates")
    x0, y0, x1, y1 = map(float, points)
    if x1 <= x0 or y1 <= y0 or (x1 - x0) * (y1 - y0) == 0:
        raise ValueError("Expected a box with positive area")
    return x0, y0, x1, y1


@dataclass
class _Counts:
    frames: int = 0
    clips: int = 0
    positives: int = 0
    true_present: int = 0
    false_present: int = 0
    false_absent: int = 0
    uncertain: int = 0
    localized_iou_half: int = 0
    iou_sum: float = 0.0
    stable_edges: int = 0
    flicker_edges: int = 0
    confident_reversal_edges: int = 0
    false_present_runs: int = 0
    delay_appearances: int = 0
    delays_us: list[int] = field(default_factory=list)
    jitter_pairs: int = 0
    jitter_sum: float = 0.0
    paired_count: int = 0
    paired_jitter_sum: float = 0.0
    paired_baseline_sum: float = 0.0

    def add(self, other):
        for attribute, value in vars(other).items():
            if attribute == "delays_us":
                self.delays_us.extend(value)
            else:
                setattr(self, attribute, getattr(self, attribute) + value)

    def summary(self):
        queries = self.frames * _BANDS
        detected = len(self.delays_us)
        positive_iou = _ratio(self.iou_sum, self.positives)
        return {
            "frames": self.frames,
            "clips": self.clips,
            "queries": queries,
            "positives": self.positives,
            "negatives": queries - self.positives,
            "true_present": self.true_present,
            "false_present": self.false_present,
            "false_absent": self.false_absent,
            "uncertain": self.uncertain,
            "coverage": _ratio(queries - self.uncertain, queries),
            "recall": _ratio(self.true_present, self.positives),
            "precision": _ratio(self.true_present, self.true_present + self.false_present),
            "localized_iou_half": self.localized_iou_half,
            "positive_iou": positive_iou,
            "macro_positive_iou": positive_iou,
            "stable_edges": self.stable_edges,
            "flicker_edges": self.flicker_edges,
            "confident_reversal_edges": self.confident_reversal_edges,
            "false_present_runs": self.false_present_runs,
            "delay_appearances": self.delay_appearances,
            "delay_detected": detected,
            "delay_median_us": _median_us(self.delays_us),
            "delay_never_detected": self.delay_appearances - detected,
            "jitter_pairs": self.jitter_pairs,
            "mean_jitter": _ratio(self.jitter_sum, self.jitter_pairs),
            "paired_jitter": {
                "count": self.paired_count,
                "candidate_mean": _ratio(self.paired_jitter_sum, self.paired_count),
                "baseline_mean": _ratio(self.paired_baseline_sum, self.paired_count),
            },
        }


def _measure_recording(rows, targets, max_gap_us):
    counts = _Counts()
    previous = previous_target = previous_baseline = None
    pending = [None] * _BANDS
    for row, baseline in rows:
        target = targets[row.sample_id]
        connected = _adjacent(previous, row, max_gap_us)
        counts.frames += 1
        counts.clips += previous is None or previous.clip_id != row.clip_id
        if not connected:
            pending = [None] * _BANDS
        for band, (state, present) in enumerate(zip(row.states, target.present)):
            _count_query(counts, state, present, row.boxes[band], target.boxes[band])
            if present:
                if connected and not previous_target.present[band]:
                    counts.delay_appearances += 1
                    pending[band] = row.timestamp_us
                if state == PRESENT and pending[band] is not None:
                    counts.delays_us.append(row.timestamp_us - pending[band])
                    pending[band] = None
            else:
                pending[band] = None
                previous_false = connected and not previous_target.present[band] and previous.states[band] == PRESENT
                counts.false_present_runs += state == PRESENT and not previous_false
            if connected:
                _count_edge(counts, band, previous, row, previous_target, target, previous_baseline, baseline)
        previous, previous_target, previous_baseline = row, target, baseline
    return counts


def _adjacent(previous, current, max_gap_us):
    return (
        previous is not None
        and previous.clip_id == current.clip_id
        and previous.source_index + 1 == current.source_index
        and current.timestamp_us - previous.timestamp_us <= max_gap_us
    )


def _count_query(counts, state, present, box, target):
    counts.uncertain += state == UNCERTAIN
    if present:
        iou = _iou(box, target)
        counts.positives += 1
        counts.iou_sum += iou
        counts.true_present += state == PRESENT
        counts.false_absent += state == ABSENT
        counts.localized_iou_half += state == PRESENT and iou >= 0.5
    else:
        counts.false_present += state == PRESENT


def _count_edge(counts, band, previous, current, previous_target, target, previous_baseline, baseline):
    if previous_target.present[band] != target.present[band]:
        return
    first, second = previous.states[band], current.states[band]
    counts.stable_edges += 1
    counts.flicker_edges += first != second
    counts.confident_reversal_edges += first * second == -1
    if not target.present[band] or first != PRESENT or second != PRESENT:
        return
    gt0, gt1 = previous_target.boxes[band], target.boxes[band]
    change = _center_error_change(previous.boxes[band], current.boxes[band], gt0, gt1)
    counts.jitter_pairs += 1
    counts.jitter_sum += change
    if baseline is None or previous_baseline.states[band] != PRESENT or baseline.states[band] != PRESENT:
        return
    counts.paired_count += 1
    counts.paired_jitter_sum += change
    counts.paired_baseline_sum += _center_error_change(previous_baseline.boxes[band], baseline.boxes[band], gt0, gt1)


def _iou(a, b):
    intersection = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union


def _center_error_change(a, b, target_a, target_b):
    dx = ((b[0] - target_b[0]) + (b[2] - target_b[2])) - ((a[0] - target_a[0]) + (a[2] - target_a[2]))
    dy = ((b[1] - target_b[1]) + (b[3] - target_b[3])) - ((a[1] - target_a[1]) + (a[3] - target_a[3]))
    return hypot(dx, dy) / 2


def _ratio(count, total):
    return count / total if total else None


def _median_us(values):
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    return ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) // 2


def select_temporal_recipe(dev_results, *, baseline_name="framewise") -> dict:
    """Choose a recipe on matched development data, otherwise retain the baseline."""
    if not isinstance(dev_results, Mapping) or baseline_name not in dev_results:
        raise ValueError("Development metrics must include the framewise baseline")
    base = dev_results[baseline_name]
    assessments, eligible = {}, []
    for name, trial in sorted(dev_results.items()):
        if name == baseline_name:
            continue
        if base["evaluation_identity"] != trial["evaluation_identity"]:
            assessments[name] = {"eligible": False, "reasons": ["different_evaluation_frames_labels_or_gap"]}
            continue
        if trial["paired_jitter"]["count"] and trial["paired_baseline_identity"] != base["prediction_identity"]:
            assessments[name] = {"eligible": False, "reasons": ["different_paired_baseline"]}
            continue
        changes = _changes(base, trial)
        reasons = _decline_reasons(changes)
        assessments[name] = {"eligible": not reasons, "reasons": reasons, **changes}
        if not reasons:
            # Prefer reducing phantom evidence, then flicker and comparable jitter.
            eligible.append((
                trial["false_present"], trial["flicker_edges"], -changes["paired_jitter_gain"],
                trial["delay_never_detected"], trial["delay_median_us"] or 0, -trial["recall"],
                -trial["coverage"], name,
            ))
    selected = min(eligible)[-1] if eligible else baseline_name
    return {
        "name": selected,
        "baseline_name": baseline_name,
        "summary": {
            "scope": "development_only",
            "reason": "matched_development_improvement" if eligible else "retain_framewise_no_eligible_improvement",
            "eligible_count": len(eligible),
            "max_recall_loss": _MAX_RECALL_LOSS,
            "max_extra_median_delay_us": _MAX_EXTRA_MEDIAN_DELAY_US,
            "min_paired_jitter_gain": _MIN_JITTER_GAIN,
            "max_box_iou_loss": _MAX_IOU_LOSS,
            "max_lost_localized_boxes": _MAX_LOCALIZED_LOSS,
            "max_extra_false_absent": 0,
            "tie_break": [
                "fewer_false_present", "less_flicker", "larger_matched_jitter_gain", "fewer_missed_appearances",
                "lower_median_delay", "higher_recall", "higher_coverage", "lexical_recipe_name",
            ],
        },
        "candidates": assessments,
    }


def _changes(base, trial):
    jitter = trial["paired_jitter"]
    gain = jitter["baseline_mean"] - jitter["candidate_mean"] if jitter["count"] else 0.0
    return {
        "fewer_false_present": base["false_present"] - trial["false_present"],
        "fewer_flicker_edges": base["flicker_edges"] - trial["flicker_edges"],
        "paired_jitter_pairs": jitter["count"],
        "paired_jitter_gain": gain,
        "recall_change": trial["recall"] - base["recall"]
        if trial["recall"] is not None and base["recall"] is not None else None,
        "additional_false_absent": trial["false_absent"] - base["false_absent"],
        "lost_localized_boxes": base["localized_iou_half"] - trial["localized_iou_half"],
        "box_iou_change": trial["macro_positive_iou"] - base["macro_positive_iou"]
        if trial["macro_positive_iou"] is not None and base["macro_positive_iou"] is not None else None,
        "additional_undetected": trial["delay_never_detected"] - base["delay_never_detected"],
        "median_delay_change_us": trial["delay_median_us"] - base["delay_median_us"]
        if trial["delay_median_us"] is not None and base["delay_median_us"] is not None else None,
    }


def _decline_reasons(changes):
    reasons = []
    if changes["fewer_false_present"] < 0:
        reasons.append("more_false_present")
    if changes["recall_change"] is None or changes["recall_change"] < -_MAX_RECALL_LOSS - 1e-12:
        reasons.append("recall_loss_or_unmeasured_recall")
    if changes["additional_false_absent"] > 0:
        reasons.append("more_false_absent")
    if changes["lost_localized_boxes"] > _MAX_LOCALIZED_LOSS:
        reasons.append("fewer_localized_boxes")
    if changes["box_iou_change"] is not None and changes["box_iou_change"] < -_MAX_IOU_LOSS - 1e-12:
        reasons.append("box_iou_regression")
    if changes["additional_undetected"] > 0:
        reasons.append("more_undetected_appearances")
    delay = changes["median_delay_change_us"]
    if delay is not None and delay > _MAX_EXTRA_MEDIAN_DELAY_US:
        reasons.append("appearance_delay_regression")
    if not (
        changes["fewer_false_present"] > 0
        or changes["fewer_flicker_edges"] > 0
        or changes["paired_jitter_gain"] > _MIN_JITTER_GAIN
    ):
        reasons.append("no_measured_improvement")
    return reasons
