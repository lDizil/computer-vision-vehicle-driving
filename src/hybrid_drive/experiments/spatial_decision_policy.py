"""Offline three-way decisions for conditioned per-zone presence scores."""

from bisect import bisect_right
from dataclasses import dataclass
import math
from numbers import Real

import numpy as np


ABSENT = -1
UNCERTAIN = 0
PRESENT = 1


@dataclass(frozen=True)
class ZoneDecisionPolicy:
    absent_below: float
    present_at_or_above: float

    def __post_init__(self):
        low = _finite_float(self.absent_below, "absent_below")
        high = _finite_float(self.present_at_or_above, "present_at_or_above")
        if low > high:
            raise ValueError("absent_below must not exceed present_at_or_above")
        object.__setattr__(self, "absent_below", low)
        object.__setattr__(self, "present_at_or_above", high)

    def decide(self, scores: np.ndarray) -> np.ndarray:
        """Return int8 states; the interval [absent_below, present_at_or_above) abstains."""
        values = _checked_scores(scores)
        decisions = np.full(values.shape, UNCERTAIN, dtype=np.int8)
        decisions[values < self.absent_below] = ABSENT
        decisions[values >= self.present_at_or_above] = PRESENT
        return decisions


def calibrate_zone_policy(
    scores, present, recording_ids, *, max_false_positive_rate=0.20, max_false_negative_rate=0.10
) -> ZoneDecisionPolicy:
    """Fit one pair of dev cutoffs, bounding confident errors in each recording.

    Minimize the presence cutoff by advancing one float64 step above the
    greatest negative score that must be excluded. If no recording constrains
    the cutoff, use the minimum observed score. Maximize the absence cutoff
    up to that cutoff, keeping tied positives at the boundary out of ABSENT.
    """
    values = _checked_scores(scores)
    groups = _labeled_groups(values.shape, present, recording_ids)
    max_fp = _budget(max_false_positive_rate, "max_false_positive_rate")
    max_fn = _budget(max_false_negative_rate, "max_false_negative_rate")
    if len(groups) < 2 or not present.any() or present.all():
        raise ValueError("Calibration requires at least two recordings and both classes")

    negatives_to_exclude = []
    positive_ceiling = math.inf
    for rows in groups.values():
        labels, grouped = present[rows], values[rows]
        neg, pos = grouped[~labels], grouped[labels]
        allowed_fp = _allowed_errors(len(neg), max_fp)
        allowed_fn = _allowed_errors(len(pos), max_fn)
        if allowed_fp < len(neg):
            rank = len(neg) - allowed_fp - 1
            negatives_to_exclude.append(float(np.partition(neg, rank)[rank]))
        if allowed_fn < len(pos):
            cut = float(np.partition(pos, allowed_fn)[allowed_fn])
            positive_ceiling = min(positive_ceiling, cut)

    high = (
        math.nextafter(max(negatives_to_exclude), math.inf)
        if negatives_to_exclude else float(values.min())
    )
    if not math.isfinite(high):
        raise ValueError("No finite float64 presence cutoff satisfies the false-positive budget")
    return ZoneDecisionPolicy(min(high, positive_ceiling), high)


def measure_zone_policy(scores, present, recording_ids, policy: ZoneDecisionPolicy) -> dict:
    """Assess confident decisions against all labels, counting abstentions separately."""
    if not isinstance(policy, ZoneDecisionPolicy):
        raise ValueError("Expected a ZoneDecisionPolicy")
    decisions = policy.decide(scores)
    groups = _labeled_groups(decisions.shape, present, recording_ids)
    result = _measure(decisions, present)
    result["per_recording"] = {
        identity: _measure(decisions[rows], present[rows]) for identity, rows in groups.items()
    }
    return result


def _measure(decisions, labels):
    positives = int(labels.sum())
    negatives = labels.size - positives
    marked_present = decisions == PRESENT
    marked_absent = decisions == ABSENT
    present_count, absent_count = int(marked_present.sum()), int(marked_absent.sum())
    true_present = int((marked_present & labels).sum())
    false_present = present_count - true_present
    true_absent = int((marked_absent & ~labels).sum())
    false_absent = absent_count - true_absent
    return {
        "present_count": present_count,
        "absent_count": absent_count,
        "uncertain_count": labels.size - present_count - absent_count,
        "true_present": true_present,
        "false_present": false_present,
        "true_absent": true_absent,
        "false_absent": false_absent,
        "positives": positives,
        "negatives": negatives,
        "positive_recall": _ratio(true_present, positives),
        "false_positive_rate": _ratio(false_present, negatives),
        "false_negative_rate": _ratio(false_absent, positives),
        "precision": _ratio(true_present, present_count),
        "coverage": _ratio(present_count + absent_count, labels.size),
    }


def _allowed_errors(count, rate):
    # Compare the same rounded ratios reported by _measure, not rate * count
    # (which can round an integer just below the admissible boundary).
    return bisect_right(range(1, count + 1), rate, key=lambda errors: errors / count)


def _ratio(count, total):
    return count / total if total else None


def _checked_scores(scores):
    if (
        not isinstance(scores, np.ndarray)
        or scores.ndim != 2
        or scores.shape[1] != 4
        or len(scores) == 0
        or scores.dtype.kind != "f"
    ):
        raise ValueError("scores must be a nonempty floating [N,4] array")
    with np.errstate(over="ignore", invalid="ignore"):
        values = scores.astype(np.float64, copy=False)
    if not np.isfinite(values).all():
        raise ValueError("scores must be finite in float64")
    return values


def _labeled_groups(shape, present, recording_ids):
    if not isinstance(present, np.ndarray) or present.dtype != np.bool_ or present.shape != shape:
        raise ValueError("present must be a boolean array aligned with scores")
    try:
        if isinstance(recording_ids, (str, bytes)):
            raise TypeError
        identities = tuple(recording_ids)
    except TypeError:
        raise ValueError("Expected one recording ID per score row") from None
    if len(identities) != shape[0] or any(
        not isinstance(identity, str) or not identity.strip() for identity in identities
    ):
        raise ValueError("Expected one nonempty recording ID per score row")
    groups = {}
    for row, identity in enumerate(identities):
        groups.setdefault(identity, []).append(row)
    return groups


def _budget(value, name):
    rate = _finite_float(value, name)
    if not 0 <= rate <= 1:
        raise ValueError(f"{name} must be between zero and one")
    return rate


def _finite_float(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    try:
        converted = float(value)
    except OverflowError:
        raise ValueError(f"{name} must be a finite real number") from None
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be a finite real number")
    return converted
