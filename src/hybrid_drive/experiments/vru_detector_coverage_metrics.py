"""Paired detector coverage on one fixed set of annotated current centers."""

from collections import defaultdict
from dataclasses import asdict, dataclass
from math import dist, fsum
import re
from statistics import median

from hybrid_drive.experiments.object_vru_metrics import (
    MATCH_GAP_M, MATCH_RADIUS_M, MOVING_M, QUIET_M, _finite, _identifier, _xy, match_current_centers,
)


ARMS = ("reference", "candidate")
_CATEGORIES = ("moving", "quiet", "intermediate")
MIN_ADDITIONAL_MATCHED = 15
MIN_IMPROVED_RECORDINGS = 4
MIN_COMMON_MATCHES = 12
MAX_COMMON_MEAN_ERROR_INCREASE_M = .20
_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class _Match:
    object_id: str
    center_error_m: float


@dataclass(frozen=True)
class _Pair:
    target_index: int
    category: str
    reference: _Match | None
    candidate: _Match | None


@dataclass(frozen=True)
class _Window:
    sample_token: str
    recording: str
    reference_detections: int
    candidate_detections: int
    pairs: tuple[_Pair, ...]


def summarize_detector_coverage(reference_rows, candidate_rows, labels):
    """Match the two detector profiles independently, compare exact shared targets.

    Only the current target center affects assignment. Future centers classify
    the resulting pairs for reporting; they do not enter the detector match.
    An unmatched detection is not necessarily false: the labels describe a
    curated subset, not every observable actor.
    """
    windows = _bind(reference_rows, candidate_rows, labels)
    overall = _summarize(windows)
    grouped = defaultdict(list)
    for row in windows:
        grouped[row.recording].append(row)
    by_recording = {key: _summarize(rows) for key, rows in sorted(grouped.items())}
    return {
        **overall,
        "by_recording": by_recording,
        "rows": [{"sample_token": row.sample_token, "recording": row.recording,
                  "pairs": [asdict(pair) for pair in row.pairs]} for row in windows],
        "policy": {"current_match": "frozen_object_vru_mutual_nearest_two_sided_margin",
                   "match_radius_m": MATCH_RADIUS_M, "match_gap_m": MATCH_GAP_M,
                   "moving_threshold_m": MOVING_M, "quiet_threshold_m": QUIET_M,
                   "motion_category_from": "current_to_last_label_displacement",
                   "unmatched_detections": "not_precision_incompletely_labeled_scene"},
        "gate": _gate(overall, by_recording),
    }


def _bind(reference_rows, candidate_rows, labels):
    reference_rows, candidate_rows, labels = map(tuple, (reference_rows, candidate_rows, labels))
    if not reference_rows or len(reference_rows) != len(candidate_rows) or len(reference_rows) != len(labels):
        raise ValueError("Both detector profiles and labels must cover the same nonempty window set")
    windows, seen = [], set()
    for reference, candidate, label in zip(reference_rows, candidate_rows, labels, strict=True):
        binding = _binding(reference)
        token, _, recording, _ = binding
        if token in seen or token != label["sample_token"] or binding != _binding(candidate):
            raise ValueError("Detector or label observations differ or repeat a window")
        seen.add(token)
        targets = _read_targets(label)
        reference_objects = _read_objects(reference["objects"])
        candidate_objects = _read_objects(candidate["objects"])
        centers = [track[0] for track in targets]
        a, b = (_matches(objects, centers) for objects in (reference_objects, candidate_objects))
        pairs = tuple(_Pair(i, _category(dist(track[0], track[-1])), a.get(i), b.get(i))
                      for i, track in enumerate(targets))
        windows.append(_Window(token, recording, len(reference_objects), len(candidate_objects), pairs))
    return windows


def _binding(row):
    current = _identifier(row["sample_token"], "current camera observation token")
    previous = _identifier(row["previous_token"], "previous camera observation token")
    recording = _identifier(row["recording"], "recording")
    sha = row["current_sha256"]
    if current == previous or not isinstance(sha, str) or not _SHA256.fullmatch(sha):
        raise ValueError("Expected distinct observations and a canonical current camera SHA256")
    return current, previous, recording, sha


def _read_targets(label):
    elapsed = _xy(label["target_dt_s"], "target intervals")
    if abs(elapsed[0] - .5) > .1 + 1e-12 or abs(elapsed[1] - 1.) > .1 + 1e-12:
        raise ValueError("Labels must provide intervals around 0.5 s and 1.0 s")
    tracks = tuple(tuple(_xy(xy, "true center") for xy in track["centers_xy_m"]) for track in label["tracks"])
    if any(len(track) != 3 for track in tracks):
        raise ValueError("Each target must have current and two future centers")
    return tracks


def _read_objects(rows):
    items, seen = [], set()
    for row in rows:
        object_id = _identifier(row["object_id"], "detection ID")
        if object_id in seen or row["kind"] not in ("person", "bicycle"):
            raise ValueError("Duplicate detection ID or unexpected object class")
        seen.add(object_id)
        if not 0 <= _finite(row["confidence"], "detection confidence") <= 1:
            raise ValueError("Detection confidence must be in [0,1]")
        items.append((object_id, _xy(row["xy_m"], "detected center")))
    return items


def _matches(objects, centers):
    matches = match_current_centers([xy for _, xy in objects], centers)
    return {j: _Match(objects[i][0], dist(objects[i][1], centers[j])) for i, j in matches}


def _category(movement):
    if movement >= MOVING_M:
        return "moving"
    if movement <= QUIET_M:
        return "quiet"
    return "intermediate"


def _summarize(windows):
    pairs = [p for row in windows for p in row.pairs]
    slices = {"all": _slice(pairs),
              **{key: _slice([p for p in pairs if p.category == key]) for key in _CATEGORIES}}
    counts = {"windows": len(windows), "recordings": len({row.recording for row in windows}),
              "annotated_tracks": len(pairs), **{key + "_tracks": slices[key]["targets"] for key in _CATEGORIES}}
    arms = {}
    for name in ARMS:
        detections = sum(getattr(row, name + "_detections") for row in windows)
        coverage = slices["all"]["arms"][name]
        matched = coverage["matched_tracks"]
        arms[name] = {**coverage, "detections": detections, "unmatched_tracks": len(pairs) - matched,
                      "unmatched_detections": detections - matched,
                      "matched_windows": sum(any(getattr(p, name) is not None for p in row.pairs) for row in windows)}
    return {"counts": counts, "arms": arms, "slices": slices}


def _slice(pairs):
    both = [p for p in pairs if p.reference is not None and p.candidate is not None]
    matched = {name: sum(getattr(p, name) is not None for p in pairs) for name in ARMS}
    a, b = ([getattr(p, arm).center_error_m for p in both] for arm in ARMS)
    total = len(pairs)
    return {
        "targets": total, "both": len(both), "only_reference": matched["reference"] - len(both),
        "only_candidate": matched["candidate"] - len(both), "neither": total - sum(matched.values()) + len(both),
        "arms": {arm: {"matched_tracks": count, "matched_annotated_fraction": count / total if total else None}
                 for arm, count in matched.items()},
        "paired_center_error": {"common_tracks": len(both),
                                "reference_mean_m": _mean(a), "candidate_mean_m": _mean(b),
                                "reference_median_m": median(a) if a else None,
                                "candidate_median_m": median(b) if b else None,
                                "candidate_minus_reference_mean_m": _mean([y - x for x, y in zip(a, b, strict=True)])},
    }


def _mean(values):
    return fsum(values) / len(values) if values else None


def _matched_gain(summary):
    arms = summary["arms"]
    return arms["candidate"]["matched_tracks"] - arms["reference"]["matched_tracks"]


def _gate(overall, by_recording):
    gain = _matched_gain(overall)
    spread = sum(_matched_gain(row) > 0 for row in by_recording.values())
    paired = overall["slices"]["all"]["paired_center_error"]
    shared = paired["common_tracks"]
    center_change = paired["candidate_minus_reference_mean_m"]
    if shared < MIN_COMMON_MATCHES:
        status = "insufficient_common_support"
    elif (gain >= MIN_ADDITIONAL_MATCHED and spread >= MIN_IMPROVED_RECORDINGS
          and center_change <= MAX_COMMON_MEAN_ERROR_INCREASE_M + 1e-12):
        status = "coverage_signal"
    else:
        status = "coverage_gain_not_established"
    return {"status": status, "additional_matched": gain, "improved_recordings": spread,
            "common_matches": shared, "common_mean_error_change_m": center_change,
            "minimum_additional_matched": MIN_ADDITIONAL_MATCHED,
            "minimum_improved_recordings": MIN_IMPROVED_RECORDINGS, "minimum_common_matches": MIN_COMMON_MATCHES,
            "maximum_common_mean_error_increase_m": MAX_COMMON_MEAN_ERROR_INCREASE_M}
