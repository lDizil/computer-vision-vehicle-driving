"""Paired coverage and tracking diagnostics for a frozen VRU fusion policy."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import dist, fsum

from hybrid_drive.experiments.triple_vru_metrics import summarize_three_frame
from hybrid_drive.experiments.vru_detector_coverage_metrics import summarize_detector_coverage


ARMS = ("ssdlite", "faster_rcnn", "ssd_primary")
_FUSED = "ssd_primary"
_MIN_ADDITIONAL_MATCHED = 5
_MIN_IMPROVED_RECORDINGS = 4
_MIN_COMMON_MATCHES = 12
_MAX_COMMON_CENTER_INCREASE_M = .20
_EPS = 1e-12
_OBSERVATION_FIELDS = ("sample_token", "previous_token", "older_token", "recording", "current_sha256",
                       "ref_time_us", "history_s", "older_history_s")
_OPTIONAL_HISTORY_HASHES = ("previous_sha256", "older_sha256")
_FORECAST = "three_frame_trend"
_LAST_HORIZON = "1.0s"


@dataclass(frozen=True)
class _Errors:
    displacement_m: float
    future_center_m: float
    nonzero_velocity: bool
    confirmed: bool


@dataclass(frozen=True)
class _Pair:
    recording: str
    category: str
    single: _Errors
    fused: _Errors


def summarize_fusion_holdout(arms, labels):
    """Use the frozen current-center matcher and compare motion on common targets.

    Per-arm tracking counts measure support, not paired accuracy. An abstention
    has zero forecast displacement for scoring; it is not a stationary label.
    """
    rows = _aligned_rows(arms)
    labels = tuple(labels)
    singles = summarize_detector_coverage(rows["ssdlite"], rows["faster_rcnn"], labels)
    best = ("faster_rcnn" if singles["arms"]["candidate"]["matched_tracks"]
            >= singles["arms"]["reference"]["matched_tracks"] else "ssdlite")
    fusion = summarize_detector_coverage(rows[best], rows[_FUSED], labels)
    recordings = {
        name: {"targets": scores["counts"]["annotated_tracks"], "matched": _counts(scores, fusion["by_recording"][name]),
               "only_single": fusion["by_recording"][name]["slices"]["all"]["only_reference"],
               "only_fusion": fusion["by_recording"][name]["slices"]["all"]["only_candidate"]}
        for name, scores in singles["by_recording"].items()
    }
    tracking = {name: _tracking_support(summarize_three_frame(history, labels)) for name, history in rows.items()}
    motion = _paired_motion(rows[best], rows[_FUSED], labels, fusion["rows"], best)
    return {
        "counts": singles["counts"],
        "best_single": best,
        "coverage": {
            "arms": _coverage_arms(singles, fusion),
            "single_model_overlap": singles["slices"],
            "vs_best_single": fusion["slices"],
            "by_recording": recordings,
        },
        "gate": _coverage_gate(fusion),
        "tracking": {"by_arm": tracking, "paired_against_best_single": motion},
        "policy": {**singles["policy"], "single_selection": "most_current_matches_tie_favors_faster_rcnn",
                   "motion_comparison": "targets_matched_by_both_selected_profiles",
                   "unknown_motion": "zero_displacement_comparator_not_stationary_evidence",
                   "scope": "offline_diagnostic_only"},
    }


def _aligned_rows(arms):
    if not isinstance(arms, Mapping) or set(arms) != set(ARMS):
        raise ValueError("Expected exactly the three frozen detector profiles")
    rows = {name: tuple(arms[name]) for name in ARMS}
    count = len(rows[ARMS[0]])
    if not count or any(len(row) != count for row in rows.values()):
        raise ValueError("Detector profiles must cover the same nonempty window set")
    for profiles in zip(*(rows[name] for name in ARMS), strict=True):
        first = _observation(profiles[0])
        if any(_observation(row) != first for row in profiles[1:]):
            raise ValueError("Detector observations or history timestamps differ across profiles")
    return rows


def _observation(row):
    stamp = row.get("ref_time_us")
    if any(key not in row for key in _OBSERVATION_FIELDS) or type(stamp) is not int or stamp < 0:
        raise ValueError("Expected complete detector observation bindings and an integer reference timestamp")
    return tuple(row[key] for key in _OBSERVATION_FIELDS) + tuple(row.get(key) for key in _OPTIONAL_HISTORY_HASHES)


def _coverage_arms(singles, fusion):
    return {
        name: {**summary["arms"][side], **{
            group + "_matched": summary["slices"][group]["arms"][side]["matched_tracks"]
            for group in ("moving", "quiet", "intermediate")
        }}
        for name, summary, side in (("ssdlite", singles, "reference"), ("faster_rcnn", singles, "candidate"),
                                    (_FUSED, fusion, "candidate"))
    }


def _counts(singles, fusion):
    return {"ssdlite": singles["arms"]["reference"]["matched_tracks"],
            "faster_rcnn": singles["arms"]["candidate"]["matched_tracks"],
            _FUSED: fusion["arms"]["candidate"]["matched_tracks"]}


def _coverage_gate(fusion):
    prior = fusion["gate"]
    additional = prior["additional_matched"]
    improved = prior["improved_recordings"]
    shared = prior["common_matches"]
    center_change = prior["common_mean_error_change_m"]
    if shared < _MIN_COMMON_MATCHES:
        status = "insufficient_common_support"
    elif (additional >= _MIN_ADDITIONAL_MATCHED and improved >= _MIN_IMPROVED_RECORDINGS
          and center_change <= _MAX_COMMON_CENTER_INCREASE_M + _EPS):
        status = "coverage_signal"
    else:
        status = "coverage_gain_not_established"
    return {"status": status, "additional_matched": additional, "improved_recordings": improved,
            "common_matches": shared, "common_mean_error_change_m": center_change,
            "minimum_additional_matched": _MIN_ADDITIONAL_MATCHED, "minimum_improved_recordings": _MIN_IMPROVED_RECORDINGS,
            "minimum_common_matches": _MIN_COMMON_MATCHES,
            "maximum_common_mean_error_increase_m": _MAX_COMMON_CENTER_INCREASE_M}


def _tracking_support(report):
    horizon = report["horizons"][_LAST_HORIZON]
    moving = horizon["confirmed"]["moving"]
    quiet = horizon["all_matched"]["quiet"]
    return {"counts": report["counts"], "moving_confirmed": moving["pairs"],
            "moving_confirmed_recordings": moving["recordings"],
            "moving_matched": horizon["all_matched"]["moving"]["pairs"],
            "quiet_matched": quiet["pairs"], "quiet_nonzero_velocity": quiet["arms"][_FORECAST]["nonzero_velocity"],
            "gate": report["gate"]}


def _paired_motion(single, fused, labels, assignments, best):
    pairs = []
    for a, b, label, window in zip(single, fused, labels, assignments, strict=True):
        left, right = ({obj["object_id"]: obj for obj in row["objects"]} for row in (a, b))
        for match in window["pairs"]:
            if match["reference"] is None or match["candidate"] is None:
                continue
            actual = label["tracks"][match["target_index"]]["centers_xy_m"]
            elapsed = label["target_dt_s"][1]
            pairs.append(_Pair(window["recording"], match["category"],
                               _errors(left[match["reference"]["object_id"]], actual, elapsed),
                               _errors(right[match["candidate"]["object_id"]], actual, elapsed)))
    return {"horizon_s": 1., "all": _paired_summary(pairs, best),
            **{kind: _paired_summary([row for row in pairs if row.category == kind], best)
               for kind in ("moving", "quiet")}}


def _errors(obj, actual, elapsed):
    velocity = obj["forecasts"][_FORECAST]
    delta = tuple(v * elapsed for v in velocity)
    target_delta = tuple(b - a for a, b in zip(actual[0], actual[-1], strict=True))
    future = tuple(a + b for a, b in zip(obj["xy_m"], delta, strict=True))
    return _Errors(dist(delta, target_delta), dist(future, actual[-1]), any(velocity),
                   obj["confirmation"]["status"] == "confirmed")


def _paired_summary(pairs, best):
    arms = {}
    for name, field in ((best, "single"), (_FUSED, "fused")):
        values = [getattr(pair, field) for pair in pairs]
        arms[name] = {"displacement_mean_m": _mean(v.displacement_m for v in values),
                      "future_center_mean_m": _mean(v.future_center_m for v in values),
                      "nonzero_velocity": sum(v.nonzero_velocity for v in values),
                      "confirmed": sum(v.confirmed for v in values)}
    return {"pairs": len(pairs), "recordings": len({p.recording for p in pairs}), "arms": arms}


def _mean(values):
    values = tuple(values)
    return fsum(values) / len(values) if values else None
