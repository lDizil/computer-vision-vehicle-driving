"""Paired box changes and conservative offline selection at IoU 0.5."""

from collections.abc import Mapping
from dataclasses import replace
from itertools import product

from .spatial_decision_policy import PRESENT
from .spatial_track_metrics import TrackingTarget, match_tracking_frame
from .spatial_zoom_metrics import _decisions, _fixed_summary, _valid_count


GUARD_MODES = ("baseline", "zoom_full", "zoom_half", "gate", "gate_supported")
_MIN_IOU = 0.5
_MIN_GAIN = 2
_MAX_RECORDING_DROP = 1
_MATCHED = "detections_matched"
_CHANGE_FIELDS = ("frames", "gained_targets", "lost_targets", "changed_present_slots")
_ORACLE_FIELDS = (
    "frames", "labeled_boxes", "output_boxes",
    "baseline_detections_matched", "zoom_half_detections_matched", "zoom_full_detections_matched",
    "oracle_detections_matched", "recoverable_matches",
)


def evaluate_guard_changes(original, adjusted, targets) -> dict:
    """Count changes to matched targets, not changes to assignment partners.

    Matching is run afresh for each method. Losing one target and acquiring
    another in the same frame records both events, even with no net gain.
    """
    before, after = _fixed_geometry_cohorts(targets, original, adjusted)
    by_recording = {}
    for source, revised in zip(before, after):
        target = targets[source.observation.sample_id]
        old = {label for _, label in match_tracking_frame(source, target, min_iou=_MIN_IOU)}
        new = {label for _, label in match_tracking_frame(revised, target, min_iou=_MIN_IOU)}
        changed = sum(a != b for a, b in zip(source.boxes, revised.boxes))
        _add_counts(
            by_recording, source.observation.recording_id, _CHANGE_FIELDS,
            (1, len(new - old), len(old - new), changed),
        )
    return _total_counts(by_recording, _CHANGE_FIELDS)


def oracle_guard_match_ceiling(baseline, half, full, targets) -> dict:
    """Find the exact label-aware count ceiling of the three fixed box choices.

    For each PRESENT slot choose only its original, half- or full-zoom
    coordinates. All slots in the frame must share one one-to-one assignment;
    independently counting the union of recovered labels would overstate it.
    This is an offline bound, not a candidate or deployable prediction.
    """
    cohorts = _fixed_geometry_cohorts(targets, baseline, half, full)
    by_recording = {}
    for frame in zip(*cohorts):
        source = frame[0].observation
        target = targets[source.sample_id]
        labels = sum(box is not None for box in target.boxes)
        visible = frame[0].states.count(PRESENT)
        singles = tuple(len(match_tracking_frame(row, target, min_iou=_MIN_IOU)) for row in frame)
        ceiling = _oracle_frame(frame, target, max(singles), min(labels, visible))
        _add_counts(
            by_recording, source.recording_id, _ORACLE_FIELDS,
            (1, labels, visible, *singles, ceiling, ceiling - singles[0]),
        )
    return {
        "scope": "labels_only_fixed_present_box_choices",
        "min_iou": _MIN_IOU,
        **_total_counts(by_recording, _ORACLE_FIELDS),
    }


def _oracle_frame(frame, target, best, capacity):
    if best == capacity:
        return best
    source = frame[0]
    alternatives = [tuple(dict.fromkeys(slot)) for slot in zip(*(row.boxes for row in frame))]
    singles = {row.boxes for row in frame}
    for boxes in product(*alternatives):
        if boxes in singles:
            continue
        matches = match_tracking_frame(replace(source, boxes=boxes), target, min_iou=_MIN_IOU)
        best = max(best, len(matches))
        if best == capacity:
            break
    return best


def _fixed_geometry_cohorts(targets, *values):
    original, *alternatives = tuple(_decisions(value) for value in values)
    if not isinstance(targets, Mapping) or set(targets) != {d.observation.sample_id for d in original}:
        raise ValueError("Expected exactly one tracking target per evaluated sample ID")
    if any(not isinstance(target, TrackingTarget) for target in targets.values()):
        raise ValueError("Expected TrackingTarget values by sample ID")
    for alternative in alternatives:
        if len(alternative) != len(original):
            raise ValueError("Geometry comparison requires the same ordered frame count")
        for before, after in zip(original, alternative):
            if before.observation != after.observation:
                raise ValueError("Geometry comparison requires the same ordered observations")
            if (before.states, before.scores, before.reasons, before.reset_reason) != (
                after.states, after.scores, after.reasons, after.reset_reason,
            ):
                raise ValueError("Geometry comparison must preserve occupancy, scores and policy reasons")
            if any(
                state != PRESENT and old != new
                for state, old, new in zip(before.states, before.boxes, after.boxes)
            ):
                raise ValueError("Geometry comparison must not change inactive boxes")
    return original, *alternatives


def _add_counts(by_recording, recording, keys, values):
    group = by_recording.setdefault(recording, dict.fromkeys(keys, 0))
    for key, value in zip(keys, values):
        group[key] += value


def _total_counts(by_recording, keys):
    return {
        **{key: sum(row[key] for row in by_recording.values()) for key in keys},
        "per_recording": by_recording,
    }


def select_guard_mode(development: Mapping[str, Mapping[str, Mapping]]) -> dict:
    """Freeze a recipe on development metrics, with no tracking-based ranking."""
    if (
        not isinstance(development, Mapping) or not development
        or any(not isinstance(role, str) or not role.strip() for role in development)
    ):
        raise ValueError("Expected named development roles with baseline metrics")
    roles = dict(sorted(development.items()))
    baselines = {role: _baseline(stats, role) for role, stats in roles.items()}
    candidates = {
        mode: _assess(mode, roles, baselines)
        for mode in GUARD_MODES[1:]
    }
    eligible = [
        (-row["matched_gain"], row["changed_present_slots"], name)
        for name, row in candidates.items() if row["eligible"]
    ]
    return {
        "name": min(eligible)[-1] if eligible else "baseline",
        "candidates": candidates,
        "summary": {
            "scope": "development_only",
            "roles": list(roles),
            "min_iou": _MIN_IOU,
            "minimum_matched_gain": _MIN_GAIN,
            "maximum_role_net_drop": 0,
            "maximum_recording_net_drop": _MAX_RECORDING_DROP,
            "tie_break": ["higher_matched_gain", "fewer_changed_present_slots", "lexical_recipe_name"],
        },
    }


def check_guard_holdout(chosen: str, single_role_stats: Mapping[str, Mapping]) -> dict:
    """Test only the preselected recipe; a failed holdout cannot nominate another."""
    if chosen not in GUARD_MODES:
        raise ValueError("Unknown frozen guard recipe")
    if chosen == "baseline":
        return {"evaluated_recipe": chosen, "outcome": "baseline_retained_on_dev", "reasons": []}
    baseline = _baseline(single_role_stats, "holdout")
    result = _assess(chosen, {"holdout": single_role_stats}, {"holdout": baseline})
    return {
        "evaluated_recipe": chosen,
        "outcome": "preselected_passed" if result["eligible"] else "no_replacement",
        "reasons": result["reasons"],
    }


def _baseline(stats, role):
    baseline = stats.get("baseline") if isinstance(stats, Mapping) else None
    fixed = _fixed_summary(baseline)
    if fixed is None or not _valid_changes(baseline, baseline, baseline=True):
        raise ValueError(f"Invalid frozen baseline metrics: {role}")
    return baseline, fixed


def _assess(mode, roles, baselines):
    reasons, per_role = [], {}
    for role, stats in roles.items():
        baseline, fixed = baselines[role]
        trial = stats.get(mode)
        invalid = _incompatibility(trial, baseline, fixed)
        if invalid is not None:
            reasons.append(f"{invalid}:{role}")
            per_role[role] = {"error": invalid}
            continue

        gain = int(trial[_MATCHED] - baseline[_MATCHED])
        base_records = baseline["per_recording"]
        drops = {}
        for recording, after in sorted(trial["per_recording"].items()):
            drop = int(base_records[recording][_MATCHED] - after[_MATCHED])
            if drop > 0:
                drops[recording] = drop
            if drop > _MAX_RECORDING_DROP:
                reasons.append(f"recording_drop:{role}:{recording}")
        if gain < 0:
            reasons.append(f"role_regression:{role}")
        per_role[role] = {
            "matched_gain": gain,
            "recording_drops": drops,
            **{key: int(trial["changes"][key]) for key in _CHANGE_FIELDS[1:]},
        }

    totals = {
        key: sum(row[key] for row in per_role.values()) if all("error" not in row for row in per_role.values()) else None
        for key in ("matched_gain", *_CHANGE_FIELDS[1:])
    }
    if totals["matched_gain"] is not None and totals["matched_gain"] < _MIN_GAIN:
        reasons.append("insufficient_matched_gain")
    return {
        "eligible": not reasons, "reasons": reasons,
        **totals, "per_role": per_role,
    }


def _incompatibility(trial, baseline, fixed):
    if trial is None:
        return "missing_metrics"
    if not isinstance(trial, Mapping):
        return "invalid_metrics"
    if trial.get("evaluation_identity") != baseline["evaluation_identity"]:
        return "different_frozen_inputs_or_labels"
    signature = _fixed_summary(trial)
    if signature is None:
        return "invalid_metrics"
    if signature != fixed:
        return "different_fixed_counts"
    if not _valid_changes(trial, baseline):
        return "invalid_change_counts"
    return None


def _valid_changes(trial, original, *, baseline=False):
    changes = trial.get("changes")
    if not _valid_change_row(changes, trial, original, baseline):
        return False
    groups = changes.get("per_recording")
    trial_groups, original_groups = trial["per_recording"], original["per_recording"]
    if not isinstance(groups, Mapping) or set(groups) != set(trial_groups):
        return False
    if any(
        not _valid_change_row(row, trial_groups[recording], original_groups[recording], baseline)
        for recording, row in groups.items()
    ):
        return False
    return all(sum(row[key] for row in groups.values()) == changes[key] for key in _CHANGE_FIELDS)


def _valid_change_row(changes, score, initial, baseline):
    if not isinstance(changes, Mapping) or any(not _valid_count(changes.get(key)) for key in _CHANGE_FIELDS):
        return False
    gained, lost, moved = (changes[key] for key in _CHANGE_FIELDS[1:])
    before, after = initial[_MATCHED], score[_MATCHED]
    return (
        changes["frames"] == score["frames"]
        and moved <= score["output_boxes"]
        and (not (gained or lost) or moved > 0)
        and (not baseline or not (gained or lost or moved))
        and lost <= before and gained <= after
        and before + gained <= score["labeled_boxes"]
        and gained - lost == after - before
        and abs(after - before) <= moved
    )
