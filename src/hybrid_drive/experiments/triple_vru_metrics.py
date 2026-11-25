"""Compare confirmation and three-frame trends on the frozen #51 current matches."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from math import fsum, hypot
from statistics import median

from hybrid_drive.experiments.object_vru_metrics import (
    HORIZONS, MATCH_GAP_M, MATCH_RADIUS_M, MIN_MOVING_GAIN_M, MIN_QUIET, MOVING_M,
    QUIET_M, TEST_MIN_MOVING, TEST_MIN_RECORDINGS, _bind, _finite, _identifier, _xy,
)


ARMS = ("stationary", "two_frame", "confirmed_two_frame", "three_frame_trend")
_CATEGORIES = ("all", "moving", "quiet", "intermediate")
_ZERO = (0., 0.)
_EPS = 1e-12


@dataclass(frozen=True)
class _Forecast:
    accepted_two_frame: bool
    status: str
    velocity_by_arm: dict[str, tuple[float, float]]

    @property
    def confirmed(self):
        return self.status == "confirmed"


@dataclass(frozen=True)
class _Comparison:
    recording: str
    category: str
    forecast: _Forecast
    errors: dict[str, tuple[float, float]]


def summarize_three_frame(rows, labels):
    """Keep all #51 current matches even when a history link is unavailable.

    The stationarity arm and abstaining forecasts use zero displacement as a
    comparison baseline, not as evidence that the object is stationary.
    """
    rows = tuple(rows)
    base = _bind(rows, labels)
    windows = [_window(row, pair) for row, pair in zip(rows, base)]
    overview = _summarize(windows)
    recordings = defaultdict(list)
    for window in windows:
        recordings[window[0].recording].append(window)
    return {
        **overview,
        "policy": {
            "current_match": "frozen_object_vru_mutual_nearest_two_sided_margin",
            "match_radius_m": MATCH_RADIUS_M, "match_gap_m": MATCH_GAP_M,
            "moving_threshold_m": MOVING_M, "quiet_threshold_m": QUIET_M,
            "unknown_motion": "zero_displacement_comparator_not_stationary_evidence",
        },
        "by_recording": {name: _summarize(values) for name, values in sorted(recordings.items())},
        "gate": _gate(overview["horizons"]["1.0s"]),
    }


def _window(row, base):
    current = row["sample_token"]
    previous = _identifier(row["previous_token"], "previous observation token")
    older = row["older_token"]
    if current == previous:
        raise ValueError("Current and previous observations must be different")
    if older is None:
        if row.get("older_history_s") is not None:
            raise ValueError("An absent third observation cannot have a history interval")
    else:
        _identifier(older, "older observation token")
        if older in (current, previous):
            raise ValueError("The third observation must be different from the first two")
        if not .4 <= _finite(row.get("older_history_s"), "older history interval") <= .6:
            raise ValueError("Expected older history interval in [0.4,0.6] s")
    seen_older = set()
    forecasts = tuple(_forecast(obj, older is not None, seen_older) for obj in row["objects"])
    return base, older is not None, forecasts


def _forecast(row, has_older, seen_older):
    first = row["motions"]["appearance"]
    accepted = first["status"] == "estimated"
    confirmation = row["confirmation"]
    status = _identifier(confirmation["status"], "confirmation status")
    older = confirmation["older_object_id"]
    cameras = confirmation["common_cameras"]
    if (not isinstance(cameras, (tuple, list))
            or any(not isinstance(name, str) or not name.strip() for name in cameras)
            or len(cameras) != len(set(cameras))):
        raise ValueError("Expected unique common-camera names in three-frame confirmation")
    trend = _xy(confirmation["trend_velocity_xy_mps"], "confirmed trend velocity")
    if status == "confirmed":
        if not has_older or not accepted:
            raise ValueError("Confirmation requires a third observation and an accepted two-frame link")
        _identifier(older, "confirmed older detection ID")
        if older in seen_older:
            raise ValueError("Two confirmed objects cannot reuse one older detection")
        seen_older.add(older)
        if not cameras or not set(cameras) <= set(first["common_cameras"]):
            raise ValueError("Confirmation cameras must be shared with the accepted two-frame link")
    elif older is not None or cameras or trend != _ZERO:
        raise ValueError("An abstaining confirmation must not supply an older link, camera or velocity")

    raw = row["forecasts"]
    if set(raw) != set(ARMS):
        raise ValueError("Expected exactly the four fixed motion forecasts")
    velocities = {name: _xy(raw[name], name + " velocity") for name in ARMS}
    two_frame = velocities["two_frame"]
    if velocities["stationary"] != _ZERO or (not accepted and two_frame != _ZERO):
        raise ValueError("Stationary and unavailable two-frame forecasts must have zero velocity")
    kept = two_frame if status == "confirmed" else _ZERO
    if velocities["confirmed_two_frame"] != kept or velocities["three_frame_trend"] != trend:
        raise ValueError("Confirmed forecast velocities must match the exact retained link and measured trend")
    return _Forecast(accepted, status, velocities)


def _summarize(windows):
    annotations = sum(len(base.tracks) for base, _, _ in windows)
    forecasts = [forecast for _, _, items in windows for forecast in items]
    matched = [items[i] for base, _, items in windows for i, _ in base.matches]
    with_older = sum(has_older for _, has_older, _ in windows)
    counts = {
        "windows": len(windows), "recordings": len({row.recording for row, _, _ in windows}),
        "windows_with_older": with_older, "windows_without_older": len(windows) - with_older,
        "annotated_tracks": annotations, "detections": len(forecasts), "matched_tracks": len(matched),
        "matched_windows": sum(bool(row.matches) for row, _, _ in windows),
        "unmatched_tracks": annotations - len(matched), "unmatched_detections": len(forecasts) - len(matched),
        "matched_with_older": sum(len(row.matches) for row, has_older, _ in windows if has_older),
        "accepted_two_frame": sum(row.accepted_two_frame for row in matched),
        "confirmed_three_frame": sum(row.confirmed for row in matched),
        "detections_with_accepted_two_frame": sum(row.accepted_two_frame for row in forecasts),
        "detections_with_confirmed_three_frame": sum(row.confirmed for row in forecasts),
    }
    horizons = {key: _horizon(windows, step) for step, key in enumerate(HORIZONS)}
    return {"counts": counts, "horizons": horizons,
            "confirmation_statuses": _statuses(forecasts), "matched_confirmation_statuses": _statuses(matched)}


def _statuses(forecasts):
    return dict(sorted(Counter(row.status for row in forecasts).items()))


def _horizon(windows, step):
    elapsed = [row.elapsed[step] for row, _, _ in windows]
    comparisons = _compare(windows, step)
    groups = (("all_matched", comparisons),
              ("accepted_two_frame", [x for x in comparisons if x.forecast.accepted_two_frame]),
              ("confirmed", [x for x in comparisons if x.forecast.confirmed]))
    subsets = {name: {category: _aggregate([x for x in values if category == "all" or x.category == category])
                     for category in _CATEGORIES} for name, values in groups}
    return {"seconds": .5 * (step + 1),
            "observed_target_dt_s": {"minimum": min(elapsed), "maximum": max(elapsed)}, **subsets}


def _compare(windows, step):
    comparisons = []
    for base, _, forecasts in windows:
        elapsed = base.elapsed[step]
        for i, j in base.matches:
            detection, track = base.detections[i], base.tracks[j]
            now, future = track[0], track[step + 1]
            move = tuple(b - a for a, b in zip(now, future))
            distance = hypot(*move)
            category = "moving" if distance >= MOVING_M else "quiet" if distance <= QUIET_M else "intermediate"
            forecast = forecasts[i]
            errors = {}
            for arm, (vx, vy) in forecast.velocity_by_arm.items():
                dx, dy = vx * elapsed, vy * elapsed
                errors[arm] = (hypot(move[0] - dx, move[1] - dy),
                               hypot(future[0] - detection.xy[0] - dx, future[1] - detection.xy[1] - dy))
            comparisons.append(_Comparison(base.recording, category, forecast, errors))
    return comparisons


def _aggregate(pairs):
    arms = {}
    for arm in ARMS:
        displacement = [pair.errors[arm][0] for pair in pairs]
        center = [pair.errors[arm][1] for pair in pairs]
        arms[arm] = {"displacement_mean_m": fsum(displacement) / len(pairs) if pairs else None,
                     "displacement_median_m": median(displacement) if pairs else None,
                     "future_center_mean_m": fsum(center) / len(pairs) if pairs else None,
                     "future_center_median_m": median(center) if pairs else None,
                     "nonzero_velocity": sum(pair.forecast.velocity_by_arm[arm] != _ZERO for pair in pairs)}
    return {"pairs": len(pairs), "recordings": len({pair.recording for pair in pairs}), "arms": arms}


def _gain(section, baseline, field):
    a, b = (section["arms"][name][field] for name in (baseline, "three_frame_trend"))
    return None if a is None or b is None else a - b


def _gate(horizon):
    moving = horizon["confirmed"]["moving"]
    quiet = horizon["accepted_two_frame"]["quiet"]
    policy = {"minimum_confirmed_moving_pairs": TEST_MIN_MOVING,
              "minimum_confirmed_moving_recordings": TEST_MIN_RECORDINGS,
              "minimum_two_frame_quiet_pairs": MIN_QUIET, "minimum_moving_gain_m": MIN_MOVING_GAIN_M,
              "moving_comparison": "all_matched", "quiet_comparison": "accepted_two_frame",
              "require_no_center_regression_vs_stationary": True, "horizon_s": 1.}
    quiet_speeds = quiet["arms"]["two_frame"]["nonzero_velocity"] - quiet["arms"]["three_frame_trend"]["nonzero_velocity"]
    motion_gain = _gain(horizon["all_matched"]["moving"], "stationary", "displacement_mean_m")
    quiet_gain = _gain(quiet, "two_frame", "displacement_mean_m")
    center_gain = _gain(horizon["all_matched"]["all"], "stationary", "future_center_mean_m")
    if moving["pairs"] < TEST_MIN_MOVING or moving["recordings"] < TEST_MIN_RECORDINGS or quiet["pairs"] < MIN_QUIET:
        status = "insufficient_support"
    elif (motion_gain + _EPS < MIN_MOVING_GAIN_M or not quiet_gain > _EPS or quiet_speeds <= 0
          or center_gain < -_EPS):
        status = "motion_gain_not_established"
    else:
        status = "confirmation_signal"
    return {"status": status, **policy, "confirmed_moving_pairs": moving["pairs"],
            "confirmed_moving_recordings": moving["recordings"], "two_frame_quiet_pairs": quiet["pairs"],
            "moving_displacement_gain_m": motion_gain,
            "moving_displacement_gain_vs_two_frame_m": _gain(horizon["all_matched"]["moving"], "two_frame", "displacement_mean_m"),
            "quiet_displacement_gain_vs_two_frame_m": quiet_gain,
            "quiet_nonzero_velocity_reduction": quiet_speeds, "future_center_gain_m": center_gain}
