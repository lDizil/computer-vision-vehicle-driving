"""Offline diagnostics for object motion, on one fixed set of current matches.

An unassigned detection is not necessarily false: the curated labels contain
only anchor-visible, persistent targets. Motion gains are fit on development
rows supplied by the caller, never selected by the held-out evaluator.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from math import fsum, hypot, isfinite
from numbers import Real
from statistics import median


ARMS = ("stationary", "geometry", "appearance", "calibrated")
HORIZONS = ("0.5s", "1.0s")
MATCH_RADIUS_M = 2.
MATCH_GAP_M = .25
MOVING_M = .75
QUIET_M = .25
GAIN_GRID = (0., .25, .5, .75, 1.)
DEV_MIN_MOVING = 8
DEV_MIN_RECORDINGS = 2
MIN_QUIET = 4
TEST_MIN_MOVING = 20
TEST_MIN_RECORDINGS = 4
MIN_MOVING_GAIN_M = .10
MAX_QUIET_PENALTY_M = .10
_EPS = 1e-12
_ZERO = (0., 0.)


@dataclass(frozen=True)
class _Motion:
    status: str
    previous_id: str | None
    velocity: tuple
    radial: tuple | None


@dataclass(frozen=True)
class _Detection:
    xy: tuple
    geometry: _Motion
    appearance: _Motion


@dataclass(frozen=True)
class _Window:
    recording: str
    elapsed: tuple
    tracks: tuple
    detections: tuple
    matches: tuple


@dataclass(frozen=True)
class _Comparison:
    recording: str
    category: str
    status: str
    errors: dict


def match_current_centers(detections, targets):
    """Mutual closest centers within 2 m, with a 0.25 m gap on both sides.

    Returns detection/target indexes. Confidence, history, velocity and future
    centers do not influence this one-to-one, conservative assignment.
    """
    targets = tuple(targets)
    distances = [[hypot(dx - tx, dy - ty) for tx, ty in targets] for dx, dy in detections]
    detection_choices = [_unique_nearest(row) for row in distances]
    target_choices = [_unique_nearest(column) for column in zip(*distances)]
    return tuple((i, j) for i, j in enumerate(detection_choices) if j is not None and target_choices[j] == i)


def _unique_nearest(distances):
    order = sorted(enumerate(distances), key=lambda pair: pair[1])
    if not order or order[0][1] > MATCH_RADIUS_M:
        return None
    if len(order) > 1 and order[1][1] - order[0][1] + _EPS < MATCH_GAP_M:
        return None
    return order[0][0]


def fit_motion_gains(rows, labels):
    """Choose radial and tangential velocity gains from a small frozen grid."""
    windows = _bind(rows, labels)
    baseline = _horizon(_compare(windows, 1, _ZERO))
    counts = _selection_counts(baseline)
    counts.update(development_recordings=len({scene.recording for scene in windows}))
    support = (counts["moving_pairs"] >= DEV_MIN_MOVING and counts["quiet_pairs"] >= MIN_QUIET
               and counts["moving_recordings"] == counts["development_recordings"] >= DEV_MIN_RECORDINGS)
    policy = {"minimum_moving_pairs": DEV_MIN_MOVING, "minimum_moving_recordings": DEV_MIN_RECORDINGS,
              "require_moving_support_in_each_development_recording": True, "minimum_quiet_pairs": MIN_QUIET,
              "maximum_quiet_penalty_m": MAX_QUIET_PENALTY_M, "require_no_center_regression": True,
              "horizon_s": 1., "gain_grid": list(GAIN_GRID)}
    if not support:
        return {"status": "insufficient_support", "radial_gain": 0., "tangential_gain": 0.,
                "counts": counts, "policy": policy, "candidates": [], "selected": _candidate(_ZERO, baseline)}
    candidates = [_candidate((radial, tangential), _horizon(_compare(windows, 1, (radial, tangential))))
                  for radial in GAIN_GRID for tangential in GAIN_GRID]
    selected = min((candidate for candidate in candidates if candidate["eligible"]), key=_fit_order)
    radial, tangential = selected["radial_gain"], selected["tangential_gain"]
    return {"status": "calibrated_on_development" if radial or tangential else "stationary_selected",
            "radial_gain": radial, "tangential_gain": tangential,
            "counts": counts, "policy": policy, "candidates": candidates, "selected": selected}


def _fit_order(candidate):
    radial, tangential = candidate["radial_gain"], candidate["tangential_gain"]
    return round(candidate["moving_displacement_mean_m"], 12), radial + tangential, radial, tangential


def _candidate(gains, summary):
    radial, tangential = gains
    metrics = _deltas(summary)
    return {"radial_gain": radial, "tangential_gain": tangential,
            "moving_displacement_mean_m": _stat(summary, "accepted_appearance", "moving", "calibrated", "displacement_mean_m"),
            **metrics, "eligible": _valid_tradeoff(metrics)}


def summarize_object_motion(rows, labels, gains):
    """Compare the four fixed forecasts at the observed label time offsets."""
    gains = tuple(_finite(gains[key], key) for key in ("radial_gain", "tangential_gain"))
    if any(gain not in GAIN_GRID for gain in gains):
        raise ValueError("Calibration gains must come from the fixed development grid")
    windows = _bind(rows, labels)
    overall = _summary(windows, gains)
    groups = defaultdict(list)
    for window in windows:
        groups[window.recording].append(window)
    by_recording = {key: _summary(values, gains) for key, values in sorted(groups.items())}

    # Group the already fixed current matches by the appearance branch status;
    # keep the same stationary and geometry scores inside each subgroup.
    status = {name: {} for name in overall["association_statuses"]["appearance"]}
    for step, key in enumerate(HORIZONS):
        comparisons = _compare(windows, step, gains)
        for name, result in status.items():
            result[key] = _horizon([pair for pair in comparisons if pair.status == name])["all_matched"]
    return {**overall, "by_recording": by_recording, "by_motion_status": status,
            "policy": {"current_match": "mutual_nearest_two_sided_margin", "match_radius_m": MATCH_RADIUS_M,
                       "match_gap_m": MATCH_GAP_M, "moving_threshold_m": MOVING_M, "quiet_threshold_m": QUIET_M,
                       "radial_gain": gains[0], "tangential_gain": gains[1]},
            "gate": _gate(overall["horizons"]["1.0s"])}


def _bind(rows, labels):
    rows, labels = tuple(rows), tuple(labels)
    if not rows or len(rows) != len(labels):
        raise ValueError("Predictions and labels must contain the same nonempty window set")
    windows, seen = [], set()
    for row, label in zip(rows, labels):
        token = _identifier(row["sample_token"], "sample token")
        if token in seen or token != label["sample_token"]:
            raise ValueError("Prediction and label window identities differ")
        seen.add(token)
        recording = _identifier(row["recording"], "recording")
        history = _finite(row["history_s"], "history interval")
        if not .4 <= history <= .6:
            raise ValueError("Expected a past interval in [0.4,0.6] s")
        elapsed = _xy(label["target_dt_s"], "target intervals")
        if abs(elapsed[0] - .5) > .1 + _EPS or abs(elapsed[1] - 1.) > .1 + _EPS:
            raise ValueError("Labels must provide intervals around 0.5 s and 1.0 s")
        tracks = tuple(tuple(_xy(xy, "true center") for xy in track["centers_xy_m"]) for track in label["tracks"])
        if any(len(centers) != 3 for centers in tracks):
            raise ValueError("Each track must have current and two future centers")
        detections = _read_detections(row["objects"])
        matches = match_current_centers([obj.xy for obj in detections], [track[0] for track in tracks])
        windows.append(_Window(recording, elapsed, tracks, detections, matches))
    return windows


def _identifier(value, meaning):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Expected nonempty {meaning}")
    return value


def _xy(value, meaning):
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"Expected two finite components for {meaning}")
    return tuple(_finite(part, meaning) for part in value)


def _finite(value, meaning):
    if not isinstance(value, Real) or isinstance(value, bool) or abs(value) > 10_000 or not isfinite(value):
        raise ValueError(f"Expected bounded finite {meaning}")
    return float(value)


def _read_detections(objects):
    seen, previous = set(), {name: set() for name in ("geometry", "appearance")}
    detections = []
    for row in objects:
        key = _identifier(row["object_id"], "detection ID")
        if key in seen or row["kind"] not in ("person", "bicycle"):
            raise ValueError("Duplicate detection ID or unexpected object class")
        seen.add(key)
        if not 0 <= _finite(row["confidence"], "detection confidence") <= 1:
            raise ValueError("Detection confidence must be in [0,1]")
        motions = {name: _read_motion(row["motions"][name]) for name in previous}
        for name, motion in motions.items():
            if motion.previous_id is not None:
                if motion.previous_id in previous[name]:
                    raise ValueError("An association branch reused a previous detection")
                previous[name].add(motion.previous_id)
        detections.append(_Detection(_xy(row["xy_m"], "detected center"), **motions))
    return tuple(detections)


def _read_motion(row):
    status = _identifier(row["status"], "association status")
    velocity = _xy(row["velocity_xy_mps"], "velocity")
    previous, radial, past = (row[key] for key in ("previous_object_id", "radial_unit_xy", "past_xy_m"))
    cameras = row["common_cameras"]
    if not isinstance(cameras, (tuple, list)) or any(not isinstance(x, str) or not x.strip() for x in cameras):
        raise ValueError("Expected valid common-camera names")
    if len(cameras) != len(set(cameras)):
        raise ValueError("Duplicate camera in an association")
    if status == "estimated":
        _identifier(previous, "previous detection ID")
        _xy(past, "previous center")
        radial = _xy(radial, "radial unit vector")
        norm = hypot(*radial)
        if not cameras or abs(norm - 1.) > 1e-6:
            raise ValueError("Motion estimates need a shared camera and a unit radial vector")
        radial = tuple(x / norm for x in radial)
    elif velocity != _ZERO or any(x is not None for x in (previous, radial, past)) or cameras:
        raise ValueError("An abstaining branch must not supply a velocity or association")
    return _Motion(status, previous, velocity, radial)


def _summary(windows, gains):
    detections = [obj for window in windows for obj in window.detections]
    matched = [(window.detections[i], window.tracks[j][0]) for window in windows for i, j in window.matches]
    annotated = sum(len(window.tracks) for window in windows)
    current_error = [hypot(obj.xy[0] - target[0], obj.xy[1] - target[1]) for obj, target in matched]
    metrics = {"windows": len(windows), "recordings": len({row.recording for row in windows}),
               "annotated_tracks": annotated, "detections": len(detections), "matched_tracks": len(matched),
               "unmatched_tracks": annotated - len(matched), "unmatched_detections": len(detections) - len(matched),
               "matched_annotated_fraction": len(matched) / annotated if annotated else None,
               "matched_windows": sum(bool(window.matches) for window in windows),
               "accepted_appearance": sum(obj.appearance.status == "estimated" for obj, _ in matched),
               "accepted_geometry": sum(obj.geometry.status == "estimated" for obj, _ in matched)}
    horizons = {}
    for j, key in enumerate(HORIZONS):
        actual = [row.elapsed[j] for row in windows]
        horizons[key] = {"seconds": .5 * (j + 1), "observed_target_dt_s": {"minimum": min(actual), "maximum": max(actual)},
                         **_horizon(_compare(windows, j, gains))}
    return {"counts": metrics, "horizons": horizons,
            "current_center_error_m": {"mean": _mean(current_error), "median": _median(current_error)},
            "association_statuses": {name: dict(sorted(Counter(getattr(obj, name).status for obj in detections).items()))
                                     for name in ("geometry", "appearance")}}


def _scaled_velocity(motion, gains):
    if motion.radial is None:
        return _ZERO
    radial, tangential = gains
    # Orthogonal projection avoids applying one gain to both vector components.
    parallel = fsum(value * axis for value, axis in zip(motion.velocity, motion.radial))
    return tuple(tangential * value + (radial - tangential) * parallel * axis
                 for value, axis in zip(motion.velocity, motion.radial))


def _compare(windows, step, gains):
    result = []
    for window in windows:
        elapsed = window.elapsed[step]
        for i, j in window.matches:
            obj, track = window.detections[i], window.tracks[j]
            now, future = track[0], track[step + 1]
            move = tuple(b - a for a, b in zip(now, future))
            distance = hypot(*move)
            category = "moving" if distance >= MOVING_M else "quiet" if distance <= QUIET_M else "intermediate"
            velocities = (_ZERO, obj.geometry.velocity, obj.appearance.velocity, _scaled_velocity(obj.appearance, gains))
            errors = {}
            for arm, (vx, vy) in zip(ARMS, velocities):
                dx, dy = vx * elapsed, vy * elapsed
                errors[arm] = (hypot(move[0] - dx, move[1] - dy), hypot(future[0] - obj.xy[0] - dx, future[1] - obj.xy[1] - dy))
            result.append(_Comparison(window.recording, category, obj.appearance.status, errors))
    return result


def _horizon(pairs):
    accepted = [pair for pair in pairs if pair.status == "estimated"]
    return {name: {kind: _aggregate([pair for pair in group if kind == "all" or pair.category == kind])
                   for kind in ("all", "moving", "quiet", "intermediate")}
            for name, group in (("all_matched", pairs), ("accepted_appearance", accepted))}


def _aggregate(pairs):
    arms = {}
    for key in ARMS:
        displacement = [pair.errors[key][0] for pair in pairs]
        center = [pair.errors[key][1] for pair in pairs]
        arms[key] = {"displacement_mean_m": _mean(displacement), "displacement_median_m": _median(displacement),
                     "future_center_mean_m": _mean(center), "future_center_median_m": _median(center)}
    return {"pairs": len(pairs), "recordings": len({pair.recording for pair in pairs}), "arms": arms}


def _mean(values):
    return fsum(values) / len(values) if values else None


def _median(values):
    return median(values) if values else None


def _selection_counts(horizon):
    moving, quiet = (horizon["accepted_appearance"][group] for group in ("moving", "quiet"))
    return {"moving_pairs": moving["pairs"], "moving_recordings": moving["recordings"],
            "quiet_pairs": quiet["pairs"], "matched_pairs": horizon["all_matched"]["all"]["pairs"]}


def _stat(horizon, set_name, category, arm, metric):
    return horizon[set_name][category]["arms"][arm][metric]


def _deltas(horizon):
    targets = (("moving_displacement_gain_m", "accepted_appearance", "moving", "displacement_mean_m", 1),
               ("quiet_displacement_penalty_m", "accepted_appearance", "quiet", "displacement_mean_m", -1),
               ("future_center_gain_m", "all_matched", "all", "future_center_mean_m", 1))
    result = {}
    for output, pair_set, category, field, sign in targets:
        old = _stat(horizon, pair_set, category, "stationary", field)
        new = _stat(horizon, pair_set, category, "calibrated", field)
        result[output] = None if old is None or new is None else sign * (old - new)
    return result


def _valid_tradeoff(metrics):
    quiet, center = (metrics[key] for key in ("quiet_displacement_penalty_m", "future_center_gain_m"))
    return quiet is not None and center is not None and quiet <= MAX_QUIET_PENALTY_M + _EPS and center >= -_EPS


def _gate(horizon):
    policy = {"minimum_moving_pairs": TEST_MIN_MOVING, "minimum_moving_recordings": TEST_MIN_RECORDINGS,
              "minimum_quiet_pairs": MIN_QUIET, "minimum_moving_gain_m": MIN_MOVING_GAIN_M,
              "maximum_quiet_penalty_m": MAX_QUIET_PENALTY_M, "require_no_center_regression": True,
              "horizon_s": 1.}
    counts = _selection_counts(horizon)
    metrics = _deltas(horizon)
    if counts["moving_pairs"] < TEST_MIN_MOVING or counts["moving_recordings"] < TEST_MIN_RECORDINGS or counts["quiet_pairs"] < MIN_QUIET:
        status = "insufficient_support"
    elif _valid_tradeoff(metrics) and metrics["moving_displacement_gain_m"] + _EPS >= MIN_MOVING_GAIN_M:
        status = "motion_signal"
    else:
        status = "motion_gain_not_established"
    return {"status": status, **policy, **counts, **metrics}
