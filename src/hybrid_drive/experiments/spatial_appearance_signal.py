"""Offline retrieval diagnostics on matched boxes, independent of tracker selection."""

from collections import Counter
from collections.abc import Mapping
from math import fsum, hypot

from .spatial_appearance_features import APPEARANCE_KINDS, FrameAppearance
from .spatial_decision_policy import PRESENT
from .spatial_track_metrics import TrackingTarget, match_tracking_frame
from .temporal_spatial_filter import DEFAULT_MAX_GAP_US
from .temporal_spatial_types import FrameDecision


_MIN_IOU = 0.5
_MAX_CENTER_DISTANCE = 0.35
_MAX_SIZE_RATIO = 3.0
_TIE = 1e-8


def measure_appearance_signal(decisions, targets, features, shuffled_features) -> dict:
    """Rank the same true and competing matches with original and permuted appearance.

    An anchor must recur among confidently present IoU-matched boxes in the
    next adjacent frame of its clip. Compare it against other matched cars in
    the same or neighboring band passing the measured-box center and extent
    gates. This is a geometry neighborhood, not the tracker's motion forecast.
    Each profile requires complete descriptors for that query; the control
    keeps the same query and permutes only available vectors within each frame.
    """
    decisions = _checked_timeline(decisions)
    samples = {decision.observation.sample_id for decision in decisions}
    if not isinstance(targets, Mapping) or set(targets) != samples or any(
        not isinstance(target, TrackingTarget) for target in targets.values()
    ):
        raise ValueError("Expected tracking targets for exactly the evaluated appearance frames")
    _check_features(decisions, features, shuffled_features)

    matched = []
    for decision in decisions:
        target = targets[decision.observation.sample_id]
        matched.append({
            target.track_ids[truth_band]: band
            for band, truth_band in match_tracking_frame(decision, target, min_iou=_MIN_IOU)
        })

    eligible = 0
    actual = {kind: [] for kind in APPEARANCE_KINDS}
    shuffled = {kind: [] for kind in APPEARANCE_KINDS}
    for index, (before, after) in enumerate(zip(decisions, decisions[1:])):
        if not _adjacent(before, after):
            continue
        for alias, old_band in matched[index].items():
            new_band = matched[index + 1].get(alias)
            if new_band is None or not _geometry_neighbor(before, old_band, after, new_band):
                continue
            competitors = tuple(
                band for other, band in matched[index + 1].items()
                if other != alias and _geometry_neighbor(before, old_band, after, band)
            )
            if not competitors:
                continue
            eligible += 1
            for kind in APPEARANCE_KINDS:
                src = getattr(features[index], kind)[old_band]
                dst = getattr(features[index + 1], kind)
                if src is None or any(dst[band] is None for band in (new_band, *competitors)):
                    continue
                actual[kind].append(_margin(src, dst, new_band, competitors))
                permuted_src = getattr(shuffled_features[index], kind)[old_band]
                permuted_dst = getattr(shuffled_features[index + 1], kind)
                shuffled[kind].append(_margin(permuted_src, permuted_dst, new_band, competitors))

    return {
        "eligible_geometry": eligible,
        "profiles": {kind: _summary(actual[kind], shuffled[kind]) for kind in APPEARANCE_KINDS},
    }


def _checked_timeline(decisions):
    if not isinstance(decisions, (tuple, list)) or not decisions or any(
        not isinstance(decision, FrameDecision) for decision in decisions
    ):
        raise ValueError("Expected a nonempty sequence of original FrameDecision values")
    samples, recordings, clips = set(), set(), set()
    previous = None
    for decision in decisions:
        obs = decision.observation
        if obs.sample_id in samples:
            raise ValueError("Duplicate sample ID in appearance timeline")
        samples.add(obs.sample_id)
        if previous is None or obs.recording_id != previous.recording_id:
            if obs.recording_id in recordings:
                raise ValueError("Each appearance recording must occupy one source block")
            recordings.add(obs.recording_id)
        elif obs.source_index <= previous.source_index or obs.timestamp_us <= previous.timestamp_us:
            raise ValueError("Appearance source indices and timestamps must increase within each recording")
        key = obs.recording_id, obs.clip_id
        if previous is None or key != (previous.recording_id, previous.clip_id):
            if key in clips:
                raise ValueError("Each appearance clip must occupy one source block")
            clips.add(key)
        previous = obs
    return decisions


def _check_features(decisions, actual, shuffled):
    if any(not isinstance(rows, (tuple, list)) or len(rows) != len(decisions) for rows in (actual, shuffled)):
        raise ValueError("Appearance and shuffled control must cover all original decisions")
    dimensions = {}
    for decision, original, control in zip(decisions, actual, shuffled):
        expected = decision.observation.sample_id
        if any(not isinstance(frame, FrameAppearance) or frame.sample_id != expected for frame in (original, control)):
            raise ValueError("Appearance and shuffled control must follow exact sample order")
        for kind in APPEARANCE_KINDS:
            a, b = getattr(original, kind), getattr(control, kind)
            if any((left is None) != (right is None) for left, right in zip(a, b)):
                raise ValueError("Shuffled appearance must preserve the missing descriptor mask")
            if any(vector is not None and state != PRESENT for vector, state in zip(a, decision.states)):
                raise ValueError("Only PRESENT boxes may carry appearance evidence")
            if Counter(a) != Counter(b):
                raise ValueError("Shuffled appearance must be a within-frame permutation of available vectors")
            for vector in a:
                if vector is not None and dimensions.setdefault(kind, len(vector)) != len(vector):
                    raise ValueError("Appearance dimensions must be fixed within each profile")


def _adjacent(before, after):
    left, right = before.observation, after.observation
    return (
        (left.recording_id, left.clip_id) == (right.recording_id, right.clip_id)
        and after.reset_reason is None
        and right.source_index == left.source_index + 1
        and 0 < right.timestamp_us - left.timestamp_us <= DEFAULT_MAX_GAP_US
    )


def _geometry_neighbor(before, a, after, b):
    if abs(a - b) > 1:
        return False
    left, right = before.boxes[a], after.boxes[b]
    extents = tuple((left[i + 2] - left[i], right[i + 2] - right[i]) for i in (0, 1))
    if any(max(u, v) > _MAX_SIZE_RATIO * min(u, v) for u, v in extents):
        return False
    dx = ((left[0] + left[2]) - (right[0] + right[2])) / 2
    dy = ((left[1] + left[3]) - (right[1] + right[3])) / 2
    return hypot(dx, dy) <= _MAX_CENTER_DISTANCE


def _cosine(a, b):
    similarity = fsum(x * y for x, y in zip(a, b)) / (hypot(*a) * hypot(*b))
    return max(-1.0, min(1.0, similarity))


def _margin(anchor, candidates, true_band, competitors):
    return _cosine(anchor, candidates[true_band]) - max(_cosine(anchor, candidates[band]) for band in competitors)


def _summary(actual, shuffled):
    count = len(actual)
    return {
        "queries": count,
        "actual_correct": sum(value > _TIE for value in actual),
        "shuffled_correct": sum(value > _TIE for value in shuffled),
        "actual_ties": sum(abs(value) <= _TIE for value in actual),
        "shuffled_ties": sum(abs(value) <= _TIE for value in shuffled),
        "actual_mean_margin": fsum(actual) / count if count else None,
        "shuffled_mean_margin": fsum(shuffled) / count if count else None,
    }
