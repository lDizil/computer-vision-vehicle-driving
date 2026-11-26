"""Score a fixed clip set with three observed frames and annotated identities."""

from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from math import dist

from hybrid_drive.experiments.object_vru_metrics import (
    MATCH_GAP_M,
    MATCH_RADIUS_M,
    _identifier,
    _xy,
    match_current_centers,
)


_FRAMES = ("current", "previous", "older")
_KINDS = ("person", "bicycle")
_DISTANCE_BANDS = ("lt_8m", "8_to_lt_16m", "gte_16m")


@dataclass(frozen=True)
class _Point:
    kind: str
    xy: tuple[float, float]
    previous_id: str | None = None


def summarize_vru_clips(predictions, labels, *, expected_count=120, required_pass=80):
    """Check current localization and the causal history of each fixed primary.

    All centers are expressed in the current ego frame. A link is evidence
    only if it spatially matches the same annotated track. An unmatched
    detection is unknown, not necessarily false or part of another track.
    """
    if type(expected_count) is not int or type(required_pass) is not int or not 1 <= required_pass <= expected_count:
        raise ValueError("Expected a positive pass requirement no larger than the fixed clip count")
    predictions, labels = tuple(predictions), tuple(labels)
    if len(predictions) != expected_count or len(labels) != expected_count:
        raise ValueError("Predictions and labels must both cover the complete fixed clip count")

    rows, seen = [], set()
    recordings = defaultdict(list)
    distances = {name: [] for name in _DISTANCE_BANDS}
    for predicted, label in zip(predictions, labels, strict=True):
        token, recording = _window_key(predicted)
        if token in seen or (token, recording) != _window_key(label):
            raise ValueError("Ordered clip identities must agree without repeats")
        seen.add(token)
        row = _score_clip(token, recording, predicted, label)
        rows.append(row)
        recordings[recording].append(row)
        distances[row["distance_band"]].append(row)

    totals = _summary(rows)
    passed = totals["counts"]["passed"]
    return {**totals,
            "gate": {"status": "passed" if passed >= required_pass else "failed", "required_pass": required_pass,
                     "expected_count": expected_count, "additional_passes_needed": max(0, required_pass - passed)},
            "by_recording": {name: _summary(group) for name, group in sorted(recordings.items())},
            "by_distance": {name: _summary(group) for name, group in distances.items()},
            "rows": rows,
            "policy": {"scope": "offline_perception_only", "coordinate_frame": "current_ego",
                       "matching": "same_class_mutual_nearest_two_sided_margin", "match_radius_m": MATCH_RADIUS_M,
                       "match_gap_m": MATCH_GAP_M, "distance_band_edges_m": [8., 16.],
                       "requires_current_and_one_earlier_match": True, "allows_known_other_track_link": False}}


def _window_key(row):
    if not isinstance(row, Mapping):
        raise ValueError("Expected a clip row")
    return _identifier(row.get("sample_token"), "clip token"), _identifier(row.get("recording"), "recording")


def _score_clip(token, recording, predicted, label):
    observations = _read_frames(predicted, "object_id")
    truth = _read_frames(label, "track_id")
    _validate_links(observations)
    primary_id = _identifier(label.get("primary_id"), "fixed primary track ID")
    primary = truth["current"].get(primary_id)
    if primary is None or any(primary_id not in frame or frame[primary_id].kind != primary.kind for frame in truth.values()):
        raise ValueError("The fixed primary must have one consistent class in all three frames")

    current = observations["current"]
    current_links = _match_ids(current, truth["current"])
    current_id = next((key for key, track_id in current_links.items() if track_id == primary_id), None)
    error = None if current_id is None else dist(current[current_id].xy, primary.xy)

    history = {}
    link_id = None if current_id is None else current[current_id].previous_id
    for time in _FRAMES[1:]:
        linked_track = None
        if link_id is None:
            status = "no_link"
        else:
            linked_track = _match_ids(observations[time], truth[time]).get(link_id)
            status = "unverified" if linked_track is None else "same_track" if linked_track == primary_id else "other_track"
        history[time] = {"status": status, "object_id": link_id, "track_id": linked_track}
        link_id = None if link_id is None else observations[time][link_id].previous_id

    reason = _decision(current_id, history)
    x = primary.xy[0]
    band = _DISTANCE_BANDS[0 if x < 8. else 1 if x < 16. else 2]
    return {"sample_token": token, "recording": recording, "primary_id": primary_id, "primary_kind": primary.kind,
            "primary_x_m": x, "distance_band": band, "current_object_id": current_id, "current_error_m": error,
            "history": history, "passed": reason == "passed", "reason": reason}


def _decision(current_id, history):
    if current_id is None:
        return "current_not_matched"
    statuses = {link["status"] for link in history.values()}
    if "other_track" in statuses:
        return "historical_link_other_track"
    if "same_track" in statuses:
        return "passed"
    if "unverified" in statuses:
        return "historical_link_unverified"
    return "no_historical_link"


def _read_frames(row, id_field):
    frames = row.get("frames")
    if not isinstance(frames, Mapping) or set(frames) != set(_FRAMES):
        raise ValueError("Expected exactly the current, previous and older frames")
    return {name: _read_points(frames[name], id_field) for name in _FRAMES}


def _read_points(items, id_field):
    if not isinstance(items, (list, tuple)):
        raise ValueError("Expected a list of centers in each frame")
    points = {}
    for item in items:
        if not isinstance(item, Mapping):
            raise ValueError("Expected a center row")
        key = _identifier(item.get(id_field), id_field)
        kind = item.get("kind")
        if key in points or kind not in _KINDS:
            raise ValueError("Duplicate frame ID or unexpected VRU class")
        previous_id = None
        if id_field == "object_id":
            if "previous_object_id" not in item:
                raise ValueError("Prediction must explicitly provide a previous link or null")
            previous_id = item["previous_object_id"]
            if previous_id is not None:
                _identifier(previous_id, "previous object ID")
        points[key] = _Point(kind, _xy(item.get("xy_m"), "frame center"), previous_id)
    return points


def _validate_links(frames):
    if any(point.previous_id is not None for point in frames["older"].values()):
        raise ValueError("The oldest frame cannot link outside the observed clip")
    for now, before in zip(_FRAMES, _FRAMES[1:]):
        earlier, used = frames[before], set()
        for point in frames[now].values():
            key = point.previous_id
            if key is None:
                continue
            if key not in earlier or earlier[key].kind != point.kind or key in used:
                raise ValueError("Predicted links must be same-class and one-to-one into the adjacent earlier frame")
            used.add(key)


def _match_ids(objects, tracks):
    pairs = {}
    for kind in _KINDS:
        observed = [(key, point.xy) for key, point in objects.items() if point.kind == kind]
        truth = [(key, point.xy) for key, point in tracks.items() if point.kind == kind]
        for i, j in match_current_centers((xy for _, xy in observed), (xy for _, xy in truth)):
            pairs[observed[i][0]] = truth[j][0]
    return pairs


def _summary(rows):
    total = len(rows)
    passed = sum(row["passed"] for row in rows)
    current = sum(row["current_object_id"] is not None for row in rows)
    failures = Counter(row["reason"] for row in rows if not row["passed"])
    return {"counts": {"clips": total, "passed": passed, "failed": total - passed, "current_matched": current},
            "pass_rate": passed / total if total else None, "failures": dict(sorted(failures.items()))}
