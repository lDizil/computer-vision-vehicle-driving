"""Attribute offline VRU tracking losses on fixed, conservative spatial matches."""

from collections import Counter, defaultdict
from dataclasses import dataclass
from math import dist, fsum

from hybrid_drive.experiments.object_vru_metrics import MATCH_GAP_M, MATCH_RADIUS_M, _identifier, _xy, match_current_centers
from hybrid_drive.experiments.vru_detector_coverage_metrics import _category, _read_objects, _read_targets
from hybrid_drive.experiments.vru_fusion_holdout_metrics import _observation


BASELINE = "ssd_primary"
SOURCE_ARMS = ("ssdlite", "faster_rcnn", BASELINE)
RESCUE_ARMS = ("native_rescue", "bridge_rescue")
ARMS = SOURCE_ARMS + RESCUE_ARMS
_GROUPS = ("all", "moving", "quiet", "intermediate")
_TRUTH_STATES = {"available", "missing_annotations", "missing_frame"}
_LINK_STATES = {"estimated", "no_reliable_history", "ambiguous_history", "nonmutual_history"}
_GATE = {"minimum_matched_moving": 12, "minimum_verified_moving_gain": 2,
         "minimum_verified_gain_recordings": 2, "minimum_moving_error_reduction_m": .03,
         "maximum_future_center_error_increase_m": .05, "maximum_added_quiet_nonzero": 0,
         "maximum_added_wrong_links": 0}
_ZERO = (0., 0.)
_EPS = 1e-12


@dataclass(frozen=True)
class _Frame:
    objects: dict
    centers: dict
    target_by_id: dict
    id_by_target: dict
    nearby_targets: frozenset


def summarize_chain_replay(arms, labels, history_truth):
    """Compare constant current coverage; audit each selected chain in its source arm.

    Historical truth verifies spatial correspondence to the same annotated
    target, not an exact detection identity. Unassigned and unlabeled links are
    unknown. A forecast abstention has zero scoring displacement, not a label.
    """
    if set(arms) != set(ARMS):
        raise ValueError("Expected the five fixed chain replay arms")
    labels, history_truth = tuple(labels), tuple(history_truth)
    arms = {name: tuple(arms[name]) for name in ARMS}
    count = len(labels)
    if not count or len(history_truth) != count or any(len(arm) != count for arm in arms.values()):
        raise ValueError("Chain forecasts and both truth artifacts must cover the same nonempty window set")

    windows, seen = [], set()
    for i, label in enumerate(labels):
        token = _identifier(label["sample_token"], "label window")
        if token in seen:
            raise ValueError("Duplicate chain replay window")
        seen.add(token)
        windows.append(_window({name: arm[i] for name, arm in arms.items()}, label, history_truth[i]))
    recordings = defaultdict(list)
    for window in windows:
        recordings[window["recording"]].append(window)
    return {**_summary(windows), "by_recording": {name: _summary(rows) for name, rows in sorted(recordings.items())},
            "cases": [case for window in windows for case in window["cases"]],
            "policy": {"scope": "offline_diagnostic_only", "horizon_s": 1., "match_radius_m": MATCH_RADIUS_M,
                       "match_gap_m": MATCH_GAP_M, "matching": "mutual_nearest_two_sided_margin_at_each_observation",
                       "verified": "same_annotated_track_spatially_matched_at_three_observations_not_exact_id",
                       "wrong_link_unit": "matched_target_with_a_link_to_another_spatially_matched_target",
                       "unknown_motion": "zero_displacement_comparator_not_stationary_evidence", "gate": dict(_GATE)}}


def _window(profiles, label, history_truth):
    row = profiles[BASELINE]
    token, previous, older = (row[key] for key in ("sample_token", "previous_token", "older_token"))
    recording = _identifier(row["recording"], "recording")
    identity = (token, previous, older)
    expected_count = 2 if older is None else 3
    observation = _observation(row)
    if (token != label["sample_token"] or len(set(identity) - {None}) != expected_count
            or any(_observation(other) != observation for other in profiles.values())
            or tuple(history_truth[key] for key in ("sample_token", "previous_token", "older_token")) != identity):
        raise ValueError("Chain replay observation bindings differ")
    current_tracks = _read_targets(label)
    truths = {"current": {i: track[0] for i, track in enumerate(current_tracks)},
              **_historical_truth(history_truth, older is not None, len(current_tracks))}
    current_frames, tracked_frames, detection_counts = {}, {}, {}
    for name, profile in profiles.items():
        history = profile["history"]
        if (history["older"] is None) != (older is None):
            raise ValueError("Expected the same third-observation availability in every chain replay arm")
        current_frames[name] = _frame(profile["objects"], truths["current"])
        detection_counts[name] = len(current_frames[name].objects)
        if name in SOURCE_ARMS:
            tracked_frames[name] = {"current": current_frames[name],
                                    "previous": _frame(history["previous"], truths["previous"]),
                                    "older": _frame(history["older"], truths["older"]) if older is not None else None}

    for name in RESCUE_ARMS:
        if (_geometry(current_frames[name]) != _geometry(current_frames[BASELINE])
                or current_frames[name].id_by_target != current_frames[BASELINE].id_by_target
                or profiles[name]["history"] != row["history"]):
            raise ValueError("Rescue must keep exact current geometry, assignments and observation history")
    rescue_counts = {name: _validate_evidence(name, frame.objects, tracked_frames) for name, frame in current_frames.items()}
    _validate_augmentation(current_frames)

    cases = [_case(token, recording, i, track, label["target_dt_s"][1], truths, history_truth,
                   current_frames, tracked_frames) for i, track in enumerate(current_tracks)]
    return {"recording": recording, "has_older": older is not None, "cases": cases,
            "detections": detection_counts, "rescues": rescue_counts}


def _historical_truth(bundle, has_older, count):
    if len(bundle["tracks"]) != count:
        raise ValueError("Historical truth must cover the same ordered targets")
    result = {}
    for time in ("previous", "older"):
        status = bundle[f"{time}_status"]
        present = time == "previous" or has_older
        if status not in _TRUTH_STATES or (status != "missing_frame") != present:
            raise ValueError("Historical truth availability contradicts the observation index")
        points = {}
        for i, row in enumerate(bundle["tracks"]):
            point = row[f"{time}_xy_m"]
            if point is not None:
                if status != "available":
                    raise ValueError("Unavailable historical truth cannot include a labeled center")
                points[i] = _xy(point, f"{time} true center")
        result[time] = points
    return result


def _frame(rows, targets):
    objects = {row["object_id"]: row for row in rows}
    centers = dict(_read_objects(rows))
    target_ids, detection_ids = tuple(targets), tuple(centers)
    matches = match_current_centers(centers.values(), targets.values())
    assigned = {detection_ids[i]: target_ids[j] for i, j in matches}
    nearby = frozenset(key for key, gt in targets.items() if any(dist(gt, xy) <= MATCH_RADIUS_M for xy in centers.values()))
    return _Frame(objects, centers, assigned, {target: key for key, target in assigned.items()}, nearby)


def _geometry(frame):
    return {key: (frame.centers[key], row["kind"], row["confidence"], tuple(row["cameras"]))
            for key, row in frame.objects.items()}


def _validate_evidence(name, objects, tracked_frames):
    used_origins, rescued = set(), Counter()
    used_hops = set()
    native_arms = SOURCE_ARMS[:2]
    for key, obj in objects.items():
        trace = obj["trace"]
        source, origin, rescue = (trace[field] for field in ("origin_arm", "origin_object_id", "rescue"))
        same_origin = key == origin
        if rescue is None:
            valid = same_origin and source == (name if name in SOURCE_ARMS else BASELINE)
        elif rescue == "native":
            valid = name in RESCUE_ARMS and source in native_arms and same_origin
        else:
            valid = rescue == "same_snapshot" and name == "bridge_rescue" and source == "faster_rcnn" and not same_origin
        if not valid:
            raise ValueError("Invalid chain evidence origin or rescue type")
        origin_key = (source, origin)
        frames = tracked_frames[source]
        origin_obj = frames["current"].objects.get(origin)
        if origin_obj is None or origin_obj["kind"] != obj["kind"] or origin_key in used_origins:
            raise ValueError("Chain evidence must refer to one distinct, same-class current source detection")
        used_origins.add(origin_key)

        if name not in SOURCE_ARMS:
            if ({**trace, "rescue": None} != origin_obj["trace"] or obj["confirmation"] != origin_obj["confirmation"]
                    or obj["forecasts"]["three_frame_trend"] != origin_obj["forecasts"]["three_frame_trend"]):
                raise ValueError("A rescue must preserve the exact source links and forecast")
            if rescue is not None:
                if obj["confirmation"]["status"] != "confirmed":
                    raise ValueError("A rescue must have confirmed source motion")
                rescued[rescue] += 1
        else:
            _validate_source_trace(obj, frames)
        if obj["confirmation"]["status"] == "confirmed":
            # Source profiles share their detector-namespaced observation IDs.
            hops = {(part, trace[part]["previous_object_id"]) for part in ("initial", "older")}
            if used_hops & hops:
                raise ValueError("Confirmed chains cannot reuse a historical detection at the same time")
            used_hops.update(hops)
    return dict(sorted(rescued.items()))


def _validate_augmentation(frames):
    for name, original in (("native_rescue", BASELINE), ("bridge_rescue", "native_rescue")):
        objects = frames[name].objects
        for key, previous in frames[original].objects.items():
            item = objects[key]
            if item["trace"] != previous["trace"]:
                if previous["confirmation"]["status"] != "no_older_link":
                    raise ValueError("Rescue cannot replace motion unless only the older link is missing")
                if name == "native_rescue" and item["trace"]["initial"] != previous["trace"]["initial"]:
                    raise ValueError("Native rescue must keep the same recent link")
            if name == "bridge_rescue" and item["trace"]["rescue"] != "same_snapshot" and item["trace"] != previous["trace"]:
                raise ValueError("Bridge arm must keep every native rescue")


def _validate_source_trace(obj, frames):
    trace, confirmation = obj["trace"], obj["confirmation"]
    latest = _link_id(trace["initial"], frames["previous"], obj["kind"])
    older = trace["older"]
    if (older is not None) != (latest is not None and frames["older"] is not None):
        raise ValueError("Second link trace must follow an estimated first link and an available third observation")
    oldest = _link_id(older, frames["older"], obj["kind"]) if older is not None else None
    confirmed = confirmation["status"] == "confirmed"
    velocity = _xy(obj["forecasts"]["three_frame_trend"], "chain trend velocity")
    if (velocity != _xy(confirmation["trend_velocity_xy_mps"], "confirmed chain velocity")
            or confirmed and (oldest is None or oldest != confirmation["older_object_id"])
            or not confirmed and (velocity != _ZERO or confirmation["older_object_id"] is not None)):
        raise ValueError("Expected confirmation and trend bound to the exact source trace")


def _link_id(link, frame, kind):
    key, status = link["previous_object_id"], link["status"]
    if (status not in _LINK_STATES or (status == "estimated") != (key is not None)
            or key is not None and (key not in frame.objects or frame.objects[key]["kind"] != kind)):
        raise ValueError("Chain link must identify a detection in its source observation or explicitly abstain")
    return key


def _case(token, recording, i, track, dt, truths, history, frames, tracked_frames):
    known = {f"{time}_xy_m": points.get(i) for time, points in truths.items()}
    known.update({"future_xy_m": track[-1], "target_dt_s": dt,
                  **{f"{time}_status": history[f"{time}_status"] for time in ("previous", "older")}})
    return {"sample_token": token, "recording": recording, "target_index": i, "category": _category(dist(track[0], track[-1])),
            "truth": known, "arms": {name: _attribution(i, frame, tracked_frames, known) for name, frame in frames.items()}}


def _attribution(i, frame, tracked_frames, truth):
    key = frame.id_by_target.get(i)
    if key is None:
        return {"matched": False, "cause": "current_ambiguous_or_nonmutual" if i in frame.nearby_targets else "current_missing_center"}
    obj = frame.objects[key]
    trace = obj["trace"]
    source, origin = trace["origin_arm"], trace["origin_object_id"]
    source_frames = tracked_frames[source]
    origin_status = _assignment(i, source_frames["current"], origin)
    history = {time: _history_stage(i, source_frames[time], trace[edge], truth[f"{time}_status"], truth[f"{time}_xy_m"])
               for time, edge in (("previous", "initial"), ("older", "older"))}
    statuses = {"origin": origin_status, **{time: row["assignment"] for time, row in history.items()}}
    confirmation = obj["confirmation"]["status"]
    cause = next((f"{time}_{status}" for time, status in statuses.items() if status != "verified"), f"confirmation_{confirmation}")
    velocity = _xy(obj["forecasts"]["three_frame_trend"], "chain trend velocity")
    center = frame.centers[key]
    shift = tuple(v * truth["target_dt_s"] for v in velocity)
    actual = tuple(b - a for a, b in zip(truth["current_xy_m"], truth["future_xy_m"], strict=True))
    future = tuple(a + b for a, b in zip(center, shift, strict=True))
    return {"matched": True, "object_id": key, "center_xy_m": center, "future_xy_m": future,
            "origin_arm": source, "origin_object_id": origin, "origin_xy_m": source_frames["current"].centers[origin],
            "cameras": obj["confirmation"].get("common_cameras") or obj["cameras"],
            "origin_assignment": origin_status, "history": history, "rescue": trace["rescue"],
            "cause": cause, "confirmation": confirmation, "verified": cause == "confirmation_confirmed",
            "wrong_link": "other_target" in statuses.values(), "velocity_xy_mps": velocity,
            "displacement_error_m": dist(shift, actual), "future_center_error_m": dist(future, truth["future_xy_m"])}


def _assignment(i, frame, key):
    target = frame.target_by_id.get(key)
    if target is not None:
        return "verified" if target == i else "other_target"
    return "unverified_link"


def _history_stage(i, frame, link, truth_status, point):
    key = link["previous_object_id"] if link else None
    linked_center = frame.centers.get(key) if frame else None
    assignment = _assignment(i, frame, key) if key is not None else "unverified_link"
    if assignment not in ("other_target", "verified"):
        if truth_status != "available":
            assignment = truth_status
        elif point is None:
            assignment = "unlabeled_target"
        elif i not in frame.id_by_target:
            assignment = "ambiguous_or_nonmutual_center" if i in frame.nearby_targets else "missing_center"
        elif not link:
            assignment = "link_not_attempted"
        elif key is None:
            assignment = link["status"]
    return {"assignment": assignment, "linked_object_id": key, "linked_xy_m": linked_center,
            "link_status": link["status"] if link else None}


def _summary(windows):
    cases = [case for window in windows for case in window["cases"]]
    groups = {kind: [x for x in cases if kind == "all" or kind == x["category"]] for kind in _GROUPS}
    arms, comparisons = {}, {}
    for name in ARMS:
        rescues = Counter()
        for window in windows:
            rescues.update(window["rescues"][name])
        arms[name] = {"detections": sum(window["detections"][name] for window in windows),
                      "rescue_detections": dict(sorted(rescues.items())),
                      **{kind: _arm_summary(rows, name) for kind, rows in groups.items()}}
    for name in RESCUE_ARMS:
        stats = {kind: _paired(rows, name) for kind, rows in groups.items()}
        comparisons[name] = {**stats, "gate": _decision(stats)}
    with_older = sum(window["has_older"] for window in windows)
    return {"counts": {"windows": len(windows), "windows_with_older": with_older,
                       "windows_without_older": len(windows) - with_older,
                       "recordings": len({window["recording"] for window in windows}), "annotated_tracks": len(cases),
                       **{f"{kind}_tracks": len(groups[kind]) for kind in _GROUPS[1:]}},
            "arms": arms, "comparisons": comparisons}


def _arm_summary(cases, name):
    rows = [case["arms"][name] for case in cases]
    matched = [row for row in rows if row["matched"]]
    return {"targets": len(rows), "matched": len(matched),
            "confirmation_statuses": _counts(row["confirmation"] for row in matched),
            "causes": _counts(row["cause"] for row in rows), "verified_confirmed": sum(row["verified"] for row in matched),
            "observed_wrong_links": sum(row["wrong_link"] for row in matched),
            "confirmed": sum(row["confirmation"] == "confirmed" for row in matched),
            "nonzero_velocity": sum(row["velocity_xy_mps"] != _ZERO for row in matched),
            "displacement_mean_m": _mean(row["displacement_error_m"] for row in matched),
            "future_center_mean_m": _mean(row["future_center_error_m"] for row in matched)}


def _paired(cases, name):
    pairs = [(case["recording"], case["arms"][BASELINE], case["arms"][name]) for case in cases
             if case["arms"][BASELINE]["matched"] and case["arms"][name]["matched"]]
    gained = {"confirmed": [], "verified": [], "nonzero": [], "wrong": []}
    lost = Counter()
    for recording, a, b in pairs:
        for key, flag in (("confirmed", lambda row: row["confirmation"] == "confirmed"),
                          ("verified", lambda row: row["verified"]),
                          ("nonzero", lambda row: row["velocity_xy_mps"] != _ZERO),
                          ("wrong", lambda row: row["wrong_link"])):
            before, after = flag(a), flag(b)
            if after and not before:
                gained[key].append(recording)
            if before and not after:
                lost[key] += 1
    return {"pairs": len(pairs), "recordings": len({name for name, _, _ in pairs}),
            **{key + "_gained": len(records) for key, records in gained.items()},
            **{key + "_lost": lost[key] for key in gained},
            "verified_gain_recordings": len(set(gained["verified"])),
            "displacement_change_m": _mean(b["displacement_error_m"] - a["displacement_error_m"] for _, a, b in pairs),
            "future_center_change_m": _mean(b["future_center_error_m"] - a["future_center_error_m"] for _, a, b in pairs)}


def _decision(stats):
    moving, quiet, overall = (stats[key] for key in ("moving", "quiet", "all"))
    support = moving["pairs"] >= _GATE["minimum_matched_moving"]
    verified_gain = moving["verified_gained"] - moving["verified_lost"]
    displacement, future = moving["displacement_change_m"], moving["future_center_change_m"]
    checks = {"coverage_invariant": True, "matched_moving_support": support,
              "verified_moving_gain": verified_gain >= _GATE["minimum_verified_moving_gain"],
              "verified_gain_spread": moving["verified_gain_recordings"] >= _GATE["minimum_verified_gain_recordings"],
              "moving_error_reduction": displacement is not None and displacement <= -_GATE["minimum_moving_error_reduction_m"] + _EPS,
              "future_center_bound": future is not None and future <= _GATE["maximum_future_center_error_increase_m"] + _EPS,
              "no_added_quiet_motion": quiet["nonzero_gained"] <= _GATE["maximum_added_quiet_nonzero"],
              "no_added_wrong_links": overall["wrong_gained"] <= _GATE["maximum_added_wrong_links"]}
    status = "insufficient_support" if not support else "rescue_signal" if all(checks.values()) else "gain_not_established"
    return {"status": status, "checks": checks, "verified_moving_net_gain": verified_gain}


def _counts(items):
    return dict(sorted(Counter(items).items()))


def _mean(items):
    items = tuple(items)
    return fsum(items) / len(items) if items else None
