"""Measure observed OpenScene contacts after the transferred forecasts are sealed."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import atan2, comb, cos, hypot, sin

import numpy as np

from hybrid_drive.contracts._validation import finite_real, identifier
from hybrid_drive.contracts.sensors import timestamp_us
from hybrid_drive.experiments.bounded_io import require_hex
from hybrid_drive.experiments.box_motion_data import CAMERAS, _pose
from hybrid_drive.experiments.frozen_transfer_data import VULNERABLE_CLASSES
from hybrid_drive.experiments.frozen_transfer_protocol import SCORE_ARMS
from hybrid_drive.experiments.paired_slow_natural import _STEPS, _STEP_US, _lower, _overlapping_ids, _timing
from hybrid_drive.geometry.camera import CameraCalibration
from hybrid_drive.geometry.rigid import invert_pose, matrix


_ANNOTATION_FIELDS = {"gt_names", "track_tokens", "gt_boxes"}
_GRID_S = _STEP_US / 1e6


@dataclass(frozen=True)
class _Candidate:
    key: str
    poses: np.ndarray
    fast_overlap_s: float | None
    fast_gap_m: float | None
    margins: dict[str, float]


def measure_recording(frames, selection_rows, predictions) -> list[dict]:
    """Measure sampled annotated contacts, not continuous collision or clearance.

    Vehicle boxes are used whenever annotated. Vulnerable boxes count only for
    tracks with a pedestrian/bicycle centroid in a rectified camera frustum at
    the anchor. Frustum coverage is not evidence of visibility or no occlusion.
    """
    rows = []
    for selection, prediction in zip(selection_rows, predictions, strict=True):
        row, index = _bind(frames, selection, prediction)
        if prediction["status"] == "out_of_envelope":
            if prediction.get("candidates") != []:
                raise ValueError("Out-of-envelope predictions cannot have candidates")
            rows.append({**row, "window_status": "out_of_envelope", "candidate_count": 0, "candidate_pairs": 0})
            continue
        if prediction["status"] != "ok":
            raise ValueError("Unknown prediction status for transfer contact measurement")
        candidates = _parse_candidates(prediction["candidates"])
        row.update(candidate_count=len(candidates), candidate_pairs=comb(len(candidates), 2))
        rows.append(_measure_window(row, frames[index:index + _STEPS + 1], candidates))
    return rows


def _bind(frames, selection, prediction) -> tuple[dict, int]:
    token = selection["sample_token"]
    require_hex(token, 16, "transfer sample token")
    digest = prediction["snapshot_sha256"]
    require_hex(digest, 64, "transfer snapshot digest")
    recording = identifier(selection["recording"], "transfer recording")
    stamp = timestamp_us(selection["ref_time_us"], "transfer anchor time")
    index = timestamp_us(selection["frame_index"], "transfer frame index")
    population = selection["population"]
    if population not in ("uniform", "vulnerable"):
        raise ValueError("Unknown transfer population")
    if (token, recording, stamp) != (
        prediction["sample_token"], prediction["recording"],
        timestamp_us(prediction["ref_time_us"], "transfer prediction time"),
    ):
        raise ValueError("Transfer selection and prediction identities differ")
    if index >= len(frames) or (frames[index]["token"], frames[index]["timestamp"]) != (token, stamp):
        raise ValueError("Transfer selection does not match the exact metadata anchor")
    return {
        "sample_token": token, "recording": recording, "ref_time_us": stamp,
        "snapshot_sha256": digest, "frame_index": index, "population": population, "pairs": [],
    }, index


def _parse_candidates(values) -> tuple[_Candidate, ...]:
    if not isinstance(values, list) or not 1 <= len(values) <= 8:
        raise ValueError("Expected between one and eight transfer candidates")
    result = []
    for item in values:
        key = identifier(item["candidate_id"], "candidate_id")
        if finite_real(item["dt_s"], "dt_s") != _GRID_S:
            raise ValueError("Transfer trajectories must use the 0.5 s annotation grid")
        poses = matrix(item["poses"], (_STEPS, 3), "transfer candidate poses")
        if np.abs(poses[:, :2]).max() > 1_000:
            raise ValueError("Transfer candidate pose exceeds the local scoring extent")
        overlap, gap = item["fast_overlap_s"], item["fast_axis_gap_m"]
        if overlap is not None:
            overlap = finite_real(overlap, "fast_overlap_s")
            if not 0 <= overlap <= _STEPS * _GRID_S:
                raise ValueError("Fast contact lies outside the transfer horizon")
        if gap is not None:
            gap = finite_real(gap, "fast_axis_gap_m")
        margins = {name: finite_real(item["margins"][name], name) for name in SCORE_ARMS}
        result.append(_Candidate(key, poses, overlap, gap, margins))
    if len({candidate.key for candidate in result}) != len(result):
        raise ValueError("Duplicate transfer candidate ID")
    return tuple(result)


def _measure_window(row, window, candidates):
    failure, lags = _timing(window)
    row["future_lag_us"] = lags
    if failure:
        return {**row, "window_status": failure}
    if any(not isinstance(frame.get("anns"), dict) or _ANNOTATION_FIELDS - frame["anns"].keys()
           for frame in window):
        return {**row, "window_status": "missing_annotations"}

    annotations = [_tracked_boxes(frame["anns"]) for frame in window]
    eligible = _eligible_vulnerable(window[0], annotations[0])
    world_to_anchor = invert_pose(_pose(window[0]))
    boxes = [_boxes_in_anchor(world_to_anchor @ _pose(frame), observed, eligible)
             for frame, observed in zip(window, annotations, strict=True)]
    vehicles, vulnerable = zip(*boxes)
    row.update(eligible_vulnerable_anchor_count=len(eligible),
               annotated_vehicle_count_by_sample=[len(sample) for sample in vehicles],
               annotated_eligible_vulnerable_count_by_sample=[len(sample) for sample in vulnerable])
    if _overlapping_ids((0, 0, 0), vehicles[0]) or _overlapping_ids((0, 0, 0), vulnerable[0]):
        return {**row, "window_status": "shared_observed_initial_contact"}

    contacts = [{
        "candidate_id": candidate.key,
        "vehicle_contact_steps_s": _contact_samples(candidate.poses, vehicles[1:]),
        "vulnerable_contact_steps_s": _contact_samples(candidate.poses, vulnerable[1:]),
    } for candidate in candidates]
    row["pairs"] = [_compare(a, b) for a, b in combinations(zip(candidates, contacts, strict=True), 2)]
    return {**row, "window_status": "complete", "candidates": contacts}


def _tracked_boxes(annotation):
    result = {}
    for name, token, raw in zip(annotation["gt_names"], annotation["track_tokens"],
                                annotation["gt_boxes"], strict=True):
        if name != "vehicle" and name not in VULNERABLE_CLASSES:
            continue
        require_hex(token, 16, "annotated track token")
        if token in result:
            raise ValueError("Duplicate in-scope track token in an OpenScene frame")
        box = matrix(raw, (7,), "annotated oriented box")
        if (box[3:6] <= 0).any():
            raise ValueError("Annotated box dimensions must be positive")
        result[token] = (name, box)
    return result


def _eligible_vulnerable(anchor, observed):
    vulnerable = {token: (name, box) for token, (name, box) in observed.items() if name in VULNERABLE_CLASSES}
    if not vulnerable:
        return {}
    # Match the predictor's same-K rectification without reading RGB. For
    # OpenScene these legacy sensor2lidar fields actually describe camera-to-ego.
    calibrations = []
    for name in CAMERAS:
        source = anchor["cams"][name]
        camera_to_ego = np.eye(4)
        camera_to_ego[:3, :3] = matrix(source["sensor2lidar_rotation"], (3, 3), "camera rotation")
        camera_to_ego[:3, 3] = matrix(source["sensor2lidar_translation"], (3,), "camera translation")
        calibrations.append(CameraCalibration(
            name, 1920, 1080, source["cam_intrinsic"], invert_pose(camera_to_ego),
        ))
    return {token: name for token, (name, box) in vulnerable.items()
            if any(_in_frustum(box[:3], camera) for camera in calibrations)}


def _in_frustum(center, camera):
    ego_to_camera = camera.ego_to_camera
    point = ego_to_camera[:3, :3] @ center + ego_to_camera[:3, 3]
    if point[2] < camera.near_m:
        return False
    u, v, depth = camera.intrinsic @ point
    return 0 <= u <= (camera.width - 1) * depth and 0 <= v <= (camera.height - 1) * depth


def _boxes_in_anchor(transform, observed, eligible):
    rotation, offset = transform[:3, :3], transform[:3, 3]
    vehicles, vulnerable = {}, {}
    for token, (name, box) in observed.items():
        if name != "vehicle" and eligible.get(token) != name:
            continue
        center = rotation @ box[:3] + offset
        heading = rotation @ np.array((cos(box[6]), sin(box[6]), 0.))
        if not np.isfinite(center).all() or hypot(*heading[:2]) <= 1e-6:
            raise ValueError("Annotated box cannot be projected into the anchor plane")
        result = vehicles if name == "vehicle" else vulnerable
        result[token] = (center[:2], atan2(heading[1], heading[0]), (box[3] / 2, box[4] / 2))
    return vehicles, vulnerable


def _contact_samples(poses, observations):
    return [step * _GRID_S for step, (pose, boxes) in enumerate(zip(poses, observations, strict=True), start=1)
            if _overlapping_ids(pose, boxes)]


def _compare(left, right):
    a, sa = left
    b, sb = right
    vehicle = _contact_outcome(sa["vehicle_contact_steps_s"], sb["vehicle_contact_steps_s"])
    vulnerable = _contact_outcome(sa["vulnerable_contact_steps_s"], sb["vulnerable_contact_steps_s"])
    gap_choice = None
    if a.fast_gap_m is not None and b.fast_gap_m is not None:
        gap_choice = _lower(a.key, b.key, -a.fast_gap_m, -b.fast_gap_m)
    return {
        "candidate_ids": [a.key, b.key],
        "vehicle_outcome": vehicle,
        "vulnerable_outcome": vulnerable,
        "fast_binary_tied": (a.fast_overlap_s is None) == (b.fast_overlap_s is None),
        "preferred": {
            "fast": a.key, "fast_gap": gap_choice,
            **{key: _lower(a.key, b.key, a.margins[key], b.margins[key]) for key in SCORE_ARMS},
        },
        "vulnerable_only": vehicle == "neither_observed" and vulnerable in ("left_contact", "right_contact"),
    }


def _contact_outcome(left, right):
    if left and right:
        return "both_contact"
    if left:
        return "left_contact"
    if right:
        return "right_contact"
    return "neither_observed"
