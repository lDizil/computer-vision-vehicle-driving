"""Sampled diagnostic VRU contacts and image-ground evidence for the same probe pair."""

import numpy as np

from hybrid_drive.contracts.sensors import timestamp_us
from hybrid_drive.experiments.box_motion_data import CAMERAS, _pose
from hybrid_drive.experiments.frozen_transfer_contacts import _boxes_in_anchor, _eligible_vulnerable, _tracked_boxes
from hybrid_drive.experiments.frozen_transfer_data import VULNERABLE_CLASSES
from hybrid_drive.experiments.paired_slow_natural import (
    _EGO_CENTER_OFFSET_M,
    _EGO_HALF_EXTENT_M,
    _EGO_PAD_M,
    _GAP_RANGE_US,
    _STEP_US,
    _STEPS,
    _overlapping_ids,
    _timing,
)
from hybrid_drive.geometry.rigid import invert_pose, matrix


GRID_X_MIN, GRID_Y_MIN = 2., -7.
GRID_STEP_M = 1.
GRID_ROWS, GRID_COLS = 25, 14
_GRID_SHAPE = (GRID_ROWS, GRID_COLS)
_SPEED_MPS = (2.5, 9.)
_LATERAL_M = 2.
_CELL_PAD_M = .35
_ORIGIN = (0., 0., 0.)
_ANNOTATION_FIELDS = {"gt_names", "track_tokens", "gt_boxes"}
_MOTION_FIELDS = {"log_token", "log_name", "timestamp", "ego2global_translation", "ego2global_rotation"}


def visible_vru_cells(frame) -> tuple[tuple[int, int], ...]:
    """Return unique grid cells of current VRU centroids in a camera frustum.

    This uses annotation and calibration, not a test of unoccluded RGB pixels.
    Missing annotations or an unavailable needed rig are not negative labels.
    """
    if not _has_annotations(frame):
        raise ValueError("Current annotations are required for VRU cell labels")
    observed = _tracked_boxes(frame["anns"])
    return tuple(sorted(set(_eligible_cells(frame, observed).values())))


def probe_pair(frames, index):
    """Return the past-derived [left, right] paths and two independent labels.

    The required label comes from five sampled footprints against fixed anchor
    boxes. The optional label uses exact future coassociations, even if it
    reverses the comparison. These paths are probes, not drivable-route claims.
    """
    if not 0 < index < len(frames):
        return None
    previous, anchor = frames[index - 1:index + 1]
    if not _has_annotations(anchor):
        return None
    poses = _probe_poses(previous, anchor)
    if poses is None:
        return None
    observed = _tracked_boxes(anchor["anns"])
    cells = _eligible_cells(anchor, observed)
    if not cells:
        return None
    vehicles, vulnerable = _boxes_in_anchor(np.eye(4), observed, _vulnerable_classes(observed))
    if _overlapping_ids(_ORIGIN, vehicles) or _overlapping_ids(_ORIGIN, vulnerable):
        return None
    if any(_touched(path, vehicles) for path in poses):
        return None

    hits = tuple(_touched(path, vulnerable) for path in poses)
    blocked = _exclusive_hit(hits)
    if blocked is None or not hits[blocked] <= cells.keys():
        return None
    corridors = exclusive_corridors(poses)
    if not any(corridors[blocked][cells[token]] for token in hits[blocked]):
        return None
    if any(corridors[1 - blocked][cell] for cell in cells.values()):
        return None

    eligible = {token: observed[token][0] for token in cells}
    future = _future_blocked(frames[index:index + _STEPS + 1], poses, eligible)
    return poses.tolist(), blocked, future


def _probe_poses(previous, anchor):
    if _MOTION_FIELDS - previous.keys() or _MOTION_FIELDS - anchor.keys():
        return None
    if (previous["log_token"], previous["log_name"]) != (anchor["log_token"], anchor["log_name"]):
        return None
    dt_us = timestamp_us(anchor["timestamp"], "anchor time") - timestamp_us(previous["timestamp"], "history time")
    if not _GAP_RANGE_US[0] <= dt_us <= _GAP_RANGE_US[1]:
        return None
    past, current = _pose(previous), _pose(anchor)
    speed = float(current[:3, 0] @ (current[:3, 3] - past[:3, 3])) / (dt_us / 1e6)
    if not _SPEED_MPS[0] <= speed <= _SPEED_MPS[1]:
        return None

    step_s = _STEP_US / 1e6
    times = np.arange(1, _STEPS + 1) * step_s
    phase = times / (_STEPS * step_s)
    shift = _LATERAL_M * phase**3 * (10 - 15*phase + 6*phase**2)
    velocity_y = 30 * _LATERAL_M * phase**2 * (1 - phase)**2 / (_STEPS * step_s)
    left = np.column_stack((speed * times, shift, np.arctan2(velocity_y, speed)))
    return np.stack((left, left * [1, -1, -1]))


def _eligible_cells(frame, observed):
    cells = {}
    for token, (name, box) in observed.items():
        if name not in VULNERABLE_CLASSES:
            continue
        index = np.floor((box[:2] - [GRID_X_MIN, GRID_Y_MIN]) / GRID_STEP_M)
        if np.all((0 <= index) & (index < _GRID_SHAPE)):
            cells[token] = (int(index[0]), int(index[1]))
    if not cells:
        return cells
    cameras = frame.get("cams")
    if not isinstance(cameras, dict) or any(name not in cameras for name in CAMERAS):
        raise ValueError("Six calibrated cameras are required for VRU cell labels")
    eligible = _eligible_vulnerable(frame, {token: observed[token] for token in cells})
    return {token: cells[token] for token in eligible}


def _future_blocked(window, poses, eligible):
    if any(_MOTION_FIELDS - frame.keys() or not _has_annotations(frame) for frame in window):
        return None
    failure, _ = _timing(window)
    if failure:
        return None
    world_to_anchor = invert_pose(_pose(window[0]))
    hits = (set(), set())
    for step, frame in enumerate(window[1:]):
        observed = _tracked_boxes(frame["anns"])
        if any(token not in observed or observed[token][0] != name for token, name in eligible.items()):
            return None
        vehicles, vulnerable = _boxes_in_anchor(world_to_anchor @ _pose(frame), observed, _vulnerable_classes(observed))
        for arm, path in enumerate(poses):
            if _overlapping_ids(path[step], vehicles):
                return None
            touched = _overlapping_ids(path[step], vulnerable)
            if any(token not in eligible for token in touched):
                return None
            hits[arm].update(touched)
    return _exclusive_hit(hits)


def _has_annotations(frame):
    annotations = frame.get("anns")
    return isinstance(annotations, dict) and _ANNOTATION_FIELDS <= annotations.keys()


def _vulnerable_classes(observed):
    return {token: name for token, (name, _) in observed.items() if name in VULNERABLE_CLASSES}


def _touched(poses, boxes):
    return {token for pose in poses for token in _overlapping_ids(pose, boxes)}


def _exclusive_hit(hits):
    if bool(hits[0]) == bool(hits[1]):
        return None
    return 0 if hits[0] else 1


def exclusive_corridors(poses) -> np.ndarray:
    """Interpolate footprints at 0.1 s and mask shared cells on the 1 m grid."""
    paths = np.concatenate((np.zeros((2, 1, 3)), matrix(poses, (2, _STEPS, 3), "probe pair")), axis=1)
    grid_t = np.arange(_STEPS + 1)
    substeps = np.linspace(0, _STEPS, 5*_STEPS + 1)
    dense = np.array([np.column_stack((np.interp(substeps, grid_t, x), np.interp(substeps, grid_t, y),
                                       np.interp(substeps, grid_t, np.unwrap(yaw)))) for x, y, yaw in paths.transpose(0, 2, 1)])
    theta = dense[:, :, 2, None, None]
    c, s = np.cos(theta), np.sin(theta)
    rows = GRID_X_MIN + (np.arange(GRID_ROWS) + .5)*GRID_STEP_M
    cols = GRID_Y_MIN + (np.arange(GRID_COLS) + .5)*GRID_STEP_M
    dx = rows[None, None, :, None] - dense[:, :, 0, None, None]
    dy = cols[None, None, None, :] - dense[:, :, 1, None, None]
    along = c*dx + s*dy - _EGO_CENTER_OFFSET_M
    lateral = -s*dx + c*dy
    length, width = (extent + _EGO_PAD_M + _CELL_PAD_M for extent in _EGO_HALF_EXTENT_M)
    swept = ((np.abs(along) <= length) & (np.abs(lateral) <= width)).any(axis=1)
    return swept & ~swept[::-1]
