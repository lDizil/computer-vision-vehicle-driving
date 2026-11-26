"""Camera-only center estimates from an explicit upright object-height prior."""

from collections.abc import Mapping
from itertools import product
from math import dist
from statistics import median

import numpy as np

from hybrid_drive.contracts._validation import finite_real, identifier
from hybrid_drive.experiments.object_vru_geometry import (
    _EDGE_MARGIN_PX,
    _FOOT_PERTURBATION_PX,
    _MAX_FOOT_SENSITIVITY_M,
    GroundHit,
    _appearance,
    _check_confidence,
    _check_kind,
)
from hybrid_drive.geometry.camera import CameraCalibration
from hybrid_drive.geometry.rigid import invert_pose, matrix


_HEIGHT_BOUNDS_M = {"person": (.6, 2.5), "bicycle": (.35, 2.5)}
_MAX_DEPTH_M = 48.
_MAX_ABS_BOTTOM_Z_M = 2.
_MIN_SINGULAR_RATIO = .01
_MAX_RELATIVE_RESIDUAL = .12


def upright_hit(view, box, height_m, *, detection_id) -> GroundHit | None:
    """Solve two optical depths, without intersecting the rays with a road plane.

    The box (mapping or detection object) supplies ``xyxy``, ``kind`` and
    ``confidence``; view supplies rectified RGB and its calibration. The
    projected top and bottom centers are proxies for an upright segment. A
    small residual and pixel sensitivity do not rule out occlusion or a wrong
    physical height for this instance.
    """
    calibration, rgb = getattr(view, "calibration", None), getattr(view, "rgb", None)
    if (not isinstance(calibration, CameraCalibration) or not isinstance(rgb, np.ndarray)
            or rgb.dtype != np.uint8 or rgb.shape != (calibration.height, calibration.width, 3)):
        raise ValueError("Expected rectified uint8 RGB matching the view calibration")
    identifier(detection_id, "detection ID")
    xyxy, kind, confidence = _box_fields(box)
    confidence = _check_confidence(confidence)
    height = _checked_height(kind, height_m)
    left, top, right, bottom = matrix(xyxy, (4,), "box XYXY").tolist()
    if right < left or bottom < top:
        raise ValueError("Box corners must be in XYXY order")
    if right - left < 4 or bottom - top < 8:
        return None
    margin = _EDGE_MARGIN_PX
    if left < margin or top < margin or right > calibration.width - 1 - margin or bottom > calibration.height - 1 - margin:
        return None

    camera_to_ego = invert_pose(calibration.ego_to_camera)
    origin = camera_to_ego[:3, 3]
    with np.errstate(over="ignore", invalid="ignore"):
        try:
            rays_to_ego = camera_to_ego[:3, :3] @ np.linalg.inv(calibration.intrinsic)
        except np.linalg.LinAlgError:
            return None
    if not np.isfinite(rays_to_ego).all():
        return None

    u = (left + right) / 2
    point = _solve_center(rays_to_ego, origin, u, top, bottom, height, calibration.near_m)
    if point is None:
        return None
    sensitivity = 0.
    delta = _FOOT_PERTURBATION_PX
    for du, dt, db in product((-delta, delta), repeat=3):
        alternative = _solve_center(rays_to_ego, origin, u + du, top + dt, bottom + db, height, calibration.near_m)
        if alternative is None:
            return None
        sensitivity = max(sensitivity, dist(point, alternative))
    if sensitivity > _MAX_FOOT_SENSITIVITY_M:
        return None
    return GroundHit(detection_id, kind, calibration.name, point, tuple(origin.tolist()), confidence,
                     max(.15, sensitivity), _appearance(rgb, left, top, right, bottom))


def _solve_center(rays_to_ego, origin, u, top, bottom, height, near):
    with np.errstate(over="ignore", invalid="ignore"):
        ray_top, ray_bottom = (rays_to_ego @ np.array(((u, u), (top, bottom), (1., 1.)))).T
    if not np.isfinite((ray_top, ray_bottom)).all():
        return None
    # The columns use optical-Z scale, not unit ray length.
    axes = np.column_stack((ray_top, -ray_bottom))
    upright = np.array((0., 0., height))
    try:
        depths, _, rank, singular = np.linalg.lstsq(axes, upright, rcond=None)
    except np.linalg.LinAlgError:
        return None
    if rank != 2 or singular[-1] <= _MIN_SINGULAR_RATIO * singular[0]:
        return None
    if not np.isfinite(depths).all() or np.any(depths < near) or np.any(depths > _MAX_DEPTH_M):
        return None
    top_offset, bottom_offset = depths[0] * ray_top, depths[1] * ray_bottom
    if np.linalg.norm(top_offset - bottom_offset - upright) > _MAX_RELATIVE_RESIDUAL * height:
        return None
    if abs(origin[2] + bottom_offset[2]) > _MAX_ABS_BOTTOM_Z_M:
        return None
    x, y = origin[:2] + .5 * (top_offset[:2] + bottom_offset[:2])
    if not (0 <= x <= 32 and abs(y) <= 12):
        return None
    return float(x), float(y)


def _box_fields(box):
    names = ("xyxy", "kind", "confidence")
    if isinstance(box, Mapping):
        return tuple(box.get(name) for name in names)
    return tuple(getattr(box, name, None) for name in names)


def typical_heights(projected_truth_rows) -> dict[str, float]:
    """Fit per-class median ``dz`` from the caller's development truth rows.

    The caller is responsible for binding these rows to the disjoint fit
    split. Classes with no support are absent rather than assigned defaults.
    """
    heights = {kind: [] for kind in _HEIGHT_BOUNDS_M}
    for row in projected_truth_rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("targets"), (list, tuple)):
            raise ValueError("Expected projected truth rows with target lists")
        for target in row["targets"]:
            if not isinstance(target, Mapping):
                raise ValueError("Expected a projected truth target")
            kind = target.get("kind")
            if kind == "unsupported":
                continue
            _check_kind(kind)
            oriented = matrix(target.get("box_ego"), (7,), "truth box in ego coordinates")
            if np.any(oriented[3:6] <= 0.):
                raise ValueError("Expected positive truth box dimensions")
            heights[kind].append(float(oriented[5]))
    return {kind: _checked_height(kind, median(values)) for kind, values in heights.items() if values}


def _checked_height(kind, height_m):
    _check_kind(kind)
    height = finite_real(height_m, "upright height")
    minimum, maximum = _HEIGHT_BOUNDS_M[kind]
    if not minimum <= height <= maximum:
        raise ValueError(f"Expected a plausible {kind} height in [{minimum}, {maximum}] m")
    return height
