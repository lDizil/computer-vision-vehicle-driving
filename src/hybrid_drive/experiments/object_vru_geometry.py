"""Conservative ground estimates from individual rectified-camera detections."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import ceil, dist, floor, fsum, isclose
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from hybrid_drive.contracts._validation import finite_real, identifier, tuple_snapshot
from hybrid_drive.geometry.camera import CameraCalibration
from hybrid_drive.geometry.rigid import invert_pose, matrix, rigid_transform


Kind = Literal["person", "bicycle"]
XY = tuple[float, float]
XYZ = tuple[float, float, float]
_RGB_BINS = 12
_EDGE_MARGIN_PX = 1.5
_FOOT_PERTURBATION_PX = 1.5
_MAX_FOOT_SENSITIVITY_M = 4.0
_FUSION_RADIUS_M = .9


@dataclass(frozen=True)
class GroundHit:
    detection_id: str
    kind: Kind
    camera_name: str
    xy: XY
    camera_xyz: XYZ
    confidence: float
    uncertainty_m: float
    appearance: tuple[float, ...]

    def __post_init__(self):
        identifier(self.detection_id, "detection ID")
        identifier(self.camera_name, "camera name")
        _check_kind(self.kind)
        xy = tuple(matrix(self.xy, (2,), "hit XY").tolist())
        camera_xyz = tuple(matrix(self.camera_xyz, (3,), "camera XYZ").tolist())
        uncertainty = finite_real(self.uncertainty_m, "footpoint sensitivity")
        appearance = tuple(matrix(self.appearance, (3 * _RGB_BINS,), "appearance histogram").tolist())
        if uncertainty <= 0 or min(appearance) < 0 or not isclose(fsum(appearance), 1., abs_tol=1e-9):
            raise ValueError("Expected positive sensitivity and a nonnegative L1-normalized appearance")
        object.__setattr__(self, "xy", xy)
        object.__setattr__(self, "camera_xyz", camera_xyz)
        object.__setattr__(self, "confidence", _check_confidence(self.confidence))
        object.__setattr__(self, "uncertainty_m", uncertainty)
        object.__setattr__(self, "appearance", appearance)


@dataclass(frozen=True)
class GroundObject:
    object_id: str
    kind: Kind
    xy: XY
    confidence: float
    hits: tuple[GroundHit, ...]

    def __post_init__(self):
        identifier(self.object_id, "local object ID")
        _check_kind(self.kind)
        hits = tuple_snapshot(self.hits, "object hits")
        if not hits or any(not isinstance(hit, GroundHit) or hit.kind != self.kind for hit in hits):
            raise ValueError("Ground object needs nonempty same-kind observations")
        if len({hit.camera_name for hit in hits}) != len(hits) or len({hit.detection_id for hit in hits}) != len(hits):
            raise ValueError("Ground object may contain only one observation per camera and detection")
        object.__setattr__(self, "hits", hits)
        object.__setattr__(self, "xy", tuple(matrix(self.xy, (2,), "object XY").tolist()))
        object.__setattr__(self, "confidence", _check_confidence(self.confidence))


def project_detection(calibration: CameraCalibration, rgb: NDArray[np.uint8], box_xyxy: ArrayLike,
                      kind: Kind, confidence: float, *, detection_id: str) -> GroundHit | None:
    """Intersect the box's bottom-center ray with ego z=0, or abstain.

    The reported sensitivity is a worst planar displacement under a small
    vertical pixel perturbation, not a calibrated position covariance. The
    descriptor is three 12-bin inner-crop RGB histograms normalized jointly;
    compare descriptors only within the same camera.
    """
    if not isinstance(calibration, CameraCalibration):
        raise ValueError("Expected a rectified camera calibration")
    identifier(detection_id, "detection ID")
    _check_kind(kind)
    confidence = _check_confidence(confidence)
    if (not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or
            rgb.shape != (calibration.height, calibration.width, 3)):
        raise ValueError("Expected uint8 RGB that matches the rectified calibration")
    left, top, right, bottom = matrix(box_xyxy, (4,), "box XYXY").tolist()
    if right < left or bottom < top:
        raise ValueError("Box corners must be in XYXY order")
    if (right - left < 4 or bottom - top < 8 or left < _EDGE_MARGIN_PX or top < _EDGE_MARGIN_PX
            or right > calibration.width - 1 - _EDGE_MARGIN_PX
            or bottom > calibration.height - 1 - _EDGE_MARGIN_PX):
        return None
    camera_to_ego = invert_pose(calibration.ego_to_camera)
    u = (left + right) / 2
    intrinsic = calibration.intrinsic
    point = _ground_point(u, bottom, intrinsic, camera_to_ego, calibration.near_m)
    if point is None or not (0 <= point[0] <= 32 and abs(point[1]) <= 12):
        return None
    perturbed = [_ground_point(u, bottom + d, intrinsic, camera_to_ego, calibration.near_m)
                 for d in (-_FOOT_PERTURBATION_PX, _FOOT_PERTURBATION_PX)]
    if any(p is None for p in perturbed):
        return None
    sensitivity = max(dist(point, p) for p in perturbed)
    if sensitivity > _MAX_FOOT_SENSITIVITY_M:
        return None
    return GroundHit(detection_id, kind, calibration.name, point, tuple(camera_to_ego[:3, 3]),
                     confidence, max(.15, sensitivity), _appearance(rgb, left, top, right, bottom))


def _ground_point(u, v, intrinsic, camera_to_ego, near) -> XY | None:
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        vertical = (v - intrinsic[1, 2]) / intrinsic[1, 1]
        horizontal = (u - intrinsic[0, 2] - intrinsic[0, 1] * vertical) / intrinsic[0, 0]
        ray = np.array([horizontal, vertical, 1.])
        if not np.isfinite(ray).all() or abs(horizontal) > 4 or abs(vertical) > 4:
            return None
        origin = camera_to_ego[:3, 3]
        direction = camera_to_ego[:3, :3] @ ray
        if origin[2] <= 0 or direction[2] >= -.02 * np.linalg.norm(direction):
            return None
        depth = -origin[2] / direction[2]
        if not near <= depth <= 48:
            return None
        xy = origin[:2] + direction[:2] * depth
    return tuple(xy.tolist()) if np.isfinite(xy).all() else None


def _appearance(rgb, left, top, right, bottom):
    width, height = right - left, bottom - top
    x0, x1 = floor(left + width * .2), ceil(right - width * .2)
    y0, y1 = floor(top + height * .15), ceil(bottom - height * .25)
    inner = rgb[y0:y1, x0:x1]
    quantized = inner.astype(np.uint16) * _RGB_BINS // 256
    counts = np.concatenate([np.bincount(quantized[..., c].ravel(), minlength=_RGB_BINS) for c in range(3)])
    return tuple((counts / counts.sum()).tolist())


def fuse_ground_hits(hits) -> tuple[GroundObject, ...]:
    """Confidence-first complete-link fusion, without treating IDs as tracks."""
    hits = tuple_snapshot(hits, "ground hits")
    if any(not isinstance(hit, GroundHit) for hit in hits):
        raise ValueError("Fusion takes only projected ground hits")
    if len({hit.detection_id for hit in hits}) != len(hits):
        raise ValueError("Detection IDs must be unique within an observation")
    clusters = []
    for hit in sorted(hits, key=lambda h: (-h.confidence, h.uncertainty_m, h.camera_name, h.detection_id)):
        best = None
        for index, cluster in enumerate(clusters):
            if cluster[0].kind != hit.kind or any(member.camera_name == hit.camera_name for member in cluster):
                continue
            furthest = max(dist(member.xy, hit.xy) for member in cluster)
            if furthest <= _FUSION_RADIUS_M and (best is None or (furthest, index) < best):
                best = (furthest, index)
        if best is None:
            clusters.append([hit])
        else:
            clusters[best[1]].append(hit)
    return tuple(_make_object(index, cluster) for index, cluster in enumerate(clusters))


def _make_object(index, cluster):
    scale = max(.15, min(hit.uncertainty_m for hit in cluster))
    weights = [max(hit.confidence, .05) * (scale / max(.15, hit.uncertainty_m)) ** 2 for hit in cluster]
    total = fsum(weights)
    xy = tuple(fsum((w / total) * hit.xy[axis] for w, hit in zip(weights, cluster)) for axis in range(2))
    return GroundObject(f"d{index:02d}", cluster[0].kind, xy, cluster[0].confidence, tuple(cluster))


def align_ground_objects(objects, past_to_now: ArrayLike) -> tuple[GroundObject, ...]:
    """Re-express previous inferred ground points in the current ego frame."""
    pose = rigid_transform(past_to_now, "past-to-now")
    rotation, shift = pose[:3, :3], pose[:3, 3]
    transformed = []
    for item in tuple_snapshot(objects, "ground objects"):
        if not isinstance(item, GroundObject):
            raise ValueError("Alignment takes only fused ground objects")
        xy = tuple((rotation[:2, :2] @ item.xy + shift[:2]).tolist())
        hits = tuple(replace(hit, xy=tuple((rotation[:2, :2] @ hit.xy + shift[:2]).tolist()),
                             camera_xyz=tuple((rotation @ hit.camera_xyz + shift).tolist()))
                     for hit in item.hits)
        transformed.append(replace(item, xy=xy, hits=hits))
    return tuple(transformed)


def _check_kind(kind):
    if not isinstance(kind, str) or kind not in ("person", "bicycle"):
        raise ValueError("Ground detections must be person or bicycle")


def _check_confidence(confidence):
    confidence = finite_real(confidence, "detector confidence")
    if not 0 <= confidence <= 1:
        raise ValueError("Detector confidence must be in [0, 1]")
    return confidence
