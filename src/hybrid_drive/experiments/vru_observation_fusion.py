"""Fuse projected detections from one camera snapshot, before association."""

from collections.abc import Iterable
from dataclasses import replace
from math import dist

from hybrid_drive.experiments.object_vru_geometry import GroundObject


MIN_CROSS_DETECTOR_DISTANCE_M = 1.


def fuse_vru_observations(
    ssdlite: Iterable[GroundObject], faster_rcnn: Iterable[GroundObject],
) -> tuple[GroundObject, ...]:
    """Keep all SSDLite centers, adding only nonconflicting Faster R-CNN centers.

    The caller supplies both sets in the same frame. Class and confidence do
    not resolve collisions: scores from the two detectors are not calibrated
    to each other. A returned ID is still local to a snapshot, not a track ID.
    """
    primary = _namespaced(ssdlite, "ssd")
    secondary = _namespaced(faster_rcnn, "rcnn")
    additions = tuple(candidate for candidate in secondary if all(
        dist(candidate.xy, center.xy) >= MIN_CROSS_DETECTOR_DISTANCE_M for center in primary
    ))
    return primary + additions


def _namespaced(objects, source):
    objects = tuple(objects)
    if any(not isinstance(obj, GroundObject) for obj in objects):
        raise ValueError("Fusion requires projected ground objects")
    if len({obj.object_id for obj in objects}) != len(objects):
        raise ValueError("Duplicate object identity within one detector")
    hit_ids = [hit.detection_id for obj in objects for hit in obj.hits]
    if len(set(hit_ids)) != len(hit_ids):
        raise ValueError("Duplicate hit identity within one detector")
    return tuple(replace(obj, object_id=f"{source}:{obj.object_id}", hits=tuple(
        replace(hit, detection_id=f"{source}:{hit.detection_id}") for hit in obj.hits
    )) for obj in objects)
