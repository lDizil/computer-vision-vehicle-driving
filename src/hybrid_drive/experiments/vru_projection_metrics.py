"""Attribute current-centre loss to the first stage supported by camera evidence."""

from collections import Counter, defaultdict
from dataclasses import dataclass
from math import dist
from statistics import median

from hybrid_drive.experiments.object_vru_metrics import MATCH_GAP_M, MATCH_RADIUS_M, _finite, _xy, match_current_centers
from hybrid_drive.experiments.vru_chain_capture import read_ground_objects
from hybrid_drive.experiments.vru_detector_coverage_metrics import _category, _read_targets
from hybrid_drive.experiments.vru_observation_fusion import profile_vru_observations


IOU = .30
IOU_MARGIN = .05
IOU_CHECKS = (.20, IOU, .45)
SIZE_WH = (768, 432)
_MODELS = ("ssdlite", "faster_rcnn")
_STATES = {"score_floor", "crop_edge", "deduplicated", "box_budget", "projection_rejected", "projected"}
_REASONS = {"size", "edge", "invalid_ray", "outside_roi", "unstable"}
_KINDS = {"person", "bicycle"}


@dataclass(frozen=True)
class _Box:
    detector: str
    camera: str
    kind: str
    confidence: float
    rectangle: tuple
    state: str
    reason: str | None
    ground: tuple | None


def summarize_projection_audit(traces, truth_rows, labels, frames):
    seen = set()
    episodes = []
    for row, truth, label in zip(traces, truth_rows, labels, strict=True):
        token = row["sample_token"]
        if token in seen or token != truth["sample_token"] or token != label["sample_token"]:
            raise ValueError("Projection audit inputs and both annotations must cover the same windows in order")
        seen.add(token)
        episodes.append(_episode(row, truth, label, frames[token]))
    if not episodes:
        raise ValueError("Expected at least one bound projection window")
    cases = [case for episode in episodes for case in episode["cases"]]
    result = _summary(cases, episodes)
    grouped = defaultdict(list)
    for episode in episodes:
        grouped[episode["recording"]].append(episode)
    return {
        **result, "cases": cases,
        "by_recording": {name: _summary([case for item in rows for case in item["cases"]], rows)
                         for name, rows in sorted(grouped.items())},
        "policy": {
            "current_match_radius_m": MATCH_RADIUS_M, "current_match_margin_m": MATCH_GAP_M,
            "box_overlap_iou": IOU, "box_overlap_margin_iou": IOU_MARGIN, "box_overlap_sensitivity": IOU_CHECKS,
            "emitted_boxes": "torchvision_output_after_model_threshold_and_nms",
            "box_association": "same_camera_and_class_unique_target_against_projected_cuboid_envelopes",
            "visibility": "projected_envelope_not_proof_of_unoccluded_rgb",
            "plane_oracle": "annotation_bottom_height_only_a_diagnostic_not_a_prediction",
        },
    }


def _episode(row, truth, label, frame):
    camera_metadata = {view["camera"]: view for view in frame["views"]}
    tracks = _read_targets(label)
    targets = truth["targets"]
    positions = [item[0] for item in tracks]
    if len(targets) != len(tracks) or any(tuple(target["box_ego"][:2]) != xy for target, xy in zip(targets, positions)):
        raise ValueError("Projection truth must preserve the order and positions of current targets")
    raw, objects = [], {}
    for name in _MODELS:
        profile = row["profiles"][name]
        objects[name] = read_ground_objects(profile["objects"])
        raw.extend(_validate_boxes(name, profile["cameras"], objects[name], camera_metadata))
    fused = profile_vru_observations(*(objects[name] for name in _MODELS)).combined
    coordinates = [obj.xy for obj in fused]
    near = [[i for i, center in enumerate(coordinates) if dist(xy, center) <= MATCH_RADIUS_M] for xy in positions]
    matched = {j: i for i, j in match_current_centers(coordinates, positions)}
    evidence = {cutoff: _assign_evidence(raw, targets, cutoff) for cutoff in IOU_CHECKS}
    cases = []
    for i, (track, target) in enumerate(zip(tracks, targets, strict=True)):
        group = _category(dist(track[0], track[-1]))
        boxes, ties = evidence[IOU]
        owned = boxes.get(i, [])
        projected = [item for item in owned if item[0].ground is not None]
        ground_distances = [dist(box.ground, track[0]) for box, _ in projected]
        cause = _cause(i, target, matched, near, owned, i in ties, ground_distances)
        cases.append({
            "sample_token": row["sample_token"], "recording": row["recording"], "target_index": i, "category": group,
            "kind": target["kind"], "center_xy_m": track[0], "box_ego": target["box_ego"],
            "cause": cause, "matched": i in matched,
            "near_centers": [{"object_id": fused[j].object_id, "xy_m": coordinates[j],
                              "error_m": dist(track[0], coordinates[j])} for j in near[i]],
            "views": target["views"], "owned_2d_counts": {str(k): len(x[0].get(i, [])) for k, x in evidence.items()},
            "projection_rejections": sorted({box.reason for box, _ in owned if box.reason is not None}),
            "filter_states": sorted({box.state for box, _ in owned if box.state not in ("projected", "projection_rejected")}),
            "boxes": _examples(owned, track[0]),
            "best_owned_ground_error_m": min(ground_distances, default=None),
            "plane_probe": _plane_probe(target["box_ego"], projected, camera_metadata),
        })
    return {"recording": row["recording"], "cases": cases, "fused_objects": len(fused),
            "emitted_box_states": Counter(box.state for box in raw)}


def _validate_boxes(detector, cameras, objects, camera_metadata):
    if tuple(row["camera_name"] for row in cameras) != tuple(camera_metadata):
        raise ValueError("Detector trace changed the ordered camera rig")
    hits = [hit for obj in objects for hit in obj.hits]
    untraced = {hit.detection_id: hit for hit in hits}
    if len(untraced) != len(hits):
        raise ValueError("Duplicate ground hit in detector trace")
    for row in cameras:
        camera = row["camera_name"]
        for raw in row["boxes"]:
            kind, state, reason, identifier = (raw[k] for k in ("kind", "state", "projection_reason", "detection_id"))
            confidence = _finite(raw["confidence"], "camera box score")
            if (kind not in _KINDS or state not in _STATES or not 0 <= confidence <= 1
                    or (state == "projection_rejected" and reason not in _REASONS)
                    or (state != "projection_rejected" and reason is not None)):
                raise ValueError("Unrecognized detector trace state")
            ground = None
            if state == "projected":
                hit = untraced.pop(identifier, None)
                ground = _xy(raw["xy_m"], "camera hit position")
                if hit is None or (hit.camera_name, hit.kind, hit.xy, hit.confidence) != (camera, kind, ground, confidence):
                    raise ValueError("The camera trace is not linked to its exact projected hit")
            elif identifier is not None or raw["xy_m"] is not None:
                raise ValueError("A rejected camera box cannot have an accepted hit")
            yield _Box(detector, camera, kind, confidence, _xyxy(raw["xyxy"]), state, reason, ground)
    if untraced:
        raise ValueError("Projected hits are missing from the camera trace")


def _assign_evidence(raw, targets, cutoff):
    footprints = defaultdict(list)
    for i, target in enumerate(targets):
        for view in target["views"]:
            footprints[view["camera_name"], target["kind"]].append((i, _xyxy(view["xyxy"])))
    owned, ties = defaultdict(list), set()
    for box in raw:
        ranks = sorted(((_iou(box.rectangle, rect), i) for i, rect in footprints[box.camera, box.kind]), reverse=True)
        if not ranks or ranks[0][0] < cutoff:
            continue
        highest, selected = ranks[0]
        if len(ranks) > 1 and highest - ranks[1][0] < IOU_MARGIN:
            ties.update(i for value, i in ranks if value >= cutoff and highest - value < IOU_MARGIN)
            continue
        owned[selected].append((box, highest))
    return owned, ties


def _cause(i, target, matched, near, owned, tied, distances):
    if i in matched:
        return "matched"
    if near[i]:
        return "ground_center_ambiguous"
    if target["kind"] not in _KINDS:
        return "unsupported_class"
    if not target["views"]:
        return "no_projected_target_envelope"
    if not owned:
        return "camera_box_ambiguous" if tied else "no_reliable_camera_box"
    if distances:
        return "lost_after_camera_projection" if min(distances) <= MATCH_RADIUS_M else "camera_projection_far"
    if any(box.state == "projection_rejected" for box, _ in owned):
        return "camera_projection_rejected"
    return "camera_box_filtered"


def _plane_probe(world_box, projected, camera_metadata):
    xy = world_box[:2]
    bottom = world_box[2] - world_box[5] / 2.
    errors = []
    for box, _ in projected:
        origin = camera_metadata[box.camera]["camera_to_ego_translation"]
        if origin[2] <= 0 or bottom >= origin[2]:
            continue
        before = dist(box.ground, xy)
        factor = (origin[2] - bottom) / origin[2]
        after = dist([start + (point - start) * factor for point, start in zip(box.ground, origin[:2], strict=True)], xy)
        errors.append((before, after))
    return {"box_bottom_z_m": bottom, "paired_hits": len(errors),
            "best_before_m": min((a for a, _ in errors), default=None), "best_oracle_m": min((b for _, b in errors), default=None)}


def _examples(owned, center):
    rows = []
    for state in sorted({box.state for box, _ in owned}):
        evidence = sorted((item for item in owned if item[0].state == state), key=lambda item: -item[1])
        for box, overlap in evidence[:2]:
            rows.append({"detector": box.detector, "camera_name": box.camera, "box_xyxy": box.rectangle,
                         "score": box.confidence, "iou": overlap, "state": box.state, "reason": box.reason,
                         "xy_m": box.ground, "ground_error_m": None if box.ground is None else dist(box.ground, center)})
    return rows


def _summary(cases, episodes):
    moving = [case for case in cases if case["category"] == "moving"]
    counts = Counter()
    for episode in episodes:
        counts.update(episode["emitted_box_states"])
    return {
        "windows": len(episodes), "recordings": len({ep["recording"] for ep in episodes}),
        "fused_objects": sum(ep["fused_objects"] for ep in episodes), "emitted_box_states": dict(sorted(counts.items())),
        "all": _counts(cases), "moving": _counts(moving),
        "moving_by_forward_range": {name: _counts([case for case in moving if lo <= case["center_xy_m"][0] < hi])
                                    for name, lo, hi in (("under_8m", 0., 8.), ("8_to_16m", 8., 16.), ("16m_plus", 16., 1e4))},
    }


def _counts(rows):
    unplaced = [item for item in rows if not item["matched"] and not item["near_centers"]]
    projections = [item["plane_probe"] for item in unplaced if item["plane_probe"]["paired_hits"]]
    return {
        "targets": len(rows), "causes": dict(sorted(Counter(item["cause"] for item in rows).items())),
        "unmatched_with_no_near_center": len(unplaced),
        "no_near_with_camera_witness": {str(i): sum(item["owned_2d_counts"][str(i)] != 0 for item in unplaced)
                                        for i in IOU_CHECKS},
        "projection_rejection_reasons": dict(sorted(Counter(reason for item in unplaced
                                                            for reason in item["projection_rejections"]).items())),
        "box_filter_states": dict(sorted(Counter(state for item in unplaced for state in item["filter_states"]).items())),
        "diagnostic_plane_oracle": {
            "unplaced_targets_with_owned_projected_hit": len(projections),
            "misses_moved_within_radius": sum(item["best_before_m"] > MATCH_RADIUS_M >= item["best_oracle_m"]
                                             for item in projections),
            "median_best_error_before_m": median(item["best_before_m"] for item in projections) if projections else None,
            "median_best_error_with_annotation_bottom_m": median(item["best_oracle_m"] for item in projections) if projections else None,
        },
    }


def _xyxy(raw):
    if not isinstance(raw, (tuple, list)) or len(raw) != 4:
        raise ValueError("Expected a bounded camera rectangle")
    a, b, c, d = (_finite(value, "camera rectangle") for value in raw)
    if not (0 <= a <= c <= SIZE_WH[0] and 0 <= b <= d <= SIZE_WH[1]):
        raise ValueError("Camera rectangle lies outside the rectified view")
    return a, b, c, d


def _iou(a, b):
    shared = max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(0., min(a[3], b[3]) - max(a[1], b[1]))
    if not shared:
        return 0.
    return shared / ((a[2]-a[0]) * (a[3]-a[1]) + (b[2]-b[0]) * (b[3]-b[1]) - shared)
