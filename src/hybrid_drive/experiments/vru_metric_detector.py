"""Offline D-FINE and metric-depth adapter for the existing projected-object tracker."""

from dataclasses import dataclass, replace
from hashlib import sha1, sha256
from importlib.metadata import version
from math import ceil, floor, isfinite
from pathlib import Path

import numpy as np

from hybrid_drive.assets.files import check_components
from hybrid_drive.experiments.bounded_io import decode_json, read_bounded
from hybrid_drive.experiments.object_vru_geometry import GroundHit, _appearance, fuse_ground_hits, project_detection
from hybrid_drive.geometry.rigid import invert_pose


MODEL_MANIFEST_SHA256 = "d33cf82ad2ebdaca45bb29b36c045d742b645b5ed0d46ddde1a6389f9151b8ac"
FRAMES_WH = (768, 432)
MODEL_VERSIONS = {"torch": "2.8.0", "torchvision": "0.23.0", "transformers": "4.57.3"}
SCORE_FLOOR = {"person": .25, "bicycle": .20}
DEPTH_POLICY = {"quantile": .35, "min_valid_fraction": .70, "max_range_m": 48., "max_spread_m": 4.}


@dataclass(frozen=True)
class DepthSample:
    meters_z: float
    spread_z: float


def model_profiles(root, manifest_path):
    """Validate exactly the prepared upstream configurations before loading weights."""
    payload = read_bounded(manifest_path, 16384)
    if sha256(payload).hexdigest() != MODEL_MANIFEST_SHA256:
        raise ValueError("Expected the pinned metric perception model manifest")
    manifest = decode_json(payload)
    verified = {}
    for name in ("detector", "depth"):
        spec = manifest[name]
        directory = Path(root) / name
        check_components(directory)
        if {item.name for item in directory.iterdir()} != set(spec["files"]):
            raise ValueError("Expected only the three pinned files per model")
        for filename, metadata in spec["files"].items():
            path = directory / filename
            size = metadata["bytes"]
            data = read_bounded(path, size)
            if len(data) != size:
                raise ValueError(f"Unexpected file size for {name}/{filename}")
            if "sha256" in metadata:
                expected, actual = metadata["sha256"], sha256(data).hexdigest()
            else:
                expected = metadata["git_blob_sha1"]
                actual = sha1(f"blob {size}".encode() + bytes((0,)) + data).hexdigest()
            if actual != expected:
                raise ValueError(f"Wrong checksum for {name}/{filename}")
        verified[name] = {"repository": spec["repo"], "revision": spec["revision"],
                          "weights_sha256": spec["files"]["model.safetensors"]["sha256"]}
    return verified


class MetricVruDetector:
    """Same camera images and 2D boxes for the old-plane and depth treatments."""

    def __init__(self, root, manifest, *, device, batch_size=2, tiling="full"):
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation, AutoModelForObjectDetection

        if device not in ("cpu", "mps") or device == "mps" and not torch.backends.mps.is_available():
            raise ValueError("Expected an available local CPU or MPS backend")
        if type(batch_size) is not int or not 1 <= batch_size <= 3:
            raise ValueError("Expected a small bounded camera batch")
        if tiling not in ("full", "front"):
            raise ValueError("Expected full views alone or the fixed front overlap tiles")
        installed = {package: version(package) for package in MODEL_VERSIONS}
        if installed != MODEL_VERSIONS:
            raise RuntimeError(f"Expected the pinned model runtime, got {installed}")
        self._torch = torch
        self._device = device
        self._batch = batch_size
        self._tiling = tiling
        profiles = model_profiles(root, manifest)
        detector_path, depth_path = (Path(root) / name for name in ("detector", "depth"))
        local = {"local_files_only": True, "trust_remote_code": False}
        self._det_processor = AutoImageProcessor.from_pretrained(detector_path, use_fast=False, local_files_only=True)
        self._dep_processor = AutoImageProcessor.from_pretrained(depth_path, use_fast=False, local_files_only=True)
        self._detector = AutoModelForObjectDetection.from_pretrained(
            detector_path, use_safetensors=True, **local,
        ).eval().to(device)
        self._depth = AutoModelForDepthEstimation.from_pretrained(
            depth_path, use_safetensors=True, **local,
        ).eval().to(device)
        self._classes = {int(key): value.lower() for key, value in self._detector.config.id2label.items()
                         if value.lower() in SCORE_FLOOR}
        if set(self._classes.values()) != set(SCORE_FLOOR):
            raise ValueError("Model must expose person and bicycle separately")
        config = self._depth.config
        if config.depth_estimation_type != "metric" or config.max_depth != 80:
            raise ValueError("Relative-depth or indoor checkpoints cannot supply metric ground points")
        self.provenance = {"models": profiles, "runtime": installed, "device": device,
                           "rectified_wh": list(FRAMES_WH), "batch_size": batch_size, "depth_policy": DEPTH_POLICY,
                           "box_floor": SCORE_FLOOR, "tiling": tiling}

    def __call__(self, snapshot):
        from PIL import Image
        from hybrid_drive.experiments.object_vru_detector import deduplicate_boxes

        if len(snapshot.views) != 6 or any((v.calibration.width, v.calibration.height) != FRAMES_WH for v in snapshot.views):
            raise ValueError("Expected the same six calibrated rectified camera views")
        frames = tuple(Image.fromarray(item.rgb) for item in snapshot.views)
        observations = []
        with self._torch.inference_mode():
            boxes = self._detect_views(frames, snapshot.views)
            for offset in range(0, len(frames), self._batch):
                current = frames[offset:offset+self._batch]
                detections = boxes[offset:offset+self._batch]
                keep = [i for i, values in enumerate(detections) if values]
                depths = dict(zip(keep, self._depth_maps([current[i] for i in keep]), strict=True)) if keep else {}
                for i, candidates in enumerate(detections):
                    view = snapshot.views[offset+i]
                    observations.append(_camera_observations(view, deduplicate_boxes(candidates), depths.get(i), offset+i))
        hits = {"plane": [], "metric": []}
        for _, plane, metric in observations:
            hits["plane"].extend(plane)
            hits["metric"].extend(metric)
        return {key: fuse_ground_hits(values) for key, values in hits.items()}, [row for row, _, _ in observations]

    def _detect_views(self, images, views):
        from hybrid_drive.experiments.object_vru_detector import _Tile, _tiles

        tiles = (_tiles(views) if self._tiling == "front"
                 else tuple(_Tile(i, 0, (0, 0, *FRAMES_WH)) for i in range(len(images))))
        collected = [[] for _ in views]
        for offset in range(0, len(tiles), self._batch):
            batch = tiles[offset:offset+self._batch]
            candidates = self._boxes(tuple(images[tile.view_index].crop(tile.crop) for tile in batch))
            for tile, hits in zip(batch, candidates, strict=True):
                for hit in hits:
                    mapped = place_tile_box(tile, hit)
                    if mapped is not None:
                        collected[tile.view_index].append(mapped)
        return collected

    def _boxes(self, images):
        from hybrid_drive.experiments.object_vru_detector import DetectedBox

        inputs = self._det_processor(images=list(images), return_tensors="pt").to(self._device)
        result = self._detector(**inputs)
        target_sizes = [(image.height, image.width) for image in images]
        rows = self._det_processor.post_process_object_detection(result, threshold=min(SCORE_FLOOR.values()),
                                                                  target_sizes=target_sizes)
        grouped = []
        for image, row in zip(images, rows, strict=True):
            boxes = []
            raw = zip(row["labels"].tolist(), row["scores"].tolist(), row["boxes"].tolist(), strict=True)
            for label, score, box in raw:
                kind = self._classes.get(label)
                if kind is None or not isfinite(score) or score < SCORE_FLOOR[kind]:
                    continue
                xyxy = _clip_box(box, image.width, image.height)
                if xyxy is not None:
                    boxes.append(DetectedBox(kind, score, xyxy, 0))
            grouped.append(boxes)
        return grouped

    def _depth_maps(self, images):
        inputs = self._dep_processor(images=list(images), return_tensors="pt").to(self._device)
        result = self._depth(**inputs).predicted_depth
        full = self._torch.nn.functional.interpolate(result.unsqueeze(1), FRAMES_WH[::-1], mode="bilinear",
                                                     align_corners=False)
        return full[:, 0].float().cpu().numpy()


def _camera_observations(view, boxes, depth, camera_index):
    output, flat, metric = [], [], []
    for i, box in enumerate(boxes):
        key = f"v{camera_index}-b{i}"
        plane = project_detection(view.calibration, view.rgb, box.xyxy, box.kind, box.confidence, detection_id=key)
        measured = depth_sample(depth, box.xyxy) if depth is not None else None
        geo = metric_hit(view, box, measured, detection_id=key) if measured is not None else None
        if plane is not None:
            flat.append(plane)
        if geo is not None:
            metric.append(geo)
        output.append({"id": key, "kind": box.kind, "confidence": box.confidence, "xyxy": box.xyxy,
                       "depth_z_m": None if measured is None else measured.meters_z,
                       "depth_spread_z_m": None if measured is None else measured.spread_z,
                       "plane_xy_m": None if plane is None else plane.xy,
                       "metric_xy_m": None if geo is None else geo.xy})
    return {"camera_name": view.calibration.name, "boxes": output}, flat, metric


def _clip_box(box, width, height):
    if not all(isfinite(value) for value in box):
        return None
    x0, y0, x1, y1 = box
    x0, x1 = max(0., x0), min(float(width), x1)
    y0, y1 = max(0., y0), min(float(height), y1)
    return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None


def place_tile_box(tile, box):
    left, top, right, bottom = tile.crop
    a, b, c, d = box.xyxy
    if tile.tile and min(a, b, right-left-c, bottom-top-d) <= 2.:
        return None
    return replace(box, tile=tile.tile, xyxy=(left+a, top+b, left+c, top+d))


def depth_sample(depth, box):
    if depth.ndim != 2 or depth.shape != FRAMES_WH[::-1]:
        raise ValueError("Metric depth must use the same pixel grid as the box")
    x0, y0, x1, y1 = box
    w, h = x1-x0, y1-y0
    if w < 4. or h < 8.:
        return None
    inside = depth[floor(y0+h*.22):ceil(y0+h*.70), floor(x0+w*.30):ceil(x0+w*.70)]
    valid = inside[np.isfinite(inside) & (inside >= .5) & (inside <= DEPTH_POLICY["max_range_m"])]
    if len(valid) < max(6, DEPTH_POLICY["min_valid_fraction"] * inside.size):
        return None
    low, middle, high = np.quantile(valid, (.20, DEPTH_POLICY["quantile"], .80))
    spread = float(high - low)
    if spread > DEPTH_POLICY["max_spread_m"]:
        return None
    return DepthSample(float(middle), spread)


def metric_hit(view, box, depth, *, detection_id):
    cal = view.calibration
    x0, y0, x1, y1 = box.xyxy
    point = np.array(((x0+x1)/2., (y0+y1)/2., 1.))
    camera_to_ego = invert_pose(cal.ego_to_camera)
    origin = camera_to_ego[:3, 3]
    ray = camera_to_ego[:3, :3] @ np.linalg.solve(cal.intrinsic, point)
    x, y, z = origin + ray * depth.meters_z
    if not (0. <= x <= 32. and abs(y) <= 12. and -.75 <= z <= 3.25):
        return None
    jitter = max(.15, float(np.linalg.norm(ray[:2])) * depth.spread_z)
    return GroundHit(detection_id, box.kind, cal.name, (x, y), tuple(origin), box.confidence, jitter,
                     _appearance(view.rgb, x0, y0, x1, y1))
