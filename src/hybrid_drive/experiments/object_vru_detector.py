"""Pinned offline TorchVision detector for camera-specific VRU boxes."""

from dataclasses import dataclass
from hashlib import sha256
from importlib.metadata import version
from io import BytesIO
from math import isfinite
from pathlib import Path

import numpy as np
import torch

from hybrid_drive.contracts.sensors import CameraSnapshot
from hybrid_drive.experiments.bounded_io import read_bounded
from hybrid_drive.experiments.object_vru_geometry import fuse_ground_hits, project_detection, trace_projection


WEIGHTS_NAME = "ssdlite320_mobilenet_v3_large_coco-a79551df.pth"
WEIGHTS_SIZE = 14_069_355
WEIGHTS_SHA256 = "a79551df90c79834bcd3bb3845ef9d966b5449a3a9b2833ae8404778ca5d65d2"
WEIGHTS_URL = "https://download.pytorch.org/models/" + WEIGHTS_NAME
TORCHVISION_VERSION = "0.15.2"
RECTIFIED_WH = (768, 432)
COCO_CLASSES = {1: "person", 2: "bicycle"}
SCORE_FLOOR = {"person": .25, "bicycle": .20}
_MAX_IMAGE_BOXES = 24
_TILE_NMS = .48
_BATCH = 4


@dataclass(frozen=True)
class DetectedBox:
    kind: str
    confidence: float
    xyxy: tuple[float, float, float, float]
    tile: int


@dataclass(frozen=True)
class _Tile:
    view_index: int
    tile: int
    crop: tuple[int, int, int, int]


class LocalVruDetector:
    def __init__(self, weights_root: Path):
        installed = version("torchvision")
        if installed.removesuffix("+cpu") != TORCHVISION_VERSION:
            raise RuntimeError("Use the project's pinned TorchVision CPU image")
        raw = read_bounded(weights_root / WEIGHTS_NAME, WEIGHTS_SIZE)
        if len(raw) != WEIGHTS_SIZE or sha256(raw).hexdigest() != WEIGHTS_SHA256:
            raise ValueError("SSDLite weights differ from the pinned publisher file")
        from torchvision.models.detection import SSDLite320_MobileNet_V3_Large_Weights, ssdlite320_mobilenet_v3_large

        official = SSDLite320_MobileNet_V3_Large_Weights.COCO_V1
        if official.url != WEIGHTS_URL or any(official.meta["categories"][i] != name for i, name in COCO_CLASSES.items()):
            raise ValueError("Unexpected official detector class or provenance mapping")
        weights = torch.load(BytesIO(raw), map_location="cpu", weights_only=True)
        if (not isinstance(weights, dict) or len(weights) > 1024 or not all(
            isinstance(name, str) and isinstance(value, torch.Tensor) and torch.isfinite(value).all()
            for name, value in weights.items()
        )):
            raise ValueError("Expected a finite local tensor-only detector state")
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(0)
            detector = ssdlite320_mobilenet_v3_large(weights=None, weights_backbone=None, num_classes=91,
                                                     score_thresh=.16, detections_per_img=80)
        detector.load_state_dict(weights, strict=True)
        self._model = detector.eval().requires_grad_(False)
        self.provenance = {"weights_sha256": WEIGHTS_SHA256, "weights_bytes": WEIGHTS_SIZE,
                           "url": WEIGHTS_URL, "torchvision": installed, "torch": str(torch.__version__),
                           "size_wh": list(RECTIFIED_WH), "profile": "six_full_views_plus_front_overlap_tiles"}

    def __call__(self, snapshot: CameraSnapshot):
        return self._detect(snapshot, audit=False)

    def audit(self, snapshot: CameraSnapshot):
        """Return unchanged objects and trace the model's emitted VRU boxes."""
        return self._detect(snapshot, audit=True)

    @torch.no_grad()
    def _detect(self, snapshot, *, audit):
        views = snapshot.views
        if len(views) != 6 or any((v.calibration.width, v.calibration.height) != RECTIFIED_WH for v in views):
            raise ValueError("Expected the six calibrated detector views at the declared resolution")
        tile_specs = _tiles(views)
        collected = [[] for _ in views]
        traces = [[] if audit else None for _ in views]
        for offset in range(0, len(tile_specs), _BATCH):
            batch = tile_specs[offset:offset + _BATCH]
            raster = []
            for tile in batch:
                x1, y1, x2, y2 = tile.crop
                rgb = views[tile.view_index].rgb[y1:y2, x1:x2]
                raster.append(torch.from_numpy(np.array(rgb)).permute(2, 0, 1).float().div_(255))
            for tile, result in zip(batch, self._model(raster), strict=True):
                idx = tile.view_index
                collected[idx].extend(_model_boxes(tile, result, traces[idx]))
        hits, retained, cameras = [], 0, []
        for idx, (view, boxes, trace) in enumerate(zip(views, collected, traces, strict=True)):
            kept = _project_boxes(view, idx, boxes, project_detection, trace)
            retained += len(kept)
            hits.extend(hit for _, hit in kept if hit is not None)
            if trace is not None:
                cameras.append({"camera_name": view.calibration.name, "boxes": trace})
        objects = fuse_ground_hits(tuple(hits))
        if audit:
            return objects, cameras
        return objects, {"tiles": len(tile_specs), "detector_boxes": retained, "projected_camera_hits": len(hits)}


def _tiles(views):
    specs = []
    w, h = RECTIFIED_WH
    for i, view in enumerate(views):
        specs.append(_Tile(i, 0, (0, 0, w, h)))
        if view.direction.value.startswith("front"):
            specs.append(_Tile(i, 1, (0, h//5, (5*w)//8, (19*h)//20)))
            specs.append(_Tile(i, 2, ((3*w)//8, h//5, w, (19*h)//20)))
    return specs


def remap_box(tile, label, confidence, box):
    remapped, state = _classify_box(tile, label, confidence, box, audit=False)
    return remapped if state is None else None


def _model_boxes(tile, result, trace):
    packed = zip(result["labels"].tolist(), result["scores"].tolist(), result["boxes"].tolist(), strict=True)
    for label, confidence, box in packed:
        remapped, state = _classify_box(tile, label, confidence, box, audit=trace is not None)
        if remapped is None:
            continue
        if state is None:
            yield remapped
        elif trace is not None:
            trace.append(_trace_box(remapped, state))


def _classify_box(tile, label, confidence, box, *, audit):
    kind = COCO_CLASSES.get(label)
    if kind is None:
        return None, None
    if not isfinite(confidence) or not 0 <= confidence <= 1 or len(box) != 4 or not all(map(isfinite, box)):
        raise ValueError("Detector returned nonfinite or invalid box evidence")
    below_floor = confidence < SCORE_FLOOR[kind]
    if below_floor and not audit:
        return None, "score_floor"
    start_x, start_y, end_x, end_y = tile.crop
    w, h = end_x - start_x, end_y - start_y
    x1, y1, x2, y2 = (float(v) for v in box)
    if not (0 <= x1 < x2 <= w + .01 and 0 <= y1 < y2 <= h + .01):
        raise ValueError("Detector box lies outside its input crop")
    x2, y2 = min(x2, w), min(y2, h)
    state = None
    if below_floor:
        state = "score_floor"
    elif x1 >= x2 or y1 >= y2 or (tile.tile and min(x1, y1, w-x2, h-y2) <= 2.):
        state = "crop_edge"
    return DetectedBox(kind, confidence, (start_x+x1, start_y+y1, start_x+x2, start_y+y2), tile.tile), state


def deduplicate_boxes(boxes):
    return tuple(box for box, state in _box_states(boxes, audit=False) if state is None)


def _box_states(boxes, *, audit):
    kept = []
    for box in sorted(boxes, key=lambda b: (-b.confidence, b.kind, b.tile, b.xyxy)):
        if len(kept) == _MAX_IMAGE_BOXES:
            if not audit:
                break
            yield box, "box_budget"
        elif any(box.kind == prior.kind and _iou(box.xyxy, prior.xyxy) >= _TILE_NMS for prior in kept):
            if audit:
                yield box, "deduplicated"
        else:
            kept.append(box)
            yield box, None


def _project_boxes(view, index, boxes, project, trace):
    kept = []
    for box, state in _box_states(boxes, audit=trace is not None):
        if state is not None:
            trace.append(_trace_box(box, state))
            continue
        detection_id = f"v{index}-b{len(kept)}"
        args = (view.calibration, view.rgb, box.xyxy, box.kind, box.confidence)
        if trace is None:
            hit = project(*args, detection_id=detection_id)
        else:
            hit, reason = trace_projection(*args, detection_id=detection_id)
            trace.append(_trace_box(box, "projected" if hit is not None else "projection_rejected",
                                    projection_reason=reason, hit=hit))
        kept.append((box, hit))
    return kept


def _trace_box(box, state, *, projection_reason=None, hit=None):
    return {"kind": box.kind, "confidence": box.confidence, "xyxy": box.xyxy, "tile": box.tile,
            "state": state, "projection_reason": projection_reason,
            "xy_m": None if hit is None else hit.xy, "detection_id": None if hit is None else hit.detection_id}


def _iou(a, b):
    shared = max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(0., min(a[3], b[3]) - max(a[1], b[1]))
    return shared / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - shared)
