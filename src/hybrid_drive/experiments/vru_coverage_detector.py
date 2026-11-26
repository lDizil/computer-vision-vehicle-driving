"""A second frozen CPU detection profile; geometry remains the #51 projection."""

from hashlib import sha256
from importlib.metadata import version
from io import BytesIO

import numpy as np
import torch

from hybrid_drive.experiments.bounded_io import read_bounded
from hybrid_drive.experiments.object_vru_detector import (
    COCO_CLASSES, RECTIFIED_WH, TORCHVISION_VERSION, _Tile, _model_boxes, _project_boxes,
)
from hybrid_drive.experiments.object_vru_geometry import fuse_ground_hits, project_detection


WEIGHTS_NAME = "fasterrcnn_mobilenet_v3_large_fpn-fb6a3cc7.pth"
WEIGHTS_SHA256 = "fb6a3cc702b1df54c18a44b26708cd083614211062d0c36d2ca7bf9270df3533"
WEIGHTS_SIZE = 77_844_807
WEIGHTS_URL = "https://download.pytorch.org/models/" + WEIGHTS_NAME
MIN_SIDE, MAX_SIDE = 640, 1120
REGION_BUDGET = 300
BOX_BUDGET = 80
PROFILE = "fasterrcnn_mobilenet_v3_large_fpn_six_full_640x1120"


class CoverageDetector:
    """TorchVision regions → unchanged VRU box filters → camera ground hits."""

    def __init__(self, root):
        installed = version("torchvision")
        if installed.removesuffix("+cpu") != TORCHVISION_VERSION:
            raise RuntimeError("Use the project's pinned TorchVision CPU image")
        raw = read_bounded(root / WEIGHTS_NAME, WEIGHTS_SIZE)
        if len(raw) != WEIGHTS_SIZE or sha256(raw).hexdigest() != WEIGHTS_SHA256:
            raise ValueError("Faster R-CNN weights differ from the pinned publisher file")
        from torchvision.models.detection import (
            FasterRCNN_MobileNet_V3_Large_FPN_Weights, fasterrcnn_mobilenet_v3_large_fpn,
        )

        official = FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1
        if official.url != WEIGHTS_URL or any(official.meta["categories"][i] != name for i, name in COCO_CLASSES.items()):
            raise ValueError("Unexpected publisher checkpoint and class mapping")
        weights = torch.load(BytesIO(raw), map_location="cpu", weights_only=True)
        if (not isinstance(weights, dict) or not 1 <= len(weights) <= 1024
                or not all(isinstance(k, str) and isinstance(v, torch.Tensor) and torch.isfinite(v).all()
                           for k, v in weights.items())):
            raise ValueError("Expected a local finite detector state")
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(0)
            model = fasterrcnn_mobilenet_v3_large_fpn(
                weights=None, weights_backbone=None, num_classes=91, min_size=MIN_SIDE, max_size=MAX_SIDE,
                rpn_pre_nms_top_n_test=600, rpn_post_nms_top_n_test=REGION_BUDGET,
                box_score_thresh=.16, box_detections_per_img=BOX_BUDGET,
            )
        model.load_state_dict(weights, strict=True)
        self._model = model.eval().requires_grad_(False)
        self.provenance = {"url": WEIGHTS_URL, "weights_sha256": WEIGHTS_SHA256, "weights_bytes": WEIGHTS_SIZE,
                           "torchvision": installed, "torch": str(torch.__version__), "profile": PROFILE,
                           "size_wh": list(RECTIFIED_WH), "model_min_side": MIN_SIDE, "model_max_side": MAX_SIDE,
                           "regions_per_image": REGION_BUDGET, "detections_per_image": BOX_BUDGET}

    def __call__(self, snapshot):
        return self._detect(snapshot, audit=False)

    def audit(self, snapshot):
        """Return unchanged objects and trace the model's emitted VRU boxes."""
        return self._detect(snapshot, audit=True)

    @torch.no_grad()
    def _detect(self, snapshot, *, audit):
        if len(snapshot.views) != 6 or any((v.calibration.width,v.calibration.height) != RECTIFIED_WH for v in snapshot.views):
            raise ValueError("Coverage comparison requires the original six-view rectification")
        w,h=RECTIFIED_WH
        hits, boxes_by_camera = [], []
        for i, view in enumerate(snapshot.views):
            raster = torch.from_numpy(np.array(view.rgb)).permute(2, 0, 1).float().div_(255)
            result, = self._model([raster])
            tile = _Tile(i, 0, (0, 0, w, h))
            trace = [] if audit else None
            kept = _project_boxes(view, i, _model_boxes(tile, result, trace), project_detection, trace)
            hits.extend(hit for _, hit in kept if hit is not None)
            if trace is None:
                trace = [{"kind": item.kind, "confidence": item.confidence, "xyxy": item.xyxy,
                          "ground_projection": hit is not None} for item, hit in kept]
            boxes_by_camera.append({"camera_name": view.calibration.name, "boxes": trace})
        return fuse_ground_hits(tuple(hits)), boxes_by_camera
