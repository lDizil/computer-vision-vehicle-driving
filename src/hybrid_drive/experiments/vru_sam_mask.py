"""One-frame SAM 2 box masks for a foreground-only depth ablation."""

from hashlib import sha1, sha256
from importlib.metadata import version
from math import ceil, floor, isfinite
from pathlib import Path

import numpy as np
from PIL import Image

from hybrid_drive.assets.files import check_components
from hybrid_drive.contracts._validation import finite_real, tuple_snapshot
from hybrid_drive.experiments.bounded_io import decode_json, read_bounded
from hybrid_drive.experiments.vru_metric_detector import DEPTH_POLICY, FRAMES_WH, DepthSample


SAM_MANIFEST_SHA256 = "fe25e3dbac9f30798c917dd858d2d1deeefa1d278e70dc6b54c81d7fdba52be5"
TRANSFORMERS_VERSION = "4.57.3"
MIN_MASK_IOU_SCORE = .80
MIN_INTERIOR_MASK_FRACTION = .25
_MAX_BOXES = 24
_LOW_RES_MASK_HW = (256, 256)


class SamBoxMasks:
    """Return one local foreground mask or an abstention per input box."""

    def __init__(self, local_model_root, manifest_path, *, device):
        import torch
        from transformers import Sam2Model, Sam2Processor

        if device not in ("cpu", "mps") or device == "mps" and not torch.backends.mps.is_available():
            raise ValueError("Expected an available local CPU or MPS backend")
        if version("transformers") != TRANSFORMERS_VERSION:
            raise RuntimeError("SAM 2 requires the pinned Transformers runtime")
        path, profile = _verified_model(local_model_root, manifest_path)
        local = {"local_files_only": True, "trust_remote_code": False}
        # The publisher uses a video checkpoint for both APIs. Select the image
        # classes explicitly, not the video architectures declared in config.
        self._processor = Sam2Processor.from_pretrained(path, **local)
        model, info = Sam2Model.from_pretrained(
            path, use_safetensors=True, output_loading_info=True, **local,
        )
        if info["missing_keys"] or info["mismatched_keys"] or info["error_msgs"]:
            raise ValueError("The pinned checkpoint did not populate the SAM 2 image model")
        self._model = model.eval().to(device)
        self._torch = torch
        self._device = device
        self.provenance = {
            **profile, "mode": "independent_image_box_prompts",
            "manifest_sha256": SAM_MANIFEST_SHA256, "transformers": TRANSFORMERS_VERSION, "device": device,
            "min_model_iou_score": MIN_MASK_IOU_SCORE, "min_interior_mask_fraction": MIN_INTERIOR_MASK_FRACTION,
        }

    def predict(self, rgb, boxes_xyxy) -> tuple[np.ndarray | None, ...]:
        if not isinstance(rgb, np.ndarray) or rgb.shape != (*FRAMES_WH[::-1], 3) or rgb.dtype != np.uint8:
            raise ValueError("SAM 2 requires a rectified uint8 768x432 RGB frame")
        boxes = tuple_snapshot(boxes_xyxy, "SAM 2 prompt boxes")
        if len(boxes) > _MAX_BOXES:
            raise ValueError("Expected at most 24 boxes for the current camera")
        if not boxes:
            return ()
        prompts = [list(_rectified_box(box)) for box in boxes]
        batch = self._processor(images=Image.fromarray(rgb), input_boxes=[prompts], return_tensors="pt").to(self._device)
        with self._torch.inference_mode():
            result = self._model(**batch, multimask_output=False)
        scores = result.iou_scores.float().cpu().numpy()
        if scores.shape != (1, len(boxes), 1) or tuple(result.pred_masks.shape) != (1, len(boxes), 1, *_LOW_RES_MASK_HW):
            raise ValueError("Expected one SAM 2 prediction per box in prompt order")
        sizes = batch["original_sizes"].cpu()
        restored = self._processor.post_process_masks(result.pred_masks.float().cpu(), sizes, binarize=True)
        if len(restored) != 1:
            raise ValueError("Expected SAM 2 masks only for the current camera")
        masks = restored[0].numpy()
        if masks.shape != (len(boxes), 1, *FRAMES_WH[::-1]) or masks.dtype != np.bool_:
            raise ValueError("Expected boolean SAM 2 masks on the original camera raster")
        return tuple(
            mask.copy() if isfinite(score) and MIN_MASK_IOU_SCORE <= score <= 1. and mask.any() else None
            for score, mask in zip(scores[0, :, 0], masks[:, 0], strict=True)
        )


def masked_depth_sample(depth, box, mask) -> DepthSample | None:
    """Use the baseline torso crop and depth statistics on masked pixels only."""
    if not isinstance(depth, np.ndarray) or depth.shape != FRAMES_WH[::-1] or depth.dtype.kind != "f":
        raise ValueError("Metric depth must use the same floating-point grid as the box")
    x0, y0, x1, y1 = _rectified_box(box)
    if mask is None:
        return None
    if not isinstance(mask, np.ndarray) or mask.shape != depth.shape or mask.dtype != np.bool_:
        raise ValueError("SAM 2 foreground must be a boolean mask on the same pixel grid")
    width, height = x1 - x0, y1 - y0
    if width < 4. or height < 8.:
        return None
    region = (slice(floor(y0 + height * .22), ceil(y0 + height * .70)),
              slice(floor(x0 + width * .30), ceil(x0 + width * .70)))
    inside = mask[region]
    values = depth[region][inside]
    if values.size < max(6, MIN_INTERIOR_MASK_FRACTION * inside.size):
        return None
    valid = values[np.isfinite(values) & (values >= .5) & (values <= DEPTH_POLICY["max_range_m"])]
    if len(valid) < max(6, DEPTH_POLICY["min_valid_fraction"] * values.size):
        return None
    low, middle, high = np.quantile(valid, (.20, DEPTH_POLICY["quantile"], .80))
    spread = float(high - low)
    return None if spread > DEPTH_POLICY["max_spread_m"] else DepthSample(float(middle), spread)


def _rectified_box(box):
    coordinates = tuple_snapshot(box, "SAM 2 prompt box")
    if len(coordinates) != 4:
        raise ValueError("Expected a four-coordinate SAM 2 box")
    x0, y0, x1, y1 = (finite_real(part, "SAM 2 box coordinate") for part in coordinates)
    if not (0. <= x0 < x1 <= FRAMES_WH[0] and 0. <= y0 < y1 <= FRAMES_WH[1]):
        raise ValueError("SAM 2 box falls outside the rectified camera raster")
    return x0, y0, x1, y1


def _verified_model(local_model_root, manifest_path):
    payload = read_bounded(manifest_path, 16384)
    if sha256(payload).hexdigest() != SAM_MANIFEST_SHA256:
        raise ValueError("Expected the pinned SAM 2 model manifest")
    manifest = decode_json(payload)
    root = Path(local_model_root)
    check_components(root)
    if {item.name for item in root.iterdir()} != set(manifest["files"]):
        raise ValueError("Expected only the four pinned SAM 2 files")
    for name, metadata in manifest["files"].items():
        size = metadata["bytes"]
        data = read_bounded(root / name, size)
        if len(data) != size:
            raise ValueError(f"Unexpected size for SAM 2/{name}")
        if "sha256" in metadata:
            expected, actual = metadata["sha256"], sha256(data).hexdigest()
        else:
            digest = sha1(f"blob {size}".encode() + bytes((0,)))
            digest.update(data)
            expected, actual = metadata["git_blob_sha1"], digest.hexdigest()
        if actual != expected:
            raise ValueError(f"Wrong checksum for SAM 2/{name}")
    return root, {"repository": manifest["repo"], "revision": manifest["revision"],
                  "weights_sha256": manifest["files"]["model.safetensors"]["sha256"]}
