"""Label-free appearance descriptors bound to unchanged camera decisions."""

import base64
import binascii
from dataclasses import asdict, dataclass
from hashlib import sha256
from math import ceil
from numbers import Real
from pathlib import Path
from typing import TYPE_CHECKING
from types import MappingProxyType

import numpy as np

from hybrid_drive.assets.files import AtomicFile
from hybrid_drive.assets.spatial_manifest import BAND_COUNT
from .bounded_io import canonical_json, decode_json, read_bounded, require_hex, require_keys
from .spatial_decision_policy import PRESENT
from .temporal_spatial_types import FrameDecision

if TYPE_CHECKING:
    from torch import Tensor


APPEARANCE_KINDS = ("rgb", "dino", "contrast")
_RGB_BINS = 8
_MIN_CROP_PIXELS_PER_AXIS = 3
_CONTEXT_BOX_FRACTION = 0.25
_CONTEXT_PATCH_FRACTION = 0.5
_MIN_CONTEXT_AREA = 1e-12
_MIN_VECTOR_NORM = 1e-12
_CONTRAST_RELATIVE_FLOOR = 1e-6
_RGB_DIM = 3 * _RGB_BINS

_MAX_BATCH = 16
_MAX_IMAGE_SIDE = 512
_MAX_PATCH_SIDE = 64
_MAX_FEATURE_CHANNELS = 1024
_MAX_FRAMES = 1024
_MAX_TRACE_BYTES = 20 * 1024 * 1024
_MAX_FP32 = float(np.finfo(np.float32).max)
_FLOAT32_LE = np.dtype("<f4")
_TRACE_SCHEMA = "hybrid-drive-spatial-appearance-v1"

DESCRIPTOR_PROTOCOL = MappingProxyType({
    "version": 1,
    "slots": "original PRESENT boxes only",
    "rgb": "concatenated per-channel marginal histogram, half-open pixel-centre crop",
    "rgb_bins_per_channel": _RGB_BINS,
    "min_rgb_pixels_per_axis": _MIN_CROP_PIXELS_PER_AXIS,
    "dino": "patch vectors averaged by xyxy fractional cell overlap",
    "contrast": "ROI mean minus the weighted mean of an expanded outside ring",
    "context_padding_box_fraction_per_side": _CONTEXT_BOX_FRACTION,
    "context_min_padding_patch_fraction_per_side": _CONTEXT_PATCH_FRACTION,
    "context_clip": "normalized image bounds",
    "min_context_ring_area": _MIN_CONTEXT_AREA,
    "normalization": "L2 with float64 accumulation, cast to float32",
    "min_vector_norm": _MIN_VECTOR_NORM,
    "contrast_relative_floor": _CONTRAST_RELATIVE_FLOOR,
    "missing": "null, never an invented zero vector",
    "encoding": "canonical base64 of little-endian float32 vector",
})

Embedding = tuple[float, ...]
AppearanceSlots = tuple[Embedding | None, Embedding | None, Embedding | None, Embedding | None]


def descriptor_protocol() -> dict:
    """Return an independent JSON-friendly copy for capture protocols."""
    return dict(DESCRIPTOR_PROTOCOL)


@dataclass(frozen=True, slots=True)
class FrameAppearance:
    sample_id: str
    rgb: AppearanceSlots
    dino: AppearanceSlots
    contrast: AppearanceSlots

    def __post_init__(self):
        if (
            not isinstance(self.sample_id, str) or not 0 < len(self.sample_id) <= 256
            or not self.sample_id.strip() or any(ch in self.sample_id for ch in ("\0", "\n", "\r"))
        ):
            raise ValueError("Appearance sample ID must be bounded nonempty text")
        for kind in APPEARANCE_KINDS:
            rows = _four(getattr(self, kind), kind + " slots")
            checked = tuple(
                _checked_vector(row, dimensions=_RGB_DIM if kind == "rgb" else None)
                if row is not None else None for row in rows
            )
            object.__setattr__(self, kind, checked)
        _trace_dimensions((self,))


def extract_spatial_appearance(decisions, rgb: "Tensor", patch_maps: "Tensor") -> tuple[FrameAppearance, ...]:
    """Extract deterministic appearance from frozen CPU tensors, without targets.

    RGB arrives already resized and normalized to [0, 1]. DINO grid cells
    cover equal rectangles in that same normalized image. Neither input is
    modified, and missing evidence never creates an embedding.
    """
    frames = _checked_decisions(decisions, maximum=_MAX_BATCH)
    pixels = _tensor_array(rgb, "RGB", len(frames), max_channels=3, max_side=_MAX_IMAGE_SIDE)
    patches = _tensor_array(
        patch_maps, "DINO grids", len(frames), max_channels=_MAX_FEATURE_CHANNELS, max_side=_MAX_PATCH_SIDE,
    )
    if pixels.shape[1] != 3 or pixels.min() < 0 or pixels.max() > 1:
        raise ValueError("Expected exactly three RGB channels in [0, 1]")
    return tuple(_extract_frame(frame, image, patch) for frame, image, patch in zip(frames, pixels, patches))


def _tensor_array(value, name, batch, *, max_channels, max_side):
    import torch

    if (
        not isinstance(value, torch.Tensor) or value.ndim != 5 or value.shape[:2] != (batch, 1)
        or not 1 <= value.shape[2] <= max_channels or not all(1 <= side <= max_side for side in value.shape[-2:])
        or value.dtype != torch.float32 or value.device.type != "cpu" or value.layout != torch.strided
        or value.requires_grad
    ):
        raise ValueError(f"Expected bounded frozen CPU FP32 {name} with shape [B,1,C,H,W]")
    array = value[:, 0].detach().numpy()
    if not np.isfinite(array).all():
        raise ValueError(f"Nonfinite {name} in appearance capture")
    return array


def _extract_frame(decision, rgb, patches):
    colors, local, contrast = [], [], []
    for state, box in zip(decision.states, decision.boxes):
        histogram = inside = difference = None
        if state == PRESENT:
            crop = _pixel_crop(rgb, box)
            if crop is not None:
                histogram = _color_histogram(crop)
                inside, difference = _patch_appearance(patches, box)
        colors.append(histogram)
        local.append(inside)
        contrast.append(difference)
    return FrameAppearance(decision.observation.sample_id, tuple(colors), tuple(local), tuple(contrast))


def _pixel_crop(image, box):
    _, height, width = image.shape
    x1, y1, x2, y2 = box
    # A pixel with normalized centre (i + .5) / size contributes only
    # inside the half-open rectangle. ROI geometry is never rounded outward.
    xlo, xhi = (min(width, max(0, ceil(x * width - 0.5))) for x in (x1, x2))
    ylo, yhi = (min(height, max(0, ceil(y * height - 0.5))) for y in (y1, y2))
    if min(xhi - xlo, yhi - ylo) < _MIN_CROP_PIXELS_PER_AXIS:
        return None
    return image[:, ylo:yhi, xlo:xhi]


def _color_histogram(crop):
    bins = np.minimum((crop * _RGB_BINS).astype(np.intp), _RGB_BINS - 1)
    counts = np.concatenate([np.bincount(channel.ravel(), minlength=_RGB_BINS) for channel in bins])
    return _unit(counts)


def _patch_appearance(grid, box):
    _, height, width = grid.shape
    inner = _cell_areas(box, height, width)
    inner_mean = _weighted_mean(grid, inner)
    if inner_mean is None:
        return None, None

    x1, y1, x2, y2 = box
    dx = max((x2 - x1) * _CONTEXT_BOX_FRACTION, _CONTEXT_PATCH_FRACTION / width)
    dy = max((y2 - y1) * _CONTEXT_BOX_FRACTION, _CONTEXT_PATCH_FRACTION / height)
    exterior = (max(0, x1 - dx), max(0, y1 - dy), min(1, x2 + dx), min(1, y2 + dy))
    ring = np.maximum(_cell_areas(exterior, height, width) - inner, 0.0)
    outside = _weighted_mean(grid, ring, minimum_area=_MIN_CONTEXT_AREA)
    contrast = None
    if outside is not None:
        floor = _CONTRAST_RELATIVE_FLOOR * max(1.0, np.linalg.norm(inner_mean), np.linalg.norm(outside))
        contrast = _unit(inner_mean - outside, minimum_norm=floor)
    return _unit(inner_mean), contrast


def _cell_areas(box, height, width):
    x1, y1, x2, y2 = box
    x = np.arange(width + 1, dtype=np.float64) / width
    y = np.arange(height + 1, dtype=np.float64) / height
    col = np.maximum(np.minimum(x[1:], x2) - np.maximum(x[:-1], x1), 0)
    row = np.maximum(np.minimum(y[1:], y2) - np.maximum(y[:-1], y1), 0)
    return row[:, None] * col[None, :]


def _weighted_mean(grid, coverage, *, minimum_area=0.0):
    mass = coverage.sum()
    if mass <= minimum_area:
        return None
    return np.einsum("chw,hw->c", grid, coverage / mass, dtype=np.float64, optimize=False)


def _unit(values, *, minimum_norm=_MIN_VECTOR_NORM):
    values = np.asarray(values, dtype=np.float64)
    norm = float(np.linalg.norm(values))
    if norm <= minimum_norm:
        return None
    return tuple(np.asarray(values / norm, dtype=np.float32).tolist())


def write_appearance_trace(path, decisions, frames, *, model_hash, capture_hash) -> str:
    """Publish a no-overwrite artifact; return the SHA-256 of its exact bytes."""
    require_hex(model_hash, 64, "appearance model hash")
    require_hex(capture_hash, 64, "appearance capture hash")
    decisions = _checked_decisions(decisions)
    frames = _matched_frames(decisions, frames)
    encoded_body = sum(
        4 * ((len(vec) * _FLOAT32_LE.itemsize + 2) // 3)
        for frame in frames for kind in APPEARANCE_KINDS for vec in getattr(frame, kind)
        if vec is not None
    )
    if encoded_body > _MAX_TRACE_BYTES:
        raise ValueError("Appearance vectors exceed trace byte budget")
    data = canonical_json({
        "schema": _TRACE_SCHEMA,
        "protocol": descriptor_protocol(),
        "model_hash": model_hash,
        "capture_hash": capture_hash,
        "decisions_hash": _decision_hash(decisions),
        "dimensions": _trace_dimensions(frames),
        "frames": [
            {
                "sample_id": frame.sample_id,
                **{kind: [_encode(vec) for vec in getattr(frame, kind)] for kind in APPEARANCE_KINDS},
            }
            for frame in frames
        ],
    }) + b"\n"
    if len(data) > _MAX_TRACE_BYTES:
        raise ValueError("Appearance trace exceeds byte budget")
    with AtomicFile(Path(path).absolute()) as target:
        target.out.write(data)
        target.commit()
    return sha256(data).hexdigest()


def read_appearance_trace(path, decisions, *, model_hash, capture_hash, expected_sha256) -> tuple[FrameAppearance, ...]:
    """Require both the exact byte receipt and full original-decision alignment."""
    require_hex(model_hash, 64, "appearance model hash")
    require_hex(capture_hash, 64, "appearance capture hash")
    require_hex(expected_sha256, 64, "appearance byte hash")
    decisions = _checked_decisions(decisions)
    raw = read_bounded(Path(path).absolute(), _MAX_TRACE_BYTES)
    if sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("Appearance trace byte hash differs")
    payload = decode_json(raw)
    require_keys(payload, {"schema", "protocol", "model_hash", "capture_hash", "decisions_hash",
                           "dimensions", "frames"}, "appearance trace")
    if (
        payload["schema"] != _TRACE_SCHEMA
        or canonical_json(payload["protocol"]) != canonical_json(descriptor_protocol())
        or payload["model_hash"] != model_hash or payload["capture_hash"] != capture_hash
        or payload["decisions_hash"] != _decision_hash(decisions)
        or not isinstance(payload["frames"], list) or len(payload["frames"]) != len(decisions)
    ):
        raise ValueError("Appearance trace does not match protocol or bound inputs")

    dims = payload["dimensions"]
    require_keys(dims, set(APPEARANCE_KINDS), "appearance dimensions")
    if any(dim is not None and (type(dim) is not int or not 1 <= dim <= _MAX_FEATURE_CHANNELS)
           for dim in dims.values()) or dims["rgb"] not in (None, _RGB_DIM):
        raise ValueError("Invalid appearance descriptor dimensions")

    restored = []
    for row, decision in zip(payload["frames"], decisions):
        require_keys(row, {"sample_id", *APPEARANCE_KINDS}, "appearance frame")
        if row["sample_id"] != decision.observation.sample_id:
            raise ValueError("Appearance frame identity or order changed")
        restored.append(FrameAppearance(row["sample_id"], **{
            kind: tuple(_decode(value, dims[kind]) for value in _four(row[kind], kind + " slots"))
            for kind in APPEARANCE_KINDS
        }))
    restored = _matched_frames(decisions, restored)
    if _trace_dimensions(restored) != dims:
        raise ValueError("Appearance dimensions do not describe the observed vectors")
    return restored


def _checked_decisions(values, *, maximum=_MAX_FRAMES):
    if not isinstance(values, (tuple, list)) or not 1 <= len(values) <= maximum:
        raise ValueError("Expected a bounded nonempty sequence of appearance decisions")
    if not all(isinstance(value, FrameDecision) for value in values):
        raise ValueError("Appearance capture requires original FrameDecision values")
    if len({value.observation.sample_id for value in values}) != len(values):
        raise ValueError("Appearance decisions contain duplicate sample identities")
    return tuple(values)


def _matched_frames(decisions, frames):
    if not isinstance(frames, (tuple, list)) or len(frames) != len(decisions):
        raise ValueError("Appearance count differs from the original decisions")
    for d, f in zip(decisions, frames):
        if not isinstance(f, FrameAppearance) or f.sample_id != d.observation.sample_id:
            raise ValueError("Appearance frame identity or order changed")
        if any(
            vector is not None and state != PRESENT
            for kind in APPEARANCE_KINDS for state, vector in zip(d.states, getattr(f, kind))
        ):
            raise ValueError("Cannot attach appearance to a non-present decision")
    return tuple(frames)


def _decision_hash(decisions):
    return sha256(canonical_json([asdict(decision) for decision in decisions])).hexdigest()


def _four(values, label):
    if not isinstance(values, (list, tuple)) or len(values) != BAND_COUNT:
        raise ValueError(f"Expected four {label}")
    return values


def _checked_vector(values, *, dimensions):
    if (
        not isinstance(values, (tuple, list)) or not 1 <= len(values) <= _MAX_FEATURE_CHANNELS
        or (dimensions is not None and len(values) != dimensions)
        or any(not isinstance(v, Real) or isinstance(v, bool) for v in values)
    ):
        raise ValueError("Invalid appearance vector type or dimension")
    try:
        values = np.asarray(values, dtype=np.float64)
    except (ValueError, TypeError, OverflowError):
        raise ValueError("Invalid float32 appearance vector") from None
    if not np.isfinite(values).all() or np.max(np.abs(values)) > _MAX_FP32:
        raise ValueError("Appearance vectors must be finite float32")
    values = np.asarray(values, dtype=np.float32)
    if not np.any(values):
        raise ValueError("Missing appearance must be represented as null, not a zero vector")
    return tuple(values.tolist())


def _trace_dimensions(frames):
    answer = {}
    for kind in APPEARANCE_KINDS:
        sizes = {len(vector) for frame in frames for vector in getattr(frame, kind) if vector is not None}
        if len(sizes) > 1:
            raise ValueError(f"Inconsistent {kind} appearance dimensions")
        answer[kind] = next(iter(sizes)) if sizes else None
    if None not in (answer["dino"], answer["contrast"]) and answer["dino"] != answer["contrast"]:
        raise ValueError("DINO and contrast must use the same patch channels")
    return answer


def _encode(vector):
    if vector is None:
        return None
    return base64.b64encode(np.asarray(vector, dtype=_FLOAT32_LE).tobytes()).decode("ascii")


def _decode(value, dimension):
    if value is None:
        return None
    raw_bytes = 0 if dimension is None else dimension * _FLOAT32_LE.itemsize
    if raw_bytes == 0 or not isinstance(value, str) or len(value) != 4 * ((raw_bytes + 2) // 3):
        raise ValueError("Appearance vector exceeds or differs from its byte shape")
    try:
        binary = base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error):
        raise ValueError("Invalid appearance vector base64") from None
    if len(binary) != raw_bytes or base64.b64encode(binary).decode("ascii") != value:
        raise ValueError("Appearance vector base64 shape or encoding differs")
    return tuple(np.frombuffer(binary, dtype=_FLOAT32_LE).tolist())
