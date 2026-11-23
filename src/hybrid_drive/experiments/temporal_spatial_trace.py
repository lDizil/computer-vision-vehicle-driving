"""Capture model outputs without opening any temporally aligned labels."""

from dataclasses import asdict, dataclass, fields
import hashlib
from pathlib import Path
from time import perf_counter

import torch

from hybrid_drive.assets.spatial_manifest import read_spatial_rgb
from .bounded_io import decode_json, read_bounded, require_hex, require_keys
from .io import write_json
from .neural_checkpoint import module_state_hash
from .protocol import content_hash
from .temporal_spatial_types import FrameObservation


_TRACE_SCHEMA = "hybrid-drive-temporal-spatial-observations-v1"
_MAX_TRACE_BYTES = 4 * 1024 * 1024
_FRAME_FIELDS = {item.name for item in fields(FrameObservation)}


@dataclass(frozen=True)
class SpatialTrace:
    role: str
    pack_hash: str
    model_hash: str
    frames: tuple[FrameObservation, ...]
    inference_wall_s: float


@torch.no_grad()
def capture_spatial_trace(pack, vision, model, *, batch_size=8):
    if type(batch_size) is not int or not 1 <= batch_size <= 16:
        raise ValueError("Trace batch size must be 1–16")
    if not pack.frames:
        raise ValueError("Cannot capture an empty clip pack")
    before = _vision_state(vision)
    observations, t0 = [], perf_counter()
    for start in range(0, len(pack.frames), batch_size):
        refs = pack.frames[start:start + batch_size]
        images = [
            torch.from_numpy(read_spatial_rgb(pack.root, row.rgb)).permute(2, 0, 1).float().div_(255)
            for row in refs
        ]
        patches = vision(torch.stack(images).unsqueeze(1))
        scores, boxes = model.predict(patches)
        if (
            scores.shape != (len(refs), 4) or boxes.shape != (len(refs), 4, 4)
            or scores.requires_grad or boxes.requires_grad
        ):
            raise ValueError("A frozen model must return one score and box per frame/quarter")
        for ref, score, frame_boxes in zip(refs, scores.tolist(), boxes.tolist()):
            observations.append(FrameObservation(*_identity(ref), score, frame_boxes))
    if _vision_state(vision) != before:
        raise ValueError("Frozen vision changed during trace capture")
    model.verify_unchanged()
    return SpatialTrace(
        _role(pack), pack.manifest_hash, content_hash(model.receipt), tuple(observations), perf_counter() - t0,
    )


def _vision_state(vision):
    if vision.training or any(p.requires_grad or p.grad is not None for p in vision.parameters()):
        raise ValueError("Trace capture needs frozen eval-mode vision")
    return module_state_hash(vision)


def write_spatial_trace(path, trace):
    """Write a local capture and return its exact file SHA-256 receipt."""
    write_json(path, {"schema": _TRACE_SCHEMA, **asdict(trace)})
    return hashlib.sha256(read_bounded(Path(path), _MAX_TRACE_BYTES)).hexdigest()


def read_spatial_trace(path, pack, *, expected_sha256, expected_model_hash):
    require_hex(expected_sha256, 64, "trace byte hash")
    require_hex(expected_model_hash, 64, "trace model hash")
    raw = read_bounded(Path(path), _MAX_TRACE_BYTES)
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("Observation trace byte hash differs")
    trace = decode_json(raw)
    require_keys(
        trace, {"schema", "role", "pack_hash", "model_hash", "frames", "inference_wall_s"}, "observation trace",
    )
    seconds = trace["inference_wall_s"]
    if (
        trace["schema"] != _TRACE_SCHEMA or trace["role"] != _role(pack)
        or trace["pack_hash"] != pack.manifest_hash or trace["model_hash"] != expected_model_hash
        or not isinstance(trace["frames"], list) or len(trace["frames"]) != len(pack.frames)
        or type(seconds) not in (int, float) or not 0 <= seconds < float("inf")
    ):
        raise ValueError("Observation trace does not match the bound role, pack or model")
    observations = []
    for row, ref in zip(trace["frames"], pack.frames):
        require_keys(row, _FRAME_FIELDS, "observation frame")
        frame = FrameObservation(**row)
        if _identity(frame) != _identity(ref):
            raise ValueError("Observation trace frame identity or order changed")
        observations.append(frame)
    return SpatialTrace(trace["role"], trace["pack_hash"], trace["model_hash"], tuple(observations), seconds)


def _role(pack):
    return pack.manifest["sources"]["temporal_role"]


def _identity(value):
    return value.recording_id, value.clip_id, value.sample_id, value.source_index, value.timestamp_us
