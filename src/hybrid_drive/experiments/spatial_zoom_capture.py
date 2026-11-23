"""Freeze close-up and global feature grids on existing camera decisions."""

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path

from safetensors.torch import load, save
import torch

from hybrid_drive.assets.files import AtomicFile
from hybrid_drive.assets.spatial_manifest import read_spatial_rgb, verify_disjoint_spatial_manifests
from hybrid_drive.neural.pretrained_alignment import load_alignment_vision
from hybrid_drive.neural.spatial_zoom_crops import encode_zoom_pair, make_zoom_windows, zoom_protocol
from .bounded_io import decode_json, read_bounded, require_hex, require_keys
from .io import write_json
from .neural_checkpoint import _header, module_state_hash
from .proposal_alignment import _new_result_directory
from .protocol import content_hash
from .spatial_appearance_capture import reopen_appearance_capture
from .spatial_decision_policy import PRESENT


TRAIN_ROLES = ("development", "stress")
DEV_ROLES = ("holdout",)
REFERENCE_ROLES = ("tracking_holdout", "appearance_holdout")
ZOOM_ROLES = (*TRAIN_ROLES, *DEV_ROLES, *REFERENCE_ROLES)
_SCHEMA = "hybrid-drive-spatial-zoom-capture-v1"
_FILE_SCHEMA = "hybrid-drive-zoom-feature-tensors-v1"
_HEADER = "zoom-capture.json"
_MAX_META = 64 * 1024
_MAX_FEATURE_BYTES = 64 * 1024**2


@dataclass(frozen=True)
class ZoomInputs:
    packs: dict
    decisions: dict
    source_model_hash: str
    vision_hash: str
    source_capture_sha256: str
    protected_roots: tuple


@dataclass(frozen=True)
class ZoomGrids:
    windows: torch.Tensor
    global_features: torch.Tensor
    zoom_features: torch.Tensor


@dataclass(frozen=True)
class ZoomCapture:
    inputs: ZoomInputs
    grids: dict
    manifest: dict
    sha256: str


def open_zoom_inputs(appearance_scope, capture_path, *, expected_capture_sha256):
    previous = reopen_appearance_capture(capture_path, appearance_scope, expected_capture_sha256=expected_capture_sha256)
    packs = previous.inputs.packs
    if set(packs) != set(ZOOM_ROLES) or set(previous.decisions) != set(ZOOM_ROLES):
        raise ValueError("Expected the five verified existing clip roles")
    for i, role in enumerate(ZOOM_ROLES):
        pack = packs[role]
        if tuple(x.observation.sample_id for x in previous.decisions[role]) != tuple(f.sample_id for f in pack.frames):
            raise ValueError("Zoom inputs must use the exact previous prediction order")
        for other in ZOOM_ROLES[:i]:
            verify_disjoint_spatial_manifests(
                pack.manifest, packs[other].manifest, role=f"other zoom split ({other})", compare_selection=False,
            )
    return ZoomInputs(
        packs, previous.decisions, previous.inputs.model_hash, previous.inputs.model_receipt["vision_hash"],
        previous.sha256, (*previous.inputs.protected_roots, capture_path),
    )


def capture_zoom_grids(inputs, model_root, output_root, *, revision, frame_batch_size=4):
    require_hex(revision, 40, "crop capture revision")
    if type(frame_batch_size) is not int or not 1 <= frame_batch_size <= 4:
        raise ValueError("Crop capture takes 1–4 frames per image batch")
    root = _new_result_directory(output_root, (*inputs.protected_roots, model_root))
    vision, receipt = load_alignment_vision(model_root)
    if content_hash(receipt) != inputs.vision_hash:
        raise ValueError("Crop features must use the exact previous frozen vision encoder")
    before = _vision_state(vision)
    dim = vision.visual_dim
    _check_dim(dim)
    roles = {}
    for role in ZOOM_ROLES:
        print(f"Encoding same windows on full frames and crops: {role}", flush=True)
        fixed = inputs.decisions[role]
        pack = inputs.packs[role]
        original, state, grids = capture_zoom_role(vision, pack, fixed, frame_batch_size=frame_batch_size)
        path = root / _features_name(role)
        digest = _write_compact(path, grids, original, state, dim, _binding(inputs, role))
        roles[role] = {
            "pack_hash": pack.manifest_hash, "sha256": digest, "frames": len(fixed), "visible_boxes": int(state.sum()),
        }
    if _vision_state(vision) != before:
        raise ValueError("Frozen DINO state changed during crop extraction")
    manifest = {
        "schema": _SCHEMA, "status": "captured", "revision": revision,
        "source_model_hash": inputs.source_model_hash, "vision_hash": inputs.vision_hash,
        "source_capture_sha256": inputs.source_capture_sha256, "feature_dim": dim,
        "protocol": zoom_protocol(), "roles": roles, "labels_opened": False,
    }
    path = root / _HEADER
    write_json(path, manifest)
    digest = hashlib.sha256(read_bounded(path, _MAX_META)).hexdigest()
    reopen_zoom_capture(root, inputs, expected_sha256=digest)
    return root, digest


def capture_zoom_role(vision, pack, decisions, *, frame_batch_size=4):
    """Encode the exact ordered visible boxes of an already verified clip pack."""
    if type(frame_batch_size) is not int or not 1 <= frame_batch_size <= 4:
        raise ValueError("Crop capture takes 1–4 frames per image batch")
    if (
        not decisions
        or tuple(d.observation.sample_id for d in decisions) != tuple(f.sample_id for f in pack.frames)
    ):
        raise ValueError("Crop frames differ from the fixed prediction order")
    state = torch.tensor([tuple(s == PRESENT for s in d.states) for d in decisions], dtype=torch.bool)
    original = torch.tensor([d.boxes for d in decisions], dtype=torch.float32)
    batches = []
    for offset in range(0, len(decisions), frame_batch_size):
        sl = slice(offset, offset + frame_batch_size)
        rgb = torch.stack([
            torch.from_numpy(read_spatial_rgb(pack.root, frame.rgb)).permute(2, 0, 1).float().div_(255)
            for frame in pack.frames[sl]
        ]).unsqueeze(1)
        batches.append(encode_zoom_pair(vision, rgb, original[sl], state[sl]))
    grids = ZoomGrids(*(torch.cat(items, dim=0) for items in zip(*batches)))
    return original, state, grids


def reopen_zoom_capture(root, inputs, *, expected_sha256):
    require_hex(expected_sha256, 64, "zoom capture byte hash")
    root = Path(root).absolute()
    raw = read_bounded(root / _HEADER, _MAX_META)
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("Zoom capture byte hash differs")
    manifest = decode_json(raw)
    require_keys(manifest, {
        "schema", "status", "revision", "source_model_hash", "vision_hash", "source_capture_sha256",
        "feature_dim", "protocol", "roles", "labels_opened",
    }, "zoom capture manifest")
    require_hex(manifest["revision"], 40, "zoom capture revision")
    if (
        manifest["schema"] != _SCHEMA or manifest["status"] != "captured" or manifest["labels_opened"] is not False
        or manifest["source_model_hash"] != inputs.source_model_hash or manifest["vision_hash"] != inputs.vision_hash
        or manifest["source_capture_sha256"] != inputs.source_capture_sha256 or manifest["protocol"] != zoom_protocol()
    ):
        raise ValueError("Zoom capture disagrees with the original frozen decisions or recipe")
    dim = manifest["feature_dim"]
    _check_dim(dim)
    require_keys(manifest["roles"], set(ZOOM_ROLES), "zoom capture roles")
    grids = {}
    for role in ZOOM_ROLES:
        fixed = inputs.decisions[role]
        active = torch.tensor([tuple(s == PRESENT for s in d.states) for d in fixed], dtype=torch.bool)
        boxes = torch.tensor([d.boxes for d in fixed], dtype=torch.float32)
        row = manifest["roles"][role]
        require_keys(row, {"pack_hash", "sha256", "frames", "visible_boxes"}, "zoom capture role")
        if (
            row["pack_hash"] != inputs.packs[role].manifest_hash
            or row["frames"] != len(fixed) or row["visible_boxes"] != int(active.sum())
        ):
            raise ValueError("Zoom role receipt has different frames or visibility")
        grids[role] = _read_compact(root / _features_name(role), boxes, active, dim,
                                   _binding(inputs, role), row["sha256"])
    return ZoomCapture(inputs, grids, manifest, expected_sha256)


def _check_dim(dim):
    if type(dim) is not int or not 1 <= dim <= 512:
        raise ValueError("Crop feature dimensions must be between 1 and 512")


def _vision_state(vision):
    if vision.training or any(p.requires_grad or p.grad is not None for p in vision.parameters()):
        raise ValueError("Crop encoding requires frozen eval-mode vision")
    return module_state_hash(vision)


def _binding(inputs, role):
    return content_hash({
        "role": role, "capture": inputs.source_capture_sha256, "pack": inputs.packs[role].manifest_hash,
        "vision": inputs.vision_hash, "protocol": zoom_protocol(), "decisions": [asdict(d) for d in inputs.decisions[role]],
    })


def _features_name(role):
    return f"zoom-features-{role}.safetensors"


def _write_compact(path, data, boxes, active, dim, binding):
    _verify_dense(data, boxes, active, dim)
    encoded = save({k: getattr(data, k)[active].contiguous() for k in (
        "windows", "global_features", "zoom_features",
    )}, metadata={"schema": _FILE_SCHEMA, "binding": binding})
    if len(encoded) > _MAX_FEATURE_BYTES:
        raise ValueError("Zoom tensors exceed the role byte budget")
    _header(encoded)
    with AtomicFile(path) as output:
        output.out.write(encoded)
        output.commit()
    return hashlib.sha256(encoded).hexdigest()


def _read_compact(path, boxes, active, dim, binding, expected_sha256):
    require_hex(expected_sha256, 64, "zoom feature byte hash")
    raw = read_bounded(path, _MAX_FEATURE_BYTES)
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("Zoom feature byte hash differs")
    if _header(raw).get("__metadata__") != {"schema": _FILE_SCHEMA, "binding": binding}:
        raise ValueError("Zoom tensor provenance differs")
    arrays = load(raw)
    if not isinstance(arrays, dict) or arrays.keys() != {"windows", "global_features", "zoom_features"}:
        raise ValueError("Zoom capture tensor keys differ")
    count = int(active.sum())
    dense = []
    for field, shape in (("windows", (4,)), ("global_features", (dim, 4, 4)), ("zoom_features", (dim, 4, 4))):
        packed = arrays[field]
        if packed.dtype != torch.float32 or tuple(packed.shape) != (count, *shape) or not torch.isfinite(packed).all():
            raise ValueError("Invalid compact zoom tensor")
        values = torch.zeros((*active.shape, *shape), dtype=torch.float32)
        values[active] = packed
        dense.append(values)
    data = ZoomGrids(*dense)
    _verify_dense(data, boxes, active, dim)
    return data


def _verify_dense(data, boxes, active, dim):
    if (
        data.windows.device.type != "cpu" or data.windows.dtype != torch.float32 or data.windows.requires_grad
        or data.windows.shape != boxes.shape or len(boxes) == 0
    ):
        raise ValueError("Expected one frozen crop window per original visible box slot")
    for start in range(0, len(boxes), 16):
        indices = slice(start, start + 16)
        if not torch.equal(data.windows[indices], make_zoom_windows(boxes[indices], active[indices])):
            raise ValueError("Saved crop windows do not match the original visible boxes")
    for values in (data.global_features, data.zoom_features):
        if (
            values.dtype != torch.float32 or values.device.type != "cpu"
            or tuple(values.shape) != (*active.shape, dim, 4, 4) or values.requires_grad
            or not torch.isfinite(values).all() or torch.count_nonzero(values[~active])
        ):
            raise ValueError("Zoom grids must contain only finite frozen features for visible boxes")
