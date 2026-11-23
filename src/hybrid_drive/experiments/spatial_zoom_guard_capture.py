"""Bind a new label-free capture to the exact camera and cached zoom experiment."""

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path

import torch

from hybrid_drive.assets.spatial_manifest import ZOOM_GATE_HOLDOUT_ROLE, read_spatial_manifest
from hybrid_drive.assets.spatial_temporal import read_temporal_spatial_pack, verify_independent_temporal_final
from hybrid_drive.neural.pretrained_alignment import load_alignment_vision
from hybrid_drive.neural.spatial_zoom_crops import zoom_protocol
from .bounded_io import decode_json, read_bounded, require_hex, require_keys
from .frozen_spatial_accuracy import load_frozen_spatial_accuracy
from .io import write_json
from .proposal_alignment import _new_result_directory
from .proposal_prior_source import load_spatial_prior_source
from .protocol import content_hash
from .spatial_decision_policy import PRESENT, ZoneDecisionPolicy
from .spatial_zoom_capture import _read_compact, _vision_state, _write_compact, capture_zoom_role, reopen_zoom_capture
from .temporal_spatial_filter import RECIPES
from .temporal_spatial_replay import replay_temporal
from .temporal_spatial_trace import capture_spatial_trace, read_spatial_trace, write_spatial_trace


FRESH_ROLE = ZOOM_GATE_HOLDOUT_ROLE
_SCHEMA = "hybrid-drive-zoom-guard-fresh-capture-v1"
_HEADER = "fresh-capture.json"
_TRACE = "fresh-observations.json"
_FEATURES = "fresh-zoom.safetensors"
_MAX_META = 64 * 1024


@dataclass(frozen=True)
class GuardInputs:
    known: object
    fresh_pack: object
    model_receipt: dict
    protected_roots: tuple


@dataclass(frozen=True)
class FreshZoomCapture:
    decisions: tuple
    grids: object
    manifest: dict
    sha256: str


def open_guard_inputs(
    known_inputs, model_receipt, known_zoom_root, fresh_root, previous_roots, *, known_zoom_sha256,
):
    if content_hash(model_receipt) != known_inputs.source_model_hash:
        raise ValueError("Guard input model or camera thresholds differ from the original zoom capture")
    earlier = tuple(previous_roots)
    if len(earlier) != 8 or len({Path(path).absolute() for path in earlier}) != 8:
        raise ValueError("Blind admission requires exactly eight different previously used pack roots")
    bound = {
        *model_receipt["accuracy_data_hashes"].values(),
        *(pack.manifest_hash for pack in known_inputs.packs.values()),
    }
    if len(bound) != 8 or {read_spatial_manifest(r)["manifest_hash"] for r in earlier} != bound:
        raise ValueError("Blind admission requires the eight exact packs from the previous capture lineage")
    fresh = read_temporal_spatial_pack(fresh_root, role=FRESH_ROLE)
    verify_independent_temporal_final(fresh, earlier, expected_role=FRESH_ROLE)
    known = reopen_zoom_capture(known_zoom_root, known_inputs, expected_sha256=known_zoom_sha256)
    return GuardInputs(known, fresh, model_receipt, (*known_inputs.protected_roots, known_zoom_root, fresh_root))


def capture_guard_holdout(
    inputs, model_root, prior_root, accuracy_root, output_root, *,
    revision, prior_result_sha256, accuracy_result_sha256,
):
    require_hex(revision, 40, "fresh capture revision")
    root = _new_result_directory(output_root, (*inputs.protected_roots, model_root, prior_root, accuracy_root))
    with torch.device("cpu"), torch.no_grad():
        vision, receipt = load_alignment_vision(model_root)
        source = inputs.known.inputs
        if content_hash(receipt) != source.vision_hash or vision.visual_dim != inputs.known.manifest["feature_dim"]:
            raise ValueError("Fresh frames require the same archived vision encoder")
        reference = inputs.model_receipt["reference_dataset_hash"]
        prior = load_spatial_prior_source(
            prior_root, expected_result_sha256=prior_result_sha256, dataset_hash=reference, vision_provenance=receipt,
        )
        model = load_frozen_spatial_accuracy(
            accuracy_root, expected_result_sha256=accuracy_result_sha256, reference_dataset_hash=reference,
            vision_receipt=receipt, visual_dim=vision.visual_dim, prior=prior,
        )
        if content_hash(model.receipt) != source.source_model_hash:
            raise ValueError("Fresh frames require the unchanged detector, refiner and presence thresholds")
        before = _vision_state(vision)
        print("Capturing frozen detections on fresh recordings", flush=True)
        trace = capture_spatial_trace(inputs.fresh_pack, vision, model, batch_size=8)
        raw_sha = write_spatial_trace(root / _TRACE, trace)
        decisions = _decisions(inputs, trace)
        print("Capturing matched global and crop grids on the same visible slots", flush=True)
        original, visible, grids = capture_zoom_role(vision, inputs.fresh_pack, decisions)
        features = _write_compact(
            root / _FEATURES, grids, original, visible, vision.visual_dim, _binding(inputs, decisions, raw_sha),
        )
        if _vision_state(vision) != before:
            raise ValueError("Frozen vision changed during fresh capture")
    header = {
        "schema": _SCHEMA, "status": "captured", "revision": revision, "labels_opened": False,
        "known_zoom_sha256": inputs.known.sha256, "pack_hash": inputs.fresh_pack.manifest_hash,
        "source_model_hash": source.source_model_hash, "vision_hash": source.vision_hash,
        "protocol": zoom_protocol(), "observation_sha256": raw_sha, "feature_sha256": features,
        "frames": len(decisions), "visible_boxes": int(visible.sum()),
    }
    path = root / _HEADER
    write_json(path, header)
    sha = hashlib.sha256(read_bounded(path, _MAX_META)).hexdigest()
    reopen_guard_holdout(root, inputs, expected_sha256=sha)
    return root, sha


def reopen_guard_holdout(root, inputs, *, expected_sha256):
    require_hex(expected_sha256, 64, "fresh zoom capture byte hash")
    root = Path(root).absolute()
    raw = read_bounded(root / _HEADER, _MAX_META)
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("Fresh zoom capture byte hash differs")
    meta = decode_json(raw)
    require_keys(meta, {
        "schema", "status", "revision", "labels_opened", "known_zoom_sha256", "pack_hash", "source_model_hash",
        "vision_hash", "protocol", "observation_sha256", "feature_sha256", "frames", "visible_boxes",
    }, "fresh capture manifest")
    require_hex(meta["revision"], 40, "fresh capture revision")
    source = inputs.known.inputs
    if (
        meta["schema"] != _SCHEMA or meta["status"] != "captured" or meta["labels_opened"] is not False
        or meta["known_zoom_sha256"] != inputs.known.sha256 or meta["pack_hash"] != inputs.fresh_pack.manifest_hash
        or meta["source_model_hash"] != source.source_model_hash or meta["vision_hash"] != source.vision_hash
        or meta["protocol"] != zoom_protocol() or meta["frames"] != len(inputs.fresh_pack.frames)
    ):
        raise ValueError("Fresh zoom capture differs from the bound source, model or holdout pack")
    trace = read_spatial_trace(
        root / _TRACE, inputs.fresh_pack, expected_sha256=meta["observation_sha256"],
        expected_model_hash=source.source_model_hash,
    )
    fixed = _decisions(inputs, trace)
    boxes = torch.tensor([f.boxes for f in fixed], dtype=torch.float32)
    active = torch.tensor([[s == PRESENT for s in f.states] for f in fixed], dtype=torch.bool)
    if meta["visible_boxes"] != int(active.sum()):
        raise ValueError("Fresh capture visibility differs from frozen decisions")
    grids = _read_compact(
        root / _FEATURES, boxes, active, inputs.known.manifest["feature_dim"],
        _binding(inputs, fixed, meta["observation_sha256"]), meta["feature_sha256"],
    )
    return FreshZoomCapture(fixed, grids, meta, expected_sha256)


def _decisions(inputs, trace):
    policy = ZoneDecisionPolicy(**inputs.model_receipt["calibrated_policy"])
    return replay_temporal(trace.frames, policy, RECIPES["framewise"]).decisions


def _binding(inputs, decisions, observation_sha):
    return content_hash({
        "role": FRESH_ROLE, "known": inputs.known.sha256, "pack": inputs.fresh_pack.manifest_hash,
        "observation_sha256": observation_sha, "vision": inputs.known.inputs.vision_hash,
        "protocol": zoom_protocol(), "decisions": [asdict(d) for d in decisions],
    })
