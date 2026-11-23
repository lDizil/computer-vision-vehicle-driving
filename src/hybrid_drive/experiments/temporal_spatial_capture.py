"""Capture and reopen a content-bound, labels-free input for repeated video experiments."""

from dataclasses import dataclass
import hashlib
from pathlib import Path

import torch

from hybrid_drive.assets.spatial_manifest import read_spatial_manifest
from hybrid_drive.assets.spatial_temporal import read_temporal_spatial_pack, verify_independent_temporal_final
from hybrid_drive.neural.pretrained_alignment import load_alignment_vision
from .bounded_io import decode_json, read_bounded, require_hex, require_keys
from .frozen_spatial_accuracy import load_frozen_spatial_accuracy
from .io import write_json
from .proposal_alignment import _new_result_directory
from .proposal_prior_source import load_spatial_prior_source
from .protocol import content_hash
from .temporal_spatial_trace import capture_spatial_trace, read_spatial_trace, write_spatial_trace


ROLES = ("development", "holdout", "stress")
PRIOR_ROLES = ("reference", "previous_final_reused_as_dev", "evaluation")
SCHEMA = "hybrid-drive-spatial-video-capture-v1"
MANIFEST_NAME = "capture.json"
_MAX_MANIFEST = 64 * 1024


@dataclass(frozen=True)
class VideoCapture:
    packs: dict
    traces: dict
    manifest: dict
    manifest_sha256: str


def open_video_packs(roots, previous_roots):
    if set(roots) != set(ROLES) or set(previous_roots) != set(PRIOR_ROLES):
        raise ValueError("Expected all four-band model data packs and three temporal roles")
    packs = {role: read_temporal_spatial_pack(roots[role], role=role) for role in ROLES}
    if len({pack.manifest_hash for pack in packs.values()}) != len(ROLES):
        raise ValueError("Temporal pack identities must differ")
    verify_independent_temporal_final(
        packs["holdout"], (*previous_roots.values(), roots["development"], roots["stress"]),
    )
    hashes = {
        role: read_spatial_manifest(previous_roots[role])["manifest_hash"]
        for role in PRIOR_ROLES
    }
    if len(set(hashes.values())) != len(PRIOR_ROLES):
        raise ValueError("The prior model data packs must be distinct")
    return packs, hashes


def capture_video(
    roots, previous_roots, model_root, prior_root, accuracy_root, output_root,
    *, revision, prior_result_sha256, accuracy_result_sha256, batch_size=8,
):
    require_hex(revision, 40, "capture revision")
    packs, previous = open_video_packs(roots, previous_roots)
    root = _new_result_directory(
        output_root, (*roots.values(), *previous_roots.values(), model_root, prior_root, accuracy_root),
    )
    with torch.device("cpu"):
        vision, receipt = load_alignment_vision(model_root)
        prior = load_spatial_prior_source(
            prior_root, expected_result_sha256=prior_result_sha256, dataset_hash=previous["reference"],
            vision_provenance=receipt,
        )
        model = load_frozen_spatial_accuracy(
            accuracy_root, expected_result_sha256=accuracy_result_sha256,
            reference_dataset_hash=previous["reference"], vision_receipt=receipt, visual_dim=vision.visual_dim,
            prior=prior,
        )
        model_receipt = model.receipt
        if model_receipt["accuracy_data_hashes"] != previous:
            raise ValueError("Supplied prior packs do not match the accuracy artifact")
        model_hash = content_hash(model_receipt)
        captures = {}
        for role, pack in packs.items():
            print(f"Capturing {role}: {len(pack.frames)} frames", flush=True)
            trace = capture_spatial_trace(pack, vision, model, batch_size=batch_size)
            path = root / _trace_file(role)
            sha256 = write_spatial_trace(path, trace)
            reread = read_spatial_trace(path, pack, expected_sha256=sha256, expected_model_hash=model_hash)
            if reread != trace:
                raise ValueError("Saved video observations did not round-trip")
            captures[role] = {
                "pack_hash": pack.manifest_hash, "trace_sha256": sha256,
                "frames": len(reread.frames), "inference_wall_s": reread.inference_wall_s,
            }

    manifest = {
        "schema": SCHEMA, "revision": revision, "status": "captured",
        "model_receipt": model_receipt, "model_hash": model_hash, "captures": captures,
        "batch_size": batch_size, "labels_opened": False,
    }
    path = root / MANIFEST_NAME
    write_json(path, manifest)
    return root, hashlib.sha256(read_bounded(path, _MAX_MANIFEST)).hexdigest()


def reopen_video_capture(root, roots, previous_roots, *, expected_capture_sha256):
    require_hex(expected_capture_sha256, 64, "capture manifest byte hash")
    packs, previous = open_video_packs(roots, previous_roots)
    root = Path(root).absolute()
    raw = read_bounded(root / MANIFEST_NAME, _MAX_MANIFEST)
    if hashlib.sha256(raw).hexdigest() != expected_capture_sha256:
        raise ValueError("Video capture manifest byte hash differs")
    manifest = decode_json(raw)
    require_keys(manifest, {
        "schema", "revision", "status", "model_receipt", "model_hash", "captures", "batch_size", "labels_opened",
    }, "video capture manifest")
    require_hex(manifest["revision"], 40, "capture revision")
    require_hex(manifest["model_hash"], 64, "frozen model hash")
    model = manifest["model_receipt"]
    if (
        manifest["schema"] != SCHEMA or manifest["status"] != "captured" or manifest["labels_opened"] is not False
        or not isinstance(model, dict) or content_hash(model) != manifest["model_hash"]
        or model.get("accuracy_data_hashes") != previous
        or type(manifest["batch_size"]) is not int or not 1 <= manifest["batch_size"] <= 16
    ):
        raise ValueError("Capture manifest is incompatible with the bound model and prior datasets")
    require_keys(manifest["captures"], set(ROLES), "capture roles")
    traces = {}
    for role, pack in packs.items():
        claimed = manifest["captures"][role]
        require_keys(claimed, {"pack_hash", "trace_sha256", "frames", "inference_wall_s"}, "capture receipts")
        if claimed["pack_hash"] != pack.manifest_hash or claimed["frames"] != len(pack.frames):
            raise ValueError("Capture pack or frame count differs")
        trace = read_spatial_trace(
            root / _trace_file(role), pack,
            expected_sha256=claimed["trace_sha256"], expected_model_hash=manifest["model_hash"],
        )
        if trace.inference_wall_s != claimed["inference_wall_s"]:
            raise ValueError("Capture durations differ from the bound trace")
        traces[role] = trace
    return VideoCapture(packs, traces, manifest, expected_capture_sha256)


def _trace_file(role):
    return f"observations-{role}.json"
