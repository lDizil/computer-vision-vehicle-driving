"""Bind a previously unseen clip pack to the exact frozen camera model."""

import hashlib
from pathlib import Path

import torch

from hybrid_drive.assets.spatial_temporal import read_temporal_spatial_pack, verify_independent_temporal_final
from hybrid_drive.neural.pretrained_alignment import load_alignment_vision
from .bounded_io import decode_json, read_bounded, require_hex, require_keys
from .frozen_spatial_accuracy import load_frozen_spatial_accuracy
from .io import write_json
from .proposal_alignment import _new_result_directory
from .proposal_prior_source import load_spatial_prior_source
from .protocol import content_hash
from .temporal_spatial_capture import reopen_video_capture
from .temporal_spatial_trace import capture_spatial_trace, read_spatial_trace, write_spatial_trace


TRACKING_ROLE = "tracking_holdout"
_CAPTURE_SCHEMA = "hybrid-drive-vehicle-tracking-capture-v1"
_MAX_MANIFEST = 32 * 1024
_MANIFEST = "tracking-capture.json"
_TRACE = "tracking-observations.json"


def open_tracking_packs(video_roots, previous_roots, video_capture_root, tracking_root, *, video_capture_sha256):
    previous = reopen_video_capture(
        video_capture_root, video_roots, previous_roots, expected_capture_sha256=video_capture_sha256,
    )
    final = read_temporal_spatial_pack(tracking_root, role=TRACKING_ROLE)
    verify_independent_temporal_final(
        final, (*video_roots.values(), *previous_roots.values()), expected_role=TRACKING_ROLE,
    )
    return previous, final


def capture_tracking_holdout(
    previous, final, model_root, prior_root, accuracy_root, output_root, *,
    revision, prior_result_sha256, accuracy_result_sha256, protected_roots=(), batch_size=8,
):
    require_hex(revision, 40, "tracking capture revision")
    prior_manifest = previous.manifest
    policy_receipt = prior_manifest["model_receipt"]
    reference = policy_receipt["reference_dataset_hash"]
    root = _new_result_directory(
        output_root, (final.root, model_root, prior_root, accuracy_root, *protected_roots),
    )
    with torch.device("cpu"):
        vision, receipt = load_alignment_vision(model_root)
        prior = load_spatial_prior_source(
            prior_root, expected_result_sha256=prior_result_sha256, dataset_hash=reference,
            vision_provenance=receipt,
        )
        model = load_frozen_spatial_accuracy(
            accuracy_root, expected_result_sha256=accuracy_result_sha256, reference_dataset_hash=reference,
            vision_receipt=receipt, visual_dim=vision.visual_dim, prior=prior,
        )
        if content_hash(model.receipt) != prior_manifest["model_hash"]:
            raise ValueError("Tracking capture must use the same frozen camera model and thresholds")
        trace = capture_spatial_trace(final, vision, model, batch_size=batch_size)

    manifest = {
        "schema": _CAPTURE_SCHEMA, "status": "captured", "revision": revision,
        "prior_video_capture_sha256": previous.manifest_sha256,
        "pack_hash": final.manifest_hash, "model_hash": prior_manifest["model_hash"],
        "trace_sha256": write_spatial_trace(root / _TRACE, trace),
        "frames": len(trace.frames), "inference_wall_s": trace.inference_wall_s, "labels_opened": False,
    }
    path = root / _MANIFEST
    write_json(path, manifest)
    digest = hashlib.sha256(read_bounded(path, _MAX_MANIFEST)).hexdigest()
    reopened, _ = reopen_tracking_capture(root, previous, final, expected_capture_sha256=digest)
    if reopened != trace:
        raise ValueError("Tracking observations did not survive the capture round-trip")
    return root, digest


def reopen_tracking_capture(root, previous, final, *, expected_capture_sha256):
    require_hex(expected_capture_sha256, 64, "tracking capture byte hash")
    root = Path(root).absolute()
    raw = read_bounded(root / _MANIFEST, _MAX_MANIFEST)
    if hashlib.sha256(raw).hexdigest() != expected_capture_sha256:
        raise ValueError("Tracking capture manifest byte hash differs")
    manifest = decode_json(raw)
    require_keys(manifest, {
        "schema", "status", "revision", "prior_video_capture_sha256", "pack_hash", "model_hash",
        "trace_sha256", "frames", "inference_wall_s", "labels_opened",
    }, "tracking capture manifest")
    require_hex(manifest["revision"], 40, "tracking capture revision")
    if (
        manifest["schema"] != _CAPTURE_SCHEMA or manifest["status"] != "captured"
        or manifest["labels_opened"] is not False or manifest["prior_video_capture_sha256"] != previous.manifest_sha256
        or manifest["pack_hash"] != final.manifest_hash or manifest["model_hash"] != previous.manifest["model_hash"]
        or manifest["frames"] != len(final.frames)
    ):
        raise ValueError("Tracking receipt belongs to a different frozen input, model or final pack")
    trace = read_spatial_trace(
        root / _TRACE, final,
        expected_sha256=manifest["trace_sha256"], expected_model_hash=manifest["model_hash"],
    )
    if trace.role != TRACKING_ROLE or trace.inference_wall_s != manifest["inference_wall_s"]:
        raise ValueError("Tracking trace has a different role or duration")
    return trace, manifest
