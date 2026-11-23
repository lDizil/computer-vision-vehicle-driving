"""Capture local descriptors on fixed predictions, without opening truth."""

from dataclasses import dataclass
import hashlib
from pathlib import Path

import torch

from hybrid_drive.assets.spatial_manifest import APPEARANCE_HOLDOUT_ROLE, read_spatial_rgb
from hybrid_drive.assets.spatial_temporal import read_temporal_spatial_pack, verify_independent_temporal_final
from hybrid_drive.neural.pretrained_alignment import load_alignment_vision
from .bounded_io import decode_json, read_bounded, require_hex, require_keys
from .frozen_spatial_accuracy import load_frozen_spatial_accuracy
from .io import write_json
from .neural_checkpoint import module_state_hash
from .proposal_alignment import _new_result_directory
from .proposal_prior_source import load_spatial_prior_source
from .protocol import content_hash
from .spatial_appearance_features import (
    APPEARANCE_KINDS, descriptor_protocol, extract_spatial_appearance, read_appearance_trace, write_appearance_trace,
)
from .spatial_decision_policy import PRESENT, ZoneDecisionPolicy
from .spatial_tracking_capture import TRACKING_ROLE, open_tracking_packs, reopen_tracking_capture
from .temporal_spatial_capture import ROLES
from .temporal_spatial_filter import RECIPES
from .temporal_spatial_replay import replay_temporal
from .temporal_spatial_trace import capture_spatial_trace, read_spatial_trace, write_spatial_trace


APPEARANCE_ROLE = APPEARANCE_HOLDOUT_ROLE
INPUT_ROLES = (*ROLES, TRACKING_ROLE, APPEARANCE_ROLE)
_SCHEMA = "hybrid-drive-spatial-appearance-capture-v1"
_MANIFEST = "appearance-capture.json"
_NEW_TRACE = "appearance-holdout-observations.json"
_MAX_HEADER = 128 * 1024


@dataclass(frozen=True)
class AppearanceInputs:
    packs: dict
    previous_traces: dict
    previous_capture_hashes: dict
    model_hash: str
    model_receipt: dict
    protected_roots: tuple


@dataclass(frozen=True)
class AppearanceCapture:
    inputs: AppearanceInputs
    decisions: dict
    appearances: dict
    manifest: dict
    sha256: str


def open_appearance_inputs(
    video_roots, previous_roots, video_capture_root, tracking_root, tracking_capture_root, appearance_root,
    *, video_capture_sha256, tracking_capture_sha256,
):
    prior, tracking = open_tracking_packs(
        video_roots, previous_roots, video_capture_root, tracking_root, video_capture_sha256=video_capture_sha256,
    )
    tracking_trace, _ = reopen_tracking_capture(
        tracking_capture_root, prior, tracking, expected_capture_sha256=tracking_capture_sha256,
    )
    new = read_temporal_spatial_pack(appearance_root, role=APPEARANCE_ROLE)
    verify_independent_temporal_final(
        new, (*video_roots.values(), *previous_roots.values(), tracking_root), expected_role=APPEARANCE_ROLE,
    )
    protected = (
        *video_roots.values(), *previous_roots.values(), video_capture_root, tracking_root, tracking_capture_root,
        appearance_root,
    )
    packs = {**prior.packs, TRACKING_ROLE: tracking, APPEARANCE_ROLE: new}
    traces = {**prior.traces, TRACKING_ROLE: tracking_trace}
    hashes = {**{role: prior.manifest_sha256 for role in ROLES}, TRACKING_ROLE: tracking_capture_sha256}
    return AppearanceInputs(
        packs, traces, hashes, prior.manifest["model_hash"], prior.manifest["model_receipt"], protected,
    )


def capture_appearance(
    scope, model_root, prior_root, accuracy_root, output_root, *,
    revision, prior_result_sha256, accuracy_result_sha256, batch_size=8,
):
    require_hex(revision, 40, "appearance capture revision")
    if type(batch_size) is not int or not 1 <= batch_size <= 16:
        raise ValueError("Appearance capture batch size must be 1–16")
    root = _new_result_directory(output_root, (*scope.protected_roots, model_root, prior_root, accuracy_root))
    with torch.device("cpu"), torch.no_grad():
        vision, provenance = load_alignment_vision(model_root)
        reference = scope.model_receipt["reference_dataset_hash"]
        prior = load_spatial_prior_source(
            prior_root, expected_result_sha256=prior_result_sha256, dataset_hash=reference,
            vision_provenance=provenance,
        )
        model = load_frozen_spatial_accuracy(
            accuracy_root, expected_result_sha256=accuracy_result_sha256, reference_dataset_hash=reference,
            vision_receipt=provenance, visual_dim=vision.visual_dim, prior=prior,
        )
        if content_hash(model.receipt) != scope.model_hash:
            raise ValueError("Appearance inputs must use the unchanged frozen camera model")
        before = _frozen_state(vision)
        print("Capturing only the new holdout detections", flush=True)
        new = capture_spatial_trace(scope.packs[APPEARANCE_ROLE], vision, model, batch_size=batch_size)
        fresh_hash = write_spatial_trace(root / _NEW_TRACE, new)
        traces = {**scope.previous_traces, APPEARANCE_ROLE: new}
        anchors = {**scope.previous_capture_hashes, APPEARANCE_ROLE: fresh_hash}
        roles = {}
        for role in INPUT_ROLES:
            print(f"Computing appearance on fixed {role} frames: {len(traces[role].frames)}", flush=True)
            decisions = _raw_decisions(traces[role], scope)
            features = _capture_role(scope.packs[role], decisions, vision, batch_size)
            digest = write_appearance_trace(
                root / _feature_file(role), decisions, features, model_hash=scope.model_hash, capture_hash=anchors[role],
            )
            roles[role] = {
                "pack_hash": scope.packs[role].manifest_hash, "appearance_sha256": digest, **_counts(decisions, features),
            }
        if _frozen_state(vision) != before:
            raise ValueError("Frozen vision changed during descriptor capture")
    header = {
        "schema": _SCHEMA, "status": "captured", "revision": revision, "model_hash": scope.model_hash,
        "observation_anchors": anchors, "holdout_trace_sha256": fresh_hash, "roles": roles,
        "descriptor_protocol": descriptor_protocol(), "labels_opened": False,
    }
    path = root / _MANIFEST
    write_json(path, header)
    sha = hashlib.sha256(read_bounded(path, _MAX_HEADER)).hexdigest()
    reopen_appearance_capture(root, scope, expected_capture_sha256=sha)
    return root, sha


def reopen_appearance_capture(root, scope, *, expected_capture_sha256):
    require_hex(expected_capture_sha256, 64, "appearance capture byte hash")
    root = Path(root).absolute()
    raw = read_bounded(root / _MANIFEST, _MAX_HEADER)
    if hashlib.sha256(raw).hexdigest() != expected_capture_sha256:
        raise ValueError("Appearance capture byte hash differs")
    manifest = decode_json(raw)
    require_keys(manifest, {
        "schema", "status", "revision", "model_hash", "observation_anchors", "holdout_trace_sha256",
        "roles", "descriptor_protocol", "labels_opened",
    }, "appearance capture manifest")
    require_hex(manifest["revision"], 40, "appearance capture revision")
    fresh = manifest["holdout_trace_sha256"]
    require_hex(fresh, 64, "appearance holdout observation digest")
    anchors = {**scope.previous_capture_hashes, APPEARANCE_ROLE: fresh}
    if (
        manifest["schema"] != _SCHEMA or manifest["status"] != "captured" or manifest["labels_opened"] is not False
        or manifest["model_hash"] != scope.model_hash or manifest["observation_anchors"] != anchors
        or manifest["descriptor_protocol"] != descriptor_protocol()
    ):
        raise ValueError("Appearance capture disagrees with the bound source observations or model")
    require_keys(manifest["roles"], set(INPUT_ROLES), "appearance roles")
    current = read_spatial_trace(
        root / _NEW_TRACE, scope.packs[APPEARANCE_ROLE], expected_sha256=fresh, expected_model_hash=scope.model_hash,
    )
    traces = {**scope.previous_traces, APPEARANCE_ROLE: current}
    decisions, appearances = {}, {}
    for role in INPUT_ROLES:
        info = manifest["roles"][role]
        require_keys(info, {"pack_hash", "appearance_sha256", "frames", "present", "ready"}, "descriptor role receipt")
        if info["pack_hash"] != scope.packs[role].manifest_hash:
            raise ValueError("Descriptor pack hash differs from the verified frame manifest")
        decisions[role] = fixed = _raw_decisions(traces[role], scope)
        appearances[role] = frames = read_appearance_trace(
            root / _feature_file(role), fixed, model_hash=scope.model_hash, capture_hash=anchors[role],
            expected_sha256=info["appearance_sha256"],
        )
        if any(info[k] != v for k, v in _counts(fixed, frames).items()):
            raise ValueError("Descriptor availability counts differ from the capture receipt")
    return AppearanceCapture(scope, decisions, appearances, manifest, expected_capture_sha256)


def _raw_decisions(trace, scope):
    policy = ZoneDecisionPolicy(**scope.model_receipt["calibrated_policy"])
    return replay_temporal(trace.frames, policy, RECIPES["framewise"]).decisions


def _frozen_state(vision):
    if vision.training or any(p.requires_grad or p.grad is not None for p in vision.parameters()):
        raise ValueError("Appearance capture requires frozen eval-mode vision")
    return module_state_hash(vision)


def _capture_role(pack, decisions, vision, batch_size):
    if tuple(d.observation.sample_id for d in decisions) != tuple(f.sample_id for f in pack.frames):
        raise ValueError("Appearance capture frame order differs from the bound observation trace")
    features = []
    for start in range(0, len(decisions), batch_size):
        end = start + batch_size
        images = torch.stack([
            torch.from_numpy(read_spatial_rgb(pack.root, frame.rgb)).permute(2, 0, 1).float().div_(255)
            for frame in pack.frames[start:end]
        ]).unsqueeze(1)
        features.extend(extract_spatial_appearance(decisions[start:end], images, vision(images)))
    return tuple(features)


def _counts(decisions, rows):
    return {
        "frames": len(decisions), "present": sum(s == PRESENT for d in decisions for s in d.states),
        "ready": {kind: sum(v is not None for r in rows for v in getattr(r, kind)) for kind in APPEARANCE_KINDS},
    }


def _feature_file(role):
    return f"appearance-{role}.json"
