"""Run the exact #47 checkpoint on new observations, without outcome access."""

from dataclasses import replace
from hashlib import sha256
from pathlib import Path

from safetensors.torch import load
import torch

from hybrid_drive.experiments.bounded_io import decode_json, read_bounded, require_hex
from hybrid_drive.experiments.box_motion_inputs import load_motion_snapshot
from hybrid_drive.experiments.fast_capture import artifact_hash
from hybrid_drive.experiments.frozen_transfer_data import load_transfer_inputs
from hybrid_drive.experiments.frozen_transfer_protocol import (
    CHECKPOINT_SHA256, FIT_REVISION, FIT_SHA256, PREDICTION_SCHEMA, SCORE_ARMS, donor_anchors,
)
from hybrid_drive.experiments.io import write_json
from hybrid_drive.experiments.paired_slow_pilot import FIT_SCHEMA
from hybrid_drive.experiments.paired_slow_training import initialized_model
from hybrid_drive.fast.metric_bev import past_forward_speed
from hybrid_drive.fast.scene_fast_runtime import FrozenSceneFast
from hybrid_drive.neural.paired_scene_risk import RiskRequest
from hybrid_drive.neural.pretrained_alignment import load_scene_bridge_backbones
from hybrid_drive.neural.surround_features import SharedSurroundEncoder


@torch.no_grad()
def predict_transfer(input_root, model_root, box_fit, road_fit, slow_fit, output, *, revision):
    require_hex(revision, 40, "prediction revision")
    if output.exists() or output.is_symlink():
        raise ValueError("Transfer prediction output must be new")
    manifest = load_transfer_inputs(input_root)
    fit, weight_bytes = _verified_original_fit(slow_fit)
    vision, language, provenance = load_scene_bridge_backbones(model_root)
    if fit["frozen_models"] != provenance["models"]:
        raise ValueError("Transfer must reuse exactly the original local vision and language backbones")
    info = provenance["models"]["vision"]
    encoder = SharedSurroundEncoder(vision, encoder_id=f'{info["repository"]}@{info["revision"]}')
    fast = FrozenSceneFast(box_fit, road_fit)
    trained, untrained = initialized_model().eval(), initialized_model().eval()
    trained.load_state_dict(load(weight_bytes), strict=True)
    donor_map = {}
    for receiving, row in donor_anchors(manifest["anchors"]).items():
        token = row["sample_token"]
        snapshot = load_motion_snapshot(input_root, token, manifest["frames"][token])
        donor_map[receiving] = encoder.encode(snapshot)
    rows = []
    for anchor in manifest["anchors"]:
        now = anchor["sample_token"]
        previous_id, = anchor["history_tokens"]
        snapshot = load_motion_snapshot(input_root, now, manifest["frames"][now])
        previous = load_motion_snapshot(input_root, previous_id, manifest["frames"][previous_id])
        identity = {
            "sample_token": now, "ref_time_us": snapshot.ref_time_us,
            "recording": anchor["recording"], "snapshot_sha256": snapshot.content_sha256,
        }
        speed = past_forward_speed(snapshot, previous)
        if not -.10 <= speed <= 60:
            rows.append({**identity, "status": "out_of_envelope", "candidates": []})
            continue
        features = encoder.encode(snapshot)
        result = fast(features, encoder.encode(previous))
        donor = donor_map[anchor["recording"]]
        scores = _score_requests(trained, untrained, language, features, result, donor)
        rows.append({**identity, "status": "ok", **scores})
        if len(rows) % 12 == 0:
            print(f"Frozen transfer {len(rows)}/{len(manifest['anchors'])}", flush=True)
    output.mkdir(parents=True)
    path = output / "predictions.json"
    write_json(path, {
        "schema": PREDICTION_SCHEMA, "revision": revision, "input_sha256": artifact_hash(input_root / "index.json"),
        "sources_sha256": manifest["source_sha256"], "checkpoint_sha256": CHECKPOINT_SHA256,
        "original_fit_sha256": FIT_SHA256, "original_fit_revision": FIT_REVISION,
        "future_annotations_mounted": False, "checkpoint_changed": False,
        "arms": list(SCORE_ARMS), "rows": rows,
    })
    seal = artifact_hash(path)
    write_json(output / "seal.json", {"prediction_sha256": seal})
    return {"anchors": len(rows), "predicted": sum(row["status"] == "ok" for row in rows), "sha256": seal}


def _score_requests(trained, untrained, language, features, result, donor):
    scene, ids = result.scene, result.paired_road.ordered
    if (features.snapshot.snapshot_id == donor.snapshot.snapshot_id
            or features.encoder_id != donor.encoder_id or features.feature_maps.shape != donor.feature_maps.shape):
        raise ValueError("Camera-swap donor must be a different observation encoded the same way")
    request = RiskRequest(features, scene, ids)
    no_camera = replace(request, features=replace(features, feature_maps=torch.zeros_like(features.feature_maps)))
    shuffled = replace(request, features=replace(features, feature_maps=donor.feature_maps))
    no_motion = replace(request, scene=replace(scene, boxes=tuple(
        replace(box, velocity_xy=(0., 0.)) for box in scene.boxes)))
    actual, blind, static, unrelated = trained([request, no_camera, no_motion, shuffled], language)
    original, = untrained([request], language)
    trajectories = {item.candidate_id: item.trajectory for item in scene.plan.candidates}
    geometry = {item.candidate_id: item for item in result.base.diagnostics}
    returns = {
        "trained": actual.tolist(), "untrained": original.tolist(), "no_camera": blind.tolist(),
        "no_motion": static.tolist(), "shuffled_camera": unrelated.tolist(),
    }
    return {
        "scene_sha256": scene.content_sha256,
        "donor_snapshot_id": donor.snapshot.snapshot_id, "donor_snapshot_sha256": donor.snapshot.content_sha256,
        "candidates": [{
            "candidate_id": key, "dt_s": trajectories[key].interval_s, "poses": trajectories[key].poses.tolist(),
            "fast_overlap_s": geometry[key].first_overlap_s, "fast_axis_gap_m": geometry[key].min_axis_gap_m,
            "margins": {arm: returns[arm][i] for arm in SCORE_ARMS},
        } for i, key in enumerate(ids)],
    }


def _verified_original_fit(root: Path):
    raw = read_bounded(root / "fit.json", 256 * 1024)
    if sha256(raw).hexdigest() != FIT_SHA256:
        raise ValueError("Expected the frozen #47 fit record, without any retuning")
    fit = decode_json(raw)
    weights = root / "bridge-projector.safetensors"
    weight_bytes = read_bounded(weights, 4 * 1024**2)
    if (fit["schema"] != FIT_SCHEMA or fit["revision"] != FIT_REVISION
            or fit["future_annotations_used"] is not False or fit["checkpoint_sha256"] != CHECKPOINT_SHA256
            or sha256(weight_bytes).hexdigest() != CHECKPOINT_SHA256):
        raise ValueError("The original #47 checkpoint or its fit provenance changed")
    return fit, weight_bytes
