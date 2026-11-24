"""Offline paired contact pilot; freeze estimates before opening natural labels."""

from dataclasses import asdict, replace
from hashlib import sha256
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from hybrid_drive.experiments.bounded_io import require_hex
from hybrid_drive.experiments.box_motion_inputs import load_motion_inputs, load_motion_snapshot
from hybrid_drive.experiments.fast_capture import artifact_hash, load_json
from hybrid_drive.experiments.io import write_json
from hybrid_drive.experiments.paired_slow_controls import crossing_controls
from hybrid_drive.experiments.paired_slow_metrics import CONTROL_PREDICTION_SCHEMA, CONTROL_TARGET_SCHEMA
from hybrid_drive.experiments.paired_slow_training import (
    CONTROL_COUNTS, GROUP_BATCH, LabeledControl, fit_paired, initialized_model,
)
from hybrid_drive.fast.scene_fast_runtime import FrozenSceneFast
from hybrid_drive.neural.paired_scene_risk import RiskRequest
from hybrid_drive.neural.pretrained_alignment import load_scene_bridge_backbones
from hybrid_drive.neural.surround_features import SharedSurroundEncoder


INPUT_SHA = "e04c01e8394d675d7a3fd1412658d714b4a8cdeb10e22ab093c5e64fcc704d21"
FIT_SCHEMA = "hybrid-drive-paired-slow-fit-v1"
NATURAL_PREDICTION_SCHEMA = "hybrid-drive-paired-slow-natural-predictions-v1"
_DEVELOPMENT_POSITIONS = (2, 7, 12, 17)
_PRIOR_BRIDGE_TEST_POSITIONS = (0, 6, 13, 19)


def run_fit(input_root, model_root, box_fit, road_fit, output, *, revision):
    require_hex(revision, 40, "fit revision")
    _require_new_directory(output)
    index, groups = _load_partition(input_root)
    language, provenance, encoder, runtime = _runtime(model_root, box_fit, road_fit)
    binary_error = _verify_binary_readout(language)
    controls, support = {}, {}
    for split in ("train", "development"):
        controls[split], support[split] = [], {}
        for anchor in groups[split]:
            features, result = _capture(input_root, index, anchor, encoder, runtime)
            current = _controls(features, result.scene, split)
            controls[split].extend(item[0] for item in current)
            support[split][anchor["sample_token"]] = len(current)
        print(f"{split}: {len(controls[split])} controls on {sum(n>0 for n in support[split].values())} snapshots",
              flush=True)
    model = initialized_model()
    fit = fit_paired(model, language, controls["train"], controls["development"])
    output.mkdir(parents=True)
    weights = output / "bridge-projector.safetensors"
    save_file(model.state_dict(), weights)
    record = {
        "schema": FIT_SCHEMA, "revision": revision, "input_sha256": INPUT_SHA,
        "checkpoint_sha256": artifact_hash(weights), "frozen_models": provenance["models"],
        "state_selected_on": "development_only", "training_recording": groups["train"][0]["recording"],
        "groups_by_snapshot": support, "binary_head_max_abs_delta": binary_error,
        "future_annotations_used": False, "test_camera_jpegs_opened": False,
        "natural_fast_boxes_used_for_training_targets": False, **fit,
    }
    write_json(output / "fit.json", record)
    return record


def run_predictions(input_root, model_root, box_fit, road_fit, fit_root, output, *, revision):
    require_hex(revision, 40, "prediction revision")
    _require_new_directory(output)
    index, groups = _load_partition(input_root)
    fit_record = load_json(fit_root / "fit.json")
    weights = fit_root / "bridge-projector.safetensors"
    if (fit_record["schema"] != FIT_SCHEMA or fit_record["revision"] != revision
            or fit_record["input_sha256"] != INPUT_SHA or fit_record["future_annotations_used"] is not False
            or fit_record["state_selected_on"] != "development_only"
            or fit_record["training_recording"] != groups["train"][0]["recording"]
            or not weights.is_file() or weights.is_symlink() or not 0 < weights.stat().st_size < 4 * 1024**2
            or artifact_hash(weights) != fit_record["checkpoint_sha256"]):
        raise ValueError("Expected the sealed local development-selected Bridge checkpoint")
    language, provenance, encoder, runtime = _runtime(model_root, box_fit, road_fit)
    if provenance["models"] != fit_record["frozen_models"]:
        raise ValueError("Prediction must use the same pinned vision and language models as fit")
    trained, untrained = initialized_model().eval(), initialized_model().eval()
    trained.load_state_dict(load_file(weights), strict=True)
    controls, labels, natural = [], [], []
    withheld = {anchor["sample_token"] for anchor in groups["heldout"]}
    with torch.no_grad():
        for i, anchor in enumerate(groups["natural"], 1):
            features, result = _capture(input_root, index, anchor, encoder, runtime)
            natural.append(_natural_row(features, result, anchor, language, trained, untrained))
            if anchor["sample_token"] in withheld:
                pairs = _controls(features, result.scene, "heldout")
                pred, truth = _control_rows(pairs, language, trained, untrained)
                identity = {
                    "sample_token": anchor["sample_token"], "recording": anchor["recording"],
                    "snapshot_sha256": result.scene.snapshot_sha256,
                }
                controls.append({**identity, "groups": pred})
                labels.append({**identity, "groups": truth})
            print(f"Held-out Slow predictions: {i}/{len(groups['natural'])}", flush=True)
    output.mkdir(parents=True)
    metadata = {"revision": revision, "fit_sha256": artifact_hash(fit_root / "fit.json"),
                "heldout_targets_opened": False}
    write_json(output / "natural-predictions.json", {"schema": NATURAL_PREDICTION_SCHEMA, **metadata, "rows": natural})
    write_json(output / "control-predictions.json", {"schema": CONTROL_PREDICTION_SCHEMA, **metadata, "rows": controls})
    seals = {name: artifact_hash(output / f"{name}-predictions.json") for name in ("natural", "control")}
    write_json(output / "seals.json", seals)
    write_json(output / "control-targets.json", {
        "schema": CONTROL_TARGET_SCHEMA, "revision": revision, "control_prediction_sha256": seals["control"],
        "source": "synthetic_conditional_sampled_actor_contact_not_observed_pixels", "rows": labels,
    })
    return seals


def partition_observations(index):
    by_recording = {}
    for row in index["anchors"]:
        by_recording.setdefault(row["recording"], []).append(row)
    if len(by_recording) != 2 or any(len(rows) != 20 for rows in by_recording.values()):
        raise ValueError("Expected two pinned recordings of twenty observation windows")
    first, second = (by_recording[key] for key in sorted(by_recording))
    partitions = {
        "train": tuple(row for i, row in enumerate(first) if i not in _DEVELOPMENT_POSITIONS),
        "development": tuple(row for i, row in enumerate(first) if i in _DEVELOPMENT_POSITIONS),
        "natural": tuple(second),
        "heldout": tuple(row for i, row in enumerate(second) if i not in _PRIOR_BRIDGE_TEST_POSITIONS),
    }
    seen_frames, seen_images = set(), set()
    for split in ("train", "development", "natural"):
        frames = {token for row in partitions[split] for token in (*row["history_tokens"], row["sample_token"])}
        images = {camera["sha256"] for frame in frames for camera in index["frames"][frame]["views"]}
        if seen_frames & frames or seen_images & images:
            raise ValueError("A frame or JPEG crossed the training, development or held-out boundary")
        seen_frames.update(frames)
        seen_images.update(images)
    return partitions


def _load_partition(root):
    if artifact_hash(root / "index.json") != INPUT_SHA:
        raise ValueError("Paired Slow requires the pinned observation-only OpenScene input index")
    index = load_motion_inputs(root)
    return index, partition_observations(index)


def _runtime(model_root, box_fit, road_fit):
    vision, language, provenance = load_scene_bridge_backbones(model_root)
    identity = provenance["models"]["vision"]
    encoder = SharedSurroundEncoder(vision, encoder_id=f'{identity["repository"]}@{identity["revision"]}')
    return language, provenance, encoder, FrozenSceneFast(box_fit, road_fit)


@torch.no_grad()
def _verify_binary_readout(language):
    prefix = torch.linspace(-.2, .3, 16 * language.llm_dim).reshape(2, 8, language.llm_dim)
    _, full = language(prefix, language.prompt_ids(len(prefix)))
    reference = full[:, -1, [language.yes_token_id, language.no_token_id]]
    small = language.binary_answer_logits(prefix)
    torch.testing.assert_close(small, reference, rtol=1e-5, atol=1e-5)
    return float((small - reference).abs().max())


def _capture(root, index, anchor, encoder, runtime):
    before, now = anchor["history_tokens"][-1], anchor["sample_token"]
    previous = encoder.encode(load_motion_snapshot(root, before, index["frames"][before]))
    features = encoder.encode(load_motion_snapshot(root, now, index["frames"][now]))
    return features, runtime(features, previous)


def _controls(features, scene, split):
    seed = int.from_bytes(sha256(f"paired-slow-v1/{split}/{scene.snapshot_id}".encode()).digest()[:8], "little")
    controls = crossing_controls(scene, seed=seed, count=CONTROL_COUNTS[split])
    return [(LabeledControl(tuple(RiskRequest(features, world, item.candidate_ids) for world in item.worlds),
                            tuple(item.candidate_ids.index(key) for key in item.blocked)), item)
            for item in controls]


def _control_rows(pairs, language, trained, untrained):
    predictions, targets = [], []
    for i in range(0, len(pairs), GROUP_BATCH):
        batch = pairs[i:i + GROUP_BATCH]
        requests = [item for labels, _ in batch for item in labels.requests]
        full = trained(requests, language)
        untouched = untrained(requests, language)
        still = trained([_zero_motion(item) for item in requests[::2]], language)
        for j, ((labels, control), first_world) in enumerate(zip(batch, requests[::2], strict=True)):
            if _zero_motion(labels.requests[1]).scene.content_sha256 != _zero_motion(first_world).scene.content_sha256:
                raise ValueError("Control worlds differed in a value other than motion")
            identity = {"world_sha256": [item.scene.content_sha256 for item in labels.requests],
                        "candidate_ids": list(control.candidate_ids)}
            predictions.append({**identity, "margins": {
                "trained": [scores.tolist() for scores in full[j*2:j*2+2]],
                "untrained": [scores.tolist() for scores in untouched[j*2:j*2+2]],
                "no_motion": [still[j].tolist(), still[j].tolist()],
            }})
            targets.append({**identity, "blocked_index": list(labels.blocked), "signed_axis_gaps_m": control.gaps_m,
                            "hypothetical_actors": [asdict(item.scene.boxes[0]) for item in labels.requests]})
    return predictions, targets


def _zero_motion(request):
    return replace(request, scene=replace(request.scene, boxes=tuple(
        replace(box, velocity_xy=(0., 0.)) for box in request.scene.boxes)))


def _natural_row(features, result, anchor, language, trained, untrained):
    scene = result.scene
    order = result.paired_road.ordered
    request = RiskRequest(features, scene, order)
    actual, = trained([request], language)
    before, = untrained([request], language)
    trajectories = {item.candidate_id: item.trajectory for item in scene.plan.candidates}
    fast_contact = {item.candidate_id: item.first_overlap_s for item in result.base.diagnostics}
    return {
        "sample_token": scene.snapshot_id, "recording": anchor["recording"], "ref_time_us": scene.ref_time_us,
        "snapshot_sha256": scene.snapshot_sha256,
        "candidates": [{"candidate_id": key, "dt_s": trajectories[key].interval_s,
                        "poses": trajectories[key].poses.tolist(), "fast_overlap_s": fast_contact[key],
                        "slow_margin": float(score), "untrained_margin": float(initial)}
                       for key, score, initial in zip(order, actual, before, strict=True)],
    }


def _require_new_directory(path: Path):
    if path.exists() or path.is_symlink():
        raise ValueError("Output directory must be new")
