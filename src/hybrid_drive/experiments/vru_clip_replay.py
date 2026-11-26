"""Freeze sensor-only camera predictions and replay the same causal association."""

from dataclasses import asdict
from math import ceil

from hybrid_drive.experiments.bounded_io import require_hex
from hybrid_drive.experiments.io import write_json
from hybrid_drive.experiments.object_vru_association import associate_objects
from hybrid_drive.experiments.object_vru_geometry import align_ground_objects, fuse_ground_hits
from hybrid_drive.experiments.vru_chain_capture import (
    ensure_new_output, index_sha, read_ground_objects, save_sealed, sealed_read, validate_anchor,
)
from hybrid_drive.experiments.vru_clip_metrics import summarize_vru_clips
from hybrid_drive.experiments.vru_observation_fusion import fuse_vru_observations
from hybrid_drive.geometry.rigid import invert_pose, rigid_transform


CAPTURE_SCHEMA = "hybrid-drive-vru-clip-observations-v1"
DEVELOPMENT_SCORE = "hybrid-drive-vru-clip-development-score-v1"
FRAMES = ("older", "previous", "current")
TOKENS = ("older_token", "previous_token", "sample_token")
FRAME_SIZE = (768, 432)
FITTED_PLANE_Z_M = -.190
MODEL_ARMS = ("dfine_plane", "dfine_depth")


def capture_candidate(input_root, models, manifest, output, *, split, revision, device):
    from hybrid_drive.experiments.box_motion_inputs import load_motion_snapshot
    from hybrid_drive.experiments.vru_metric_detector import MetricVruDetector

    ensure_new_output(output)
    require_hex(revision, 40, "candidate revision")
    source = _sensor_index(input_root, split)
    detector = MetricVruDetector(models, manifest, device=device)
    windows = []
    for anchor in source["anchors"]:
        windows.append(_capture_window(anchor, source, input_root, detector, load_motion_snapshot))
        if len(windows) % 8 == 0:
            print(f"Candidate depth: {len(windows)}/{len(source['anchors'])} windows", flush=True)
    seal = save_sealed(output, "observations.json", {
        "schema": CAPTURE_SCHEMA, "split": split, "revision": revision, "input_sha256": index_sha(input_root),
        "labels_read": False, "observations": windows, "profile": detector.provenance,
    })
    return {"windows": len(windows), "sha256": seal}


def _sensor_index(root, split):
    if split == "development":
        from hybrid_drive.experiments.vru_fusion_holdout_data import load_fusion_inputs

        return load_fusion_inputs(root)
    if split == "control":
        from hybrid_drive.experiments.vru_clip_data import load_clip_inputs

        return load_clip_inputs(root)
    raise ValueError("Expected the development or fixed control pack")


def _capture_window(anchor, index, root, detector, load_snapshot):
    collect = {}
    for phase, key in zip(FRAMES, TOKENS, strict=True):
        token = anchor[key]
        if token is None:
            collect[phase] = None
            continue
        shot = load_snapshot(root, token, index["frames"][token], rectified_wh=FRAME_SIZE)
        objects, camera_rows = detector(shot)
        collect[phase] = {
            "sha256": shot.content_sha256,
            "objects": {f"dfine_{name}": [asdict(obj) for obj in arm] for name, arm in objects.items()},
            "camera_evidence": camera_rows,
        }
    return {**anchor, "observations": collect}


def replay_clip(anchor, index, local):
    """Keep detector frame IDs intact while re-expressing history in one ego pose."""
    entries = [index["frames"][anchor[key]] for key in TOKENS]
    inverse = invert_pose(entries[-1]["ego_to_world"])
    poses = [rigid_transform(frame["ego_to_world"], "fixed camera ego pose") for frame in entries]
    stamps = [frame["ref_time_us"] for frame in entries]
    people = {phase: align_ground_objects(local[phase], inverse @ pose) if phase != "current" else local[phase]
              for phase, pose in zip(FRAMES, poses, strict=True)}
    out = {"older": [_point(obj, None) for obj in people["older"]]}
    for i, phase in enumerate(FRAMES[1:], start=1):
        past = FRAMES[i-1]
        age = (stamps[i] - stamps[i-1])/1e6
        motion = associate_objects(people[phase], people[past], age, mode="appearance")
        out[phase] = [_point(obj, hop.previous_object_id) for obj, hop in zip(people[phase], motion, strict=True)]
    return {"sample_token": anchor["sample_token"], "recording": anchor["recording"], "frames": out}


def _point(obj, previous_id):
    return {"object_id": obj.object_id, "kind": obj.kind, "xy_m": obj.xy, "previous_object_id": previous_id}


def score_development(root, candidate_path, baseline_path, labels_path, history_path, projection_path, output, *,
                      candidate_sha256, baseline_sha256):
    from hybrid_drive.experiments.vru_chain_capture import load_chain_observations
    from hybrid_drive.experiments.vru_chain_truth import load_vru_chain_truth
    from hybrid_drive.experiments.vru_fusion_holdout_data import load_fusion_labels
    from hybrid_drive.experiments.vru_projection_truth import load_projection_truth

    ensure_new_output(output)
    candidate = sealed_read(candidate_path, candidate_sha256)
    index, digest = _sensor_index(root, "development"), index_sha(root)
    _bind_capture(candidate, digest, "development")
    baseline = load_chain_observations(index, digest, baseline_path, baseline_sha256)
    captures = tuple(zip(index["anchors"], candidate["observations"], baseline["windows"], strict=True))
    for anchor, fresh, old in captures:
        validate_anchor(anchor, fresh)
        _validate_comparable_cameras(fresh, old)
    starts = load_fusion_labels(root, labels_path)["rows"]
    history = load_vru_chain_truth(root, labels_path, history_path)["rows"]
    projection = load_projection_truth(root, labels_path, projection_path)["rows"]
    truth, predictions = [], {}
    for (anchor, fresh, old), centers, past, box_labels in zip(captures, starts, history, projection, strict=True):
        label = _dev_label(anchor, centers, past, box_labels)
        if label is None:
            continue
        truth.append(label)
        for name, frames in _development_arms(fresh, old).items():
            predictions.setdefault(name, []).append(replay_clip(anchor, index, frames))
    if not truth:
        raise ValueError("No complete primary targets in the development pack")
    scores = {name: summarize_vru_clips(rows, truth, expected_count=len(truth), required_pass=ceil(2*len(truth)/3))
              for name, rows in predictions.items()}
    result = {"schema": DEVELOPMENT_SCORE, "input_sha256": digest, "candidate_sha256": candidate_sha256,
              "baseline_sha256": baseline_sha256, "purpose": "development_only_not_the_fixed_120_clip_control",
              "captured": len(captures), "eligible_with_primary": len(truth), "scores": scores}
    output.mkdir(parents=True)
    write_json(output/"score.json", result)
    return {name: score["counts"] for name, score in scores.items()}


def _bind_capture(candidate, digest, split):
    if (candidate["schema"] != CAPTURE_SCHEMA or candidate["split"] != split or candidate["input_sha256"] != digest
            or candidate["labels_read"] is not False):
        raise ValueError("Expected an input-only candidate capture from the fixed split")
    require_hex(candidate["revision"], 40, "candidate capture revision")


def _validate_comparable_cameras(fresh, old):
    for phase in FRAMES:
        a, b = (item["observations"][phase] for item in (fresh, old))
        if (a is None) != (b is None) or (a is not None and a["sha256"] != b["sha256"]):
            raise ValueError("Models must see identical rectified pixels, camera metadata and times")
        if a is not None and set(a["objects"]) != set(MODEL_ARMS):
            raise ValueError("Candidate lost an observation profile")


def _dev_label(anchor, centers, history, projection):
    if anchor["older_token"] is None:
        return None
    frames = {key: [] for key in FRAMES}
    eligible = []
    for i, (label, past, target) in enumerate(zip(centers["tracks"], history["tracks"], projection["targets"], strict=True)):
        kind = target["kind"]
        if kind not in ("person", "bicycle"):
            continue
        xy = label["centers_xy_m"][0]
        older, previous = (past[key+"_xy_m"] for key in FRAMES[:2])
        for phase, point in zip(FRAMES, (older, previous, xy), strict=True):
            if point is not None:
                frames[phase].append({"track_id": f"target-{i}", "kind": kind, "xy_m": point})
        if older is not None and previous is not None:
            eligible.append((abs(xy[1]), xy[0], i))
    if not eligible:
        return None
    _, _, winner = min(eligible)
    return {"sample_token": anchor["sample_token"], "recording": anchor["recording"],
            "primary_id": f"target-{winner}", "frames": frames}


def _development_arms(fresh, old):
    from hybrid_drive.experiments.vru_plane_probe import _move_hit

    arms = {name: {} for name in ("fused_baseline", "dfine_calibrated", *MODEL_ARMS)}
    for phase in FRAMES:
        baseline = old["observations"][phase]["objects"]
        arms["fused_baseline"][phase] = fuse_vru_observations(
            *(read_ground_objects(baseline[name]) for name in ("ssdlite", "faster_rcnn")),
        )
        for name in MODEL_ARMS:
            arms[name][phase] = read_ground_objects(fresh["observations"][phase]["objects"][name])
        arms["dfine_calibrated"][phase] = fuse_ground_hits(
            tuple(_move_hit(hit, FITTED_PLANE_Z_M) for obj in arms["dfine_plane"][phase] for hit in obj.hits),
        )
    return arms
