"""Blind two-frame detector replay, development-only calibration and holdout score."""

from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

from hybrid_drive.experiments.bounded_io import decode_json, read_bounded, require_hex
from hybrid_drive.experiments.box_motion_inputs import load_motion_snapshot
from hybrid_drive.experiments.fast_capture import artifact_hash
from hybrid_drive.experiments.io import write_json
from hybrid_drive.experiments.object_vru_association import MODELS, associate_objects
from hybrid_drive.experiments.object_vru_detector import LocalVruDetector, RECTIFIED_WH, WEIGHTS_SHA256
from hybrid_drive.experiments.object_vru_geometry import align_ground_objects
from hybrid_drive.experiments.object_vru_metrics import fit_motion_gains, summarize_object_motion
from hybrid_drive.experiments.temporal_vru_data import load_temporal_vru_inputs, load_temporal_vru_labels
from hybrid_drive.geometry.rigid import invert_pose


CAPTURE_SCHEMA = "hybrid-drive-object-vru-capture-v1"
GAIN_SCHEMA = "hybrid-drive-object-vru-gains-v1"
SCORE_SCHEMA = "hybrid-drive-object-vru-score-v1"
INPUTS = {
    "development": "e21b7a60d05cadea71777cdeec9b9001dbc7df075f89d4ed7401ac22134e2b71",
    "heldout": "c71dd951c4a61405c0f49c33387f5294bd22522775fb383095f56877e3533699",
}


def capture_object_motion(input_root, weights_root, output, *, revision, gain_path=None):
    require_hex(revision, 40, "capture revision")
    _new_output(output)
    observed, index_sha = _index(input_root)
    split = observed["split"]
    if (split == "heldout") != (gain_path is not None):
        raise ValueError("Holdout requires the already frozen development gain receipt")
    gain_sha = _gains(gain_path, revision)[1] if gain_path is not None else None
    detector = LocalVruDetector(weights_root)
    rows = []
    for anchor in observed["anchors"]:
        current_key, past_key = anchor["sample_token"], anchor["previous_token"]
        now = load_motion_snapshot(input_root, current_key, observed["frames"][current_key], rectified_wh=RECTIFIED_WH)
        old = load_motion_snapshot(input_root, past_key, observed["frames"][past_key], rectified_wh=RECTIFIED_WH)
        age = (now.ref_time_us - old.ref_time_us) / 1e6
        current, current_stats = detector(now)
        old_objects, previous_stats = detector(old)
        aligned = align_ground_objects(old_objects, invert_pose(now.ego_to_world) @ old.ego_to_world)
        estimates = {arm: associate_objects(current, aligned, age, mode=arm) for arm in MODELS}
        object_rows = []
        for i, obj in enumerate(current):
            object_rows.append({**_object_row(obj), "motions": {mode: asdict(estimates[mode][i]) for mode in MODELS}})
        rows.append({**anchor, "ref_time_us": now.ref_time_us, "history_s": age,
                     "current_sha256": now.content_sha256, "previous_sha256": old.content_sha256,
                     "objects": object_rows, "past_objects": [_object_row(obj) for obj in aligned],
                     "current_stats": current_stats, "past_stats": previous_stats})
        if len(rows) % 12 == 0:
            print(f"{split}: detected and associated {len(rows)}/{len(observed['anchors'])}", flush=True)
    result = {"schema": CAPTURE_SCHEMA, "revision": revision, "split": split, "input_sha256": index_sha,
              "weights": detector.provenance, "development_gain_sha256": gain_sha, "labels_mounted": False,
              "past_alignment": "current_ego_frame", "rows": rows}
    output.mkdir(parents=True)
    path = output / "predictions.json"
    write_json(path, result)
    seal = artifact_hash(path)
    write_json(output / "seal.json", {"prediction_sha256": seal})
    return {"windows": len(rows), "prediction_sha256": seal, "objects": sum(len(r["objects"]) for r in rows)}


def calibrate_object_motion(input_root, labels_path, predictions, output, *, sealed_sha256):
    _new_output(output)
    capture = _sealed_capture(input_root, predictions, sealed_sha256, require_split="development")
    if capture["development_gain_sha256"] is not None:
        raise ValueError("Calibration cannot consume a tuned capture")
    targets = load_temporal_vru_labels(input_root, labels_path)["rows"]
    fit = fit_motion_gains(capture["rows"], targets)
    output.mkdir(parents=True)
    write_json(output / "gains.json", {"schema": GAIN_SCHEMA, "revision": capture["revision"], "split": "development",
                                       "detector_sha256": WEIGHTS_SHA256, "input_sha256": INPUTS["development"],
                                       "capture_sha256": sealed_sha256, "labels_sha256": artifact_hash(labels_path),
                                       "heldout_opened": False, "fit": fit})
    return {"radial_gain": fit["radial_gain"], "tangential_gain": fit["tangential_gain"], "status": fit["status"]}


def score_object_motion(input_root, labels_path, predictions, gain_path, output, *, sealed_sha256):
    _new_output(output)
    capture = _sealed_capture(input_root, predictions, sealed_sha256, require_split="heldout")
    fit, gain_sha = _gains(gain_path, capture["revision"])
    if capture["development_gain_sha256"] != gain_sha:
        raise ValueError("Heldout forecasts must bind the gains frozen before replay")
    targets = load_temporal_vru_labels(input_root, labels_path)["rows"]
    summary = summarize_object_motion(capture["rows"], targets, fit)
    output.mkdir(parents=True)
    write_json(output / "score.json", {"schema": SCORE_SCHEMA, "revision": capture["revision"],
                                       "prediction_sha256": sealed_sha256, "targets_sha256": artifact_hash(labels_path),
                                       "gain_sha256": gain_sha, "summary": summary})
    return summary["gate"]


def _sealed_capture(input_root, predictions, seal, *, require_split):
    require_hex(seal, 64, "prediction seal")
    raw = read_bounded(predictions, 16 * 1024**2)
    if sha256(raw).hexdigest() != seal:
        raise ValueError("Object forecasts differ from their prediction seal")
    captured = decode_json(raw)
    index, digest = _index(input_root)
    if (captured["schema"] != CAPTURE_SCHEMA or captured["split"] != require_split or index["split"] != require_split
            or captured["input_sha256"] != digest or captured["weights"]["weights_sha256"] != WEIGHTS_SHA256
            or captured["labels_mounted"] is not False or captured["past_alignment"] != "current_ego_frame"):
        raise ValueError("Expected the blind object replay and its exact input cohort")
    require_hex(captured["revision"], 40, "capture revision")
    for row, anchor in zip(captured["rows"], index["anchors"], strict=True):
        if (any(row[key] != anchor[key] for key in ("sample_token", "recording", "previous_token"))
                or row["ref_time_us"] != index["frames"][anchor["sample_token"]]["ref_time_us"]):
            raise ValueError("Object-motion replay no longer matches the observation identifiers")
        require_hex(row["current_sha256"], 64, "current camera observation")
        require_hex(row["previous_sha256"], 64, "previous camera observation")
        old = {obj["object_id"]: obj for obj in row["past_objects"]}
        if len(old) != len(row["past_objects"]):
            raise ValueError("Duplicate projected past object identity")
        for arm in MODELS:
            used = set()
            for obj in row["objects"]:
                motion = obj["motions"][arm]
                if motion["status"] == "estimated":
                    key = motion["previous_object_id"]
                    if (key in used or key not in old or obj["kind"] != old[key]["kind"]
                            or motion["past_xy_m"] != old[key]["xy_m"]):
                        raise ValueError("Object replay has a duplicate or inconsistent history reference")
                    used.add(key)
    return captured


def _object_row(obj):
    return {"object_id": obj.object_id, "kind": obj.kind, "xy_m": obj.xy, "confidence": obj.confidence,
            "cameras": [hit.camera_name for hit in obj.hits]}


def _gains(path, revision):
    raw = read_bounded(path, 512 * 1024)
    gain = decode_json(raw)
    if (gain["schema"] != GAIN_SCHEMA or gain["revision"] != revision or gain["split"] != "development"
            or gain["heldout_opened"] is not False or gain["detector_sha256"] != WEIGHTS_SHA256
            or gain["input_sha256"] != INPUTS["development"]):
        raise ValueError("Velocity gain must come only from the isolated development split")
    for name in ("radial_gain", "tangential_gain"):
        if type(gain["fit"][name]) not in (int, float) or gain["fit"][name] not in (0., .25, .5, .75, 1.):
            raise ValueError("Unexpected velocity gain outside the frozen development grid")
    return gain["fit"], sha256(raw).hexdigest()


def _index(root: Path):
    pack = load_temporal_vru_inputs(root)
    measured = artifact_hash(root / "index.json")
    if measured != INPUTS.get(pack["split"]):
        raise ValueError("Expected the exact development or heldout observation pack from #50")
    return pack, measured


def _new_output(path):
    if path.exists() or path.is_symlink():
        raise ValueError("Expected a new object motion experiment directory")
