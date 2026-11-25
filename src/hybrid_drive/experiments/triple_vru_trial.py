"""Replay fixed object observations and compare two- and three-frame motion."""

from dataclasses import asdict
from hashlib import sha256

from hybrid_drive.experiments.bounded_io import canonical_json, decode_json, read_bounded, require_hex
from hybrid_drive.experiments.box_motion_inputs import load_motion_snapshot
from hybrid_drive.experiments.fast_capture import artifact_hash
from hybrid_drive.experiments.io import write_json
from hybrid_drive.experiments.object_vru_association import MODELS, associate_objects
from hybrid_drive.experiments.object_vru_detector import LocalVruDetector, RECTIFIED_WH, WEIGHTS_SHA256
from hybrid_drive.experiments.object_vru_geometry import GroundHit, GroundObject, align_ground_objects
from hybrid_drive.experiments.object_vru_trial import _index, _object_row
from hybrid_drive.experiments.temporal_vru_data import load_temporal_vru_labels
from hybrid_drive.experiments.triple_vru_confirmation import confirm_objects, frozen_two_frame_velocity
from hybrid_drive.experiments.triple_vru_inputs import load_third_observations
from hybrid_drive.experiments.triple_vru_metrics import summarize_three_frame
from hybrid_drive.geometry.rigid import invert_pose


CAPTURE_SCHEMA = "hybrid-drive-triple-vru-objects-v1"
FORECAST_SCHEMA = "hybrid-drive-triple-vru-forecast-v1"
SCORE_SCHEMA = "hybrid-drive-triple-vru-score-v1"
ORIGINAL_REVISION = "0a0b484c3455f81ab10545c882facb51b4ca0479"
ORIGINAL_GAIN_SHA256 = "16e8e38c0c604d9c8adeea445b8f38b544ef9674ddce55276438e9b409f711ac"
ORIGINAL_PREDICTION_SHA256 = {
    "development": "c577504ff84cb44f2d1dc69b4a67683f2c58d5aed09e1d75f36d757e01411b00",
    "heldout": "d2025764aad9437d9adc82685f12d36137dc2e3ecc3b7348a6d7c01feab74a41",
}


def capture_triple(base, extension, baseline_path, weights, output, *, revision):
    require_hex(revision, 40, "revision")
    _require_new(output)
    index, sha, added, added_sha = _inputs(base, extension)
    split = index["split"]
    prior = _baseline(baseline_path, split)
    detector = LocalVruDetector(weights)
    rows = []
    for anchor, saved in zip(added["anchors"], prior["rows"], strict=True):
        now = _snapshot(base, index, anchor["sample_token"])
        middle = _snapshot(base, index, anchor["previous_token"])
        age = (now.ref_time_us - middle.ref_time_us) / 1e6
        world_to_now = invert_pose(now.ego_to_world)
        current, current_stats = detector(now)
        mid_objects, middle_stats = detector(middle)
        aligned_middle = align_ground_objects(mid_objects, world_to_now @ middle.ego_to_world)
        measured = {"sample_token": now.snapshot_id, "previous_token": middle.snapshot_id,
                    "recording": anchor["recording"], "ref_time_us": now.ref_time_us, "history_s": age,
                    "current_sha256": now.content_sha256, "previous_sha256": middle.content_sha256,
                    "current_stats": current_stats, "past_stats": middle_stats}
        if (any(saved[key] != value for key, value in measured.items())
                or not _equal([_object_row(x) for x in current], [_legacy_object(x) for x in saved["objects"]])
                or not _equal([_object_row(x) for x in aligned_middle], saved["past_objects"])):
            raise ValueError("Two original detector inputs or objects drifted from #51")
        older, older_age, digest = None, None, None
        if anchor["status"] == "ready":
            oldest = _snapshot(extension, added, anchor["older_token"])
            older_age = (middle.ref_time_us - oldest.ref_time_us) / 1e6
            found, _ = detector(oldest)
            older = align_ground_objects(found, world_to_now @ oldest.ego_to_world)
            digest = oldest.content_sha256
        rows.append({**anchor, "older_history_s": older_age, "older_sha256": digest,
                     "current": [asdict(x) for x in current], "middle": [asdict(x) for x in aligned_middle],
                     "oldest": None if older is None else [asdict(x) for x in older]})
        if len(rows) % 16 == 0:
            print(f"{split}: unchanged pairs and third observations {len(rows)}/{len(added['anchors'])}", flush=True)
    result = {"schema": CAPTURE_SCHEMA, "revision": revision, "split": split, "base_input_sha256": sha,
              "extension_sha256": added_sha, "original_predictions_sha256": ORIGINAL_PREDICTION_SHA256[split],
              "weights_sha256": WEIGHTS_SHA256, "labels_mounted": False, "rows": rows}
    path = _write(output, "objects.json", result)
    return {"windows": len(rows), "ready": sum(r["oldest"] is not None for r in rows), "objects_sha256": artifact_hash(path)}


def forecast_triple(base, extension, baseline_path, evidence_path, gain_path, output, *, revision, evidence_sha256):
    require_hex(revision, 40, "revision")
    _require_new(output)
    index, sha, added, added_sha = _inputs(base, extension)
    split = index["split"]
    previous = _baseline(baseline_path, split)
    evidence = _read_sealed(evidence_path, evidence_sha256, "frozen ground objects", 32 * 1024**2)
    if (evidence["schema"] != CAPTURE_SCHEMA or evidence["revision"] != revision or evidence["split"] != split
            or evidence["base_input_sha256"] != sha or evidence["extension_sha256"] != added_sha
            or evidence["original_predictions_sha256"] != ORIGINAL_PREDICTION_SHA256[split]
            or evidence["weights_sha256"] != WEIGHTS_SHA256 or evidence["labels_mounted"] is not False):
        raise ValueError("Three-frame forecast requires the sealed compatible object observations")
    gains = _gains(gain_path)
    rows = []
    for saved, anchor, row in zip(previous["rows"], added["anchors"], evidence["rows"], strict=True):
        _verify_extension_row(anchor, row, index, added)
        if (row["oldest"] is not None) != (anchor["status"] == "ready"):
            raise ValueError("Oldest detection bundle is missing or supplied without an observation")
        current, middle = _objects(row["current"]), _objects(row["middle"])
        older = None if row["oldest"] is None else _objects(row["oldest"])
        age, age_old = saved["history_s"], row["older_history_s"]
        first = {name: associate_objects(current, middle, age, mode=name) for name in MODELS}
        confirmed = confirm_objects(current, middle, older, first["appearance"], age, age_old)
        if not _equal([_object_row(o) for o in middle], saved["past_objects"]):
            raise ValueError("The reference middle detections changed")
        output_objects = []
        for i, (obj, original, chain) in enumerate(zip(current, saved["objects"], confirmed, strict=True)):
            measured = {**_object_row(obj), "motions": {key: asdict(links[i]) for key, links in first.items()}}
            if not _equal(measured, original):
                raise ValueError("Two-frame association did not reproduce the exact #51 baseline")
            pair = frozen_two_frame_velocity(first["appearance"][i], **gains)
            tracked = chain.status == "confirmed"
            output_objects.append({**original, "confirmation": asdict(chain), "forecasts": {
                "stationary": (0., 0.), "two_frame": pair,
                "confirmed_two_frame": pair if tracked else (0., 0.), "three_frame_trend": chain.trend_velocity_xy_mps,
            }})
        rows.append({**saved, "objects": output_objects, "extension_status": anchor["status"],
                     "older_token": anchor["older_token"], "older_history_s": age_old, "older_sha256": row["older_sha256"],
                     "oldest_objects": [] if older is None else [_object_row(obj) for obj in older]})
    receipt = {"schema": FORECAST_SCHEMA, "revision": revision, "split": split, "base_input_sha256": sha,
               "extension_sha256": added_sha, "original_predictions_sha256": ORIGINAL_PREDICTION_SHA256[split],
               "gains_sha256": ORIGINAL_GAIN_SHA256, "objects_sha256": evidence_sha256,
               "labels_mounted": False, "two_frame_baseline_reproduced": True, "rows": rows}
    path = _write(output, "predictions.json", receipt)
    seal = artifact_hash(path)
    write_json(output / "seal.json", {"prediction_sha256": seal})
    return {"windows": len(rows), "prediction_sha256": seal}


def score_triple(base, extension, predictions_path, labels_path, output, *, sealed_sha256):
    _require_new(output)
    forecasts = _read_sealed(predictions_path, sealed_sha256, "three-frame predictions", 16 * 1024**2)
    index, sha, added, added_sha = _inputs(base, extension)
    split = index["split"]
    if (forecasts["schema"] != FORECAST_SCHEMA or forecasts["split"] != split or forecasts["base_input_sha256"] != sha
            or forecasts["extension_sha256"] != added_sha or forecasts["gains_sha256"] != ORIGINAL_GAIN_SHA256
            or forecasts["original_predictions_sha256"] != ORIGINAL_PREDICTION_SHA256[split]
            or forecasts["labels_mounted"] is not False or forecasts["two_frame_baseline_reproduced"] is not True):
        raise ValueError("Expected a blind prediction on the exact fixed cohort and baseline")
    require_hex(forecasts["revision"], 40, "revision")
    for anchor, pred in zip(added["anchors"], forecasts["rows"], strict=True):
        _verify_extension_row(anchor, {**pred, "status": pred["extension_status"]}, index, added)
        oldest = {obj["object_id"]: obj["kind"] for obj in pred["oldest_objects"]}
        if (len(oldest) != len(pred["oldest_objects"]) or (not pred["older_token"] and oldest)):
            raise ValueError("Invalid projected older detections")
        for obj in pred["objects"]:
            chain = obj["confirmation"]
            if chain["status"] == "confirmed" and oldest.get(chain["older_object_id"]) != obj["kind"]:
                raise ValueError("A confirmed link must refer to a measured same-kind older detection")
    targets = load_temporal_vru_labels(base, labels_path)["rows"]
    summary = summarize_three_frame(forecasts["rows"], targets)
    _write(output, "score.json", {"schema": SCORE_SCHEMA, "revision": forecasts["revision"], "split": split,
                                  "prediction_sha256": sealed_sha256, "targets_sha256": artifact_hash(labels_path),
                                  "summary": summary})
    return summary["gate"]


def _inputs(base, extension):
    index, sha = _index(base)
    added = load_third_observations(base, extension)
    return index, sha, added, artifact_hash(extension / "index.json")


def _baseline(path, split):
    prior = _read_sealed(path, ORIGINAL_PREDICTION_SHA256[split], "exact #51 forecast", 16 * 1024**2)
    if (prior["revision"] != ORIGINAL_REVISION or prior["split"] != split or prior["labels_mounted"] is not False
            or prior["weights"]["weights_sha256"] != WEIGHTS_SHA256):
        raise ValueError("Incompatible #51 baseline")
    return prior


def _gains(path):
    receipt = _read_sealed(path, ORIGINAL_GAIN_SHA256, "exact #51 development calibration", 512 * 1024)
    fit = receipt["fit"]
    if ((fit["radial_gain"], fit["tangential_gain"]) != (0., 1.) or receipt["heldout_opened"] is not False
            or receipt["split"] != "development"):
        raise ValueError("This pilot uses the original tangential-only velocity contract")
    return {key: fit[key] for key in ("radial_gain", "tangential_gain")}


def _verify_extension_row(anchor, row, index, added):
    if any(row[key] != value for key, value in anchor.items()):
        raise ValueError("Third observation row differs from its manifest")
    if anchor["older_token"] is None:
        if row["older_history_s"] is not None or row["older_sha256"] is not None:
            raise ValueError("Missing third observation cannot have a timestamp or a digest")
        return
    newer = index["frames"][anchor["previous_token"]]["ref_time_us"]
    older = added["frames"][anchor["older_token"]]["ref_time_us"]
    if row["older_history_s"] != (newer - older) / 1e6:
        raise ValueError("The third observation interval changed")
    require_hex(row["older_sha256"], 64, "oldest camera bundle hash")


def _snapshot(root, index, key):
    return load_motion_snapshot(root, key, index["frames"][key], rectified_wh=RECTIFIED_WH)


def _objects(rows):
    return tuple(GroundObject(r["object_id"], r["kind"], r["xy"], r["confidence"],
                              tuple(GroundHit(**hit) for hit in r["hits"])) for r in rows)


def _legacy_object(row):
    return {k: v for k, v in row.items() if k != "motions"}


def _equal(left, right):
    return canonical_json(left) == canonical_json(right)


def _read_sealed(path, sha, name, maximum):
    require_hex(sha, 64, name + " seal")
    raw = read_bounded(path, maximum)
    if sha256(raw).hexdigest() != sha:
        raise ValueError(name + " does not match the recorded seal")
    return decode_json(raw)


def _require_new(output):
    if output.exists() or output.is_symlink():
        raise ValueError("Expected a new three-frame output directory")


def _write(output, name, value):
    output.mkdir(parents=True)
    path = output / name
    write_json(path, value)
    return path
