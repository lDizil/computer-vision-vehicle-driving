"""Compare frozen detector profiles and the same tracker on a disjoint cohort."""

from dataclasses import asdict
from hashlib import sha256

from hybrid_drive.experiments.bounded_io import canonical_json, decode_json, read_bounded, require_hex
from hybrid_drive.experiments.box_motion_inputs import load_motion_snapshot
from hybrid_drive.experiments.fast_capture import artifact_hash
from hybrid_drive.experiments.io import write_json
from hybrid_drive.experiments.object_vru_association import MODELS, associate_objects
from hybrid_drive.experiments.object_vru_detector import LocalVruDetector, RECTIFIED_WH, WEIGHTS_SHA256 as SSD_SHA256
from hybrid_drive.experiments.object_vru_geometry import align_ground_objects
from hybrid_drive.experiments.object_vru_trial import _object_row
from hybrid_drive.experiments.triple_vru_confirmation import confirm_objects, frozen_two_frame_velocity
from hybrid_drive.experiments.vru_coverage_detector import CoverageDetector, WEIGHTS_SHA256 as FASTER_SHA256
from hybrid_drive.experiments.vru_fusion_holdout_data import load_fusion_inputs, load_fusion_labels
from hybrid_drive.experiments.vru_fusion_holdout_metrics import summarize_fusion_holdout
from hybrid_drive.experiments.vru_observation_fusion import MIN_CROSS_DETECTOR_DISTANCE_M, fuse_vru_observations
from hybrid_drive.geometry.rigid import invert_pose


CAPTURE_SCHEMA = "hybrid-drive-vru-fusion-holdout-predictions-v1"
SCORE_SCHEMA = "hybrid-drive-vru-fusion-holdout-score-v1"
ARMS = ("ssdlite", "faster_rcnn", "ssd_primary")
POLICY = {"min_distance_m": MIN_CROSS_DETECTOR_DISTANCE_M,
          "radial_gain": 0., "tangential_gain": 1., "velocity_source": "frozen_51_and_52"}


def capture_fusion_holdout(input_root, ssd_weights, faster_weights, output, *, revision):
    require_hex(revision, 40, "fusion benchmark revision")
    _new_directory(output)
    source = load_fusion_inputs(input_root)
    ssd, faster = LocalVruDetector(ssd_weights), CoverageDetector(faster_weights)
    anchors = source["anchors"]
    by_window = []
    for row in anchors:
        by_window.append(_capture_window(input_root, source, row, ssd, faster))
        if len(by_window) % 8 == 0:
            print(f"Fusion holdout: {len(by_window)}/{len(anchors)} episodes", flush=True)
    repeated = []
    for index in _repeat_indices(len(anchors)):
        again = _capture_window(input_root, source, anchors[index], ssd, faster)
        if canonical_json(again) != canonical_json(by_window[index]):
            raise ValueError("A fixed three-camera-time replay did not repeat exactly")
        repeated.append({"index": index, "sha256": sha256(canonical_json(again)).hexdigest()})
    result = {
        "schema": CAPTURE_SCHEMA, "revision": revision, "input_sha256": artifact_hash(input_root / "index.json"),
        "labels_mounted": False, "detectors": {"ssdlite": ssd.provenance, "faster_rcnn": faster.provenance},
        "policy": POLICY, "rechecks": repeated,
        "arms": {arm: [row[arm] for row in by_window] for arm in ARMS},
    }
    output.mkdir(parents=True)
    path = output / "predictions.json"
    write_json(path, result)
    seal = artifact_hash(path)
    write_json(output / "seal.json", {"sha256": seal})
    return {"windows": len(by_window), "rechecked": len(repeated), "prediction_sha256": seal}


def _capture_window(root, source, anchor, ssd, faster):
    def observe(key):
        image = load_motion_snapshot(root, key, source["frames"][key], rectified_wh=RECTIFIED_WH)
        a, _ = ssd(image)
        b, _ = faster(image)
        return image, {"ssdlite": a, "faster_rcnn": b, "ssd_primary": fuse_vru_observations(a, b)}

    now, current = observe(anchor["sample_token"])
    past, history = observe(anchor["previous_token"])
    oldest, third = observe(anchor["older_token"]) if anchor["older_token"] is not None else (None, None)
    inverse = invert_pose(now.ego_to_world)
    history_to_now = inverse @ past.ego_to_world
    oldest_to_now = inverse @ oldest.ego_to_world if oldest is not None else None
    meta = {
        **anchor, "ref_time_us": now.ref_time_us, "current_sha256": now.content_sha256,
        "previous_sha256": past.content_sha256, "older_sha256": oldest.content_sha256 if oldest is not None else None,
        "history_s": (now.ref_time_us - past.ref_time_us) / 1e6,
        "older_history_s": (past.ref_time_us - oldest.ref_time_us) / 1e6 if oldest is not None else None,
    }
    return {arm: {**meta, "objects": _track(current[arm], align_ground_objects(history[arm], history_to_now),
                                          align_ground_objects(third[arm], oldest_to_now) if third is not None else None,
                                          meta["history_s"], meta["older_history_s"])} for arm in ARMS}


def _track(now, past, older, latest_dt, older_dt):
    motions = {mode: associate_objects(now, past, latest_dt, mode=mode) for mode in MODELS}
    confirmations = confirm_objects(now, past, older, motions["appearance"], latest_dt, older_dt)
    rows = []
    for i, (obj, chain) in enumerate(zip(now, confirmations, strict=True)):
        first = motions["appearance"][i]
        pair = frozen_two_frame_velocity(first, radial_gain=POLICY["radial_gain"], tangential_gain=POLICY["tangential_gain"])
        rows.append({**_object_row(obj), "motions": {name: asdict(values[i]) for name, values in motions.items()},
                     "confirmation": asdict(chain), "forecasts": {
                         "stationary": (0., 0.), "two_frame": pair,
                         "confirmed_two_frame": pair if chain.status == "confirmed" else (0., 0.),
                         "three_frame_trend": chain.trend_velocity_xy_mps,
                     }})
    return rows


def score_fusion_holdout(input_root, predictions_path, labels_path, output, *, sealed_sha256):
    _new_directory(output)
    require_hex(sealed_sha256, 64, "prediction seal")
    raw = read_bounded(predictions_path, 32 * 1024**2)
    if sha256(raw).hexdigest() != sealed_sha256:
        raise ValueError("Fusion forecasts do not match the seal fixed before using labels")
    pred = decode_json(raw)
    source = load_fusion_inputs(input_root)
    if (pred["schema"] != CAPTURE_SCHEMA or pred["input_sha256"] != artifact_hash(input_root / "index.json")
            or pred["labels_mounted"] is not False or pred["policy"] != POLICY or set(pred["arms"]) != set(ARMS)
            or pred["detectors"]["ssdlite"]["weights_sha256"] != SSD_SHA256
            or pred["detectors"]["faster_rcnn"]["weights_sha256"] != FASTER_SHA256):
        raise ValueError("Expected exact frozen detector profiles and observation set")
    require_hex(pred["revision"], 40, "fusion benchmark revision")
    for i, anchor in enumerate(source["anchors"]):
        for rows in pred["arms"].values():
            if len(rows) != len(source["anchors"]) or any(rows[i][key] != value for key, value in anchor.items()):
                raise ValueError("Pending predictions are not bound to the exact input window")
    for row, i in zip(pred["rechecks"], _repeat_indices(len(source["anchors"])), strict=True):
        material = {arm: pred["arms"][arm][i] for arm in ARMS}
        if row != {"index": i, "sha256": sha256(canonical_json(material)).hexdigest()}:
            raise ValueError("Recheck is not bound to the scored arms")
    labels = load_fusion_labels(input_root, labels_path)["rows"]
    stats = summarize_fusion_holdout(pred["arms"], labels)
    output.mkdir(parents=True)
    write_json(output / "score.json", {"schema": SCORE_SCHEMA, "revision": pred["revision"],
                                       "prediction_sha256": sealed_sha256, "labels_sha256": artifact_hash(labels_path),
                                       "summary": stats})
    return {"windows": len(labels), "prediction_sha256": sealed_sha256}


def _repeat_indices(n):
    if n < 3:
        raise ValueError("At least three observation windows are needed")
    return (0, (n-1)//2, n-1)


def _new_directory(path):
    if path.exists() or path.is_symlink():
        raise ValueError("Expected a new fusion benchmark output directory")
