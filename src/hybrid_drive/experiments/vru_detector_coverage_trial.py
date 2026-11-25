"""Blind current-frame replay against exactly the sealed SSDLite reference."""

from hashlib import sha256

from hybrid_drive.experiments.bounded_io import canonical_json, decode_json, read_bounded, require_hex
from hybrid_drive.experiments.box_motion_inputs import load_motion_snapshot
from hybrid_drive.experiments.fast_capture import artifact_hash
from hybrid_drive.experiments.io import write_json
from hybrid_drive.experiments.object_vru_detector import RECTIFIED_WH
from hybrid_drive.experiments.object_vru_trial import _index, _object_row
from hybrid_drive.experiments.temporal_vru_data import load_temporal_vru_labels
from hybrid_drive.experiments.triple_vru_trial import ORIGINAL_PREDICTION_SHA256, _baseline
from hybrid_drive.experiments.vru_coverage_detector import CoverageDetector, PROFILE, WEIGHTS_SHA256
from hybrid_drive.experiments.vru_detector_coverage_metrics import summarize_detector_coverage


CAPTURE_SCHEMA = "hybrid-drive-vru-detector-coverage-predictions-v1"
SCORE_SCHEMA = "hybrid-drive-vru-detector-coverage-score-v1"
RECHECK_COUNT = 5


def capture_coverage(base, reference, weights, output, *, revision):
    require_hex(revision, 40, "revision")
    _require_new(output)
    index, digest = _heldout(base)
    original = _baseline(reference, "heldout")
    detector = CoverageDetector(weights)
    rows = []
    for item, previous in zip(index["anchors"], original["rows"], strict=True):
        rows.append(_capture_row(base, index, item, previous, detector))
        if len(rows) % 16 == 0:
            print(f"Six-camera coverage replay: {len(rows)}/{len(index['anchors'])}", flush=True)
    verifier = CoverageDetector(weights)
    repeated = []
    for i in recheck_indices(len(rows)):
        fresh = _capture_row(base, index, index["anchors"][i], original["rows"][i], verifier)
        if canonical_json(fresh) != canonical_json(rows[i]):
            raise ValueError("Pinned current-frame recheck is not deterministic")
        repeated.append({"index":i, "sample_token":fresh["sample_token"],
                         "sha256":sha256(canonical_json(fresh)).hexdigest()})
    record = {"schema": CAPTURE_SCHEMA, "revision": revision, "input_sha256": digest,
              "reference_sha256": ORIGINAL_PREDICTION_SHA256["heldout"], "labels_mounted": False,
              "current_only": True, "candidate": detector.provenance, "independent_recheck": repeated, "rows": rows}
    output.mkdir(parents=True)
    target = output / "predictions.json"
    write_json(target, record)
    seal = artifact_hash(target)
    write_json(output / "seal.json", {"sha256": seal})
    return {"windows":len(rows), "detector_objects":sum(len(r["objects"]) for r in rows),
            "independently_rechecked":len(repeated), "prediction_sha256":seal}


def score_coverage(base, reference, candidate, labels, output, *, sealed_sha256):
    _require_new(output)
    require_hex(sealed_sha256, 64, "coverage prediction seal")
    raw = read_bounded(candidate, 16 * 1024**2)
    if sha256(raw).hexdigest() != sealed_sha256:
        raise ValueError("Coverage forecast does not match its pre-label seal")
    predictions = decode_json(raw)
    index, digest = _heldout(base)
    original = _baseline(reference, "heldout")
    if (predictions["schema"] != CAPTURE_SCHEMA or predictions["input_sha256"] != digest
            or predictions["reference_sha256"] != ORIGINAL_PREDICTION_SHA256["heldout"]
            or predictions["labels_mounted"] is not False or predictions["current_only"] is not True
            or predictions["candidate"]["profile"] != PROFILE
            or predictions["candidate"]["weights_sha256"] != WEIGHTS_SHA256):
        raise ValueError("Expected sealed predictions on the fixed coverage cohort")
    require_hex(predictions["revision"], 40, "coverage revision")
    rows = predictions["rows"]
    for source, anchor, row in zip(original["rows"], index["anchors"], rows, strict=True):
        _bind_observation(anchor, source, row["current_sha256"], row["ref_time_us"])
        if any(row[k] != anchor[k] for k in ("sample_token", "previous_token", "recording")):
            raise ValueError("Coverage forecast observation identity changed")
    for item, i in zip(predictions["independent_recheck"], recheck_indices(len(rows)), strict=True):
        if item != {"index": i, "sample_token": rows[i]["sample_token"],
                    "sha256": sha256(canonical_json(rows[i])).hexdigest()}:
            raise ValueError("Independent recheck is not bound to the frozen prediction rows")
    targets = load_temporal_vru_labels(base, labels)["rows"]
    summary = summarize_detector_coverage(original["rows"], rows, targets)
    output.mkdir(parents=True)
    write_json(output / "score.json", {"schema": SCORE_SCHEMA, "revision": predictions["revision"],
                                       "prediction_sha256": sealed_sha256, "reference_sha256": predictions["reference_sha256"],
                                       "labels_sha256": artifact_hash(labels), "summary": summary})
    return summary["gate"]


def recheck_indices(length):
    if length < RECHECK_COUNT:
        raise ValueError("Expected enough windows for the fixed recheck")
    return tuple((length-1)*i//(RECHECK_COUNT-1) for i in range(RECHECK_COUNT))


def _capture_row(base, index, anchor, reference, detector):
    token = anchor["sample_token"]
    snapshot = load_motion_snapshot(base, token, index["frames"][token], rectified_wh=RECTIFIED_WH)
    _bind_observation(anchor, reference, snapshot.content_sha256, snapshot.ref_time_us)
    detections, boxes = detector(snapshot)
    return {**anchor, "ref_time_us": snapshot.ref_time_us, "current_sha256": snapshot.content_sha256,
            "objects": [_object_row(x) for x in detections], "boxes_by_camera": boxes}


def _bind_observation(anchor, reference, digest, stamp):
    require_hex(digest, 64, "current camera snapshot hash")
    if (any(reference[key] != anchor[key] for key in ("sample_token", "previous_token", "recording"))
            or stamp != reference["ref_time_us"] or digest != reference["current_sha256"]):
        raise ValueError("Candidate and reference must see the exact same rectified current frame")


def _heldout(base):
    index, digest = _index(base)
    if index["split"] != "heldout" or len(index["anchors"]) != 115:
        raise ValueError("Coverage uses all 115 heldout windows, without threshold selection")
    return index, digest


def _require_new(path):
    if path.exists() or path.is_symlink():
        raise ValueError("Expected new coverage result directory")
