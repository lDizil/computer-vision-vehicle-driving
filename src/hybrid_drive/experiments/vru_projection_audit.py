"""Record current-frame detector evidence and bind it to the frozen VRU replay."""

from dataclasses import asdict
from hashlib import sha256

from hybrid_drive.experiments.bounded_io import canonical_json, read_bounded, require_hex
from hybrid_drive.experiments.io import write_json
from hybrid_drive.experiments.vru_chain_capture import (
    ensure_new_output, index_sha, load_chain_observations, recheck, repeat_indices, save_sealed, sealed_read, validate_anchor,
)
from hybrid_drive.experiments.vru_chain_rescue import SINGLES
from hybrid_drive.experiments.vru_fusion_holdout_data import LABEL_LIMIT, load_fusion_inputs, load_fusion_labels


CAPTURE_SCHEMA = "hybrid-drive-vru-projection-audit-v1"
SCORE_SCHEMA = "hybrid-drive-vru-projection-audit-score-v1"


def capture_projection_audit(input_root, ssd_weights, faster_weights, output, *, revision):
    from hybrid_drive.experiments.box_motion_inputs import load_motion_snapshot
    from hybrid_drive.experiments.object_vru_detector import LocalVruDetector, RECTIFIED_WH
    from hybrid_drive.experiments.vru_coverage_detector import CoverageDetector

    require_hex(revision, 40, "projection audit revision")
    ensure_new_output(output)
    source = load_fusion_inputs(input_root)
    models = {"ssdlite": LocalVruDetector(ssd_weights), "faster_rcnn": CoverageDetector(faster_weights)}
    rows = []
    for anchor in source["anchors"]:
        rows.append(_trace_window(input_root, source, anchor, models, load_motion_snapshot, RECTIFIED_WH))
        if len(rows) % 8 == 0:
            print(f"Projection audit: {len(rows)}/{len(source['anchors'])} current frames", flush=True)
    checks = []
    for i in repeat_indices(rows):
        repeated = _trace_window(input_root, source, source["anchors"][i], models, load_motion_snapshot, RECTIFIED_WH)
        if canonical_json(repeated) != canonical_json(rows[i]):
            raise ValueError("Projection audit did not repeat on identical current images")
        checks.append(recheck(rows[i], i))
    digest = save_sealed(output, "trace.json", {
        "schema": CAPTURE_SCHEMA, "revision": revision, "input_sha256": index_sha(input_root),
        "capture_scope": "current_images_only", "labels_mounted": False, "rectified_wh": RECTIFIED_WH,
        "detectors": {name: model.provenance for name, model in models.items()}, "rows": rows, "rechecks": checks,
    })
    return {"windows": len(rows), "rechecked": len(checks), "trace_sha256": digest}


def _trace_window(root, source, anchor, models, load_snapshot, rectified_wh):
    token = anchor["sample_token"]
    current = load_snapshot(root, token, source["frames"][token], rectified_wh=rectified_wh)
    profiles = {}
    for name, model in models.items():
        objects, cameras = model.audit(current)
        profiles[name] = {"objects": [asdict(obj) for obj in objects], "cameras": cameras}
    return {**anchor, "current_sha256": current.content_sha256, "ref_time_us": current.ref_time_us, "profiles": profiles}


def score_projection_audit(input_root, trace_path, reference_path, truth_path, labels_path, output, *,
                           trace_sha256, reference_sha256):
    from hybrid_drive.experiments.vru_projection_metrics import summarize_projection_audit
    from hybrid_drive.experiments.vru_projection_truth import load_projection_truth

    ensure_new_output(output)
    captured = sealed_read(trace_path, trace_sha256)
    index, input_sha = load_fusion_inputs(input_root), index_sha(input_root)
    reference = load_chain_observations(index, input_sha, reference_path, reference_sha256)
    count = _verify_reference(captured, reference, index, input_sha)
    truth = load_projection_truth(input_root, labels_path, truth_path)
    labels = load_fusion_labels(input_root, labels_path)
    results = summarize_projection_audit(captured["rows"], truth["rows"], labels["rows"], index["frames"])
    output.mkdir(parents=True)
    write_json(output / "score.json", {
        "schema": SCORE_SCHEMA, "input_sha256": input_sha, "trace_sha256": trace_sha256,
        "reference_capture_sha256": reference_sha256, "capture_revision": captured["revision"],
        "reference_current_objects_identical": count, "labels_sha256": truth["labels_sha256"],
        "projection_truth_sha256": sha256(read_bounded(truth_path, LABEL_LIMIT)).hexdigest(),
        "summary": results,
    })
    return {"windows": len(labels["rows"]), "checked_reference_objects": count, "trace_sha256": trace_sha256}


def _verify_reference(capture, reference, index, input_sha):
    if (capture["schema"] != CAPTURE_SCHEMA or capture["input_sha256"] != input_sha
            or capture["capture_scope"] != "current_images_only" or capture["labels_mounted"] is not False
            or capture["rectified_wh"] != [768, 432] or capture["detectors"] != reference["detectors"]):
        raise ValueError("Expected the two pinned profiles replayed on the exact reference images")
    require_hex(capture["revision"], 40, "projection capture revision")
    rows = capture["rows"]
    if capture["rechecks"] != [recheck(rows[i], i) for i in repeat_indices(rows)]:
        raise ValueError("Capture rechecks are not bound to the scored traces")
    count = 0
    for anchor, row, ref in zip(index["anchors"], rows, reference["windows"], strict=True):
        validate_anchor(anchor, row)
        saved = ref["observations"]["current"]
        if (row["current_sha256"] != saved["sha256"] or set(row["profiles"]) != set(SINGLES)
                or row["ref_time_us"] != index["frames"][anchor["sample_token"]]["ref_time_us"]):
            raise ValueError("Detector input changed relative to the frozen replay")
        for name, profile in row["profiles"].items():
            if profile["objects"] != saved["objects"][name]:
                raise ValueError("Tracing changed a projected detection, feature or ID")
            count += len(profile["objects"])
    return count
