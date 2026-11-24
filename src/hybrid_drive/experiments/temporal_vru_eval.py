"""Run four blind temporal arms, seal their grids, and only then open labels."""

from hashlib import sha256

from safetensors.torch import load, save_file
import torch

from hybrid_drive.experiments.bounded_io import decode_json, read_bounded, require_hex
from hybrid_drive.experiments.fast_capture import artifact_hash
from hybrid_drive.experiments.io import write_json
from hybrid_drive.experiments.temporal_vru_capture import load_temporal_capture
from hybrid_drive.experiments.temporal_vru_data import load_temporal_vru_inputs, load_temporal_vru_labels
from hybrid_drive.experiments.temporal_vru_metrics import summarize_motion
from hybrid_drive.experiments.temporal_vru_model import ARMS, HORIZONS
from hybrid_drive.experiments.temporal_vru_training import FIT_ARMS, FIT_SCHEMA, fresh_head, infer_grid
from hybrid_drive.experiments.visual_vru_capture import require_new_output
from hybrid_drive.experiments.visual_vru_eval import donor_indices
from hybrid_drive.experiments.visual_vru_model import GRID_SHAPE
from hybrid_drive.experiments.visual_vru_training import infer_batched as predict_present, initialized_head


PREDICTION_SCHEMA = "hybrid-drive-temporal-vru-prediction-v1"
SCORE_SCHEMA = "hybrid-drive-temporal-vru-score-v1"
PERSIST_REVISION = "bf63461679cac5a57271715c4ecac2da4345afc8"
PERSIST_FIT_SHA = "dd92b5b14e5554932e2122f396302528606c4ea3586a4c03d45f02096ec0fb6a"
PERSIST_HEAD_SHA = "345c6b7aa2dea79fab7497e7f958614491d10d772abd3cc9f108bc43467bf51c"


@torch.no_grad()
def predict_temporal_vru(input_root, features_root, fit_root, persistence_root, output, *, revision):
    require_hex(revision, 40, "prediction revision")
    require_new_output(output)
    index, capture, grids = load_temporal_capture(input_root, features_root, revision)
    fit_raw = read_bounded(fit_root / "fit.json", 256 * 1024)
    fit = decode_json(fit_raw)
    recording_ids = {a["recording"] for a in index["anchors"]}
    if (index["split"] != "heldout" or fit["schema"] != FIT_SCHEMA or fit["revision"] != revision
            or fit["frozen_vision"] != capture["vision"] or fit["heldout_data_mounted"] is not False
            or fit["horizons_s"] != list(HORIZONS) or recording_ids & set(fit["train_recordings"] + fit["development_recordings"])):
        raise ValueError("Expected independent recordings and the exact development-selected fit")
    heads, digests = {}, {}
    for arm in FIT_ARMS:
        raw = read_bounded(fit_root / f"{arm}.safetensors", 2 * 1024**2)
        digests[arm] = sha256(raw).hexdigest()
        if digests[arm] != fit["arms"][arm]["checkpoint_sha256"]:
            raise ValueError("Forecast checkpoint no longer matches the exact fit")
        head = fresh_head().eval()
        head.load_state_dict(load(raw), strict=True)
        heads[arm] = head
    persisted = _frozen_present(persistence_root, capture["vision"], recording_ids)
    now_cov = grids["current_coverage"]
    present = predict_present(persisted, grids["current"], now_cov)
    donors = donor_indices(index["anchors"])
    predictions = {
        "temporal": infer_grid(heads["temporal"], grids, "temporal"),
        "single_frame": infer_grid(heads["single_frame"], grids, "single_frame"),
        "shuffled_past": infer_grid(heads["temporal"], grids, "shuffled_past", donors=donors),
        "persist_now": present[:, None].expand(-1, len(HORIZONS), -1, -1).contiguous(),
    }
    replaced = (grids["past_coverage"] > 0) & (grids["past_coverage"][donors] > 0)
    output.mkdir(parents=True)
    path = output / "margins.safetensors"
    save_file({**predictions, "coverage": now_cov}, path)
    receipt = {"schema": PREDICTION_SCHEMA, "revision": revision, "input_sha256": capture["input_sha256"],
               "fit_sha256": sha256(fit_raw).hexdigest(), "persist_head_sha256": PERSIST_HEAD_SHA,
               "checkpoint_hashes": digests, "annotations_mounted": False, "horizons_s": list(HORIZONS),
               "arms": list(ARMS), "margins_sha256": artifact_hash(path),
               "shuffled_cells": int(replaced.sum()), "history_covered_cells": int((grids["past_coverage"] > 0).sum()),
               "rows": [{**a, "donor_previous_token": capture["rows"][j]["previous_token"]}
                        for a, j in zip(capture["rows"], donors, strict=True)]}
    meta = output / "predictions.json"
    write_json(meta, receipt)
    seal = artifact_hash(meta)
    write_json(output / "seal.json", {"prediction_sha256": seal})
    return {"heldout_anchors": len(index["anchors"]), "prediction_sha256": seal}


def _frozen_present(root, vision, holdout):
    raw_fit = read_bounded(root / "fit.json", 256 * 1024)
    raw_weights = read_bounded(root / "visual.safetensors", 2 * 1024**2)
    if sha256(raw_fit).hexdigest() != PERSIST_FIT_SHA or sha256(raw_weights).hexdigest() != PERSIST_HEAD_SHA:
        raise ValueError("Persistence must use the original immutable #49 weights and receipt")
    fit = decode_json(raw_fit)
    if (fit["revision"] != PERSIST_REVISION or fit["required_vision"] != vision
            or holdout & set(fit["train_logs"] + fit["development_logs"])):
        raise ValueError("The original visual head must be independent of the test recordings")
    model = initialized_head("visual").eval()
    model.load_state_dict(load(raw_weights), strict=True)
    return model


def score_temporal_vru(input_root, labels_path, predictions, output, *, sealed_sha256):
    require_hex(sealed_sha256, 64, "prediction seal")
    require_new_output(output)
    raw = read_bounded(predictions / "predictions.json", 512 * 1024)
    if sha256(raw).hexdigest() != sealed_sha256:
        raise ValueError("Temporal predictions do not match the recorded seal")
    predicted = decode_json(raw)
    index = load_temporal_vru_inputs(input_root)
    if (predicted["schema"] != PREDICTION_SCHEMA or index["split"] != "heldout"
            or predicted["annotations_mounted"] is not False or predicted["arms"] != list(ARMS)
            or predicted["input_sha256"] != artifact_hash(input_root / "index.json")
            or predicted["horizons_s"] != list(HORIZONS) or predicted["persist_head_sha256"] != PERSIST_HEAD_SHA):
        raise ValueError("Expected a label-free forecast on the exact heldout observations")
    require_hex(predicted["revision"], 40, "forecast revision")
    anchors = index["anchors"]
    for row, anchor, donor in zip(predicted["rows"], anchors, donor_indices(anchors), strict=True):
        if (any(row[key] != anchor[key] for key in ("sample_token", "previous_token", "recording"))
                or row["donor_previous_token"] != anchors[donor]["previous_token"]):
            raise ValueError("Temporal forecast row or history donor no longer matches its observations")
        require_hex(row["current_sha256"], 64, "current camera observation")
        require_hex(row["past_sha256"], 64, "past camera observation")
    margin_bytes = read_bounded(predictions / "margins.safetensors", 8 * 1024**2)
    if sha256(margin_bytes).hexdigest() != predicted["margins_sha256"]:
        raise ValueError("Temporal score maps do not match the sealed receipt")
    arrays = load(margin_bytes)
    n = len(anchors)
    if set(arrays) != {*ARMS, "coverage"} or any(
        x.dtype != torch.float32 or not torch.isfinite(x).all() or
        x.shape != ((n, *GRID_SHAPE) if key == "coverage" else (n, len(HORIZONS), *GRID_SHAPE))
        for key, x in arrays.items()
    ):
        raise ValueError("Temporal prediction tensors are not valid")
    coverage = arrays.pop("coverage")
    if (coverage < 0).any() or (coverage > 1).any():
        raise ValueError("Temporal support outside [0,1]")
    labels = load_temporal_vru_labels(input_root, labels_path)["rows"]
    result = summarize_motion(anchors, labels, {k: v.numpy() for k, v in arrays.items()}, coverage.numpy())
    result.update({"schema": SCORE_SCHEMA, "revision": predicted["revision"], "prediction_sha256": sealed_sha256,
                   "targets_sha256": artifact_hash(labels_path)})
    output.mkdir(parents=True)
    write_json(output / "score.json", result)
    return result
