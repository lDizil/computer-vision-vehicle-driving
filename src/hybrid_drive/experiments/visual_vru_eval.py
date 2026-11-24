"""Evaluate calibrated visual cells and paired diagnostic footprints on new logs."""

from collections import Counter, defaultdict
from hashlib import sha256

from safetensors.torch import load, save_file
import torch

from hybrid_drive.experiments.bounded_io import decode_json, read_bounded, require_hex
from hybrid_drive.experiments.fast_capture import artifact_hash
from hybrid_drive.experiments.io import write_json
from hybrid_drive.experiments.visual_vru_capture import load_vru_capture, require_new_output
from hybrid_drive.experiments.visual_vru_data import load_vru_inputs, load_vru_labels
from hybrid_drive.experiments.visual_vru_model import (
    GRID_SHAPE, PAIR_TIE, candidate_masks, cell_average_precision, flank_query, pair_logits,
)
from hybrid_drive.experiments.visual_vru_training import ARMS, FIT_SCHEMA, infer_batched, initialized_head, target_cells


PREDICTION_SCHEMA = "hybrid-drive-visual-vru-predictions-v1"
SCORE_SCHEMA = "hybrid-drive-visual-vru-score-v1"


@torch.no_grad()
def predict_vru(input_root, features_root, fit_root, output, *, revision):
    require_hex(revision, 40, "prediction revision")
    require_new_output(output)
    index, cache, maps, coverage = load_vru_capture(input_root, features_root, revision)
    fit_raw = read_bounded(fit_root / "fit.json", 256 * 1024)
    fit = decode_json(fit_raw)
    if (index["split"] != "heldout" or fit["schema"] != FIT_SCHEMA or fit["revision"] != revision
            or fit["heldout_labels_mounted"] is not False or fit["required_vision"] != cache["vision"]
            or ({a["recording"] for a in index["anchors"]} & set(fit["train_logs"] + fit["development_logs"]))):
        raise ValueError("Expected an independent heldout capture and the development-only fit")
    heads, checkpoints = {}, {}
    for arm in ARMS[:2]:
        data = read_bounded(fit_root / f"{arm}.safetensors", 2 * 1024**2)
        checkpoints[arm] = sha256(data).hexdigest()
        if checkpoints[arm] != fit["heads"][arm]["checkpoint_sha256"]:
            raise ValueError("Head tensor artifact no longer matches its fit")
        model = initialized_head(arm).eval()
        model.load_state_dict(load(data), strict=True)
        heads[arm] = model
    indices = donor_indices(index["anchors"])
    predictions = {
        "visual": infer_batched(heads["visual"], maps, coverage),
        "geometry": infer_batched(heads["geometry"], None, coverage),
        "shuffled_camera": infer_batched(heads["visual"], maps[indices], coverage),
        "coverage": coverage,
    }
    output.mkdir(parents=True)
    scores = output / "scores.safetensors"
    save_file(predictions, scores)
    receipt = {
        "schema": PREDICTION_SCHEMA, "revision": revision, "checkpoint_hashes": checkpoints,
        "fit_sha256": sha256(fit_raw).hexdigest(), "input_sha256": cache["input_sha256"],
        "heldout_labels_mounted": False, "scores_sha256": artifact_hash(scores),
        "rows": [{**row, "donor_sample_token": cache["rows"][i]["sample_token"]}
                 for row, i in zip(cache["rows"], indices, strict=True)],
    }
    path = output / "predictions.json"
    write_json(path, receipt)
    seal = artifact_hash(path)
    write_json(output / "seal.json", {"prediction_sha256": seal})
    return {"heldout_anchors": len(coverage), "prediction_sha256": seal}


def donor_indices(anchors):
    by_recording = defaultdict(list)
    for i, row in enumerate(anchors):
        by_recording[row["recording"]].append(i)
    keys = sorted(by_recording)
    if len(keys) < 2:
        raise ValueError("Matched camera perturbation requires a separate recording")
    donors = dict(zip(keys, keys[1:]+keys[:1], strict=True))
    assigned = [0] * len(anchors)
    for key, ids in by_recording.items():
        other = by_recording[donors[key]]
        for rank, index in enumerate(ids):
            assigned[index] = other[rank % len(other)]
    return assigned


def score_vru(input_root, labels_path, predictions, output, *, sealed_sha256):
    require_hex(sealed_sha256, 64, "prediction seal")
    require_new_output(output)
    raw = read_bounded(predictions / "predictions.json", 256 * 1024)
    if sha256(raw).hexdigest() != sealed_sha256:
        raise ValueError("Prediction receipt differs from the recorded seal")
    sealed = decode_json(raw)
    index = load_vru_inputs(input_root)
    if (sealed["schema"] != PREDICTION_SCHEMA or index["split"] != "heldout"
            or sealed["heldout_labels_mounted"] is not False
            or sealed["input_sha256"] != artifact_hash(input_root / "index.json")):
        raise ValueError("Expected a label-free prediction against this heldout observation index")
    require_hex(sealed["revision"], 40, "prediction revision")
    anchors = index["anchors"]
    for row, anchor, donor in zip(sealed["rows"], anchors, donor_indices(anchors), strict=True):
        if ((row["sample_token"],row["recording"]) != (anchor["sample_token"], anchor["recording"])
                or row["donor_sample_token"] != anchors[donor]["sample_token"]):
            raise ValueError("Frozen row/donor identity differs from the observations")
        require_hex(row["snapshot_sha256"], 64, "snapshot digest")
    scores = read_bounded(predictions / "scores.safetensors", 4 * 1024**2)
    if sha256(scores).hexdigest() != sealed["scores_sha256"]:
        raise ValueError("Prediction grids differ from the sealed receipt")
    tensors = load(scores)
    if set(tensors) != {*ARMS, "coverage"} or any(
        value.shape != (len(anchors), *GRID_SHAPE) or value.dtype != torch.float32 or not torch.isfinite(value).all()
        for value in tensors.values()
    ):
        raise ValueError("Invalid sealed prediction grids")
    coverage = tensors.pop("coverage")
    if (coverage < 0).any() or (coverage > 1).any():
        raise ValueError("Prediction coverage lies outside [0,1]")
    # Neither the split metadata nor the target file participates in fitting/inference.
    labels = load_vru_labels(input_root, labels_path)["rows"]
    result = summarize(anchors, labels, tensors, coverage)
    result.update({"schema": SCORE_SCHEMA, "revision": sealed["revision"], "prediction_sha256": sealed_sha256,
                   "scores_sha256": sealed["scores_sha256"], "target_sha256": artifact_hash(labels_path)})
    output.mkdir(parents=True)
    write_json(output / "score.json", result)
    return result


def summarize(anchors, rows, tensors, coverage):
    target = target_cells(rows, coverage.shape)
    known = coverage > 0
    totals = {"frames": len(anchors), "frames_with_annotated_target": int(target.flatten(1).any(1).sum()),
              "frames_with_covered_target": int((known & target).flatten(1).any(1).sum()),
              "annotated_cells": int(target.sum()), "covered_annotated_cells": int((known & target).sum()),
              "covered_cells": int(known.sum())}
    by_log = defaultdict(list)
    for i, anchor in enumerate(anchors):
        by_log[anchor["recording"]].append(i)
    cells = {}
    for arm, grid in tensors.items():
        values = {name: cell_average_precision(grid[ids][known[ids]].numpy(),target[ids][known[ids]].numpy())
                  for name, ids in by_log.items()}
        nonempty = [score for score in values.values() if score is not None]
        cells[arm] = {"micro_ap": cell_average_precision(grid[known].numpy(), target[known].numpy()),
                      "recording_macro_ap": sum(nonempty)/len(nonempty) if nonempty else None,
                      "by_recording": values}
    pairs, flank_pairs = [], []
    for i, (anchor, label) in enumerate(zip(anchors, rows, strict=True)):
        query = flank_query(target[i], coverage[i])
        if query is not None:
            masks, side, band = query
            flank_pairs.append({"recording": anchor["recording"], "sample_token": anchor["sample_token"],
                                "occupied": side, "band_rows": band, "margins": _margins(tensors, i, masks)})
        if anchor["probe_poses"] is None:
            continue
        masks = candidate_masks(anchor["probe_poses"], coverage[i])
        pairs.append({"recording": anchor["recording"], "sample_token": anchor["sample_token"],
                      "current_blocked": label["blocked_index"], "future_blocked": label["future_blocked_index"],
                      "margins": _margins(tensors, i, masks), "covered_candidate_cells": [int(m.sum()) for m in masks]})
    current = pair_summary(pairs, "current_blocked")
    future = pair_summary(pairs, "future_blocked")
    flank = pair_summary(flank_pairs, "occupied")
    macro = {arm: value["recording_macro_ap"] for arm, value in cells.items()}
    supported_recordings = sum(score is not None for score in cells["visual"]["by_recording"].values())
    support = supported_recordings >= 3 and totals["frames_with_covered_target"] >= 32
    gain = (macro["visual"] - max(macro["geometry"], macro["shuffled_camera"])
            if all(x is not None for x in macro.values()) else None)
    cell_status = ("visual_localization_signal" if gain is not None and gain >= .10 else "visual_localization_not_established")
    if not support:
        cell_status = "insufficient_support"
    return {"counts": totals, "cell_metrics": cells, "current_pair_metrics": current, "future_pair_metrics": future,
            "flank_pair_metrics": flank, "flank_pair_gate": _pair_gate(flank), "flank_pairs": flank_pairs,
            "cell_gate": {"status": cell_status, "gain_over_strongest_control": gain, "minimum_gain": .10,
                          "recordings_with_covered_targets": supported_recordings,
                          "minimum_recordings": 3, "minimum_frames_with_covered_targets": 32},
            "current_pair_gate": _pair_gate(current), "future_pair_gate": _pair_gate(future), "pairs": pairs}


def _margins(tensors, index, masks):
    result = {}
    for arm, grid in tensors.items():
        logits = pair_logits(grid[index], masks)
        result[arm] = None if logits is None else logits.tolist()
    return result


def pair_summary(rows, label_key):
    actual = [r for r in rows if r[label_key] is not None]
    sides = Counter(x[label_key] for x in actual)
    choices = {}
    for method in (*ARMS, "always_left", "always_right"):
        outcomes = Counter(correct=0, incorrect=0, tied=0, unobserved=0)
        correct_sides = Counter()
        for row in actual:
            truth = row[label_key]
            if method == "always_left":
                value = [1.,0.]
            elif method == "always_right":
                value = [0.,1.]
            else:
                value = row["margins"][method]
            if value is None:
                outcomes["unobserved"] += 1
            elif abs(value[0] - value[1]) <= PAIR_TIE:
                outcomes["tied"] += 1
            elif int(value[1] > value[0]) == truth:
                outcomes["correct"] += 1
                correct_sides[truth] += 1
            else:
                outcomes["incorrect"] += 1
        balanced = (correct_sides[0]/sides[0]+correct_sides[1]/sides[1])/2 if sides[0] and sides[1] else None
        choices[method] = {**outcomes, "balanced_accuracy": balanced}
    return {"pairs": len(actual), "recordings": len({r["recording"] for r in actual}),
            "left_blocked": sides[0], "right_blocked": sides[1], "arms": choices}


def _pair_gate(summary):
    spec = {"minimum_pairs": 10, "minimum_recordings": 3, "minimum_per_side": 3, "minimum_gain": .10}
    if (summary["pairs"] < spec["minimum_pairs"] or summary["recordings"] < spec["minimum_recordings"]
            or min(summary["left_blocked"], summary["right_blocked"]) < spec["minimum_per_side"]):
        return {"status": "insufficient_support", **spec}
    a = summary["arms"]
    gain = a["visual"]["balanced_accuracy"] - max(a[m]["balanced_accuracy"] for m in a if m != "visual")
    return {"status": "visual_pair_signal" if gain >= spec["minimum_gain"] else "visual_pair_gain_not_established",
            "gain_over_strongest_control": gain, **spec}
