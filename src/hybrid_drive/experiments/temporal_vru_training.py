"""Fit equal-capacity current-only and temporal VRU heads on the same targets."""

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
from safetensors.torch import save_file
import torch
from torch.nn import functional as F

from hybrid_drive.experiments.bounded_io import require_hex
from hybrid_drive.experiments.fast_capture import artifact_hash
from hybrid_drive.experiments.io import write_json
from hybrid_drive.experiments.temporal_vru_capture import load_temporal_capture
from hybrid_drive.experiments.temporal_vru_data import load_temporal_vru_labels
from hybrid_drive.experiments.temporal_vru_model import HORIZONS, TemporalVruHead, supervised_targets
from hybrid_drive.experiments.visual_vru_capture import require_new_output
from hybrid_drive.experiments.visual_vru_model import cell_average_precision


FIT_SCHEMA = "hybrid-drive-temporal-vru-fit-v1"
FIT_ARMS = ("temporal", "single_frame")
FIT_SEED = 2407
FIT_EPOCHS = 14
BATCH_SIZE = 8
LEARNING_RATE = .0013
MOTION_WEIGHT = .24
HORIZON_LOSS_WEIGHTS = (.50, .75, 1.)


@dataclass(frozen=True, eq=False)
class FitData:
    index: dict
    receipt: dict
    grids: dict
    target: torch.Tensor
    pairs: list


def fit_temporal_vru(train_source, development_source, output, *, revision):
    require_hex(revision, 40, "fit revision")
    require_new_output(output)
    train = _load_fit_data(*train_source, revision=revision, split="train")
    development = _load_fit_data(*development_source, revision=revision, split="development")
    train_recordings = sorted({row["recording"] for row in train.index["anchors"]})
    dev_recordings = sorted({row["recording"] for row in development.index["anchors"]})
    if set(train_recordings) & set(dev_recordings) or train.receipt["vision"] != development.receipt["vision"]:
        raise ValueError("Optimization and validation must use distinct recordings and one frozen camera encoder")
    histories, weights = {}, {}
    for arm in FIT_ARMS:
        model = fresh_head()
        histories[arm] = _fit_arm(model, train, development, arm)
        weights[arm] = model.state_dict()
    output.mkdir(parents=True)
    for arm, tensors in weights.items():
        path = output / f"{arm}.safetensors"
        save_file(tensors, path)
        histories[arm]["checkpoint_sha256"] = artifact_hash(path)
    result = {
        "schema": FIT_SCHEMA, "revision": revision, "frozen_vision": train.receipt["vision"],
        "train_recordings": train_recordings, "development_recordings": dev_recordings,
        "train_anchors": len(train.target), "development_anchors": len(development.target),
        "supervised_train_cells": [int(train.target[:, j].sum()) for j in range(len(HORIZONS))],
        "train_motion_pairs_by_future_horizon": [sum(h == j for row in train.pairs for h, _, _ in row) for j in (1, 2)],
        "seed": FIT_SEED, "motion_loss_weight": MOTION_WEIGHT, "horizons_s": list(HORIZONS),
        "selection_metric": "development_mean_future_recording_macro_ap", "heldout_data_mounted": False,
        "arms": histories,
    }
    write_json(output / "fit.json", result)
    return result


def fresh_head():
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(FIT_SEED)
        return TemporalVruHead()


def _load_fit_data(input_root, labels_path, features_root, *, revision, split):
    data, receipt, grids = load_temporal_capture(input_root, features_root, revision)
    if data["split"] != split:
        raise ValueError("Fit and checkpoint selection require their declared splits")
    rows = load_temporal_vru_labels(input_root, labels_path)["rows"]
    targets, pairs = supervised_targets(rows, grids["current_coverage"])
    if not (targets & (grids["current_coverage"] > 0)[:, None]).any():
        raise ValueError("Missing covered VRU supervision in the split")
    return FitData(data, receipt, grids, targets, pairs)


def _fit_arm(model, train, dev, arm):
    rng = np.random.default_rng(FIT_SEED)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    initial = deepcopy(model.state_dict())
    best, selected, state = -1., 0, None
    epochs = []
    for epoch in range(FIT_EPOCHS + 1):
        loss_values = []
        if epoch:
            model.train()
            order = rng.permutation(len(train.target)).tolist()
            for start in range(0, len(order), BATCH_SIZE):
                idx = order[start:start + BATCH_SIZE]
                output = model(*camera_batch(train.grids, idx, arm))
                loss = forecast_loss(output, train.target[idx], train.grids["current_coverage"][idx] > 0,
                                     [train.pairs[i] for i in idx])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
                optimizer.step()
                loss_values.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            scores = infer_grid(model, dev.grids, arm)
            ap = _validation_scores(scores, dev)
        objective = (ap[1] + ap[2]) / 2
        epochs.append({"epoch": epoch, "train_loss": float(np.mean(loss_values)) if loss_values else None,
                       "development_macro_ap": ap, "validation_objective": objective})
        if objective > best:
            best, selected, state = objective, epoch, deepcopy(model.state_dict())
        if epoch in (0, FIT_EPOCHS) or epoch % 4 == 0:
            print(f"{arm} epoch {epoch}/{FIT_EPOCHS}: dev AP0.5={ap[1]:.4f}, AP1={ap[2]:.4f}", flush=True)
    model.load_state_dict(state, strict=True)
    return {"selected_epoch": selected, "selected_objective": best, "epochs": epochs,
            "changed_parameters": sum(not torch.equal(initial[key], value) for key, value in state.items()),
            "parameter_count": sum(parameter.numel() for parameter in model.parameters())}


def camera_batch(grids, ids, arm, *, history_ids=None):
    now, now_coverage = grids["current"][ids], grids["current_coverage"][ids]
    if arm == "single_frame":
        if history_ids is not None:
            raise ValueError("The present-only arm takes no observations from history")
        past, past_coverage = now, now_coverage
    else:
        if arm not in ("temporal", "shuffled_past") or (history_ids is not None) != (arm == "shuffled_past"):
            raise ValueError("Expected temporal or recording-shuffled observed history")
        past, past_coverage = grids["past_in_current"][ids], grids["past_coverage"][ids]
        if history_ids is not None:
            usable = (past_coverage > 0) & (grids["past_coverage"][history_ids] > 0)
            past = torch.where(usable[:, None], grids["past_in_current"][history_ids], past)
    return now, past, now_coverage, past_coverage, grids["history_s"][ids]


def infer_grid(model, grids, arm, *, donors=None):
    scores = []
    for i in range(0, len(grids["current"]), BATCH_SIZE):
        idx = slice(i, i+BATCH_SIZE)
        scores.append(model(*camera_batch(grids, idx, arm, history_ids=None if donors is None else donors[idx])))
    return torch.cat(scores)


def forecast_loss(scores, target, known, pairs):
    losses = []
    for j, w in enumerate(HORIZON_LOSS_WEIGHTS):
        positive, negative = known & target[:, j], known & ~target[:, j]
        evidence = [F.softplus(sign * scores[:, j][mask]).mean()
                    for mask, sign in ((positive, -1.), (negative, 1.)) if mask.any()]
        if not evidence:
            raise ValueError("No camera-covered training cells")
        losses.append(w * torch.stack(evidence).mean())
    loss = sum(losses) / sum(HORIZON_LOSS_WEIGHTS)
    changed = [F.softplus(grid[j][origin] - grid[j][destination])
               for grid, moves in zip(scores, pairs, strict=True) for j, origin, destination in moves]
    if changed:
        loss = loss + MOTION_WEIGHT * torch.stack(changed).mean()
    if not torch.isfinite(loss):
        raise ValueError("Nonfinite motion learning loss")
    return loss


def _validation_scores(logits, data):
    grouped = {}
    for i, anchor in enumerate(data.index["anchors"]):
        grouped.setdefault(anchor["recording"], []).append(i)
    results = []
    for horizon in range(len(HORIZONS)):
        ap = []
        for ids in grouped.values():
            known = data.grids["current_coverage"][ids] > 0
            result = cell_average_precision(logits[ids, horizon][known].numpy(), data.target[ids, horizon][known].numpy())
            if result is not None:
                ap.append(result)
        if not ap:
            raise ValueError("No covered validation positives at a planned horizon")
        results.append(sum(ap) / len(ap))
    return results
