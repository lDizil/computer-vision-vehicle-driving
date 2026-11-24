"""Fit the visual head and a matched position/coverage-only ablation."""

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
from safetensors.torch import save_file
import torch
from torch.nn import functional as F

from hybrid_drive.experiments.bounded_io import require_hex
from hybrid_drive.experiments.fast_capture import artifact_hash
from hybrid_drive.experiments.io import write_json
from hybrid_drive.experiments.visual_vru_capture import load_vru_capture, require_new_output
from hybrid_drive.experiments.visual_vru_data import load_vru_labels
from hybrid_drive.experiments.visual_vru_model import (
    VisualVruHead, candidate_masks, cell_average_precision, flank_query, pair_logits,
)


FIT_SCHEMA = "hybrid-drive-visual-vru-fit-v1"
FIT_SEED = 1729
FIT_EPOCHS = 14
BATCH_SIZE = 8
PAIR_WEIGHT = .2
LEARNING_RATE = .0015
ARMS = ("visual", "geometry", "shuffled_camera")


@dataclass(frozen=True, eq=False)
class _Pack:
    index: dict
    receipt: dict
    appearance: torch.Tensor
    coverage: torch.Tensor
    target: torch.Tensor
    pairs: tuple
    pair_sources: tuple


def fit_vru(train_source, development_source, output, *, revision):
    require_hex(revision, 40, "fit revision")
    require_new_output(output)
    train = _read_pack(*train_source, revision=revision, split="train")
    development = _read_pack(*development_source, revision=revision, split="development")
    train_logs = sorted({anchor["recording"] for anchor in train.index["anchors"]})
    dev_logs = sorted({anchor["recording"] for anchor in development.index["anchors"]})
    if set(train_logs) & set(dev_logs) or train.receipt["vision"] != development.receipt["vision"]:
        raise ValueError("Fit and development must use separate recordings and the same vision encoder")
    records, weights = {}, {}
    for arm in ARMS[:2]:
        model = initialized_head(arm)
        values = _fit_head(model, train, development, arm)
        records[arm] = values
        weights[arm] = model.state_dict()
    output.mkdir(parents=True)
    for name, tensors in weights.items():
        path = output / f"{name}.safetensors"
        save_file(tensors, path)
        records[name]["checkpoint_sha256"] = artifact_hash(path)
    result = {
        "schema": FIT_SCHEMA, "revision": revision, "required_vision": train.receipt["vision"],
        "heldout_labels_mounted": False, "selection_metric": "development_macro_ap_over_positive_recordings",
        "train_logs": train_logs, "development_logs": dev_logs,
        "train_anchors": len(train.coverage), "development_anchors": len(development.coverage),
        "train_annotated_cells": int(train.target.sum()), "development_annotated_cells": int(development.target.sum()),
        "train_maneuver_pairs_used": train.pair_sources.count("maneuver"),
        "train_flank_pairs_used": train.pair_sources.count("flank"), "fit_seed": FIT_SEED,
        "pair_loss_weight": PAIR_WEIGHT, "note": "frozen_dino_only_two_small_heads_fit", "heads": records,
    }
    write_json(output / "fit.json", result)
    return result


def initialized_head(arm):
    if arm not in ARMS[:2]:
        raise ValueError("Only the visual and geometry control arms are trained")
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(FIT_SEED)
        return VisualVruHead(camera_input=arm == "visual")


def _read_pack(input_root, labels_path, capture_root, *, revision, split):
    index, receipt, maps, coverage = load_vru_capture(input_root, capture_root, revision)
    if index["split"] != split:
        raise ValueError("Optimization and selection use only their own predeclared splits")
    labels = load_vru_labels(input_root, labels_path)["rows"]
    target = target_cells(labels, coverage.shape)
    pairs, pair_sources = [], []
    for i, (anchor, label) in enumerate(zip(index["anchors"], labels, strict=True)):
        blocked = label["blocked_index"]
        active = candidate_masks(anchor["probe_poses"], coverage[i]) if blocked is not None else None
        if active is not None and active.flatten(1).any(dim=1).all() and (active[blocked] & target[i]).any():
            pairs.append((active, blocked))
            pair_sources.append("maneuver")
        else:
            query = flank_query(target[i], coverage[i])
            pairs.append(query[:2] if query is not None else None)
            pair_sources.append("flank" if query is not None else None)
    if not (target & (coverage > 0)).any():
        raise ValueError("Split does not contain any covered observational positives")
    return _Pack(index, receipt, maps, coverage, target, tuple(pairs), tuple(pair_sources))


def target_cells(rows, shape):
    target = torch.zeros(shape, dtype=torch.bool)
    for index, row in enumerate(rows):
        for x, y in row["visible_cells"]:
            target[index, x, y] = True
    return target


def _fit_head(model, train, development, arm):
    rng = np.random.default_rng(FIT_SEED)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    best_value, best_epoch, state = -1, 0, None
    before = deepcopy(model.state_dict())
    history = []
    for epoch in range(FIT_EPOCHS + 1):
        losses = []
        if epoch:
            model.train()
            permutation = rng.permutation(len(train.coverage))
            for offset in range(0, len(permutation), BATCH_SIZE):
                ids = permutation[offset:offset+BATCH_SIZE].tolist()
                logits = model(train.appearance[ids] if model.camera_input else None, train.coverage[ids])
                loss = _training_loss(logits, train.coverage[ids] > 0, train.target[ids], [train.pairs[i] for i in ids])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
                optimizer.step()
                losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            logits = infer_batched(model, development.appearance if model.camera_input else None, development.coverage)
            value = macro_ap(logits, development.target, development.coverage, development.index["anchors"])
        history.append({"epoch": epoch, "development_macro_ap": value,
                        "train_mean_batch_loss": sum(losses)/len(losses) if losses else None})
        if value > best_value:
            best_value, best_epoch, state = value, epoch, deepcopy(model.state_dict())
        if epoch in (0, FIT_EPOCHS) or epoch % 4 == 0:
            print(f"{arm} epoch {epoch}/{FIT_EPOCHS}: dev AP {value:.4f}", flush=True)
    model.load_state_dict(state, strict=True)
    return {"epochs": history, "checkpoint_epoch": best_epoch, "development_macro_ap": best_value,
            "changed_parameter_tensors": sum(not torch.equal(before[k], v) for k, v in state.items()),
            "parameters": sum(p.numel() for p in model.parameters())}


def _training_loss(logits, known, target, pairs):
    classes = [(known & target, -1), (known & ~target, 1)]
    terms = [F.softplus(logits[values] * sign).mean() for values, sign in classes if values.any()]
    if not terms:
        raise ValueError("No supervised cells in batch")
    loss = torch.stack(terms).mean()
    penalties = []
    for grid, pair in zip(logits, pairs, strict=True):
        if pair is not None:
            mask, blocked = pair
            values = pair_logits(grid, mask)
            penalties.append(F.softplus(values[1-blocked] - values[blocked]))
    if penalties:
        loss = loss + PAIR_WEIGHT * torch.stack(penalties).mean()
    if not torch.isfinite(loss):
        raise ValueError("Nonfinite head optimization loss")
    return loss


def infer_batched(model, appearance, coverage):
    return torch.cat([model(None if appearance is None else appearance[i:i+BATCH_SIZE], coverage[i:i+BATCH_SIZE])
                      for i in range(0,len(coverage),BATCH_SIZE)])


def macro_ap(logits, targets, coverage, anchors):
    by_recording = {}
    for i, anchor in enumerate(anchors):
        by_recording.setdefault(anchor["recording"], []).append(i)
    scores = []
    for ids in by_recording.values():
        known = coverage[ids] > 0
        ap = cell_average_precision(logits[ids][known].numpy(), targets[ids][known].numpy())
        if ap is not None:
            scores.append(ap)
    if not scores:
        raise ValueError("No annotated positive recordings available for AP")
    return sum(scores) / len(scores)
