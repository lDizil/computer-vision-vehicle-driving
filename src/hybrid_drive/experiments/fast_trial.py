"""Fit on known observations, freeze on dev, predict without labels, then score."""

from __future__ import annotations

from dataclasses import asdict
from math import isfinite
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from hybrid_drive.contracts.trajectory import Pose2D
from hybrid_drive.experiments.bounded_io import require_hex
from hybrid_drive.experiments.fast_capture import artifact_hash, load_capture, load_json
from hybrid_drive.experiments.fast_metrics import assess_frames, supported_targets
from hybrid_drive.experiments.io import write_json
from hybrid_drive.fast.forecast_proposals import ForecastObject, propose_from_forecast
from hybrid_drive.fast.metric_forecast import MetricForecastHead, decode_boxes, forecasting_loss, raster_targets


ARMS = ("vision", "geometry_only")
THRESHOLDS = (.035, .06, .1, .15, .2, .275, .35, .45, .6)
SEED = 213


def fit_fast(capture: Path, train_targets: Path, dev_targets: Path, output: Path, *,
             revision: str, epochs: int = 18) -> None:
    require_hex(revision, 40, "revision")
    if not 1 <= epochs <= 40 or output.exists():
        raise ValueError("Expected a new output and 1 to 40 epochs")
    if load_json(capture / "capture.json")["revision"] != revision:
        raise ValueError("Capture revision does not match training revision")
    train = load_capture(capture, "train")
    dev = load_capture(capture, "dev")
    train_labels = _load_targets(train_targets, train)
    dev_labels = _load_targets(dev_targets, dev)
    target_tensors = tuple(torch.as_tensor(np.stack(values)) for values in zip(*[
        raster_targets(train_labels[o["sample_token"]], mask, train.grid)
        for o, mask in zip(train.observations, train.coverage)
    ]))
    if not target_tensors[0].any() or not target_tensors[1].any():
        raise ValueError("Training requires at least one visible current and next-step target")
    dev_truth = _truth_for_capture(dev, dev_labels)
    width = train.values.shape[1]
    output.mkdir(parents=True)
    fitted = {}
    for arm in ARMS:
        model, losses = _fit_arm(train, target_tensors, arm, epochs)
        outputs = [_forward_frame(model, dev, i, arm) for i in range(len(dev.observations))]
        thresholds = {}
        for t in THRESHOLDS:
            frames = [(list(map(asdict, decode_boxes(o, torch.from_numpy(dev.coverage[i]), dev.grid, threshold=t))),
                       t_boxes, dt) for i, (o,(t_boxes,dt)) in enumerate(zip(outputs, dev_truth))]
            thresholds[t] = assess_frames(frames)
        chosen = max(THRESHOLDS, key=lambda t: (thresholds[t]["f1"], t))
        path = output / (arm + ".safetensors")
        save_file(model.state_dict(), path)
        fitted[arm] = {
            "threshold": chosen, "dev": thresholds[chosen], "weights_sha256": artifact_hash(path),
            "loss_first": losses[0], "loss_last": losses[-1],
        }
        print(f"{arm}: dev threshold={chosen} F1={thresholds[chosen]['f1']:.3f}", flush=True)
    write_json(output / "fit.json", {
        "schema": "hybrid-drive-fast-fit-v1", "revision": revision, "channels": width,
        "capture_sha256": artifact_hash(capture / "capture.json"),
        "train_targets_sha256": artifact_hash(train_targets),
        "dev_targets_sha256": artifact_hash(dev_targets),
        "test_targets_opened": False,
        "hyperparameters": {"seed": SEED, "epochs": epochs, "batch_size": 4, "learning_rate": 0.002},
        "arms": fitted,
    })


def _fit_arm(data, targets, arm, epochs):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(SEED)
        model = MetricForecastHead(data.values.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002)
    batches = np.random.default_rng(SEED)
    losses = []
    for epoch in range(epochs):
        total = 0.0
        order = batches.permutation(len(data.observations))
        for start in range(0, len(order), 4):
            indices = order[start:start+4]
            x = _tensor(data.values[indices], arm)
            mask = torch.from_numpy(data.coverage[indices])
            loss = forecasting_loss(model(x), tuple(t[indices] for t in targets), mask)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach()) * len(indices)
        losses.append(total/len(order))
        if epoch == 0 or (epoch+1) % 4 == 0 or epoch+1 == epochs:
            print(f"{arm}: epoch {epoch+1}/{epochs}, objective {losses[-1]:.5f}", flush=True)
    return model.eval(), losses


def _tensor(values, arm):
    values = np.array(values, dtype=np.float32, copy=True)
    if arm == "geometry_only":
        values[:, :-5] = 0
    elif arm != "vision":
        raise ValueError("Unrecognized arm")
    return torch.from_numpy(values)


@torch.no_grad()
def _forward_frame(model, data, i, arm):
    return model(_tensor(data.values[i:i+1], arm))


def predict_fast(capture: Path, fitted: Path, output: Path, *, revision: str) -> None:
    if output.exists():
        raise ValueError("Output directory must be new")
    fit = load_json(fitted / "fit.json")
    if (fit["schema"] != "hybrid-drive-fast-fit-v1" or fit["revision"] != revision
            or fit["test_targets_opened"] is not False
            or fit["capture_sha256"] != artifact_hash(capture / "capture.json")):
        raise ValueError("Fit does not belong to this blind capture")
    data = load_capture(capture, "test")
    if fit["channels"] != data.values.shape[1]:
        raise ValueError("Fit does not match feature width")
    result = {}
    for arm in ARMS:
        spec = fit["arms"][arm]
        path = fitted / (arm + ".safetensors")
        if artifact_hash(path) != spec["weights_sha256"]:
            raise ValueError("Frozen weights changed")
        model = MetricForecastHead(fit["channels"])
        model.load_state_dict(load_file(path), strict=True)
        model.eval()
        rows = []
        for i, observation in enumerate(data.observations):
            boxes = decode_boxes(
                _forward_frame(model, data, i, arm), torch.from_numpy(data.coverage[i]), data.grid,
                threshold=spec["threshold"],
            )
            proposals = _plans(observation, boxes) if arm == "vision" else None
            rows.append({
                **observation, "boxes": [asdict(box) for box in boxes], "proposals": proposals,
            })
        result[arm] = rows
    output.mkdir(parents=True)
    write_json(output / "predictions.json", {
        "schema": "hybrid-drive-fast-blind-predictions-v1", "revision": revision,
        "capture_sha256": fit["capture_sha256"],
        "frozen_fit_sha256": artifact_hash(fitted / "fit.json"),
        "test_targets_opened": False,
        "frames": result,
    })


def _plans(observation, boxes):
    speed = observation["forward_speed_mps_from_past"]
    if not isfinite(speed) or not -.10 <= speed <= 60:
        return {"status": "unsupported_forward_speed"}
    plan_input = (observation["sample_token"], observation["ref_time_us"],
                  Pose2D(*observation["ego_world_xy_yaw"]), max(0, speed))
    result = propose_from_forecast(*plan_input, tuple(ForecastObject(**asdict(box)) for box in boxes))
    empty = propose_from_forecast(*plan_input, ())
    base_order = tuple(c.candidate_id for c in empty.plan.candidates)
    ranked = tuple(c.candidate_id for c in result.plan.candidates)
    return {
        "status": "proposed_not_safety_checked",
        "forward_speed_used_mps": max(0, speed),
        "small_negative_speed_clipped": speed < 0,
        "forecast_hash": result.forecast_hash,
        "all_candidates_overlap": result.all_candidates_overlap,
        "empty_forecast_first_id": base_order[0],
        "ranking_differs_from_empty_forecast": ranked != base_order,
        "candidates_hash": result.plan.candidates_hash,
        "interval_s": result.plan.candidates[0].trajectory.interval_s,
        "candidates": [
            {
                "candidate_id": can.candidate_id,
                "poses": can.trajectory.poses.tolist(),
                "diagnostic": asdict(diag),
            }
            for can, diag in zip(result.plan.candidates, result.diagnostics, strict=True)
        ],
    }


def score_fast(capture: Path, predictions: Path, targets: Path, output: Path, *,
               sealed_sha256: str) -> dict:
    if output.exists():
        raise ValueError("Output directory must be new")
    require_hex(sealed_sha256, 64, "sealed prediction SHA256")
    raw_digest = artifact_hash(predictions)
    if raw_digest != sealed_sha256:
        raise ValueError("Predictions differ from the digest recorded before mounting test targets")
    pred = load_json(predictions)
    if (pred["schema"] != "hybrid-drive-fast-blind-predictions-v1" or pred["test_targets_opened"] is not False
            or pred["capture_sha256"] != artifact_hash(capture / "capture.json")
            or pred["revision"] != load_json(capture / "capture.json")["revision"]):
        raise ValueError("Wrong or unsealed prediction capture")
    data = load_capture(capture, "test")
    labels = _load_targets(targets, data)
    truth = _truth_for_capture(data, labels)
    scenes = sorted({o["scene"] for o in data.observations})
    totals = {}
    for arm in ARMS:
        rows = pred["frames"][arm]
        if [(o["sample_token"], o["snapshot_sha256"]) for o in rows] != [
            (o["sample_token"], o["snapshot_sha256"]) for o in data.observations
        ]:
            raise ValueError("Prediction order or identity changed")
        pairs = [(row["boxes"], boxes, interval) for row, (boxes, interval) in zip(rows, truth, strict=True)]
        totals[arm] = {
            **assess_frames(pairs),
            "by_scene": {scene: assess_frames([p for p,o in zip(pairs,data.observations) if o["scene"] == scene])
                         for scene in scenes},
        }
    plans = [row["proposals"] for row in pred["frames"]["vision"]]
    planned = [p for p in plans if p["status"] == "proposed_not_safety_checked"]
    if any(len({tuple(tuple(row) for row in c["poses"]) for c in p["candidates"]}) < 2 for p in planned):
        raise ValueError("Diagnostic plan must contain distinct trajectories")
    results = {
        "schema": "hybrid-drive-fast-score-v1", "revision": pred["revision"],
        "sealed_predictions_sha256": raw_digest, "test_targets_sha256": artifact_hash(targets),
        "frames": len(data.observations), "scenes": scenes,
        "all_filtered_annotated_targets": sum(len(v["boxes"]) for v in labels.values()),
        "scored_targets": sum(len(b) for b,t in truth),
        "matching": "greedy one-to-one within 2 m XY, grid-supported annotated centers only",
        "arm_results": totals,
        "planned_frames": len(planned),
        "unsupported_speed_frames": len(plans)-len(planned),
        "plans_with_detected_objects": sum(bool(row["boxes"] and row["proposals"]["status"]=="proposed_not_safety_checked")
                                           for row in pred["frames"]["vision"]),
        "rank_changed_by_objects_frames": sum(p["ranking_differs_from_empty_forecast"] for p in planned),
        "first_choice_changed_by_objects_frames": sum(
            p["candidates"][0]["candidate_id"] != p["empty_forecast_first_id"] for p in planned
        ),
        "every_candidate_touches_forecast_frames": sum(p["all_candidates_overlap"] for p in planned),
        "candidate_histogram": {option: sum(p["candidates"][0]["candidate_id"]==option for p in planned)
                                for option in sorted({p["candidates"][0]["candidate_id"] for p in planned})},
    }
    output.mkdir(parents=True)
    write_json(output / "score.json", results)
    return results


def _load_targets(path, capture):
    labels = load_json(path)
    if set(labels) != {v["sample_token"] for v in capture.observations}:
        raise ValueError("Targets do not exactly match requested observations")
    return labels


def _truth_for_capture(data, labels):
    return [
        (supported_targets(labels[o["sample_token"]], mask, data.grid), labels[o["sample_token"]]["next_dt_s"])
        for o, mask in zip(data.observations, data.coverage, strict=True)
    ]
