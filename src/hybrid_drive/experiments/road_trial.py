"""Fit/bind/freeze and score static road layers without supplying map labels to inference."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from hybrid_drive.contracts.planning import Candidate
from hybrid_drive.contracts.trajectory import Trajectory
from hybrid_drive.experiments.bounded_io import require_hex
from hybrid_drive.experiments.fast_capture import artifact_hash, load_capture, load_json
from hybrid_drive.experiments.io import write_json
from hybrid_drive.experiments.road_scores import check_quality, layer_scores, load_road_targets, select_cut
from hybrid_drive.fast.road_candidate_rank import assess_road, rank_road_candidates
from hybrid_drive.fast.road_geometry import LAYERS, RoadGeometryHead, positive_weights, road_loss


ARMS = ("vision", "geometry_only")
SEED = 459
MONO_CONFIDENCE_MARGIN = .10
SOURCE_CAPTURE_REVISION = "d93ae54ba71d7b8873c6519e6e204a45c72ddfa8"
SOURCE_BOX_PREDICTIONS_SHA256 = "df8b35aa29b41b118f14b5058bb8e0d353e0cf2d725e418df6e3bd5648ab4720"


def fit_roads(capture: Path, train_targets: Path, dev_targets: Path, output: Path, *,
              revision: str, epochs: int = 18) -> None:
    _fresh_dir(output)
    require_hex(revision, 40, "revision")
    if not 1 <= epochs <= 40:
        raise ValueError("Expected 1 to 40 epochs")
    _verify_capture(capture)
    train, dev = [load_capture(capture, role) for role in ("train", "dev")]
    yr = load_road_targets(train_targets, capture, "train", train)
    yd = load_road_targets(dev_targets, capture, "dev", dev)
    if yr.map_sha256 != yd.map_sha256:
        raise ValueError("Train and dev labels use different map archives")
    mask = torch.from_numpy(train.coverage & yr.map_known)
    labels = torch.from_numpy(yr.labels)
    weights = positive_weights(labels, mask)
    dev_support = dev.coverage & yd.map_known
    output.mkdir(parents=True)
    arms = {}
    for arm in ARMS:
        model, losses = fit_captured_road_head(train, labels, mask, weights, arm, epochs)
        guesses = predict_captured_road_head(model, dev, arm)
        thresholds, validation = {}, {}
        for i, name in enumerate(LAYERS):
            thresholds[name], validation[name] = select_cut(guesses[:, i], yd.labels[:, i], dev_support, name)
        path = output / f"{arm}.safetensors"
        save_file(model.state_dict(), path)
        arms[arm] = {"thresholds":thresholds, "dev":validation, "weights_sha256":artifact_hash(path),
                     "initial_epoch_loss":losses[0], "final_epoch_loss":losses[-1]}
        print(f"{arm}: dev road IoU={validation['drivable']['iou']:.3f}, divider F1={validation['lane_divider']['f1']:.3f}",
              flush=True)
    write_json(output / "fit.json", {
        "schema":"hybrid-drive-road-fit-v1", "revision":revision, "test_targets_opened":False,
        "capture_sha256":artifact_hash(capture / "capture.json"),
        "training_target_manifests_sha256": {
            "train":artifact_hash(train_targets/"targets.json"), "dev":artifact_hash(dev_targets/"targets.json")},
        "channels": train.values.shape[1], "map_sha256":yr.map_sha256, "arms":arms,
        "recipe":{"seed":SEED, "epochs":epochs, "batch_size":4, "learning_rate":.002},
    })


def fit_captured_road_head(capture, labels, mask, weights, arm, epochs):
    """Fit a road head with a fixed initialization, sampler and optimizer recipe."""
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(SEED)
        model = RoadGeometryHead(capture.values.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002)
    order = np.random.default_rng(SEED)
    n = len(capture.observations)
    losses = []
    for i in range(epochs):
        total = 0.0
        ids = order.permutation(n)
        for s in range(0, len(ids), 4):
            index = ids[s:s+4]
            pred = model(_inputs(capture.values[index], arm))
            loss = road_loss(pred, labels[index], mask[index], weights)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite road objective")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach()) * len(index)
        losses.append(total/n)
        if i == 0 or (i+1)%6 == 0 or i+1 == epochs:
            print(f"{arm}: epoch {i+1}/{epochs} loss={losses[-1]:.4f}", flush=True)
    return model.eval(), losses


def _inputs(values, arm):
    tensor = torch.from_numpy(np.array(values, dtype=np.float32, copy=True))
    if arm == "geometry_only":
        # The last five channels are coverage, relative coordinates and the time interval.
        tensor[:, :-5].zero_()
    return tensor


@torch.no_grad()
def predict_captured_road_head(model, capture, arm):
    """Materialize two road probability layers in capture order."""
    return np.concatenate([
        model(_inputs(capture.values[j:j+1], arm)).sigmoid().cpu().numpy()
        for j in range(len(capture.observations))
    ])


def predict_roads(capture: Path, fitted: Path, box_predictions: Path, output: Path, *, revision: str) -> None:
    _fresh_dir(output)
    _verify_capture(capture)
    fit = load_json(fitted / "fit.json")
    if (fit["schema"] != "hybrid-drive-road-fit-v1" or fit["revision"] != revision
            or fit["test_targets_opened"] is not False
            or fit["capture_sha256"] != artifact_hash(capture / "capture.json")):
        raise ValueError("Fit does not belong to this target-free observation capture")
    test = load_capture(capture, "test")
    frozen = _load_fixed_candidates(box_predictions, test)
    results = {}
    routing = []
    for arm in ARMS:
        model = _load_model(fitted, fit, arm)
        probs = predict_captured_road_head(model, test, arm)
        results[arm] = {"probabilities":probs.tolist(), "thresholds":fit["arms"][arm]["thresholds"]}
        if arm == "vision":
            for obs, pred, visible, original in zip(test.observations, probs, test.coverage, frozen, strict=True):
                routing.append(_route(obs, pred[0], visible, original["proposals"], test.grid,
                                      fit["arms"][arm]["thresholds"]["drivable"]))
    output.mkdir(parents=True)
    write_json(output / "predictions.json", {
        "schema":"hybrid-drive-road-blind-v1", "revision":revision, "test_targets_opened":False,
        "capture_sha256":fit["capture_sha256"], "fit_sha256":artifact_hash(fitted/"fit.json"),
        "object_predictions_sha256": SOURCE_BOX_PREDICTIONS_SHA256,
        "map_label_source_sha256":fit["map_sha256"],
        "anchors":[{"sample_token":r["sample_token"],"snapshot_sha256":r["snapshot_sha256"]} for r in test.observations],
        "arms":results, "routing":routing,
    })


def _load_model(root, fit, arm):
    weights = root / f"{arm}.safetensors"
    if artifact_hash(weights) != fit["arms"][arm]["weights_sha256"]:
        raise ValueError("Frozen road weights changed")
    result = RoadGeometryHead(fit["channels"])
    result.load_state_dict(load_file(weights), strict=True)
    return result.eval()


def _load_fixed_candidates(path, test):
    if artifact_hash(path) != SOURCE_BOX_PREDICTIONS_SHA256:
        raise ValueError("Need the exact sealed object forecast and candidate set from #42")
    pred = load_json(path)
    if pred["revision"] != SOURCE_CAPTURE_REVISION or pred["test_targets_opened"] is not False:
        raise ValueError("Unrecognized object forecast provenance")
    rows = pred["frames"]["vision"]
    for a, b in zip(rows, test.observations, strict=True):
        if (a["snapshot_sha256"],a["sample_token"]) != (b["snapshot_sha256"],b["sample_token"]):
            raise ValueError("Object or road snapshot order differs")
    return rows


def _route(observation, probability, visible, original, grid, threshold):
    binding = {"sample_token":observation["sample_token"], "snapshot_sha256":observation["snapshot_sha256"]}
    if original["status"] != "proposed_not_safety_checked":
        return {**binding, "status":"no_base_candidates"}
    base = original["candidates"]
    candidates = tuple(Candidate(r["candidate_id"], Trajectory(r["poses"], original["interval_s"])) for r in base)
    order = tuple(row["candidate_id"] for row in base)
    obstacles = {c["candidate_id"]:c["diagnostic"]["first_overlap_s"] is None for c in base}
    res = rank_road_candidates(
        candidates, obstacles, probability, visible, grid,
        offroad_threshold=threshold-MONO_CONFIDENCE_MARGIN,
        onroad_threshold=threshold+MONO_CONFIDENCE_MARGIN,
    )
    return {**binding, "status":"diagnostic_only", "base_order":order, "new_order":res.ordered,
            "assessments":{k:asdict(v) for k,v in res.assessments.items()}}


def score_roads(capture: Path, predictions: Path, map_targets: Path, box_predictions: Path,
                output: Path, *, sealed_sha256: str) -> dict:
    _fresh_dir(output)
    require_hex(sealed_sha256, 64, "prediction seal")
    if artifact_hash(predictions) != sealed_sha256:
        raise ValueError("Predictions differ from the seal fixed before opening road test labels")
    pred = load_json(predictions)
    _verify_capture(capture)
    if (pred["schema"]!="hybrid-drive-road-blind-v1" or pred["test_targets_opened"] is not False
            or pred["capture_sha256"]!=artifact_hash(capture/"capture.json")
            or pred["object_predictions_sha256"]!=SOURCE_BOX_PREDICTIONS_SHA256):
        raise ValueError("Unrecognized blind road capture")
    test = load_capture(capture,"test")
    if pred["anchors"] != [{"sample_token":r["sample_token"], "snapshot_sha256":r["snapshot_sha256"]}
                           for r in test.observations]:
        raise ValueError("Predictions do not match ordered anchors")
    original = _load_fixed_candidates(box_predictions, test)
    target = load_road_targets(map_targets,capture,"test",test)
    if target.map_sha256 != pred["map_label_source_sha256"]:
        raise ValueError("Test labels do not use the frozen train/dev map release")
    support = test.coverage & target.map_known
    arms = {}
    for arm in ARMS:
        model = pred["arms"][arm]
        probabilities = np.asarray(model["probabilities"], dtype=np.float32)
        if probabilities.shape != target.labels.shape:
            raise ValueError("Malformed road forecast dimensions")
        by_recording = {}
        for scene in sorted({r["scene"] for r in test.observations}):
            ids = [i for i, row in enumerate(test.observations) if row["scene"]==scene]
            by_recording[scene] = layer_scores(probabilities[ids],target.labels[ids],support[ids],model["thresholds"])
        arms[arm] = {
            **layer_scores(probabilities,target.labels,support,model["thresholds"]),
            "by_recording":by_recording,
        }
    result = {
        "schema":"hybrid-drive-road-score-v1", "revision":pred["revision"], "prediction_seal":sealed_sha256,
        "map_target_manifest_sha256":artifact_hash(map_targets/"targets.json"), "frames":len(test.observations),
        "locations":dict(Counter(target.locations)), "supported_labeled_cells":int(support.sum()),
        "map_cells_outside_camera":int((target.map_known & ~test.coverage).sum()),
        "camera_cells_outside_map":int((test.coverage & ~target.map_known).sum()),
        "arm_metrics": arms,
        "quality_gate":check_quality(arms["vision"],arms["geometry_only"]),
        "ranking_against_map":_score_routing(pred, original, target, test),
    }
    output.mkdir(parents=True)
    write_json(output/"score.json",result)
    return result


def _score_routing(pred, original, target, captured):
    out = Counter(
        frames=0, first_changed=0, any_order_changed=0,
        before_offroad_frames=0, after_offroad_frames=0,
        before_unknown_frames=0, after_unknown_frames=0,
        frames_with_predicted_offroad=0, frames_all_candidates_have_unknown=0, frames_with_two_comparable=0,
    )
    for i, (row, base, obs) in enumerate(zip(pred["routing"],original,captured.observations,strict=True)):
        if (row["sample_token"],row["snapshot_sha256"]) != (obs["sample_token"],obs["snapshot_sha256"]):
            raise ValueError("Road ranking and original snapshot differ")
        if row["status"]!="diagnostic_only":
            continue
        before, after = tuple(row["base_order"]), tuple(row["new_order"])
        plan = base["proposals"]
        candidates = {c["candidate_id"]: c for c in plan["candidates"]}
        if before != tuple(candidates) or sorted(before)!=sorted(after) or not before:
            raise ValueError("Road ranking did not preserve the candidate set")
        if any(candidates[a]["diagnostic"]["first_overlap_s"] is not None for a,b in zip(after,before)
               if candidates[b]["diagnostic"]["first_overlap_s"] is None):
            raise ValueError("Road ranking promoted an obstacle-overlap candidate")
        evidence = row["assessments"]
        if set(evidence) != set(candidates):
            raise ValueError("Road assessments did not preserve candidate identity")
        out["frames_with_predicted_offroad"] += any(ev["offroad_samples"]>0 for ev in evidence.values())
        out["frames_all_candidates_have_unknown"] += all(ev["unknown_samples"]>0 for ev in evidence.values())
        out["frames_with_two_comparable"] += sum(
            ev["unknown_samples"]==0 and candidates[key]["diagnostic"]["first_overlap_s"] is None
            for key,ev in evidence.items()
        )>=2
        out["frames"]+=1
        out["first_changed"]+=before[0]!=after[0]
        out["any_order_changed"]+=before!=after
        for name,key in (("before",before[0]),("after",after[0])):
            metric = assess_road(
                Trajectory(candidates[key]["poses"], plan["interval_s"]),
                target.labels[i,0].astype(np.float32), target.map_known[i], captured.grid,
                offroad_threshold=.25, onroad_threshold=.75,
            )
            out[name+"_offroad_frames"] += metric.offroad_samples > 0
            out[name+"_unknown_frames"] += metric.unknown_samples > 0
    return dict(out)


def _verify_capture(root):
    meta = load_json(root/"capture.json")
    if meta["revision"] != SOURCE_CAPTURE_REVISION or meta["targets_opened"] is not False:
        raise ValueError("Road pilot requires the original target-free capture from #42")


def _fresh_dir(path):
    if path.exists():
        raise ValueError("Output must be a new directory")
