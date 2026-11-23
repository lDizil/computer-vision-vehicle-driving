"""Compare close-up features and matched controls without changing visibility decisions."""

from dataclasses import dataclass
import hashlib
from pathlib import Path

import torch

from hybrid_drive.neural.spatial_zoom_refiner import ZoomBoxRefiner, fit_zoom_refiner, zoom_head_protocol
from .bounded_io import decode_json, read_bounded, require_hex
from .io import write_json
from .neural_checkpoint import load_neural_weights, module_state_hash, save_neural_weights
from .proposal_alignment import _new_result_directory
from .protocol import content_hash
from .spatial_decision_policy import PRESENT
from .spatial_tracking_benchmark import TrackingTruthFile, _targets
from .spatial_zoom_capture import DEV_ROLES, REFERENCE_ROLES, TRAIN_ROLES, ZOOM_ROLES
from .spatial_zoom_metrics import apply_zoom_boxes, evaluate_zoom_boxes, select_zoom_candidate
from .spatial_zoom_report import write_zoom_gallery, write_zoom_report


SCOPE = "openscene_box_zoom_refinement_on_known_recordings_v1"
FAMILIES = ("geometry", "global", "zoom")
SEEDS = (0, 1)
EPOCHS = 8
BATCH_SIZE = 16
_PRIMARY_REFERENCE = "appearance_holdout"


@dataclass(frozen=True)
class _Data:
    decisions: tuple
    boxes: torch.Tensor
    active: torch.Tensor
    windows: torch.Tensor
    global_features: torch.Tensor
    zoom_features: torch.Tensor
    targets: dict

    def features(self, mode):
        if mode == "geometry":
            return torch.zeros_like(self.global_features)
        if mode == "global":
            return self.global_features
        if mode == "zoom":
            return self.zoom_features
        raise ValueError("Unknown feature condition")


def run_zoom_benchmark(capture, truth, output_root, *, revision, extra_inputs=()):
    require_hex(revision, 40, "zoom benchmark revision")
    if set(truth) != set(ZOOM_ROLES) or any(not isinstance(ref, TrackingTruthFile) for ref in truth.values()):
        raise ValueError("Five bound target files are required for the known-recording experiment")
    root = _new_result_directory(
        output_root, (*capture.inputs.protected_roots, *(ref.path for ref in truth.values()), *extra_inputs),
    )
    protocol = {
        "capture_sha256": capture.sha256, "source_model_hash": capture.inputs.source_model_hash,
        "source_capture_sha256": capture.inputs.source_capture_sha256,
        "truth_hashes": {role: ref.sha256 for role, ref in truth.items()},
        "roles": {"train": list(TRAIN_ROLES), "development": list(DEV_ROLES), "reference": list(REFERENCE_ROLES)},
        "reference_independent_of_current_fit": True, "reference_previously_observed": True, "fresh_blind_test": False,
        "feature_protocol": capture.manifest["protocol"], "feature_dim": capture.manifest["feature_dim"],
        "head": zoom_head_protocol(), "families": list(FAMILIES), "seeds": list(SEEDS), "epochs": EPOCHS,
        "batch_size": BATCH_SIZE, "visible_decisions_changed": False,
        "association_code_changed": False, "physical_control": False,
    }
    result = {"scope": SCOPE, "status": "running", "revision": revision, "protocol": protocol}
    write_json(root / "result.json", result)
    try:
        _compare(root, capture, truth, result)
        result["status"] = "passed"
    except KeyboardInterrupt:
        result.update(status="interrupted", error="Interrupted by operator")
    except Exception as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    try:
        write_zoom_report(root, result)
    except Exception as exc:
        result["report_error"] = f"{type(exc).__name__}: {exc}"
        if result["status"] == "passed":
            result.update(status="failed", error="Could not save zoom report")
    path = root / "result.json"
    write_json(path, result)
    return root, result, hashlib.sha256(read_bounded(path, 4 * 1024 * 1024)).hexdigest()


def _compare(root, capture, truth, result):
    training = _load_data(capture, truth, TRAIN_ROLES)
    dev = _load_data(capture, truth, DEV_ROLES)
    if set(training.targets) & set(dev.targets):
        raise ValueError("Crop training and development frames overlap")
    target_boxes, target_mask = _tensors_for_training(training)
    fit_results = {}
    dev_scores = {"baseline": evaluate_zoom_boxes(dev.decisions, dev.targets)}
    for seed in SEEDS:
        previous_orders = None
        for family in FAMILIES:
            name = f"{family}_{seed}"
            print(f"Training {name} on {len(training.decisions)} frames", flush=True)
            model, trace = fit_zoom_refiner(
                training.boxes, training.windows, training.features(family), training.active,
                target_boxes, target_mask, seed=seed, epochs=EPOCHS, batch_size=BATCH_SIZE,
            )
            orders = [(row["order_sha256"], row["supervised_boxes"], row["steps"]) for row in trace]
            if previous_orders is None:
                previous_orders = orders
            elif orders != previous_orders:
                raise ValueError("Matched controls trained on different rows")
            metadata = dict(
                scope=SCOPE, name=name, revision=result["revision"], capture_sha256=capture.sha256,
                protocol_hash=content_hash(result["protocol"]),
            )
            status = save_neural_weights(root / f"{name}.safetensors", {"zoom": model}, metadata=metadata)
            fit_results[name] = {
                "checkpoint": status, "metadata": metadata, "state_hash": module_state_hash(model), "trace": trace,
            }
            dev_scores[name] = evaluate_zoom_boxes(_refined(dev, model, family), dev.targets)

    selection = select_zoom_candidate(dev_scores, min_gain=2, max_recording_drop=1)
    gate = {
        "scope": SCOPE, "protocol_hash": content_hash(result["protocol"]),
        "development_hash": content_hash(dev_scores), "selection": selection,
        "trained": {k: {"sha256": v["checkpoint"]["sha256"], "state_hash": v["state_hash"]}
                    for k, v in fit_results.items()},
    }
    path = root / "development-gate.json"
    write_json(path, gate)
    saved = decode_json(read_bounded(path, 256 * 1024))
    if content_hash(saved) != content_hash(gate):
        raise ValueError("Saved development selection differs from the frozen gate")
    chosen = saved["selection"]["name"]
    if chosen not in dev_scores:
        raise ValueError("The development gate returned an unknown mode")
    result["development_gate_hash"] = content_hash(saved)

    models = {}
    for name, reference in fit_results.items():
        model = ZoomBoxRefiner(capture.manifest["feature_dim"]).eval()
        load_neural_weights(
            root / f"{name}.safetensors", {"zoom": model},
            expected_sha256=saved["trained"][name]["sha256"], expected_metadata=reference["metadata"],
        )
        if module_state_hash(model) != saved["trained"][name]["state_hash"]:
            raise ValueError("Restored box refiner differs from the trained state")
        models[name] = model

    # Reference targets are opened for the first time below. Their recordings
    # were seen in older experiments, but not in this refiner fit or gate.
    references = {}
    approved_ids = set(training.targets) | set(dev.targets)
    for role in REFERENCE_ROLES:
        data = _load_data(capture, truth, (role,))
        if approved_ids.intersection(data.targets):
            raise ValueError("A reference frame was already opened in another role")
        approved_ids.update(data.targets)
        forecasts = {"baseline": data.decisions}
        for name, model in models.items():
            forecasts[name] = _refined(data, model, name.rsplit("_", 1)[0])
        references[role] = {
            name: evaluate_zoom_boxes(frames, data.targets, include_tracking=True)
            for name, frames in forecasts.items()
        }
        if role == _PRIMARY_REFERENCE:
            write_zoom_gallery(root / "zoom.html", capture.inputs.packs[role], data.targets, forecasts, selected=chosen)
    result["gallery"] = "zoom.html"
    result["trained"] = fit_results
    matrix = {
        "development": dev_scores, "references": references, "selection": saved["selection"],
        "frame_counts": {"train": len(training.decisions), "development": len(dev.decisions),
                         "references": {role: scores["baseline"]["frames"] for role,scores in references.items()}},
        "predictions_only_move_original_visible_boxes": True,
        "tracking_is_a_secondary_replay": True, "primary_reference": _PRIMARY_REFERENCE,
    }
    result.update(matrix=matrix, matrix_hash=content_hash(matrix))


def _load_data(capture, truth, roles):
    frames, inputs, labels = [], [], {}
    for role in roles:
        targets = _targets(capture.inputs.packs[role], truth[role])
        if set(labels) & set(targets):
            raise ValueError("Zoom role contains duplicated source frames")
        labels.update(targets)
        frames.extend(capture.inputs.decisions[role])
        inputs.append(capture.grids[role])
    return _Data(
        decisions=tuple(frames), boxes=torch.tensor([d.boxes for d in frames], dtype=torch.float32),
        active=torch.tensor([[s == PRESENT for s in d.states] for d in frames], dtype=torch.bool),
        windows=torch.cat([g.windows for g in inputs], dim=0),
        global_features=torch.cat([g.global_features for g in inputs], dim=0),
        zoom_features=torch.cat([g.zoom_features for g in inputs], dim=0), targets=labels,
    )


def _tensors_for_training(data):
    absent = (float("nan"),) * 4
    labels = [data.targets[d.observation.sample_id] for d in data.decisions]
    boxes = torch.tensor([
        [absent if b is None else b for b in label.boxes] for label in labels
    ], dtype=torch.float32)
    present = torch.tensor([[x is not None for x in label.track_ids] for label in labels], dtype=torch.bool)
    return boxes, present


@torch.no_grad()
def _refined(data, model, family):
    return apply_zoom_boxes(data.decisions, model(data.boxes, data.windows, data.features(family), data.active))
