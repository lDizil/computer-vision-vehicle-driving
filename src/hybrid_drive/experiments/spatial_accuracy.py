"""Offline dev calibration and bounded residual box refinement on frozen spatial logits."""

from dataclasses import asdict, dataclass, replace

import torch

from hybrid_drive.assets.spatial_manifest import POLICY_EVALUATION_SOURCE_SCHEMA, read_spatial_manifest
from hybrid_drive.neural.pretrained_alignment import load_alignment_vision
from hybrid_drive.neural.spatial_box_refiner import SpatialBoxRefiner, spatial_box_refinement_loss
from hybrid_drive.neural.spatial_data import SpatialAlignmentDataset, verify_spatial_splits
from .bounded_io import decode_json, read_bounded, require_hex
from .fresh_spatial_holdout import read_fresh_spatial_holdout
from .io import write_json
from .localization_gallery import GalleryPredictions, write_localization_gallery
from .neural_checkpoint import load_neural_weights, module_state_hash, save_neural_weights
from .proposal_alignment import _new_result_directory
from .proposal_evaluation import encode
from .proposal_prior_source import load_spatial_prior_source
from .protocol import content_hash
from .spatial_accuracy_metrics import choose_refinement, measure_geometry, measure_strategy
from .spatial_accuracy_report import write_spatial_accuracy_report
from .spatial_decision_policy import UNCERTAIN, ZoneDecisionPolicy, calibrate_zone_policy, measure_zone_policy
from .spatial_experiment import SPLIT_FRAMES


SCOPE = "openscene_front_spatial_confidence_and_refinement_v1"
_MAX_FP = 0.20
_MAX_FN = 0.10
_MIN_IOU_GAIN = 0.01
_MAX_RECORDING_LOSS = 0.02
_LR = 0.001


@dataclass(frozen=True)
class SpatialAccuracyRecipe:
    epochs: int = 12
    batch_size: int = 8

    def __post_init__(self):
        if type(self.epochs) is not int or not 1 <= self.epochs <= 16:
            raise ValueError("Expected 1–16 training epochs")
        if type(self.batch_size) is not int or not 1 <= self.batch_size <= 16:
            raise ValueError("Expected batch size between 1 and 16")


def run_spatial_accuracy(
    reference_root, calibration_root, evaluation_root, model_root, prior_root, output_root,
    *, revision, expected_prior_result_sha256, recipe=SpatialAccuracyRecipe(),
):
    require_hex(revision, 40, "revision")
    require_hex(expected_prior_result_sha256, 64, "expected_prior_result_sha256")
    if not isinstance(recipe, SpatialAccuracyRecipe):
        raise ValueError("Expected SpatialAccuracyRecipe")
    root = _new_result_directory(
        output_root, (reference_root, calibration_root, evaluation_root, model_root, prior_root),
    )
    manifest = {
        "schema": "hybrid-drive-spatial-accuracy-v1",
        "scope": SCOPE,
        "revision": revision,
        "recipe": {**asdict(recipe), "seeds": [0, 1], "optimizer": "AdamW", "learning_rate": _LR, "weight_decay": 0},
        "policy_budgets_per_dev_recording": {"false_positive_rate": _MAX_FP, "false_negative_rate": _MAX_FN},
        "geometry_selection_on_dev": {"min_macro_iou_gain": _MIN_IOU_GAIN, "max_recording_iou_loss": _MAX_RECORDING_LOSS},
        "gallery_selection": "six midpoint samples per independent recording",
        "expected_prior_result_sha256": expected_prior_result_sha256,
        "torch_version": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "runtime_changed": False,
    }
    result = {"status": "running", "scope": SCOPE, "revision": revision, "refiners": []}
    trace = []
    write_json(root / "manifest.json", manifest)
    write_json(root / "result.json", result)
    try:
        with torch.random.fork_rng(devices=[]), torch.device("cpu"):
            _run(
                root, reference_root, calibration_root, evaluation_root,
                model_root, prior_root, recipe, manifest, result, trace,
            )
        result["status"] = "passed"
    except KeyboardInterrupt:
        result.update(status="interrupted", error="Interrupted by operator")
    except Exception as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")

    result["manifest_hash"] = content_hash(manifest)
    result["optimizer_steps_completed"] = len(trace)
    write_json(root / "manifest.json", manifest)
    write_json(root / "trace.json", trace)
    try:
        write_spatial_accuracy_report(root, manifest, result)
    except Exception as exc:
        result["report_error"] = f"{type(exc).__name__}: {exc}"
        if result["status"] == "passed":
            result.update(status="failed", error="Aggregate report could not be saved")
    write_json(root / "result.json", result)
    return root, result


def _run(
    root, reference_root, calibration_root, evaluation_root, model_root, prior_root, recipe, manifest, result, trace,
):
    original = read_spatial_manifest(reference_root)
    additional = read_fresh_spatial_holdout(reference_root, calibration_root)
    final = read_fresh_spatial_holdout(
        reference_root, evaluation_root, expected_schema=POLICY_EVALUATION_SOURCE_SCHEMA,
        previously_seen_roots=(calibration_root,),
    )
    manifest["data_hashes"] = {
        "reference": original["manifest_hash"],
        "previous_final_reused_as_dev": additional["manifest_hash"],
        "evaluation": final["manifest_hash"],
    }
    datasets = {
        role: SpatialAlignmentDataset(reference_root, role, expected_manifest_hash=original["manifest_hash"])
        for role in ("train", "dev")
    }
    verify_spatial_splits(datasets, SPLIT_FRAMES)
    datasets["additional_dev"] = SpatialAlignmentDataset(
        calibration_root, "holdout", expected_manifest_hash=additional["manifest_hash"],
    )
    vision, vision_receipt = load_alignment_vision(model_root)
    vision_hash = _frozen_vision_hash(vision)
    prior = load_spatial_prior_source(
        prior_root, expected_result_sha256=manifest["expected_prior_result_sha256"],
        dataset_hash=original["manifest_hash"], vision_provenance=vision_receipt,
    )
    manifest.update(prior_receipt=prior.receipt, vision=vision_receipt)
    manifest["recordings"] = {k: sorted(set(dataset.recording_ids)) for k, dataset in datasets.items()}
    write_json(root / "manifest.json", manifest)

    encoded = {
        role: encode(dataset, vision, prior, recipe.batch_size, capture_direct=True)
        for role, dataset in datasets.items()
    }
    train = encoded["train"]
    dev = _join_dev(encoded["dev"], encoded["additional_dev"])
    scores, labels = dev.prior_scores.numpy(), dev.presence.numpy()
    prior_threshold = prior.receipt["calibration"]["threshold"]
    previous_policy = ZoneDecisionPolicy(prior_threshold, prior_threshold)
    calibrated = calibrate_zone_policy(
        scores, labels, dev.recordings, max_false_positive_rate=_MAX_FP, max_false_negative_rate=_MAX_FN,
    )
    dev_calibrated = measure_zone_policy(scores, labels, dev.recordings, calibrated)
    _check_dev_budgets(dev_calibrated)
    base_geometry = measure_geometry(dev, dev.proposals.numpy())
    for seed in manifest["recipe"]["seeds"]:
        model = _fit(train, vision.visual_dim, recipe, seed, trace)
        metadata = {
            "scope": SCOPE, "revision": manifest["revision"], "data_hashes": manifest["data_hashes"],
            "prior_receipt_hash": content_hash(prior.receipt), "recipe": manifest["recipe"], "seed": seed,
        }
        name = f"refined-boxes-seed-{seed}.safetensors"
        checkpoint = save_neural_weights(root / name, {"refiner": model}, metadata=metadata)
        result["refiners"].append({
            "seed": seed,
            "dev_geometry": measure_geometry(dev, _refine(dev, model, recipe.batch_size)),
            "checkpoint": {**checkpoint, "file": name},
            "checkpoint_metadata": metadata,
            "state_hash": module_state_hash(model),
        })
        write_json(root / "trace.json", trace)
    selected = choose_refinement(
        base_geometry, result["refiners"], min_macro_gain=_MIN_IOU_GAIN, max_recording_loss=_MAX_RECORDING_LOSS,
    )
    result["development"] = {
        "previous": measure_zone_policy(scores, labels, dev.recordings, previous_policy),
        "calibrated": dev_calibrated, "baseline_geometry": base_geometry,
    }
    stored = _seal_policy(root, manifest, result, previous_policy, calibrated, selected)
    restored = _restore_refiners(root, result["refiners"], stored, vision.visual_dim)

    holdout = SpatialAlignmentDataset(
        evaluation_root, "holdout", expected_manifest_hash=stored["data_hashes"]["evaluation"],
    )
    final_data = encode(holdout, vision, prior, recipe.batch_size, capture_direct=True)
    manifest["recordings"]["final"] = sorted(set(holdout.recording_ids))
    _evaluate_and_publish(root, evaluation_root, final, final_data, stored, restored, recipe, result)
    result["frozen_vision_unchanged"] = _frozen_vision_hash(vision) == vision_hash
    if not result["frozen_vision_unchanged"]:
        raise ValueError("Frozen vision changed during box refinement")


def _frozen_vision_hash(vision):
    if vision.training or any(p.requires_grad or p.grad is not None for p in vision.parameters()):
        raise ValueError("Only frozen vision is allowed")
    return module_state_hash(vision)


def _join_dev(old, additional):
    if set(old.recordings).intersection(additional.recordings):
        raise ValueError("Dev packs must not share recordings")
    return replace(
        old,
        features=torch.cat((old.features, additional.features)),
        proposals=torch.cat((old.proposals, additional.proposals)),
        presence=torch.cat((old.presence, additional.presence)),
        target_boxes=torch.cat((old.target_boxes, additional.target_boxes)),
        prior_scores=torch.cat((old.prior_scores, additional.prior_scores)),
        recordings=old.recordings + additional.recordings,
    )


def _check_dev_budgets(measure):
    for values in measure["per_recording"].values():
        fp, fn = values["false_positive_rate"], values["false_negative_rate"]
        if (fp is not None and fp > _MAX_FP) or (fn is not None and fn > _MAX_FN):
            raise ValueError("Calibrated policy exceeds a declared per-recording development budget")


def _fit(data, visual_dim, recipe, seed, trace):
    torch.manual_seed(seed)
    model = SpatialBoxRefiner(visual_dim)
    initial = module_state_hash(model)
    model.train()
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=_LR, weight_decay=0)
    for epoch in range(1, recipe.epochs + 1):
        for indices in torch.randperm(len(data), generator=generator).split(recipe.batch_size):
            if not data.presence[indices].any():
                continue
            optimizer.zero_grad(set_to_none=True)
            predictions = model(data.features[indices, 0], data.proposals[indices])
            loss = spatial_box_refinement_loss(predictions, data.target_boxes[indices], data.presence[indices])
            if loss is None or not torch.isfinite(loss):
                raise ValueError("A positive training batch did not yield a finite box loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            optimizer.step()
            if any(not torch.isfinite(p).all() for p in model.parameters()):
                raise ValueError("Refinement weights changed to nonfinite values")
            trace.append({"seed": seed, "epoch": epoch, "loss": float(loss.detach()), "gradient_norm": float(norm)})
        print(f"Geometry seed {seed}, epoch {epoch}/{recipe.epochs}", flush=True)
    if module_state_hash(model) == initial:
        raise ValueError("The box refiner did not update")
    return model.eval()


@torch.no_grad()
def _refine(data, model, batch_size):
    parts = [
        model(data.features[idx, 0], data.proposals[idx])
        for idx in torch.arange(len(data)).split(batch_size)
    ]
    return torch.cat(parts).numpy()


def _seal_policy(root, manifest, result, previous, calibrated, selected):
    payload = {
        "schema": "hybrid-drive-spatial-accuracy-gate-v1",
        "data_hashes": manifest["data_hashes"],
        "selection_rule": manifest["geometry_selection_on_dev"],
        "dev_budgets": manifest["policy_budgets_per_dev_recording"],
        "dev_recordings": sorted(manifest["recordings"]["dev"] + manifest["recordings"]["additional_dev"]),
        "previous_policy": asdict(previous), "calibrated_policy": asdict(calibrated),
        "selected_refinement_seed": selected,
        "refiners": [
            {"seed": row["seed"], "sha256": row["checkpoint"]["sha256"], "state_hash": row["state_hash"]}
            for row in result["refiners"]
        ],
    }
    target = root / "frozen-policy.json"
    write_json(target, payload)
    digest = content_hash(payload)
    persisted = decode_json(read_bounded(target, 32 * 1024))
    if content_hash(persisted) != digest:
        raise ValueError("Persisted policy differs from the frozen policy receipt")
    result["holdout_gate_hash"] = digest
    result["selected_refinement_seed"] = selected
    result["calibrated_policy"] = persisted["calibrated_policy"]
    write_json(root / "result.json", result)
    return persisted


def _restore_refiners(root, rows, stored, visual_dim):
    models = {}
    for row, proof in zip(rows, stored["refiners"]):
        seed = row["seed"]
        checkpoint = row["checkpoint"]
        if seed != proof["seed"] or checkpoint["sha256"] != proof["sha256"]:
            raise ValueError("Refinement checkpoint differs from the frozen receipt")
        model = SpatialBoxRefiner(visual_dim).requires_grad_(False).eval()
        load_neural_weights(
            root / checkpoint["file"], {"refiner": model}, expected_sha256=proof["sha256"],
            expected_metadata=row["checkpoint_metadata"],
        )
        if module_state_hash(model) != proof["state_hash"]:
            raise ValueError("Restored refinement weights differ from the frozen state")
        row["restored_bitwise_before_final"] = True
        models[seed] = model
    return models


def _evaluate_and_publish(root, evaluation_root, manifest, data, stored, models, recipe, result):
    previous = ZoneDecisionPolicy(**stored["previous_policy"])
    calibrated = ZoneDecisionPolicy(**stored["calibrated_policy"])
    selected = stored["selected_refinement_seed"]
    boxes = data.proposals.numpy()
    candidates = {seed: _refine(data, model, recipe.batch_size) for seed, model in models.items()}
    chosen = boxes if selected is None else candidates[selected]

    result["final"] = {
        "previous": measure_strategy(data, boxes, previous),
        "calibrated": measure_strategy(data, boxes, calibrated),
        "selected": measure_strategy(data, chosen, calibrated),
    }
    for row in result["refiners"]:
        row["final_geometry"] = measure_geometry(data, candidates[row["seed"]])
    scores = data.prior_scores.numpy()
    uncertain = calibrated.decide(scores) == UNCERTAIN
    views = {
        "Старый порог": GalleryPredictions(scores, boxes, previous.present_at_or_above),
        "Новые решения": GalleryPredictions(scores, boxes, calibrated.present_at_or_above, uncertain=uncertain),
    }
    if selected is not None:
        views["Новые решения + уточнение рамок"] = GalleryPredictions(
            scores, chosen, calibrated.present_at_or_above, uncertain=uncertain,
        )
    else:
        # Show the original as selected; don't silently present a rejected model
        # as the recommended result. Candidate metrics remain available in JSON.
        result["fallback_reason"] = "No refiner met the development selection requirements"
    target = root / "gallery.html"
    write_localization_gallery(target, evaluation_root, manifest, views)
    result["gallery"] = target.name
    result["gallery_variant_count"] = len(views)
    result["frames"] = len(data)
