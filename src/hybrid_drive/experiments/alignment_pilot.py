"""Recording-disjoint real-data LOC pilot with a matched no-image control."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import math
import os
from pathlib import Path
import time

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from hybrid_drive.assets.files import check_components
from hybrid_drive.neural.alignment_data import AlignmentDataset, collate_alignment
from hybrid_drive.neural.loc_probe import LocAlignmentProbe, supervised_loc_loss
from hybrid_drive.neural.losses import Reason2DriveLoss
from hybrid_drive.neural.pretrained_alignment import load_alignment_backbones
from .alignment_metrics import LocMetricAccumulator
from .alignment_report import write_alignment_report
from .bounded_io import require_hex
from .io import write_json
from .neural_checkpoint import load_neural_weights, save_neural_weights
from .protocol import content_hash


SCOPE = "openscene_recording_disjoint_loc_alignment_pilot_v1"
ARMS = ("image_features", "constant_zero_features")
EXPECTED_COUNTS = {"train": 64, "holdout": 32}


@dataclass(frozen=True)
class PilotRecipe:
    epochs: int = 2
    batch_size: int = 4
    seed: int = 0

    def __post_init__(self):
        for name, lower, upper in (("epochs", 1, 4), ("batch_size", 1, 8), ("seed", 0, 2**32 - 1)):
            value = getattr(self, name)
            if type(value) is not int or not lower <= value <= upper:
                raise ValueError(f"{name} must be an integer in [{lower}, {upper}]")


@dataclass(frozen=True)
class _Examples:
    features: Tensor
    presence: Tensor
    boxes_3d: Tensor
    boxes_2d: Tensor
    feature_wall_s: float

    @property
    def count(self):
        return len(self.presence)

    def select(self, indices, arm):
        features = self.features[indices]
        if arm == "constant_zero_features":
            features = torch.zeros_like(features)
        return features, self.presence[indices], self.boxes_3d[indices], self.boxes_2d[indices]


def run_alignment_pilot(data_root, model_root, output_root, *, revision, recipe=PilotRecipe()):
    require_hex(revision, 40, "revision")
    if not isinstance(recipe, PilotRecipe):
        raise ValueError("Expected PilotRecipe")
    output_root = Path(os.path.abspath(output_root))
    for source in (data_root, model_root):
        source = Path(os.path.abspath(source))
        if output_root == source or source in output_root.parents:
            raise ValueError("Output must be separate from dataset and backbone inputs")
    datasets = {split: AlignmentDataset(data_root, split) for split in EXPECTED_COUNTS}
    _verify_splits(datasets)
    check_components(output_root)
    run_dir = output_root / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema": "hybrid-drive-alignment-pilot-v1",
        "scope": SCOPE,
        "source_revision": revision,
        "dataset_hash": datasets["train"].dataset_hash,
        "recordings": {split: dataset.recording_id for split, dataset in datasets.items()},
        "planned_examples": EXPECTED_COUNTS.copy(),
        "recipe": asdict(recipe),
        "optimizer": {"name": "AdamW", "learning_rate": 0.002, "weight_decay": 0},
        "core": {"hidden_dim": 64, "num_queries": 4, "prior_width": 768, "priors": "fixed_image_grid_2x2"},
        "arms": list(ARMS),
        "text_supervision": "yes_or_no_next_token_after_fixed_consumed_LOC",
        "objective": "LOC_presence_and_annotated_box_only_no_MOT",
        "holdout_used_for_optimization_or_checkpoint_selection": False,
        "training_label_inputs": False,
        "driving_quality_claim": False,
        "full_stage1_claim": False,
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "device": "cpu",
    }
    write_json(run_dir / "manifest.json", manifest)
    result = {"status": "running", "source_revision": revision, "scope": SCOPE, "arms": {}}
    write_json(run_dir / "result.json", result)
    trace = []
    try:
        with torch.random.fork_rng(devices=[]), torch.device("cpu"):
            _execute(run_dir, manifest, result, trace, datasets, model_root, recipe)
        result["status"] = "passed"
    except KeyboardInterrupt:
        result.update(status="interrupted", error="Interrupted by operator")
    except Exception as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    write_json(run_dir / "manifest.json", manifest)
    result["manifest_hash"] = content_hash(manifest)
    result["optimizer_steps_completed"] = len(trace)
    result["driving_quality_claim"] = False
    result["full_stage1_claim"] = False
    write_json(run_dir / "trace.json", trace)
    write_json(run_dir / "result.json", result)
    write_alignment_report(run_dir, manifest, result, trace)
    return run_dir, result


def _verify_splits(datasets):
    train, holdout = datasets["train"], datasets["holdout"]
    if (
        train.recording_id == holdout.recording_id
        or train.dataset_hash != holdout.dataset_hash
        or set(train.sample_ids) & set(holdout.sample_ids)
    ):
        raise ValueError("Alignment requires distinct recordings and one shared frozen manifest")
    for split, count in EXPECTED_COUNTS.items():
        dataset = datasets[split]
        if len(dataset) != count or len(set(dataset.sample_ids)) != count:
            raise ValueError(f"Expected all {count} fixed {split} examples, without duplicates")


def _execute(run_dir, manifest, result, trace, datasets, model_root, recipe):
    vision, language, provenance = load_alignment_backbones(model_root)
    for model in (vision, language):
        if model.training or any(p.requires_grad or p.grad is not None for p in model.parameters()):
            raise ValueError("Pretrained backbones must start frozen in eval mode")
    frozen = {"vision": _state_hash(vision), "language": _state_hash(language)}
    manifest["backbones"] = provenance
    write_json(run_dir / "manifest.json", manifest)
    examples = {split: _encode(dataset, vision, recipe.batch_size) for split, dataset in datasets.items()}
    result["feature_encoding_wall_s"] = {split: values.feature_wall_s for split, values in examples.items()}
    generator = torch.Generator().manual_seed(recipe.seed)
    orders = [torch.randperm(examples["train"].count, generator=generator) for _ in range(recipe.epochs)]
    manifest["train_order_hash"] = content_hash([order.tolist() for order in orders])
    write_json(run_dir / "manifest.json", manifest)
    criterion = Reason2DriveLoss()
    common_init = None
    for arm in ARMS:
        torch.manual_seed(recipe.seed)
        probe = LocAlignmentProbe(language, visual_dim=vision.visual_dim)
        initial_hash = _state_hash(probe)
        if common_init is None:
            common_init = initial_hash
        elif common_init != initial_hash:
            raise ValueError("Paired arms did not start from identical weights")
        print(f"Training {arm}: {recipe.epochs} epochs on {examples['train'].count} examples", flush=True)
        result["arms"][arm] = _fit_arm(
            run_dir,
            arm,
            probe,
            language,
            criterion,
            examples,
            orders,
            recipe,
            manifest,
            trace,
            initial_hash,
        )
        write_json(run_dir / "result.json", result)
    result["frozen_state_unchanged"] = frozen == {
        "vision": _state_hash(vision),
        "language": _state_hash(language),
    }
    result["frozen_parameters_with_gradients"] = sum(
        p.grad is not None for model in (vision, language) for p in model.parameters()
    )
    if not result["frozen_state_unchanged"] or result["frozen_parameters_with_gradients"]:
        raise ValueError("Frozen pretrained state or gradients changed")


@torch.no_grad()
def _encode(dataset, vision, batch_size):
    features, presence, boxes_3d, boxes_2d = [], [], [], []
    ids = []
    started = time.monotonic()
    for batch in DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_alignment
    ):
        converted = vision(batch.images)
        if (
            converted.ndim != 5
            or converted.shape[:3] != (len(batch.sample_ids), 1, vision.visual_dim)
            or min(converted.shape[3:]) <= 0
            or not torch.isfinite(converted).all()
        ):
            raise ValueError("Frozen vision returned invalid features")
        features.append(converted)
        presence.append(batch.presence)
        boxes_3d.append(batch.boxes_3d)
        boxes_2d.append(batch.boxes_2d)
        ids.extend(batch.sample_ids)
    if tuple(ids) != dataset.sample_ids:
        raise ValueError("DataLoader selection or order differs from the fixed split")
    return _Examples(
        *(torch.cat(items) for items in (features, presence, boxes_3d, boxes_2d)), time.monotonic() - started
    )


def _fit_arm(
    run_dir, arm, probe, language, criterion, examples, orders, recipe, manifest, trace, initial_hash
):
    train, holdout = examples["train"], examples["holdout"]
    initial = _evaluate(probe, language, criterion, holdout, arm, recipe.batch_size)
    optimiser = torch.optim.AdamW(probe.parameters(), lr=0.002, weight_decay=0)
    probe.train()
    # CE alone must reach the bridge through the frozen language model. The
    # decoder's direct visual path cannot satisfy this check.
    first = train.select(orders[0][: recipe.batch_size], arm)
    output = probe(first[0], language)
    text = supervised_loc_loss(criterion, output, language, *first[1:]).text
    text_gradient_norms = {}
    groups = probe.weight_modules()
    bridge = [parameter for name in ("tokenizer", "projector") for parameter in groups[name].parameters()]
    gradients = torch.autograd.grad(text, bridge)
    offset = 0
    for name in ("tokenizer", "projector"):
        end = offset + sum(1 for _ in groups[name].parameters())
        text_gradient_norms[name] = _gradient_norm(gradients[offset:end])
        offset = end
    del first, output, text, gradients

    for epoch, order in enumerate(orders, start=1):
        for indices in order.split(recipe.batch_size):
            started = time.monotonic()
            features, presence, box3, box2 = train.select(indices, arm)
            optimiser.zero_grad(set_to_none=True)
            loss = supervised_loc_loss(criterion, probe(features, language), language, presence, box3, box2)
            loss.total.backward()
            norms = {
                name: _gradient_norm([p.grad for p in module.parameters()]) for name, module in groups.items()
            }
            optimiser.step()
            if any(not torch.isfinite(parameter).all() for parameter in probe.parameters()):
                raise ValueError("Alignment optimizer produced nonfinite weights")
            trace.append(
                {
                    "arm": arm,
                    "epoch": epoch,
                    "step": len(trace) + 1,
                    "examples": len(presence),
                    "positives": int(presence.sum()),
                    "loss": float(loss.total.detach()),
                    "text": float(loss.text.detach()),
                    "classification": float(loss.classification.detach()),
                    "boxes_3d": float(loss.boxes_3d.detach()),
                    "giou_2d": float(loss.giou_2d.detach()),
                    "gradient_norms": norms,
                    "wall_s": time.monotonic() - started,
                }
            )
            write_json(run_dir / "trace.json", trace)
        print(f"{arm}: epoch {epoch}/{len(orders)} complete", flush=True)
    final_hash = _state_hash(probe)
    if final_hash == initial_hash:
        raise ValueError("The trainable LOC core did not update")
    final_train = _evaluate(probe, language, criterion, train, arm, recipe.batch_size)
    final_holdout = _evaluate(probe, language, criterion, holdout, arm, recipe.batch_size)
    metadata = {
        "scope": SCOPE,
        "arm": arm,
        "source_revision": manifest["source_revision"],
        "dataset_hash": manifest["dataset_hash"],
        "backbones_hash": content_hash(manifest["backbones"]),
        "recipe": manifest["recipe"],
        "core": manifest["core"],
        "train_order_hash": manifest["train_order_hash"],
    }
    checkpoint_path = run_dir / (arm + ".safetensors")
    checkpoint = save_neural_weights(checkpoint_path, probe.weight_modules(), metadata=metadata)
    _assert_checkpoint_roundtrip(probe, language, holdout, arm, checkpoint_path, checkpoint, metadata)
    return {
        "initial_weight_hash": initial_hash,
        "final_weight_hash": final_hash,
        "text_only_bridge_gradients": text_gradient_norms,
        "holdout_before": initial,
        "holdout_after": final_holdout,
        "train_after": final_train,
        "checkpoint": checkpoint | {"file": checkpoint_path.name},
        "checkpoint_roundtrip_bitwise": True,
        "steps": sum(row["arm"] == arm for row in trace),
    }


@torch.no_grad()
def _evaluate(probe, language, criterion, examples, arm, batch_size):
    probe.eval()
    metrics = LocMetricAccumulator()
    for indices in torch.arange(examples.count).split(batch_size):
        features, presence, box3, box2 = examples.select(indices, arm)
        prediction = probe(features, language)
        loss = supervised_loc_loss(criterion, prediction, language, presence, box3, box2)
        metrics.add(prediction, loss, language, presence, box3, box2)
    return metrics.summary()


@torch.no_grad()
def _assert_checkpoint_roundtrip(probe, language, holdout, arm, path, checkpoint, metadata):
    restored = LocAlignmentProbe(language, visual_dim=probe.tokenizer.image_channels).eval()
    load_neural_weights(
        path, restored.weight_modules(), expected_sha256=checkpoint["sha256"], expected_metadata=metadata
    )
    if _state_hash(restored) != _state_hash(probe):
        raise ValueError("Restored LOC state is not bitwise identical")
    features = holdout.select(torch.arange(1), arm)[0]
    original, replayed = probe.eval()(features, language), restored(features, language)
    pairs = [(original.logits, replayed.logits), (original.consumed_ids, replayed.consumed_ids)]
    pairs.extend(
        (value, vars(replayed.perception)[name]) for name, value in vars(original.perception).items()
    )
    if not all(torch.equal(a, b) for a, b in pairs):
        raise ValueError("Restored weights differ from the evaluated LOC model")


def _gradient_norm(gradients):
    values = []
    for gradient in gradients:
        if gradient is None or not torch.isfinite(gradient).all():
            raise ValueError("Missing or nonfinite trainable gradient")
        values.append(gradient.detach().double().square().sum())
    norm = float(torch.stack(values).sum().sqrt())
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError("A trainable group did not receive a finite nonzero gradient")
    return norm


def _state_hash(module):
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update((name + ":" + str(tensor.dtype) + ":" + str(tuple(tensor.shape)) + "\n").encode())
        digest.update(memoryview(tensor.detach().cpu().contiguous().numpy()).cast("B"))
    return digest.hexdigest()
