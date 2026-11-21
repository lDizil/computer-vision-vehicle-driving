"""Actual CPU optimization of our core through miniature frozen fixture models.

This is synthetic mathematics, not real encoder/LLM alignment, a Reason2Drive
checkpoint, road data training, held-out quality, or a driving improvement.
"""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import torch
from torch import nn
from torchvision.models.vision_transformer import VisionTransformer

from hybrid_drive.assets.files import check_components
from hybrid_drive.neural.prior_tokenizer import PriorTokenizer
from hybrid_drive.neural.vision import ViTPatchEncoder
from hybrid_drive.neural.projector import QFormerProjector
from hybrid_drive.neural.decoder import InstructedVisionDecoder
from hybrid_drive.neural.losses import PerceptionTargets, Reason2DriveLoss
from .bounded_io import require_hex
from .io import write_json
from .neural_checkpoint import load_neural_weights, save_neural_weights


SCOPE = "synthetic_frozen_encoder_lm_math_smoke"
LOC_ID, MOT_ID = 2, 3
CONFIG = {
    "tokenizer": {"image_channels": 16, "geometry_dim": 7, "token_dim": 768, "hidden_dim": 32},
    "projector": {
        "visual_dim": 16,
        "llm_dim": 32,
        "prior_dim": 768,
        "hidden_dim": 32,
        "num_queries": 4,
        "num_layers": 1,
        "num_heads": 4,
        "dropout": 0,
    },
    "decoder": {
        "visual_dim": 16,
        "llm_dim": 32,
        "loc_token_id": LOC_ID,
        "mot_token_id": MOT_ID,
        "hidden_dim": 32,
        "num_layers": 1,
        "num_heads": 4,
        "motion_steps": 2,
        "motion_interval_s": 0.5,
        "dropout": 0,
    },
}


class _FrozenFixtureLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(16, 32)
        layer = nn.TransformerEncoderLayer(32, 4, 64, dropout=0, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
        self.head = nn.Linear(32, 16)
        self.requires_grad_(False).eval()

    def forward(self, prefix, consumed_ids):
        combined = torch.cat((prefix, self.embedding(consumed_ids)), dim=1)
        mask = torch.ones(
            (combined.shape[1], combined.shape[1]), device=combined.device, dtype=torch.bool
        ).triu(1)
        hidden = self.transformer(combined, mask=mask)[:, prefix.shape[1] :]
        return hidden, self.head(hidden)


def run_neural_core_smoke(output_root, *, revision, steps=12, seed=0):
    require_hex(revision, 40, "revision")
    if type(steps) is not int or not 1 <= steps <= 200 or type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Expected 1..200 optimizer steps and a bounded integer seed")
    run_dir = Path(output_root).absolute() / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    check_components(run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema": "hybrid-drive-neural-core-smoke-v1",
        "scope": SCOPE,
        "source_revision": revision,
        "seed": seed,
        "planned_steps": steps,
        "config": CONFIG,
        "fixture": "mixed-loc-mot-v1",
        "optimizer": {"name": "AdamW", "lr": 0.003, "weight_decay": 0},
        "vision_encoder": "newly initialized frozen tiny torchvision VisionTransformer fixture, not pretrained",
        "llm": "newly initialized frozen tiny causal Transformer fixture, not Qwen/Reason2Drive",
        "real_road_data": False,
        "aligned_checkpoint_claim": False,
        "driving_quality_claim": False,
        "device": "cpu",
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
    }
    write_json(run_dir / "manifest.json", manifest)
    rows = []
    result = {"status": "failed", "scope": SCOPE, "source_revision": revision, "planned_steps": steps}
    try:
        with torch.random.fork_rng(devices=[]), torch.device("cpu"):
            torch.manual_seed(seed)
            outcome = _optimize(run_dir, manifest, rows)
        result.update(outcome, status="passed")
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["completed_steps"] = len(rows)
    write_json(run_dir / "steps.json", rows)
    write_json(run_dir / "result.json", result)
    _report(run_dir, result, rows)
    return run_dir, result


def _optimize(run_dir, manifest, rows):
    vision = (
        ViTPatchEncoder(
            VisionTransformer(
                image_size=28,
                patch_size=7,
                num_layers=1,
                num_heads=2,
                hidden_dim=16,
                mlp_dim=32,
                dropout=0,
                attention_dropout=0,
                num_classes=16,
            )
        )
        .requires_grad_(False)
        .eval()
    )
    language = _FrozenFixtureLM()
    frozen = {"vision": _weights(vision), "language": _weights(language)}
    modules = _modules()
    images, boxes, indices, geometry, prior_mask, ids, answer_mask, targets = _fixture()
    with torch.no_grad():
        features = vision(images)
    visual = features.permute(0, 1, 3, 4, 2).reshape(2, -1, 16)
    visual_mask = torch.ones(visual.shape[:2], dtype=torch.bool)
    criterion = Reason2DriveLoss()
    text_targets = torch.tensor([[LOC_ID, 4, MOT_ID, 0], [LOC_ID, 5, MOT_ID, 0]])
    text_mask = torch.ones_like(ids, dtype=torch.bool)

    def forward(models):
        priors = models["tokenizer"](features, boxes, indices, geometry, prior_mask)
        prefix = models["projector"](visual, visual_mask, priors.tokens, priors.valid_mask)
        # Frozen LLM parameters must not mean no_grad here: the prefix still
        # needs a derivative through the frozen transformer to the bridge.
        hidden, logits = language(prefix, ids)
        predictions = models["decoder"](visual, visual_mask, hidden, ids, answer_mask)
        loss = criterion(logits, text_targets, text_mask, predictions, targets)
        return prefix, logits, predictions, loss

    with torch.no_grad():
        initial = float(forward(modules)[-1].total)
    before = {name: _weights(module) for name, module in modules.items()}
    optimizer = torch.optim.AdamW(
        [p for module in modules.values() for p in module.parameters()], lr=0.003, weight_decay=0
    )
    for step in range(manifest["planned_steps"]):
        started = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        _, _, _, loss = forward(modules)
        loss.total.backward()
        gradients = {name: _gradient_norm(module) for name, module in modules.items()}
        if not all(value > 0 for value in gradients.values()):
            raise ValueError("A trainable group did not receive a nonzero gradient through the frozen LM")
        if any(p.grad is not None for module in (vision, language) for p in module.parameters()):
            raise ValueError("Frozen fixture parameters received gradients")
        optimizer.step()
        if any(not torch.isfinite(p).all() for module in modules.values() for p in module.parameters()):
            raise ValueError("An optimizer step produced nonfinite parameters")
        rows.append(
            {
                "step": step,
                "total": float(loss.total.detach()),
                "text": float(loss.text.detach()),
                "classification": float(loss.classification.detach()),
                "boxes_3d": float(loss.boxes_3d.detach()),
                "giou_2d": float(loss.giou_2d.detach()),
                "motion": float(loss.motion.detach()),
                "gradient_norms": gradients,
                "wall_s": time.monotonic() - started,
            }
        )
        write_json(run_dir / "steps.json", rows)
    for name, module in (("vision", vision), ("language", language)):
        if any(not torch.equal(value, frozen[name][key]) for key, value in module.state_dict().items()):
            raise ValueError("Frozen fixture weights changed")
    changed = {
        name: any(not torch.equal(value, before[name][key]) for key, value in module.state_dict().items())
        for name, module in modules.items()
    }
    if not all(changed.values()):
        raise ValueError("A trainable group did not update")
    for module in modules.values():
        module.eval()
    with torch.no_grad():
        original = forward(modules)
    metadata = {
        "scope": SCOPE,
        "source_revision": manifest["source_revision"],
        "seed": manifest["seed"],
        "config": CONFIG,
        "fixture": manifest["fixture"],
        "completed_steps": len(rows),
    }
    checkpoint = save_neural_weights(run_dir / "core.safetensors", modules, metadata=metadata)
    restored = _modules()
    load_neural_weights(
        run_dir / "core.safetensors",
        restored,
        expected_sha256=checkpoint["sha256"],
        expected_metadata=metadata,
    )
    for module in restored.values():
        module.eval()
    with torch.no_grad():
        replayed = forward(restored)
    for first, second in zip(_outputs(original), _outputs(replayed)):
        if not torch.equal(first, second):
            raise ValueError("Restored neural weights do not reproduce outputs bit-for-bit")
    return {
        "initial_loss": initial,
        "final_loss": float(original[-1].total),
        "loss_decreased": float(original[-1].total) < initial,
        "updated_groups": changed,
        "frozen_parameters_unchanged": True,
        "frozen_parameters_with_gradients": 0,
        "trainable_parameter_counts": {
            name: sum(p.numel() for p in module.parameters()) for name, module in modules.items()
        },
        "checkpoint": checkpoint,
        "roundtrip_bitwise_equal": True,
        "output_sha256": hashlib.sha256(
            b"".join(value.detach().cpu().numpy().tobytes() for value in _outputs(original))
        ).hexdigest(),
        "aligned_checkpoint_claim": False,
        "driving_quality_claim": False,
    }


def _modules():
    return {
        "tokenizer": PriorTokenizer(**CONFIG["tokenizer"]),
        "projector": QFormerProjector(**CONFIG["projector"]),
        "decoder": InstructedVisionDecoder(**CONFIG["decoder"]),
    }


def _weights(module):
    return {key: value.detach().clone() for key, value in module.state_dict().items()}


def _gradient_norm(module):
    values = []
    for parameter in module.parameters():
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise ValueError("A trainable parameter has absent or nonfinite gradients")
        values.append(parameter.grad.detach().double().square().sum())
    return float(torch.stack(values).sum().sqrt())


def _outputs(forward_result):
    prefix, logits, predictions, _ = forward_result
    return (
        prefix,
        logits,
        predictions.validity_logits,
        predictions.boxes_3d,
        predictions.boxes_2d,
        predictions.motion_xy,
    )


def _fixture():
    images = torch.randn(2, 2, 3, 28, 28)
    boxes = torch.tensor([0.1, 0.1, 0.8, 0.9]).expand(2, 2, 4).clone()
    indices = torch.tensor([[0, 1], [1, 0]])
    geometry = torch.randn(2, 2, 7)
    geometry[:, :, 3:6] = geometry[:, :, 3:6].abs() + 1
    prior_mask = torch.tensor([[True, False], [True, True]])
    geometry[~prior_mask] = float("nan")
    boxes[~prior_mask] = float("nan")
    indices[~prior_mask] = -100
    ids = torch.tensor([[1, LOC_ID, 4, MOT_ID], [1, LOC_ID, 5, MOT_ID]])
    answer_mask = torch.tensor([[False, True, True, True], [False, True, True, True]])
    boxes_3d = torch.full((4, 7), float("nan"))
    boxes_3d[0] = torch.tensor([1, 0, 0, 1, 2, 1, 3.04])
    boxes_3d_mask = torch.zeros(4, 7, dtype=torch.bool)
    boxes_3d_mask[0] = True
    boxes_2d = torch.full((4, 4), float("nan"))
    boxes_2d[0] = torch.tensor([0.1, 0.2, 0.6, 0.7])
    motion = torch.full((4, 2, 2), float("nan"))
    motion[1] = torch.tensor([[0.5, 0], [1, 0.2]])
    motion[3, 0] = torch.tensor([0.1, 0.3])
    targets = PerceptionTargets(
        torch.tensor([0, 0, 1, 1]),
        torch.tensor([1, 3, 1, 3]),
        torch.tensor([0, 1, 0, 1]),
        torch.tensor([1.0, 1.0, 0.0, 1.0]),
        torch.ones(4, dtype=torch.bool),
        boxes_3d,
        boxes_3d_mask,
        boxes_2d,
        torch.tensor([True, False, False, False]),
        motion,
        torch.tensor([[False, False], [True, True], [False, False], [True, False]]),
    )
    return images, boxes, indices, geometry, prior_mask, ids, answer_mask, targets


def _report(root, result, rows):
    import html

    body = "".join(
        f"<tr><td>{row['step']}</td><td>{row['total']:.6f}</td><td>{row['text']:.6f}</td>"
        f"<td>{row['classification']:.6f}</td><td>{row['boxes_3d']:.6f}</td><td>{row['giou_2d']:.6f}</td><td>{row['motion']:.6f}</td></tr>"
        for row in rows
    )
    document = (
        "<!doctype html><meta charset='utf-8'><title>Neural core mathematical smoke</title>"
        "<style>body{font:16px system-ui;max-width:1100px;margin:40px auto;padding:0 20px}table{border-collapse:collapse}"
        "th,td{padding:9px;border:1px solid #ccd}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style>"
        "<h1>Trainable neural core: CPU mathematical smoke</h1>"
        "<p>Actual Torch optimizer steps through frozen miniature fixture models. All inputs are synthetic. "
        "This is not a pretrained encoder/LLM, real-data alignment, validation quality, Reason2Drive reproduction or driving performance.</p>"
        "<pre>"
        + html.escape(json.dumps(result, indent=2))
        + "</pre>"
        + _loss_plot(rows)
        + "<table><tr><th>Step</th><th>Total</th><th>Text</th>"
        "<th>BCE</th><th>3D SmoothL1</th><th>2D GIoU</th><th>Motion</th></tr>" + body + "</table>"
    )
    (root / "report.html").write_text(document)


def _loss_plot(rows):
    if not rows:
        return ""
    maximum = max(row["total"] for row in rows) * 1.1
    if maximum <= 0:
        maximum = 1.0

    def points(key):
        return " ".join(
            f"{50+index*760/max(1,len(rows)-1):.2f},{230-190*row[key]/maximum:.2f}"
            for index, row in enumerate(rows)
        )

    ticks = "".join(
        f"<text x='7' y='{234-190*fraction:.0f}'>{maximum*fraction:.2f}</text>" for fraction in (0, 0.5, 1)
    )
    return (
        "<svg viewBox='0 0 860 280' role='img' aria-label='Synthetic training loss over optimizer steps' "
        "style='width:100%;font:13px system-ui;background:#f6f8fc'>"
        "<path d='M50 30V230H820' fill='none' stroke='#64748b'/>"
        + ticks
        + "<polyline fill='none' stroke='#2563eb' stroke-width='3' points='"
        + points("total")
        + "'/>"
        "<polyline fill='none' stroke='#7c3aed' stroke-width='2' points='" + points("text") + "'/>"
        "<text x='65' y='22' fill='#2563eb'>Total</text><text x='125' y='22' fill='#7c3aed'>Text CE</text>"
        "<text x='50' y='251'>0</text><text x='800' y='251'>" + str(len(rows) - 1) + "</text>"
        "<text x='360' y='272'>Synthetic optimizer step</text></svg>"
    )
