"""Fit only the small trainable scene-to-language stack on paired controls."""

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F

from hybrid_drive.experiments.paired_slow_metrics import TIE_MARGIN
from hybrid_drive.neural.paired_scene_risk import PairedSceneRisk, RiskRequest


FIT_SEED = 611
CONTROL_COUNTS = {"train": 8, "development": 6, "heldout": 6}
FIT_EPOCHS = 8
GROUP_BATCH = 8
LEARNING_RATE = 0.001


@dataclass(frozen=True)
class LabeledControl:
    requests: tuple[RiskRequest, RiskRequest]
    blocked: tuple[int, int]

    def __post_init__(self):
        first, second = self.requests
        if (first.features is not second.features or first.scene.snapshot_id != second.scene.snapshot_id
                or len(first.candidate_ids) != 2 or first.candidate_ids != second.candidate_ids
                or self.blocked not in ((0, 1), (1, 0))):
            raise ValueError("Expected the same two candidates and one opposite blocked choice per world")


def initialized_model(**dimensions):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(FIT_SEED)
        return PairedSceneRisk(**dimensions)


def fit_paired(model, language, train, development, *, epochs=FIT_EPOCHS):
    if not train or not development or type(epochs) is not int or epochs < 1:
        raise ValueError("Need nonempty train and development controls and a positive epoch count")
    train_ids = {row.requests[0].scene.snapshot_id for row in train}
    development_ids = {row.requests[0].scene.snapshot_id for row in development}
    if train_ids & development_ids:
        raise ValueError("Development snapshots must be separate from optimizer inputs")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    rng = np.random.default_rng(FIT_SEED)
    before = deepcopy(model.state_dict())
    best_key, best_state, best_epoch = None, None, None
    history = []
    for epoch in range(epochs + 1):
        train_loss = None
        if epoch:
            model.train()
            batch_loss = []
            ids = rng.permutation(len(train))
            for offset in range(0, len(ids), GROUP_BATCH):
                batch = [train[i] for i in ids[offset:offset + GROUP_BATCH]]
                optimizer.zero_grad(set_to_none=True)
                gaps = _labeled_gaps(model, language, batch)
                loss = F.softplus(-gaps).mean()
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite paired contact loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 4, error_if_nonfinite=True)
                optimizer.step()
                batch_loss.append((float(loss.detach()), len(batch)))
            train_loss = sum(value * count for value, count in batch_loss) / len(train)
        with torch.no_grad():
            model.eval()
            splits = [development[i:i + GROUP_BATCH] for i in range(0, len(development), GROUP_BATCH)]
            gaps = torch.cat([_labeled_gaps(model, language, rows) for rows in splits])
            loss = float(F.softplus(-gaps).mean())
            both = int((gaps > TIE_MARGIN).all(dim=1).sum())
            per_world = int((gaps > TIE_MARGIN).sum())
        history.append({"epoch": epoch, "train_loss": train_loss, "development_loss": loss,
                        "development_both_correct": both, "development_worlds_correct": per_world})
        print(f"Slow fit epoch {epoch}/{epochs}: development both={both}/{len(development)}, loss={loss:.4f}",
              flush=True)
        key = (-both, loss)
        if best_key is None or key < best_key:
            best_key, best_state, best_epoch = key, deepcopy(model.state_dict()), epoch
    if any(parameter.grad is not None or parameter.requires_grad for parameter in language.parameters()):
        raise ValueError("Frozen language parameters received optimizer gradients")
    changed = {name: sum(not torch.equal(value, before[key]) for key, value in best_state.items()
                         if key.startswith(name + ".")) for name in ("bridge", "slow", "relative")}
    model.load_state_dict(best_state, strict=True)
    return {"epochs": history, "checkpoint_epoch": best_epoch, "changed_parameter_tensors": changed,
            "trainable_parameters": sum(p.numel() for p in model.parameters()),
            "frozen_language_parameters": sum(p.numel() for p in language.parameters()),
            "train_groups": len(train), "development_groups": len(development),
            "batch_groups": GROUP_BATCH, "train_seed": FIT_SEED, "learning_rate": LEARNING_RATE}


def _labeled_gaps(model, language, controls):
    risks = torch.stack(model([req for group in controls for req in group.requests], language)).reshape(-1, 2, 2)
    target = torch.tensor([group.blocked for group in controls], device=risks.device).unsqueeze(-1)
    return (risks.gather(-1, target) - risks.gather(-1, 1 - target)).squeeze(-1)
