"""Experimental candidate-conditioned contact scores. No control or veto API."""

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from hybrid_drive.contracts.trajectory import Trajectory
from hybrid_drive.fast.forecast_proposals import ForecastObject
from hybrid_drive.fast.scene_payload import FastScene
from hybrid_drive.neural.scene_bridge import SceneBridge
from hybrid_drive.neural.scene_slow_prefix import SceneSlowPrefix
from hybrid_drive.neural.surround_features import SurroundFeatures


@dataclass(frozen=True)
class RiskRequest:
    features: SurroundFeatures
    scene: FastScene
    candidate_ids: tuple[str, ...]


class PairedSceneRisk(nn.Module):
    """Train the existing Bridge and Q-Former through a frozen binary LM readout.

    Each selected trajectory also conditions a relative actor prior at its exact
    future sample times. Relative priors contain geometry and smooth positional
    features, not labels, contact bits, risk scores, or annotated future states.
    Unseen actors remain masked. The output is a relative yes-minus-no logit,
    not a probability or clearance claim.
    """

    def __init__(self, *, visual_dim=384, prior_dim=64, llm_dim=512):
        super().__init__()
        self.bridge = SceneBridge(visual_dim, prior_dim)
        self.slow = SceneSlowPrefix(visual_dim=visual_dim, prior_dim=prior_dim, llm_dim=llm_dim)
        self.relative = nn.Sequential(nn.Linear(11, prior_dim), nn.GELU(), nn.Linear(prior_dim, prior_dim))

    def forward(self, requests, language):
        if not requests:
            raise ValueError("Need at least one risk request")
        if language.training or any(parameter.requires_grad for parameter in language.parameters()):
            raise ValueError("The language backbone must be frozen in evaluation mode")
        prior, visual, visual_mask, counts = [], [], [], []
        for request in requests:
            scene, ids = request.scene, request.candidate_ids
            candidates = {item.candidate_id: item.trajectory for item in scene.plan.candidates}
            if not ids or len(ids) != len(set(ids)) or any(key not in candidates for key in ids):
                raise ValueError("Expected unique candidate IDs in the bound Fast scene")
            tokens = self.bridge(request.features, scene)
            visible = {r.actor_id for r in tokens.records if r.kind == "actor" and r.valid}
            actors = tuple(box for box in scene.boxes if box.object_id in visible)
            counts.append(len(ids))
            for key in ids:
                owned = self.slow.candidate_mask(tokens, key)[0]
                memory = tokens.prior_tokens[0, owned]
                if actors:
                    rows = relative_actor_samples(actors, candidates[key])
                    memory = torch.cat((memory, self.relative(memory.new_tensor(rows))), dim=0)
                prior.append(memory)
                visual.append(tokens.visual_tokens[0])
                visual_mask.append(tokens.visual_mask[0])
        prior_bank = pad_sequence(prior, batch_first=True)
        lengths = torch.tensor([len(row) for row in prior], device=prior_bank.device)
        mask = torch.arange(prior_bank.shape[1], device=prior_bank.device)[None] < lengths[:, None]
        prefix = self.slow.projector(torch.stack(visual), torch.stack(visual_mask), prior_bank, mask)
        logits = language.binary_answer_logits(prefix)
        if logits.shape != (len(prior), 2) or not torch.isfinite(logits).all():
            raise ValueError("Expected finite yes/no language logits for every candidate")
        return (logits[:, 0] - logits[:, 1]).split(counts)


def relative_actor_samples(actors: tuple[ForecastObject, ...], trajectory: Trajectory) -> np.ndarray:
    """Build candidate-frame hypotheses (actor x future time), without testing contact."""
    if not actors:
        return np.empty((0, 11), dtype=np.float32)
    xy = np.array([actor.center_xyz[:2] for actor in actors])[:, None, :]
    velocity = np.array([actor.velocity_xy for actor in actors])[:, None, :]
    times = trajectory.times_s[None, :, None]
    delta = xy + times * velocity - trajectory.poses[None, :, :2]
    heading = trajectory.poses[None, :, 2]
    c, s = np.cos(heading), np.sin(heading)
    forward = delta[..., 0] * c + delta[..., 1] * s
    lateral = -delta[..., 0] * s + delta[..., 1] * c
    yaw = np.array([actor.yaw for actor in actors])[:, None] - heading
    width, length = np.array([actor.size_wlh[:2] for actor in actors]).T[:, :, None]
    # Give the projector both signed relative coordinates and continuous near-field bases.
    values = np.broadcast_arrays(
        forward / 8, lateral / 4, np.sin(yaw), np.cos(yaw), width / 4, length / 8,
        times[..., 0] / 4, 1 / (1 + ((forward - 1.25) / 3) ** 2),
        1 / (1 + (lateral / 2) ** 2), velocity[..., 0] / 20, velocity[..., 1] / 20,
    )
    return np.stack(values, axis=-1).reshape(-1, len(values)).astype(np.float32)
