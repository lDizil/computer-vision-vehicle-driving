"""A LOC-only composition; the forward path has no access to targets or IDs."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .decoder import InstructedVisionDecoder, PerceptionOutput
from .losses import PerceptionTargets
from .prior_tokenizer import PriorTokenizer
from .projector import QFormerProjector


@dataclass(frozen=True)
class LocPrediction:
    consumed_ids: Tensor
    logits: Tensor
    perception: PerceptionOutput


class LocAlignmentProbe(nn.Module):
    """Four fixed 2D grid priors and a constant consumed LOC prompt.

    The 4D prior geometry is normalized image (cx, cy, width, height), not
    ego coordinates or a target box. Only vision features vary by example.
    Camera pixels, priors, language backbones and heads retain Torch semantics.
    """

    def __init__(self, language, *, visual_dim: int, hidden_dim: int = 64, num_queries: int = 4):
        super().__init__()
        self.tokenizer = PriorTokenizer(
            visual_dim,
            geometry_dim=4,
            hidden_dim=hidden_dim,
            geometry_feature_names=("image_cx", "image_cy", "image_width", "image_height"),
        )
        self.projector = QFormerProjector(
            visual_dim,
            language.llm_dim,
            hidden_dim=hidden_dim,
            num_queries=num_queries,
            num_layers=1,
            num_heads=4,
        )
        self.decoder = InstructedVisionDecoder(
            visual_dim,
            language.llm_dim,
            language.loc_token_id,
            language.mot_token_id,
            hidden_dim=hidden_dim,
            num_layers=1,
            num_heads=4,
            motion_steps=1,
        )

    def forward(self, features: Tensor, language) -> LocPrediction:
        if not isinstance(features, Tensor) or features.ndim != 5 or features.shape[1] != 1:
            raise ValueError("The LOC pilot uses one current view [B,1,C,H,W]")
        count = features.shape[0]
        boxes = features.new_tensor(
            [
                [0, 0, 0.5, 0.5],
                [0.5, 0, 1, 0.5],
                [0, 0.5, 0.5, 1],
                [0.5, 0.5, 1, 1],
            ]
        ).expand(count, -1, -1)
        start, end = boxes[..., :2], boxes[..., 2:]
        geometry = torch.cat(((start + end) * 0.5, end - start), dim=-1)
        indices = torch.zeros((count, 4), device=features.device, dtype=torch.long)
        priors = self.tokenizer(
            features, boxes, indices, geometry, torch.ones_like(indices, dtype=torch.bool)
        )
        visual = features[:, 0].flatten(2).transpose(1, 2)
        visual_mask = torch.ones(visual.shape[:2], device=visual.device, dtype=torch.bool)
        prefix = self.projector(visual, visual_mask, priors.tokens, priors.valid_mask)
        ids = language.prompt_ids(count)
        hidden, logits = language(prefix, ids)
        answer_mask = torch.zeros_like(ids, dtype=torch.bool)
        answer_mask[:, -1] = True
        perception = self.decoder(visual, visual_mask, hidden, ids, answer_mask)
        return LocPrediction(ids, logits, perception)

    def weight_modules(self) -> dict[str, nn.Module]:
        return {"tokenizer": self.tokenizer, "projector": self.projector, "decoder": self.decoder}


def supervised_loc_loss(criterion, prediction: LocPrediction, language, presence, boxes_3d, boxes_2d):
    """Only this training-side function observes labels; CE predicts after LOC."""
    ids = prediction.consumed_ids
    count, length = ids.shape
    if (
        presence.dtype != torch.bool
        or presence.shape != (count,)
        or presence.device != ids.device
        or not (ids[:, -1] == language.loc_token_id).all()
    ):
        raise ValueError("LOC targets must align to the fixed last consumed marker")
    labels = torch.full_like(ids, -100)
    labels[:, -1] = torch.where(presence, language.yes_token_id, language.no_token_id)
    reference = prediction.logits
    targets = PerceptionTargets(
        batch_indices=torch.arange(count, device=ids.device),
        token_positions=torch.full((count,), length - 1, dtype=torch.long, device=ids.device),
        event_kinds=torch.zeros(count, dtype=torch.long, device=ids.device),
        validity=presence.to(reference.dtype),
        validity_mask=torch.ones_like(presence),
        boxes_3d=boxes_3d,
        boxes_3d_mask=presence[:, None].expand(-1, 7),
        boxes_2d=boxes_2d,
        boxes_2d_mask=presence,
        motion_xy=reference.new_zeros((count, 1, 2)),
        motion_mask=torch.zeros((count, 1), dtype=torch.bool, device=ids.device),
    )
    return criterion(reference, labels, labels != -100, prediction.perception, targets)
