"""Candidate-bound token selection for an uncalibrated frozen-language probe."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from hybrid_drive.contracts._validation import identifier
from hybrid_drive.neural.projector import QFormerProjector


@dataclass(frozen=True)
class SceneLanguageProbe:
    candidate_id: str
    scene_sha256: str
    prefix: Tensor
    answer_logits: Tensor
    selected_prior_count: int


class SceneSlowPrefix(nn.Module):
    """Project one candidate's typed memory and feed a local frozen backbone.

    These logits are a wiring probe, not VQA accuracy, a risk probability,
    or a decision. No language output is interpreted as a veto.
    """

    def __init__(self, *, visual_dim=384, prior_dim=64, llm_dim=512):
        super().__init__()
        self.projector=QFormerProjector(visual_dim,llm_dim,prior_dim=prior_dim,hidden_dim=64,
                                       num_queries=8,num_layers=1,num_heads=4,dropout=0)

    def forward(self,tokens,candidate_id,language):
        prefix,active=self.project(tokens,candidate_id)
        _,logits=language(prefix,language.prompt_ids(prefix.shape[0]))
        if (logits.ndim!=3 or logits.shape[0]!=1 or logits.shape[1]<1
                or max(language.yes_token_id,language.no_token_id)>=logits.shape[-1]):
            raise ValueError("Frozen language output does not match the answer vocabulary")
        values=logits[0,-1,[language.yes_token_id,language.no_token_id]]
        if not torch.isfinite(values).all():
            raise ValueError("Nonfinite language probe")
        return SceneLanguageProbe(candidate_id,tokens.scene_sha256,prefix,values,int(active.sum()))

    def project(self,tokens,candidate_id):
        active=self.candidate_mask(tokens,candidate_id)
        return self.projector(tokens.visual_tokens,tokens.visual_mask,tokens.prior_tokens,active),active

    @staticmethod
    def candidate_mask(tokens,candidate_id):
        candidate_id=identifier(candidate_id,"candidate_id")
        memory=tokens.prior_tokens
        if (memory.ndim!=3 or memory.shape[0]!=1 or len(tokens.records)!=memory.shape[1]
                or tokens.prior_mask.shape!=memory.shape[:2] or tokens.prior_mask.dtype!=torch.bool):
            raise ValueError("Token metadata must match one typed prior bank")
        if any((r.kind == "actor" and r.candidate_id is not None)
               or (r.kind in ("road", "trajectory") and
                   (not isinstance(r.candidate_id, str) or not r.candidate_id.strip()))
               or r.kind not in ("actor", "road", "trajectory") for r in tokens.records):
            raise ValueError("Invalid binding of typed tokens to a candidate")
        owned=torch.tensor([r.kind == "actor" or r.candidate_id == candidate_id for r in tokens.records],
                           dtype=torch.bool,device=memory.device)[None]
        active=tokens.prior_mask & owned
        poses=torch.tensor([r.kind=="trajectory" and r.candidate_id==candidate_id for r in tokens.records],
                           dtype=torch.bool,device=memory.device)
        if not (active[0] & poses).any():
            raise ValueError("No valid pose tokens for the selected candidate")
        return active
