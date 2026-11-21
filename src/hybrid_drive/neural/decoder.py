"""Perception heads conditioned on consumed LOC/MOT token hidden states."""

from dataclasses import dataclass
import math
from numbers import Integral, Real

import torch
from torch import nn
from torch.nn import functional as F

from .projector import _dropout, _floating_tensor, _mask, _positive_int, _visual_memory


@dataclass(frozen=True)
class PerceptionOutput:
    """Flat events in batch/token order; LOC=0 and MOT=1.

    boxes_3d is raw [x,y,z,width,length,height,yaw] in ego forward/left/up,
    metres/radians, without physical validity guarantees. boxes_2d is our
    auxiliary normalized-image xyxy head, not a projection of boxes_3d;
    corners may lie outside the image. motion_xy uses ego metres at dt, 2dt,
    ...; dt is the decoder's motion_interval_s configuration.
    """

    batch_indices: torch.Tensor
    token_positions: torch.Tensor
    event_kinds: torch.Tensor
    validity_logits: torch.Tensor
    boxes_3d: torch.Tensor
    boxes_2d: torch.Tensor
    motion_xy: torch.Tensor


class InstructedVisionDecoder(nn.Module):
    """Our randomly initialized decoder; no official weight compatibility.

    hidden states and consumed_token_ids must refer to the same positions:
    h_t is obtained after consuming token t, not the state predicting it.
    answer_mask excludes prompt and padding, not negative validity labels.
    """

    def __init__(self, visual_dim, llm_dim, loc_token_id, mot_token_id, *, hidden_dim=256,
                 num_layers=2, num_heads=8, motion_steps=8, motion_interval_s=0.5, dropout=0):
        super().__init__()
        self.visual_dim = _positive_int(visual_dim, "visual_dim")
        self.llm_dim = _positive_int(llm_dim, "llm_dim")
        self.loc_token_id = _token_id(loc_token_id, "loc_token_id")
        self.mot_token_id = _token_id(mot_token_id, "mot_token_id")
        if self.loc_token_id == self.mot_token_id:
            raise ValueError("loc_token_id and mot_token_id must be different")
        hidden_dim = _positive_int(hidden_dim, "hidden_dim")
        num_layers = _positive_int(num_layers, "num_layers")
        num_heads = _positive_int(num_heads, "num_heads")
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.motion_steps = _positive_int(motion_steps, "motion_steps")
        self.motion_interval_s = _motion_interval(motion_interval_s)
        dropout = _dropout(dropout)
        self.visual_projection = nn.Linear(self.visual_dim, hidden_dim)
        self.hidden_projection = nn.Sequential(nn.Linear(self.llm_dim, hidden_dim), nn.GELU(),
                                               nn.Linear(hidden_dim, hidden_dim))
        layer = nn.TransformerDecoderLayer(hidden_dim, num_heads, hidden_dim * 4,
                                           dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(layer, num_layers, norm=nn.LayerNorm(hidden_dim))
        self.validity_head = nn.Linear(hidden_dim, 1)
        self.box_3d_head = nn.Linear(hidden_dim, 7)
        self.box_2d_head = nn.Linear(hidden_dim, 4)
        self.motion_head = nn.Linear(hidden_dim, self.motion_steps * 2)

    def forward(self, visual_tokens, visual_mask, last_hidden_states, consumed_token_ids, answer_mask):
        visual = _visual_memory(visual_tokens, visual_mask, self.visual_dim, self.visual_projection.weight)
        batch_indices, token_positions, selected, event_kinds = self._events(
            last_hidden_states, consumed_token_ids, answer_mask, visual.shape[0])
        memory = self.visual_projection(visual)
        queries = self.hidden_projection(selected)
        if len(batch_indices):
            features = self._decode_events(queries, memory, visual_mask, batch_indices)
        else:
            # Keep empty losses backward-safe, including the visual input path.
            features = queries + memory.sum() * 0
        raw_boxes = self.box_2d_head(features)
        center = raw_boxes[:, :2].sigmoid()
        extent = F.softplus(raw_boxes[:, 2:]) + torch.finfo(raw_boxes.dtype).eps
        boxes_2d = torch.cat((center - extent / 2, center + extent / 2), dim=1)
        return PerceptionOutput(
            batch_indices=batch_indices, token_positions=token_positions, event_kinds=event_kinds,
            validity_logits=self.validity_head(features).squeeze(-1),
            boxes_3d=self.box_3d_head(features), boxes_2d=boxes_2d,
            motion_xy=self.motion_head(features).reshape(-1, self.motion_steps, 2),
        )

    def _events(self, hidden, token_ids, answer_mask, batch_size):
        if (not isinstance(hidden, torch.Tensor) or hidden.ndim != 3
                or hidden.shape[0] != batch_size or hidden.shape[2] != self.llm_dim):
            raise ValueError(f"last_hidden_states must have shape [B, L, {self.llm_dim}]")
        _floating_tensor(hidden, self.visual_projection.weight, "last_hidden_states")
        shape = hidden.shape[:2]
        if (not isinstance(token_ids, torch.Tensor) or token_ids.dtype != torch.long
                or token_ids.shape != shape or token_ids.device != hidden.device):
            raise ValueError("consumed_token_ids must be a long tensor with shape [B, L] on the input device")
        _mask(answer_mask, shape, hidden.device, "answer_mask")
        if (token_ids[answer_mask] < 0).any():
            raise ValueError("Answer token IDs must be nonnegative")
        events = answer_mask & ((token_ids == self.loc_token_id) | (token_ids == self.mot_token_id))
        batch_indices, positions = events.nonzero(as_tuple=True)
        selected = hidden[batch_indices, positions]
        if not torch.isfinite(selected).all():
            raise ValueError("Selected LOC/MOT hidden states must be finite")
        kinds = (token_ids[batch_indices, positions] == self.mot_token_id).long()
        return batch_indices, positions, selected, kinds

    def _decode_events(self, queries, memory, visual_mask, batch_indices):
        # h_t already contains its causal context; later events must not enter it.
        decoded = self.decoder(queries[:, None], memory[batch_indices],
                               memory_key_padding_mask=~visual_mask[batch_indices])
        return decoded[:, 0]


def _token_id(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def _motion_interval(value):
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not math.isfinite(value) or value <= 0):
        raise ValueError("motion_interval_s must be positive and finite")
    return float(value)
