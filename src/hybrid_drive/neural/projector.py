"""Trainable query projection; not a checkpoint-compatible LAVIS Q-former."""

import math
from numbers import Integral, Real

import torch
from torch import nn


class QFormerProjector(nn.Module):
    """Our Q-former-like self/cross-attention bridge, initialized from scratch.

    Masks use True for valid tokens. Positional/frame information belongs to
    the supplied visual features; this module adds only visual/prior types.
    """

    def __init__(self, visual_dim, llm_dim, *, prior_dim=768, hidden_dim=256,
                 num_queries=32, num_layers=2, num_heads=8, dropout=0):
        super().__init__()
        self.visual_dim = _positive_int(visual_dim, "visual_dim")
        self.llm_dim = _positive_int(llm_dim, "llm_dim")
        self.prior_dim = _positive_int(prior_dim, "prior_dim")
        hidden_dim = _positive_int(hidden_dim, "hidden_dim")
        self.num_queries = _positive_int(num_queries, "num_queries")
        num_layers = _positive_int(num_layers, "num_layers")
        num_heads = _positive_int(num_heads, "num_heads")
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        dropout = _dropout(dropout)
        self.visual_projection = nn.Linear(self.visual_dim, hidden_dim)
        self.prior_projection = nn.Linear(self.prior_dim, hidden_dim)
        self.type_embeddings = nn.Parameter(torch.empty(2, hidden_dim))
        self.query_tokens = nn.Parameter(torch.empty(1, self.num_queries, hidden_dim))
        nn.init.normal_(self.type_embeddings, std=0.02)
        nn.init.normal_(self.query_tokens, std=0.02)
        layer = nn.TransformerDecoderLayer(hidden_dim, num_heads, hidden_dim * 4,
                                           dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(layer, num_layers, norm=nn.LayerNorm(hidden_dim))
        self.output_projection = nn.Linear(hidden_dim, self.llm_dim)

    def forward(self, visual_tokens, visual_mask, prior_tokens=None, prior_mask=None):
        visual = _visual_memory(visual_tokens, visual_mask, self.visual_dim, self.visual_projection.weight)
        memory = self.visual_projection(visual) + self.type_embeddings[0]
        memory_mask = visual_mask
        if (prior_tokens is None) != (prior_mask is None):
            raise ValueError("prior_tokens and prior_mask must be supplied together")
        if prior_tokens is not None:
            priors = _masked_tokens(prior_tokens, prior_mask, self.prior_dim,
                                    self.prior_projection.weight, "prior", batch_size=visual.shape[0])
            projected = self.prior_projection(priors) + self.type_embeddings[1]
            memory = torch.cat((memory, projected), dim=1)
            memory_mask = torch.cat((visual_mask, prior_mask), dim=1)
        queries = self.query_tokens.expand(visual.shape[0], -1, -1)
        aligned = self.decoder(queries, memory, memory_key_padding_mask=~memory_mask)
        return self.output_projection(aligned)


def _visual_memory(tokens, mask, width, parameter):
    values = _masked_tokens(tokens, mask, width, parameter, "visual")
    if not mask.any(dim=1).all():
        raise ValueError("Every batch row must contain a valid visual token")
    return values


def _masked_tokens(tokens, mask, width, parameter, name, *, batch_size=None):
    if (not isinstance(tokens, torch.Tensor) or tokens.ndim != 3 or tokens.shape[0] == 0
            or tokens.shape[2] != width):
        raise ValueError(f"{name}_tokens must have shape [B, N, {width}] with B > 0")
    _floating_tensor(tokens, parameter, f"{name}_tokens")
    if batch_size is not None and tokens.shape[0] != batch_size:
        raise ValueError(f"{name}_tokens batch size does not match visual_tokens")
    _mask(mask, tokens.shape[:2], tokens.device, f"{name}_mask")
    if not torch.isfinite(tokens[mask]).all():
        raise ValueError(f"Valid {name}_tokens must be finite")
    # Attention masks alone cannot prevent NaN values reaching linear/matmul.
    return torch.where(mask.unsqueeze(-1), tokens, 0)


def _floating_tensor(value, parameter, name):
    if value.dtype not in (torch.float32, torch.float64):
        raise ValueError(f"{name} must contain float32 or float64 values")
    if value.dtype != parameter.dtype or value.device != parameter.device:
        raise ValueError(f"{name} must match the module dtype and device")


def _mask(value, shape, device, name):
    if (not isinstance(value, torch.Tensor) or value.dtype != torch.bool
            or tuple(value.shape) != tuple(shape) or value.device != device):
        raise ValueError(f"{name} must be a boolean tensor with shape {tuple(shape)} on the input device")


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _dropout(value):
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not math.isfinite(value) or not 0 <= value <= 1):
        raise ValueError("dropout must be finite and between zero and one")
    return float(value)
