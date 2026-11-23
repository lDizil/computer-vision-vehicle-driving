"""Four-quarter LOC through latent ROI priors and a frozen causal backbone."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .decoder import InstructedVisionDecoder, PerceptionOutput
from .losses import PerceptionTargets
from .prior_tokenizer import PriorTokenizer
from .projector import QFormerProjector


_QUARTERS = 4


@dataclass(frozen=True)
class GroundedLocPrediction:
    consumed_ids: Tensor
    logits: Tensor
    loc_positions: Tensor
    perception: PerceptionOutput


class GroundedLocProbe(nn.Module):
    """Condition the causal prefix on four left-to-right ROIs in the current view.

    In the default mode both bridge and decoder receive full image memory.
    In the experimental ROI-token mode there is no direct patch-memory path:
    both see the same four pooled region tokens.
    """

    def __init__(
        self, language, *, visual_dim: int, hidden_dim: int = 64, num_queries: int = 4,
        memory_mode: str = "patches_and_priors",
    ):
        super().__init__()
        if memory_mode not in ("patches_and_priors", "roi_tokens"):
            raise ValueError("Unknown grounded LOC memory mode")
        self.memory_mode = memory_mode
        ids, _ = _fixed_prompt(language, 1)
        self.register_buffer("_prompt", ids[0].clone(), persistent=False)
        self.tokenizer = PriorTokenizer(
            visual_dim,
            geometry_dim=4,
            token_dim=visual_dim if memory_mode == "roi_tokens" else 768,
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

    def forward(self, features: Tensor, boxes_xyxy: Tensor, language) -> GroundedLocPrediction:
        batch = _single_view_batch(features)
        ids, positions = _fixed_prompt(language, batch, device=features.device)
        if not torch.equal(ids, self._prompt.expand(batch, -1)):
            raise ValueError("Grounded LOC prompt differs from the fixed quarter order")
        if not isinstance(boxes_xyxy, Tensor) or boxes_xyxy.shape != (batch, _QUARTERS, 4):
            raise ValueError("Expected four normalized prior boxes_xyxy [B,4,4]")

        start, end = boxes_xyxy[..., :2], boxes_xyxy[..., 2:]
        geometry = torch.cat(((start + end) * 0.5, end - start), dim=-1)
        view_indices = torch.zeros((batch, _QUARTERS), device=features.device, dtype=torch.long)
        priors = self.tokenizer(
            features, boxes_xyxy, view_indices, geometry, torch.ones_like(view_indices, dtype=torch.bool)
        )
        normalized = boxes_xyxy.to(torch.float32)
        bands = ((normalized[..., 0] + normalized[..., 2]) * (0.5 * _QUARTERS)).long()
        expected_bands = torch.arange(_QUARTERS, device=boxes_xyxy.device).expand(batch, -1)
        if not torch.equal(bands, expected_bands):
            raise ValueError("Prior boxes must preserve left-to-right quarter order")
        if self.memory_mode == "roi_tokens":
            visual, visual_mask = priors.tokens, priors.valid_mask
            prefix = self.projector(visual, visual_mask)
        else:
            visual = features[:, 0].flatten(2).transpose(1, 2)
            visual_mask = torch.ones(visual.shape[:2], device=visual.device, dtype=torch.bool)
            prefix = self.projector(visual, visual_mask, priors.tokens, priors.valid_mask)
        hidden, logits = language(prefix, ids)
        _require_logits(logits, ids, language)
        answer_mask = torch.zeros_like(ids, dtype=torch.bool).scatter_(1, positions, True)
        perception = self.decoder(visual, visual_mask, hidden, ids, answer_mask)
        _require_event_order(perception, _event_keys(positions))
        return GroundedLocPrediction(ids, logits, positions, perception)

    def weight_modules(self) -> dict[str, nn.Module]:
        return {"tokenizer": self.tokenizer, "projector": self.projector, "decoder": self.decoder}


def fixed_quarter_boxes(features: Tensor) -> Tensor:
    """Return four full-height xyxy bands in normalized image coordinates."""
    batch = _single_view_batch(features)
    edges = torch.arange(_QUARTERS + 1, device=features.device, dtype=features.dtype) / _QUARTERS
    boxes = torch.stack(
        (edges[:-1], torch.zeros_like(edges[:-1]), edges[1:], torch.ones_like(edges[1:])), dim=1
    )
    return boxes.expand(batch, -1, -1)


def supervised_grounded_loc_loss(
    criterion, prediction: GroundedLocPrediction, language, presence: Tensor, boxes_2d: Tensor
):
    """Supervise the next-token logits at each LOC and only present 2D boxes."""
    batch_indices, positions, kinds = _validated_events(prediction, language)
    ids, logits = prediction.consumed_ids, prediction.logits
    shape = (ids.shape[0], _QUARTERS)
    if (
        not isinstance(presence, Tensor)
        or presence.shape != shape
        or presence.dtype != torch.bool
        or presence.device != ids.device
    ):
        raise ValueError("Expected aligned boolean presence [B,4]")
    if not _aligned_float(boxes_2d, (*shape, 4), logits):
        raise ValueError("Expected aligned target boxes_2d [B,4,4]")
    positive_frames, positive_quarters = presence.nonzero(as_tuple=True)
    # The spatial dataset assigns bands from the center in float32.
    positive_boxes = boxes_2d[positive_frames, positive_quarters].to(torch.float32)
    band_indices = ((positive_boxes[:, 0] + positive_boxes[:, 2]) * (0.5 * _QUARTERS)).long()
    if not torch.equal(band_indices, positive_quarters):
        raise ValueError("Each present target box center must lie in its assigned quarter")

    present = presence.reshape(-1)
    labels = torch.full_like(ids, -100)
    labels[batch_indices, positions] = torch.where(present, language.yes_token_id, language.no_token_id)
    events = present.numel()
    targets = PerceptionTargets(
        batch_indices=batch_indices,
        token_positions=positions,
        event_kinds=kinds,
        validity=present.to(logits.dtype),
        validity_mask=torch.ones_like(present),
        boxes_3d=logits.new_zeros((events, 7)),
        boxes_3d_mask=torch.zeros((events, 7), device=ids.device, dtype=torch.bool),
        boxes_2d=boxes_2d.reshape(events, 4),
        boxes_2d_mask=present,
        motion_xy=logits.new_zeros((events, 1, 2)),
        motion_mask=torch.zeros((events, 1), device=ids.device, dtype=torch.bool),
    )
    return criterion(logits, labels, labels != -100, prediction.perception, targets)


def grounded_presence_scores(prediction: GroundedLocPrediction, language) -> tuple[Tensor, Tensor, Tensor]:
    """Return raw decoder logits, text yes-minus-no logits, and decoded xyxy boxes."""
    batch_indices, positions, _ = _validated_events(prediction, language)
    logits, perception = prediction.logits, prediction.perception
    validity, boxes = perception.validity_logits, perception.boxes_2d
    events = batch_indices.numel()
    if not _aligned_float(validity, (events,), logits) or not _aligned_float(boxes, (events, 4), logits):
        raise ValueError("Grounded LOC decoder scores and boxes must align to the four events per frame")
    text = (
        logits[batch_indices, positions, language.yes_token_id]
        - logits[batch_indices, positions, language.no_token_id]
    )
    if not (torch.isfinite(validity).all() and torch.isfinite(text).all() and torch.isfinite(boxes).all()):
        raise ValueError("Grounded LOC scores and decoded boxes must be finite")
    shape = (prediction.consumed_ids.shape[0], _QUARTERS)
    return validity.reshape(shape), text.reshape(shape), boxes.reshape(*shape, 4)


def _single_view_batch(features):
    if (
        not isinstance(features, Tensor)
        or features.ndim != 5
        or features.shape[1] != 1
        or min(features.shape) < 1
        or features.dtype not in (torch.float32, torch.float64)
    ):
        raise ValueError("Expected nonempty float32/float64 features from one current view [B,1,C,H,W]")
    return features.shape[0]


def _fixed_prompt(language, batch, *, device=None):
    vocabulary = getattr(language, "vocab_size", None)
    aliases = (language.loc_token_id, language.mot_token_id, language.yes_token_id, language.no_token_id)
    if (
        type(vocabulary) is not int
        or vocabulary <= 0
        or any(type(token) is not int or not 0 <= token < vocabulary for token in aliases)
        or len(set(aliases)) != len(aliases)
    ):
        raise ValueError("Expected distinct native LOC/MOT/yes/no token IDs in the language vocabulary")
    loc, mot, yes, no = aliases
    ids = language.prompt_ids(batch)
    if (
        not isinstance(ids, Tensor)
        or ids.ndim != 2
        or ids.shape[0] != batch
        or ids.shape[1] < _QUARTERS
        or ids.dtype != torch.long
        or (device is not None and ids.device != device)
        or not torch.equal(ids, ids[:1].expand_as(ids))
        or ((ids < 0) | (ids >= vocabulary)).any()
    ):
        raise ValueError("Expected a shared in-vocabulary consumed prompt [B,L]")
    markers = ids == loc
    if (
        not (markers.sum(dim=1) == _QUARTERS).all()
        or not markers[:, -1].all()
        or ((ids == mot) | (ids == yes) | (ids == no)).any()
    ):
        raise ValueError("Expected four consumed LOC markers in quarter order and no MOT or answer tokens")
    return ids, markers.nonzero(as_tuple=True)[1].reshape(batch, _QUARTERS)


def _require_logits(logits, ids, language):
    if (
        not isinstance(logits, Tensor)
        or logits.shape != (*ids.shape, language.vocab_size)
        or logits.device != ids.device
        or logits.dtype not in (torch.float32, torch.float64)
    ):
        raise ValueError("Grounded LOC logits must align to [B,L,V] with the declared vocabulary width")


def _event_keys(positions):
    batches = torch.arange(positions.shape[0], device=positions.device).repeat_interleave(_QUARTERS)
    return batches, positions.reshape(-1), torch.zeros_like(batches)


def _require_event_order(perception, keys):
    if not isinstance(perception, PerceptionOutput) or not all(
        _aligned_indices(actual, expected)
        for actual, expected in zip(
            (perception.batch_indices, perception.token_positions, perception.event_kinds), keys
        )
    ):
        raise ValueError("Grounded LOC decoder events must match the expected frame/token order and LOC kind")


def _validated_events(prediction, language):
    if not isinstance(prediction, GroundedLocPrediction):
        raise ValueError("Expected GroundedLocPrediction")
    ids = prediction.consumed_ids
    if not isinstance(ids, Tensor) or ids.ndim != 2 or ids.shape[0] == 0:
        raise ValueError("Expected the fixed consumed prompt [B,L]")
    expected_ids, positions = _fixed_prompt(language, ids.shape[0], device=ids.device)
    if not _aligned_indices(ids, expected_ids) or not _aligned_indices(prediction.loc_positions, positions):
        raise ValueError("Grounded LOC prompt and positions must match the fixed consumed markers")
    _require_logits(prediction.logits, ids, language)
    keys = _event_keys(positions)
    _require_event_order(prediction.perception, keys)
    return keys


def _aligned_indices(value, expected):
    return (
        isinstance(value, Tensor)
        and value.dtype == torch.long
        and value.device == expected.device
        and torch.equal(value, expected)
    )


def _aligned_float(value, shape, reference):
    return (
        isinstance(value, Tensor)
        and value.shape == shape
        and value.dtype == reference.dtype
        and value.device == reference.device
    )
