"""Frozen DINOv2 and Pythia adapters for the bounded offline alignment probe."""

from importlib.metadata import PackageNotFoundError, version

import torch
from torch import Tensor, nn

from hybrid_drive.assets.alignment_models import verify_alignment_models, verify_alignment_vision


_IMAGE_SIZE = 224
_PATCH_SIZE = 14
_VISUAL_DIM = 384
_LLM_DIM = 512
_VOCAB_SIZE = 50304
_CONTEXT_SIZE = 2048
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)
_PROMPT = (
    "Use the front camera image. Select the nearest vehicle whose box center lies in the central "
    "horizontal half. Is there a vehicle that qualifies? LOC"
)
_GROUNDED_PROMPT = (
    "Front camera. For each vertical quarter from left to right, locate the nearest vehicle "
    "with box center in that quarter, if any.\n"
    "First quarter: LOC\nSecond quarter: LOC\nThird quarter: LOC\nFourth quarter: LOC"
)
# Pin the whole sequence so reordered quarter cues cannot pass by preserving LOC offsets.
_GROUNDED_PROMPT_IDS = (
    34552, 6568, 15, 1198, 1016, 9118, 7150, 432, 1669, 281, 987, 13, 19912, 253, 5275, 4958, 342,
    3817, 4055, 275, 326, 7150, 13, 604, 667, 15, 187,
    6407, 7150, 27, 35378, 187,
    10951, 7150, 27, 35378, 187,
    25110, 7150, 27, 35378, 187,
    48949, 7150, 27, 35378,
)
_TOKEN_ALIASES = {" LOC": 35378, " MOT": 34794, " yes": 4754, " no": 642}
_RUNTIME = {
    "torch": "2.0.1",
    "transformers": "4.40.2",
    "tokenizers": "0.19.1",
    "huggingface-hub": "0.23.5",
    "safetensors": "0.8.0",
    "regex": "2024.5.15",
}
# The original Pythia export includes generated attention buffers, not learned parameters.
_PYTHIA_LEGACY_BUFFERS = {
    f"gpt_neox.layers.{i}.attention.{name}"
    for i in range(6)
    for name in ("bias", "masked_bias", "rotary_emb.inv_freq")
}


def load_alignment_backbones(asset_root):
    """Load local immutable assets and return (patch encoder, causal adapter, provenance)."""
    return _load_alignment_backbones(asset_root, _PROMPT, _encode_prompt)


_SCENE_BRIDGE_PROMPT = "In the projected scene, is the selected candidate blocked? LOC"


def load_scene_bridge_backbones(asset_root):
    """Frozen visual and causal backbones for an uncalibrated scene handoff probe."""
    def encode(tokenizer):
        _validate_token_aliases(tokenizer)
        ids = tokenizer.encode(_SCENE_BRIDGE_PROMPT, add_special_tokens=False)
        loc = _TOKEN_ALIASES[" LOC"]
        if not ids or ids[-1] != loc or ids.count(loc) != 1 or any(
            type(token) is not int or not 0 <= token < _VOCAB_SIZE for token in ids
        ):
            raise ValueError("Expected one consumed LOC at the end of the scene probe")
        return ids

    return _load_alignment_backbones(asset_root, _SCENE_BRIDGE_PROMPT, encode)


def load_grounded_alignment_backbones(asset_root):
    """Load the same frozen assets with the four-quarter LOC prompt."""
    vision, language, provenance = _load_alignment_backbones(asset_root, _GROUNDED_PROMPT, _encode_grounded_prompt)
    provenance["vision"] = _vision_provenance(
        {
            "source": provenance["models"]["vision"],
            "sources_sha256": provenance["sources_sha256"],
        },
        provenance["runtime"],
    )
    tokenization = provenance["tokenization"]
    tokenization["loc_positions"] = [
        index for index, token in enumerate(tokenization["prompt_ids"]) if token == language.loc_token_id
    ]
    return vision, language, provenance


def _load_alignment_backbones(asset_root, prompt, encode_prompt):
    vision_path, language_path, provenance = verify_alignment_models(asset_root)
    runtime = _runtime_profile()

    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM, GPTNeoXTokenizerFast

    tokenizer = GPTNeoXTokenizerFast.from_pretrained(language_path, local_files_only=True)
    prompt_ids = encode_prompt(tokenizer)
    language_config = _checked_config(
        GPTNeoXConfig, language_path, model_type="gpt_neox", hidden_size=_LLM_DIM, vocab_size=_VOCAB_SIZE,
        max_position_embeddings=_CONTEXT_SIZE, num_hidden_layers=6, num_attention_heads=8,
    )
    vision = _load_vision(vision_path)
    language = _PythiaPrefix(
        _load_model(GPTNeoXForCausalLM, language_path, language_config, ignored=_PYTHIA_LEGACY_BUFFERS),
        prompt_ids,
    )
    provenance.update(
        runtime=runtime,
        preprocessing=_vision_preprocessing(),
        tokenization={"aliases": dict(_TOKEN_ALIASES), "prompt": prompt, "prompt_ids": prompt_ids},
    )
    return vision, language, provenance


def load_alignment_vision(asset_root):
    """Load only the verified visual encoder, without materializing the LLM."""
    vision_path, provenance = verify_alignment_vision(asset_root)
    runtime = _runtime_profile()
    return _load_vision(vision_path), _vision_provenance(provenance, runtime)


def _vision_provenance(provenance, runtime):
    return {
        **provenance,
        "runtime": dict(runtime),
        "preprocessing": _vision_preprocessing(),
    }


def _load_vision(path):
    from transformers import Dinov2Config, Dinov2Model

    config = _checked_config(
        Dinov2Config, path, model_type="dinov2", hidden_size=_VISUAL_DIM, patch_size=_PATCH_SIZE,
        image_size=518, num_channels=3, num_hidden_layers=12, num_attention_heads=6,
    )
    return _DinoPatchGrid(_load_model(Dinov2Model, path, config))


def _vision_preprocessing():
    return {
        "input": "rectified_full_image_rgb_0_1",
        "image_size": _IMAGE_SIZE,
        "patch_size": _PATCH_SIZE,
        "mean": list(_MEAN),
        "std": list(_STD),
    }


def _runtime_versions():
    runtime = {}
    for package, expected in _RUNTIME.items():
        try:
            actual = version(package)
        except PackageNotFoundError:
            raise RuntimeError(f"Missing {package}; use the Dockerfile.alignment image") from None
        comparison = actual.removesuffix("+cpu") if package == "torch" else actual
        if comparison != expected:
            raise RuntimeError(f"Expected {package}=={expected}; use the Dockerfile.alignment image")
        runtime[package] = actual
    return runtime


def _runtime_profile():
    return {**_runtime_versions(), "device": "cpu", "dtype": "float32", "attention": "eager"}


def _checked_config(config_class, path, **expected):
    config = config_class.from_pretrained(path, local_files_only=True)
    if any(getattr(config, field, None) != value for field, value in expected.items()):
        raise ValueError(f"Pinned {config_class.__name__} architecture differs")
    return config


def _load_model(model_class, path, config, *, ignored=()):
    # A concrete built-in class and the verified file list exclude Auto/remote code and pickle.
    with torch.device("cpu"):
        model, loading = model_class.from_pretrained(
            path, config=config, local_files_only=True, use_safetensors=True,
            torch_dtype=torch.float32, attn_implementation="eager", output_loading_info=True,
        )
    if (loading["missing_keys"] or set(loading["unexpected_keys"]) - set(ignored)
            or loading["mismatched_keys"] or loading["error_msgs"]):
        raise ValueError(f"Pinned {model_class.__name__} weights did not load completely")
    if any(parameter.device.type != "cpu" or parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise ValueError(f"Expected {model_class.__name__} parameters on CPU in float32")
    return model


def _encode_prompt(tokenizer):
    _validate_token_aliases(tokenizer)
    prompt_ids = tokenizer.encode(_PROMPT, add_special_tokens=False)
    loc = _TOKEN_ALIASES[" LOC"]
    if (not prompt_ids or prompt_ids[-1] != loc or loc in prompt_ids[:-1]
            or any(not 0 <= token_id < _VOCAB_SIZE for token_id in prompt_ids)):
        raise ValueError("The fixed Pythia prompt must end in a single consumed LOC")
    return prompt_ids


def _encode_grounded_prompt(tokenizer):
    _validate_token_aliases(tokenizer)
    prompt_ids = tokenizer.encode(_GROUNDED_PROMPT, add_special_tokens=False)
    if tuple(prompt_ids) != _GROUNDED_PROMPT_IDS:
        raise ValueError("The fixed Pythia prompt must retain the four consumed LOC markers in quarter order")
    return prompt_ids


def _validate_token_aliases(tokenizer):
    if not tokenizer.is_fast:
        raise ValueError("Expected the pinned fast Pythia tokenizer")
    for text, token_id in _TOKEN_ALIASES.items():
        if token_id in tokenizer.all_special_ids or tokenizer.encode(text, add_special_tokens=False) != [token_id]:
            raise ValueError(f"Pinned Pythia token alias differs for {text!r}")


class _DinoPatchGrid(nn.Module):
    visual_dim = _VISUAL_DIM

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.register_buffer("mean", torch.tensor(_MEAN, dtype=torch.float32, device="cpu").view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(_STD, dtype=torch.float32, device="cpu").view(1, 3, 1, 1))
        self.requires_grad_(False).eval()

    @torch.no_grad()
    def forward(self, images):
        if (not isinstance(images, Tensor) or images.ndim != 5 or images.shape[0] == 0
                or images.shape[1:] != (1, 3, _IMAGE_SIZE, _IMAGE_SIZE)):
            raise ValueError("Expected images with shape [B,1,3,224,224]")
        if images.device.type != "cpu" or images.dtype != torch.float32:
            raise ValueError("Expected float32 RGB images on CPU")
        if not torch.isfinite(images).all() or ((images < 0) | (images > 1)).any():
            raise ValueError("Expected RGB image values in [0,1] before normalization")
        pixels = (images[:, 0] - self.mean) / self.std
        tokens = self.backbone(pixel_values=pixels, return_dict=True).last_hidden_state
        grid = _IMAGE_SIZE // _PATCH_SIZE
        if tokens.shape != (images.shape[0], 1 + grid * grid, self.visual_dim):
            raise ValueError("DINOv2 must return one CLS token followed by the patch grid")
        return tokens[:, 1:].transpose(1, 2).reshape(-1, 1, self.visual_dim, grid, grid).contiguous()


class _PythiaPrefix(nn.Module):
    llm_dim = _LLM_DIM
    vocab_size = _VOCAB_SIZE
    loc_token_id = _TOKEN_ALIASES[" LOC"]
    mot_token_id = _TOKEN_ALIASES[" MOT"]
    yes_token_id = _TOKEN_ALIASES[" yes"]
    no_token_id = _TOKEN_ALIASES[" no"]

    def __init__(self, backbone, prompt_ids):
        super().__init__()
        self.backbone = backbone
        self.register_buffer("_prompt", torch.tensor(prompt_ids, dtype=torch.long, device="cpu"), persistent=False)
        self.requires_grad_(False).eval()

    def prompt_ids(self, batch_size):
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("Expected a positive integer prompt batch size")
        return self._prompt.expand(batch_size, -1).clone()

    def forward(self, prefix, consumed_ids):
        if (not isinstance(prefix, Tensor) or prefix.ndim != 3 or min(prefix.shape[:2]) < 1
                or prefix.shape[2] != self.llm_dim or prefix.dtype != torch.float32
                or prefix.device.type != "cpu" or not torch.isfinite(prefix).all()):
            raise ValueError("Expected finite float32 prefix [B,Q,512] on CPU with positive B,Q")
        if (not isinstance(consumed_ids, Tensor) or consumed_ids.ndim != 2
                or consumed_ids.shape[0] != prefix.shape[0] or consumed_ids.shape[1] == 0
                or consumed_ids.dtype != torch.long or consumed_ids.device.type != "cpu"
                or ((consumed_ids < 0) | (consumed_ids >= self.vocab_size)).any()):
            raise ValueError("Expected unpadded in-vocabulary consumed_ids [B,L] on CPU")
        query_count = prefix.shape[1]
        if query_count + consumed_ids.shape[1] > _CONTEXT_SIZE:
            raise ValueError("Pythia prefix and prompt exceed the pinned context size")

        # Freeze parameters, not the derivative from text/perception losses to prefix.
        embedded = self.backbone.get_input_embeddings()(consumed_ids)
        combined = torch.cat((prefix, embedded), dim=1)
        hidden = self.backbone.gpt_neox(inputs_embeds=combined, use_cache=False, return_dict=True).last_hidden_state
        if hidden.shape != combined.shape:
            raise ValueError("Pythia hidden state must correspond to every consumed position")
        hidden = hidden[:, query_count:]
        logits = self.backbone.get_output_embeddings()(hidden)
        if logits.shape != (*consumed_ids.shape, self.vocab_size):
            raise ValueError("Pythia logits do not match the pinned vocabulary")
        return hidden, logits
