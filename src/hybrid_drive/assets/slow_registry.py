"""Explicit local VLM identities shared by preparation and qualification CLIs."""

from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class SlowModelSpec:
    sources_filename: str
    asset_subdirectory: str
    manifest_filename: str
    schema: str
    text_filename: str
    projector_filename: str
    max_bytes: int


MODEL_SPECS = MappingProxyType({
    "256m": SlowModelSpec(
        "smolvlm-sources.json", "smolvlm-256m", "smolvlm-manifest.json", "hybrid-drive-smolvlm-q8-v1",
        "SmolVLM-256M-Instruct-Q8_0.gguf", "mmproj-SmolVLM-256M-Instruct-Q8_0.gguf", 320 * 1024**2,
    ),
    "500m": SlowModelSpec(
        "smolvlm-500m-sources.json", "smolvlm-500m", "smolvlm-manifest.json", "hybrid-drive-smolvlm-500m-q8-v1",
        "SmolVLM-500M-Instruct-Q8_0.gguf", "mmproj-SmolVLM-500M-Instruct-Q8_0.gguf", 600 * 1024**2,
    ),
    "qwen3vl-2b": SlowModelSpec(
        "qwen3vl-2b-sources.json", "qwen3vl-2b", "qwen3vl-2b-manifest.json", "hybrid-drive-qwen3vl-2b-q4-k-m-v1",
        "Qwen3VL-2B-Instruct-Q4_K_M.gguf", "mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf", 1536 * 1024**2,
    ),
})


def resolve_model_key(key):
    try:
        return MODEL_SPECS[key]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Unknown Slow model: {key!r}") from exc
