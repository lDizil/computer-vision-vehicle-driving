"""Bounded safetensors weights; no pickle, optimizer state or architecture factory."""

import hashlib
import os
from pathlib import Path
import re
import struct
import tempfile

from safetensors.torch import load, save
import torch
from torch import nn

from hybrid_drive.assets.files import check_components
from hybrid_drive.experiments.bounded_io import canonical_json, decode_json, read_bounded, require_hex


SCHEMA = "hybrid-drive-neural-weights-v1"
MAX_BYTES = 256 * 1024 * 1024
MAX_HEADER_BYTES = 64 * 1024
MAX_METADATA_BYTES = 16 * 1024


def save_neural_weights(path, modules, *, metadata):
    expected = _state(modules)
    metadata_json = _metadata(metadata)
    if sum(value.numel() * value.element_size() for value in expected.values()) > MAX_BYTES:
        raise ValueError("Neural tensor payload exceeds the byte budget before copying")
    weights = {name: value.detach().cpu().contiguous().clone() for name, value in expected.items()}
    if any(not torch.isfinite(value).all() for value in weights.values()):
        raise ValueError("Cannot save nonfinite model weights")
    data = save(weights, metadata={"schema": SCHEMA, "metadata_json": metadata_json})
    if len(data) > MAX_BYTES:
        raise ValueError("Neural weights exceed the byte budget")
    _header(data)
    path = Path(os.path.abspath(path))
    check_components(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".neural-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        check_components(path)
        os.link(temporary, path)
    finally:
        os.unlink(temporary)
    return {
        "schema": SCHEMA,
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
        "metadata_sha256": hashlib.sha256(metadata_json.encode()).hexdigest(),
    }


def load_neural_weights(path, modules, *, expected_sha256, expected_metadata):
    require_hex(expected_sha256, 64, "expected_sha256")
    expected = _state(modules)
    data = read_bounded(Path(path), MAX_BYTES)
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError("Neural weights differ from expected SHA256")
    header = _header(data)
    metadata = header.get("__metadata__")
    if (
        not isinstance(metadata, dict)
        or set(metadata) != {"schema", "metadata_json"}
        or metadata["schema"] != SCHEMA
        or metadata["metadata_json"] != _metadata(expected_metadata)
    ):
        raise ValueError("Neural checkpoint schema or experiment metadata differs")
    tensors = load(data)
    if set(tensors) != set(expected):
        raise ValueError("Neural checkpoint state keys differ")
    for name, value in tensors.items():
        reference = expected[name]
        if (
            value.shape != reference.shape
            or value.dtype != reference.dtype
            or not torch.isfinite(value).all()
        ):
            raise ValueError("Neural checkpoint tensor differs: " + name)
    # Validate every module before copying any state. Modules are supplied by
    # the caller, never imported/constructed from checkpoint metadata.
    for name, module in modules.items():
        prefix = name + "."
        module.load_state_dict(
            {key[len(prefix) :]: value for key, value in tensors.items() if key.startswith(prefix)},
            strict=True,
        )
    return expected_metadata.copy()


def _state(modules):
    if (
        not isinstance(modules, dict)
        or not modules
        or any(
            not isinstance(name, str)
            or not re.fullmatch("[a-z][a-z0-9_]*", name)
            or not isinstance(module, nn.Module)
            for name, module in modules.items()
        )
    ):
        raise ValueError("Named neural modules must be a nonempty dictionary")
    if len({id(module) for module in modules.values()}) != len(modules):
        raise ValueError("Each checkpoint module must have a distinct owner")
    state = {
        name + "." + key: value
        for name, module in modules.items()
        for key, value in module.state_dict().items()
    }
    if not state or any(not isinstance(value, torch.Tensor) for value in state.values()):
        raise ValueError("Neural modules must have tensor state")
    return state


def _metadata(metadata):
    if not isinstance(metadata, dict):
        raise ValueError("Checkpoint metadata must be a JSON object")
    data = canonical_json(metadata)
    if len(data) > MAX_METADATA_BYTES:
        raise ValueError("Checkpoint metadata exceeds byte budget")
    return data.decode()


def _header(data):
    if len(data) < 8:
        raise ValueError("Invalid safetensors header")
    size = struct.unpack("<Q", data[:8])[0]
    if not 0 < size <= MAX_HEADER_BYTES or 8 + size > len(data):
        raise ValueError("Safetensors header exceeds byte budget")
    header = decode_json(data[8 : 8 + size])
    if not isinstance(header, dict):
        raise ValueError("Safetensors header must be an object")
    return header


def module_state_hash(module):
    """Bitwise CPU tensor fingerprint; keep the spatial receipt format stable."""
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(repr((name, tuple(tensor.shape), str(tensor.dtype))).encode())
        digest.update(memoryview(tensor.detach().cpu().contiguous().numpy()).cast("B"))
    return digest.hexdigest()
