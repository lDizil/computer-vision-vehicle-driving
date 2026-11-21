import hashlib

import pytest
import torch
from torch import nn

from hybrid_drive.experiments.neural_checkpoint import load_neural_weights, save_neural_weights


META = {"scope": "synthetic_math_smoke", "architecture": "fixture-v1"}


def modules():
    return {"first": nn.Linear(2, 3), "second": nn.Linear(3, 1)}


def cloned(models):
    return {
        name: {key: value.clone() for key, value in module.state_dict().items()}
        for name, module in models.items()
    }


def assert_unchanged(models, before):
    for name, module in models.items():
        for key, value in module.state_dict().items():
            torch.testing.assert_close(value, before[name][key], rtol=0, atol=0)


def test_weights_and_metadata_roundtrip_without_pickle_or_implicit_factory(tmp_path):
    original = modules()
    before = cloned(original)
    path = tmp_path / "core.safetensors"
    record = save_neural_weights(path, original, metadata=META)
    assert record["size_bytes"] == path.stat().st_size
    assert record["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    restored = modules()
    assert (
        load_neural_weights(path, restored, expected_sha256=record["sha256"], expected_metadata=META) == META
    )
    assert_unchanged(restored, before)
    with pytest.raises(FileExistsError):
        save_neural_weights(path, modules(), metadata=META)
    assert record["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert not list(tmp_path.glob(".neural-*"))


@pytest.mark.parametrize("corruption", ["digest", "metadata", "shape", "keys", "dtype"])
def test_invalid_checkpoint_does_not_partially_update_any_module(tmp_path, corruption):
    source = modules()
    path = tmp_path / "core.safetensors"
    record = save_neural_weights(path, source, metadata=META)
    target = modules()
    metadata = META
    digest = record["sha256"]
    if corruption == "digest":
        digest = "0" * 64
    if corruption == "metadata":
        metadata = {"scope": "real_training"}
    if corruption == "shape":
        target["second"] = nn.Linear(4, 1)
    if corruption == "keys":
        target["third"] = target.pop("second")
    if corruption == "dtype":
        target["second"].double()
    before = cloned(target)
    with pytest.raises(ValueError):
        load_neural_weights(path, target, expected_sha256=digest, expected_metadata=metadata)
    assert_unchanged(target, before)


def test_nonfinite_weights_are_rejected_before_output_is_created(tmp_path):
    source = modules()
    with torch.no_grad():
        source["second"].weight.fill_(float("nan"))
    with pytest.raises(ValueError, match="nonfinite"):
        save_neural_weights(tmp_path / "bad.safetensors", source, metadata=META)
    assert not list(tmp_path.iterdir())


def test_symlink_and_nonregular_or_oversized_inputs_are_not_loaded(tmp_path, monkeypatch):
    from hybrid_drive.experiments import neural_checkpoint as checkpoint

    path = tmp_path / "core.safetensors"
    record = save_neural_weights(path, modules(), metadata=META)
    link = tmp_path / "link"
    link.symlink_to(path)
    with pytest.raises(ValueError):
        load_neural_weights(link, modules(), expected_sha256=record["sha256"], expected_metadata=META)
    monkeypatch.setattr(checkpoint, "MAX_BYTES", 1)
    with pytest.raises(ValueError, match="byte budget"):
        load_neural_weights(path, modules(), expected_sha256=record["sha256"], expected_metadata=META)


def test_invalid_header_is_bounded_before_safetensors_parser(tmp_path):
    path = tmp_path / "bad.safetensors"
    path.write_bytes((2**63).to_bytes(8, "little") + b"{}")
    with pytest.raises(ValueError, match="header"):
        load_neural_weights(
            path,
            modules(),
            expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            expected_metadata=META,
        )


def test_save_rejects_oversized_payload_before_copy_or_serialization(tmp_path, monkeypatch):
    from hybrid_drive.experiments import neural_checkpoint as checkpoint

    monkeypatch.setattr(checkpoint, "MAX_BYTES", 1)

    def forbidden(*args, **kwargs):
        raise AssertionError("Oversized weights reached serialization")

    monkeypatch.setattr(checkpoint, "save", forbidden)
    with pytest.raises(ValueError, match="before copying"):
        checkpoint.save_neural_weights(tmp_path / "huge.safetensors", modules(), metadata=META)
    assert not list(tmp_path.iterdir())
