import json
from pathlib import Path
from urllib.error import HTTPError

import pytest

from hybrid_drive.assets import alignment_models
from hybrid_drive.assets.files import PreparationError


_PINNED_SOURCES = Path(__file__).resolve().parents[2] / "scripts" / "alignment-model-sources.json"


@pytest.mark.parametrize("prepare", [False, True])
def test_altered_manifest_does_not_authorize_missing_models_or_new_transfer(tmp_path, prepare):
    root = tmp_path.resolve()
    raw = _PINNED_SOURCES.read_bytes()
    changed = raw.replace(b"c927226", b"0000000", 1)
    assert len(changed) == len(raw) and changed != raw
    sources = root / alignment_models.SOURCES_FILENAME
    sources.write_bytes(changed)

    def no_network(*args, **kwargs):
        raise AssertionError("A modified manifest must fail before any network call")

    with pytest.raises(PreparationError, match="pinned manifest"):
        if prepare:
            alignment_models.prepare_alignment_models(root, sources, opener=no_network)
        else:
            alignment_models.verify_alignment_models(root)


def test_local_verification_rejects_pickle_even_alongside_expected_filenames(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    raw = _PINNED_SOURCES.read_bytes()
    (root / alignment_models.SOURCES_FILENAME).write_bytes(raw)
    model = json.loads(raw)["models"]["vision"]
    folder = root / model["directory"]
    folder.mkdir()
    for name in model["files"]:
        (folder / name).touch()
    (folder / "pytorch_model.bin").touch()

    def fail_if_opened(*args, **kwargs):
        raise AssertionError("Unexpected files must fail before reading model bytes")

    monkeypatch.setattr(alignment_models, "verify_file", fail_if_opened)
    with pytest.raises(PreparationError, match="pinned file list"):
        alignment_models.verify_alignment_models(root)


def test_local_verification_rejects_truncated_weight_in_exact_file_list(tmp_path):
    root = tmp_path.resolve()
    raw = _PINNED_SOURCES.read_bytes()
    (root / alignment_models.SOURCES_FILENAME).write_bytes(raw)
    model = json.loads(raw)["models"]["vision"]
    folder = root / model["directory"]
    folder.mkdir()
    for name in model["files"]:
        (folder / name).touch()

    with pytest.raises(PreparationError, match="corrupt"):
        alignment_models.verify_alignment_models(root)


def test_preparation_sanitizes_errors_from_signed_redirects(tmp_path):
    root = tmp_path.resolve()

    def signed_redirect_error(*args, **kwargs):
        raise HTTPError(
            "https://example.org/archive?X-Amz-Signature=DO_NOT_PRINT",
            403, "presigned URL expired", None, None,
        )

    with pytest.raises(PreparationError, match=r"dinov2-small/model\.safetensors: HTTP 403") as caught:
        alignment_models.prepare_alignment_models(root, _PINNED_SOURCES, opener=signed_redirect_error)
    assert "DO_NOT_PRINT" not in str(caught.value)
    assert caught.value.__suppress_context__
    assert not (root / alignment_models.SOURCES_FILENAME).exists()
