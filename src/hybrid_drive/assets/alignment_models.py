"""Exact public backbone snapshots for the offline alignment probe."""

import hashlib
import json
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from .fetch import AssetSpec, TransferBudget, download_verified_asset
from .files import AtomicFile, PreparationError, check_components, checked_root, destination, verify_file


SOURCES_FILENAME = "alignment-model-sources.json"
_SOURCES_SIZE = 2356
_SOURCES_SHA256 = "c1665af1d0a221026baa854abfd46dda88dc4f9a37c764297d4983a15669ddb1"


def prepare_alignment_models(root, sources_path, *, rate_limit_mib=4, opener=urlopen):
    raw, sources = _read_sources(Path(sources_path))
    root = checked_root(root)
    source_copy = destination(root, SOURCES_FILENAME)
    existing = verify_file(source_copy, size_bytes=_SOURCES_SIZE, sha256=_SOURCES_SHA256)
    models = sources["models"]
    transfers = [
        (path, spec)
        for model in models.values()
        for path, spec in _model_files(root, model, preparing=True)
    ]
    budget = TransferBudget(rate_limit_mib, max_bytes=sum(spec.size_bytes for _, spec in transfers))
    transferred = 0
    for path, spec in transfers:
        try:
            result = download_verified_asset(spec, path, budget=budget, opener=opener)
        except HTTPError as exc:
            raise PreparationError(f"Cannot download {path.parent.name}/{path.name}: HTTP {exc.code}") from None
        except URLError:
            raise PreparationError(f"Cannot reach source for {path.parent.name}/{path.name}") from None
        except OSError:
            raise PreparationError(f"Cannot prepare local bytes for {path.parent.name}/{path.name}") from None
        transferred += result["bytes_transferred"]
    if existing is None:
        with AtomicFile(source_copy) as pending:
            pending.out.write(raw)
            pending.commit()
    return {"sources_sha256": _SOURCES_SHA256, "bytes_transferred": transferred}


def verify_alignment_models(root):
    """Verify the entire allowlist before any caller opens a model/config."""
    root = Path(os.path.abspath(root))
    check_components(root)
    _, sources = _read_sources(destination(root, SOURCES_FILENAME))
    directories = {}
    for role, model in sources["models"].items():
        files = _model_files(root, model)
        for path, spec in files:
            if verify_file(path, size_bytes=spec.size_bytes, sha256=spec.sha256) is None:
                raise PreparationError(f"Missing {model['directory']}/{path.name}")
        directories[role] = destination(root, model["directory"])
    provenance = {**sources, "sources_sha256": _SOURCES_SHA256}
    return directories["vision"], directories["language"], provenance


def _read_sources(path):
    check_components(path)
    try:
        with path.open("rb") as stream:
            raw = stream.read(_SOURCES_SIZE + 1)
    except OSError:
        raise PreparationError(f"Cannot read {SOURCES_FILENAME}") from None
    if len(raw) != _SOURCES_SIZE or hashlib.sha256(raw).hexdigest() != _SOURCES_SHA256:
        raise PreparationError("Alignment sources differ from the pinned manifest")
    return raw, json.loads(raw)


def _model_files(root, model, *, preparing=False):
    directory = destination(root, model["directory"])
    if preparing:
        directory.mkdir(exist_ok=True)
    if not directory.is_dir():
        raise PreparationError(f"Missing model directory {model['directory']}")
    actual = {path.name for path in directory.iterdir()}
    expected = model["files"].keys()
    if actual - expected or (not preparing and actual != expected):
        raise PreparationError(f"{model['directory']} differs from the pinned file list")
    revision = model["revision"]
    return [
        (
            destination(directory, name),
            AssetSpec(
                url=f"https://huggingface.co/{model['repository']}/resolve/{revision}/{name}",
                revision=revision,
                **fields,
            ),
        )
        for name, fields in model["files"].items()
    ]
