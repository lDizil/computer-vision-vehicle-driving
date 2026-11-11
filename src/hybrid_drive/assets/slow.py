"""Reuse the bounded opaque downloader for a pinned local VLM asset pair."""

from pathlib import Path
import os

from .fetch import AssetSpec, TransferBudget, download_verified_asset
from .files import (
    PreparationError, check_components, checked_root, destination, read_manifest, verify_file, write_manifest,
)
from .slow_registry import MODEL_SPECS


def prepare_slow(root, sources, *, rate_limit_mib=4, opener=None):
    max_bytes = _validate(sources)
    profile = _model_spec(sources)
    budget = TransferBudget(rate_limit_mib, max_bytes=max_bytes, timeout_seconds=1200)
    root = checked_root(root)
    manifest_path = destination(root, profile.manifest_filename)
    previous = read_manifest(manifest_path)
    expected = _manifest(sources)
    if previous is not None and previous != expected:
        raise PreparationError("Existing Slow manifest differs from pinned sources")
    # Check every existing target before beginning a potentially large transfer.
    for spec in sources["files"]:
        verify_file(destination(root, spec["path"]), size_bytes=spec["size_bytes"], sha256=spec["sha256"])
    for spec in sources["files"]:
        kwargs = {} if opener is None else {"opener": opener}
        download_verified_asset(AssetSpec.from_dict(spec), destination(root, spec["path"]), budget=budget, **kwargs)
    if previous is None:
        write_manifest(manifest_path, expected)
    return {**expected, "bytes_transferred_this_run": budget.received}


def verify_slow(root, sources):
    """Read-only verification; inference never downloads or repairs missing assets."""
    _validate(sources)
    root = Path(os.path.abspath(root))
    check_components(root)
    manifest = read_manifest(destination(root, _model_spec(sources).manifest_filename))
    expected = _manifest(sources)
    if manifest != expected:
        raise PreparationError("Slow asset manifest is missing or differs from pinned sources")
    for spec in sources["files"]:
        if verify_file(destination(root, spec["path"]), size_bytes=spec["size_bytes"], sha256=spec["sha256"]) is None:
            raise PreparationError("Slow model file is missing")
    return expected


def _manifest(sources):
    return {"sources": sources, "files": [
        {key: value[key] for key in ("path", "sha256", "size_bytes")} for value in sources["files"]
    ]}


def _model_spec(sources):
    if isinstance(sources, dict):
        for profile in MODEL_SPECS.values():
            if sources.get("schema") == profile.schema:
                return profile
    raise PreparationError("Expected a registered pinned Slow model source")


def _validate(sources):
    profile = _model_spec(sources)
    files = sources.get("files")
    if not isinstance(files, list) or len(files) != 2:
        raise PreparationError("Expected exactly one text model and one vision projector")
    if any(not isinstance(value, dict) for value in files) or [value.get("path") for value in files] != [
        profile.text_filename, profile.projector_filename,
    ]:
        raise PreparationError("Unexpected Slow model files")
    for value in files:
        try:
            AssetSpec.from_dict(value).validate()
        except (KeyError, TypeError, AttributeError) as exc:
            raise PreparationError("Invalid pinned Slow asset specification") from exc
    if sum(value["size_bytes"] for value in files) > profile.max_bytes:
        raise PreparationError(f"Slow assets exceed the {profile.max_bytes // 1024**2} MiB transfer budget")
    return profile.max_bytes
