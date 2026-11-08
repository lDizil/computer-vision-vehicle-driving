"""Reuse the bounded opaque downloader for the two local GGUF assets."""

from pathlib import Path
import os

from .fetch import AssetSpec, TransferBudget, download_verified_asset
from .files import (
    PreparationError, check_components, checked_root, destination, read_manifest, verify_file, write_manifest,
)


def prepare_slow(root, sources, *, rate_limit_mib=4, opener=None):
    _validate(sources)
    root = checked_root(root)
    manifest_path = destination(root, "smolvlm-manifest.json")
    previous = read_manifest(manifest_path)
    if previous is not None and previous.get("sources") != sources:
        raise PreparationError("Existing Slow manifest differs from pinned sources")
    budget = TransferBudget(rate_limit_mib, max_bytes=320 * 1024**2, timeout_seconds=1200)
    records = []
    for spec in sources["files"]:
        kwargs = {} if opener is None else {"opener": opener}
        record = download_verified_asset(AssetSpec.from_dict(spec), destination(root, spec["path"]),
                                         budget=budget, **kwargs)
        records.append({"path": spec["path"], "sha256": record["sha256"], "size_bytes": record["size_bytes"]})
    manifest = {"sources": sources, "files": records}
    write_manifest(manifest_path, manifest)
    return {**manifest, "bytes_transferred_this_run": budget.received}


def verify_slow(root, sources):
    """Read-only verification; inference never downloads or repairs missing assets."""
    _validate(sources)
    root = Path(os.path.abspath(root))
    check_components(root)
    manifest = read_manifest(destination(root, "smolvlm-manifest.json"))
    expected = {"sources": sources, "files": [
        {key: value[key] for key in ("path", "sha256", "size_bytes")} for value in sources["files"]
    ]}
    if manifest != expected:
        raise PreparationError("Slow asset manifest is missing or differs from pinned sources")
    for spec in sources["files"]:
        if verify_file(destination(root, spec["path"]), size_bytes=spec["size_bytes"], sha256=spec["sha256"]) is None:
            raise PreparationError("Slow model file is missing")
    return expected


def _validate(sources):
    if not isinstance(sources, dict) or sources.get("schema") != "hybrid-drive-smolvlm-q8-v1":
        raise PreparationError("Expected pinned SmolVLM Q8 sources")
    files = sources.get("files")
    if not isinstance(files, list) or len(files) != 2:
        raise PreparationError("Expected exactly one text model and one vision projector")
    if [value.get("path") for value in files] != [
        "SmolVLM-256M-Instruct-Q8_0.gguf", "mmproj-SmolVLM-256M-Instruct-Q8_0.gguf",
    ]:
        raise PreparationError("Unexpected Slow model files")
    for value in files:
        AssetSpec.from_dict(value).validate()
    if sum(value["size_bytes"] for value in files) > 320 * 1024**2:
        raise PreparationError("Slow assets exceed the 320 MiB transfer budget")
