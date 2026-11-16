"""Fixed 20-frame OpenScene current-camera sequence, independent of the six-case cohort."""

import os
from pathlib import Path
import tempfile

from .cohort import (
    _PINNED_LOG, _PINNED_ARCHIVE, _NAVSIM_REVISION, _NUPLAN_REVISION, CAMERA_ORDER,
    _verified_archive, _extract_cameras, _check_contents, _entries, _hash_json,
)
from .cameras import validate_jpeg, MAX_MEMBER_BYTES
from .files import PreparationError, check_components, destination, safe_relative, verify_file, write_manifest
from hybrid_drive.experiments.bounded_io import decode_json, read_bounded, require_hex, require_keys


SCHEMA = "hybrid-drive-temporal-sources-v1"
ASSET_SCHEMA = "hybrid-drive-temporal-assets-v1"
MANIFEST = "temporal-manifest.json"


def load_temporal_sources(path=None):
    path = Path(path) if path is not None else Path(__file__).resolve().parents[3] / "scripts/temporal-sources.json"
    sources = decode_json(read_bounded(path, 128 * 1024))
    validate_temporal_sources(sources)
    return sources


def validate_temporal_sources(sources):
    require_keys(sources, {"schema", "navsim_revision", "nuplan_revision", "log", "camera_archive", "selection", "cases"}, "temporal sources")
    if (sources["schema"] != SCHEMA or sources["log"] != _PINNED_LOG or sources["camera_archive"] != _PINNED_ARCHIVE
            or sources["navsim_revision"] != _NAVSIM_REVISION or sources["nuplan_revision"] != _NUPLAN_REVISION):
        raise PreparationError("Temporal sources must retain the pinned recording/archive/revisions")
    selection = sources["selection"]
    require_keys(selection, {"rule", "num_source_frames", "num_history_frames", "num_future_frames", "selected_indices",
                             "timestamp_step_us", "identity_window", "frozen_before_inference"}, "temporal selection")
    if (selection["rule"] != "earliest-twenty-full-windows-v1" or selection["num_source_frames"] != 738
            or selection["num_history_frames"] != 4 or selection["num_future_frames"] != 10
            or selection["selected_indices"] != list(range(27, 47)) or selection["timestamp_step_us"] != [499000, 501000]
            or selection["frozen_before_inference"] is not True):
        raise PreparationError("Unknown temporal selection rule")
    window = selection["identity_window"]
    if not isinstance(window, list) or len(window) != 33:
        raise PreparationError("Temporal selection requires the complete 24..56 identity window")
    tokens = set()
    for offset, frame in enumerate(window):
        require_keys(frame, {"source_list_index", "token", "scene_token", "frame_idx", "timestamp"}, "temporal identity")
        require_hex(frame["token"], 16, "token")
        require_hex(frame["scene_token"], 16, "scene_token")
        if (type(frame["source_list_index"]) is not int or frame["source_list_index"] != offset + 24
                or type(frame["frame_idx"]) is not int or frame["frame_idx"] != offset
                or type(frame["timestamp"]) is not int or not 0 < frame["timestamp"] < 2**63
                or frame["scene_token"] != "ed69ac8f92945572" or frame["token"] in tokens):
            raise PreparationError("Temporal identity window is inconsistent")
        if offset and not 499000 <= frame["timestamp"] - window[offset-1]["timestamp"] <= 501000:
            raise PreparationError("Temporal timestamp discontinuity")
        tokens.add(frame["token"])
    cases, paths = sources["cases"], set()
    if not isinstance(cases, list) or len(cases) != 20:
        raise PreparationError("Exactly twenty temporal cases are required")
    for position, case in enumerate(cases):
        require_keys(case, {"case_id", "source_list_index", "metadata_frame_idx", "metadata_scene_token", "scene_token",
                            "frame_timestamp_us", "cameras"}, "temporal case")
        identity = window[position+3]
        expected = dict(case_id=f"t{position:02d}", source_list_index=position+27, metadata_frame_idx=position+3,
                        metadata_scene_token=identity["scene_token"], scene_token=identity["token"], frame_timestamp_us=identity["timestamp"])
        if any(type(case[key]) is not type(value) or case[key] != value for key, value in expected.items()):
            raise PreparationError("Temporal case differs from its fixed identity window")
        if not isinstance(case["cameras"], list) or len(case["cameras"]) != 3:
            raise PreparationError("Three current cameras are required per temporal frame")
        for name, camera in zip(CAMERA_ORDER, case["cameras"]):
            require_keys(camera, {"camera", "member", "path"}, "temporal camera")
            relative = safe_relative(camera["path"])
            suffix = f"{Path(_PINNED_LOG['path']).stem}/{name}/"
            filename = camera["path"].removeprefix("sensor_blobs/" + suffix)
            require_hex(filename.removesuffix(".jpg"), 16, "image token")
            if (camera["camera"] != name or str(relative) != "sensor_blobs/" + suffix + filename
                    or not filename.endswith(".jpg") or camera["member"] != _PINNED_ARCHIVE["archive_prefix"] + "/" + suffix + filename
                    or camera["path"] in paths):
                raise PreparationError("Temporal camera identity/order/path differs")
            paths.add(camera["path"])


def prepare_temporal_sample(root, sources, archive_path):
    validate_temporal_sources(sources)
    root = Path(os.path.abspath(root))
    check_components(root)
    if (root / MANIFEST).exists():
        return verify_temporal_sample(root, sources)
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise PreparationError("Temporal preparation requires a new empty root")
    root.parent.mkdir(parents=True, exist_ok=True)
    check_components(root.parent)
    with _verified_archive(archive_path, sources["camera_archive"]) as archive:
        with tempfile.TemporaryDirectory(prefix=".temporal-", dir=root.parent) as temporary:
            stage = Path(temporary)
            records = _extract_cameras(archive, stage, sources)
            manifest = {"schema": ASSET_SCHEMA, "sources_hash": _hash_json(sources),
                        "archive": {"size_bytes": sources["camera_archive"]["size_bytes"],
                                    "sha256": sources["camera_archive"]["sha256"], "full_archive_sha256_verified": True}, "files": records}
            manifest["manifest_hash"] = _hash_json(manifest)
            write_manifest(stage / MANIFEST, manifest)
            check_components(root)
            os.rename(stage, root)
    return manifest


def verify_temporal_sample(root, sources):
    validate_temporal_sources(sources)
    root = Path(os.path.abspath(root))
    manifest = decode_json(read_bounded(destination(root, MANIFEST), 128 * 1024))
    require_keys(manifest, {"schema", "sources_hash", "archive", "files", "manifest_hash"}, "temporal assets")
    expected_archive = {"size_bytes": sources["camera_archive"]["size_bytes"], "sha256": sources["camera_archive"]["sha256"],
                        "full_archive_sha256_verified": True}
    if (manifest["schema"] != ASSET_SCHEMA or manifest["sources_hash"] != _hash_json(sources)
            or manifest["archive"] != expected_archive
            or manifest["manifest_hash"] != _hash_json({k:v for k,v in manifest.items() if k != "manifest_hash"})):
        raise PreparationError("Temporal asset provenance/hash differs")
    entries, records = _entries(sources), manifest["files"]
    if not isinstance(records, list) or len(records) != 60:
        raise PreparationError("Temporal assets must retain all sixty JPEGs")
    for entry, record in zip(entries, records):
        require_keys(record, set(entry) | {"size_bytes", "sha256", "width", "height", "format", "mode"}, "temporal JPEG")
        if (any(record[key] != value for key,value in entry.items()) or type(record["size_bytes"]) is not int
                or not 0 < record["size_bytes"] <= MAX_MEMBER_BYTES):
            raise PreparationError("Temporal JPEG identity/size differs")
        require_hex(record["sha256"], 64, "JPEG sha256")
        path = destination(root, entry["path"])
        if verify_file(path, size_bytes=record["size_bytes"], sha256=record["sha256"]) is None:
            raise PreparationError("Temporal JPEG is missing")
        if any(record[key] != value for key,value in validate_jpeg(path).items()):
            raise PreparationError("Temporal JPEG decoding differs")
    _check_contents(root, {entry["path"] for entry in entries} | {MANIFEST})
    return manifest
