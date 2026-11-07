"""Public preparation entry point for one real official LTF input frame."""
from __future__ import annotations

from pathlib import Path
import os

from .cameras import reuse_cameras, stream_cameras, validate_jpeg
from .fetch import AssetSpec, MAX_TRANSFER_BYTES, TransferBudget, download_verified_asset
from .files import PreparationError, check_components, checked_root, destination, read_manifest, verify_file, write_manifest


MODEL_PATHS = (
    ("checkpoint", "checkpoints/ltf_seed_0.ckpt"),
    ("encoder.config", "encoder/config.json"),
    ("encoder.weights", "encoder/model.safetensors"),
)


def _source(sources, key):
    value = sources
    for part in key.split("."):
        value = value[part]
    return value


def validate_sources(sources):
    if sources.get("scene_token") != "36dd0d0bb6f45f01" or sources.get("frame_index") != 3:
        raise PreparationError("Expected the selected real OpenScene frame")
    if sources.get("camera_order") != ["CAM_L0", "CAM_F0", "CAM_R0"]:
        raise PreparationError("Camera input order differs from the LTF feature builder")
    specs = [AssetSpec.from_dict(_source(sources, key)) for key, _ in MODEL_PATHS]
    specs.append(AssetSpec.from_dict(sources["cameras"]))
    for spec in specs:
        spec.validate()
    if sum(spec.size_bytes for spec in specs) > MAX_TRANSFER_BYTES:
        raise PreparationError("Worst-case preparation exceeds the 4 GiB transfer budget")
    entries = sources["cameras"]["files"]
    prefix = "2021.10.06.17.43.07_veh-28_00508_00877"
    expected = [
        ("CAM_L0", "a6c2e04141ff5d96.jpg"),
        ("CAM_F0", "6376edc8d4e75d0d.jpg"),
        ("CAM_R0", "d347f05f8fd956ac.jpg"),
    ]
    if len(entries) != 3:
        raise PreparationError("Expected exactly three camera entries")
    for entry, (camera, filename) in zip(entries, expected):
        suffix = f"{prefix}/{camera}/{filename}"
        if entry != {"camera": camera, "member": f"openscene-v1.1/sensor_blobs/mini/{suffix}", "path": f"sensor_blobs/{suffix}"}:
            raise PreparationError("Camera allowlist differs from selected-frame metadata")


def prepare(root, sources, *, rate_limit_mib=4, opener=None, validator=validate_jpeg):
    validate_sources(sources)
    root = checked_root(root)
    manifest_path = destination(root, "ltf-manifest.json")
    previous = read_manifest(manifest_path)
    if previous is not None and previous.get("sources") != sources:
        raise PreparationError("Existing LTF manifest differs from pinned sources")
    camera_manifest_path = destination(root, "ltf-camera-manifest.json")
    previous_cameras = read_manifest(camera_manifest_path)
    reused = reuse_cameras(root, sources["cameras"], previous_cameras, validator)
    budget = TransferBudget(rate_limit_mib)
    ceiling = sum(AssetSpec.from_dict(_source(sources, key)).size_bytes for key, _ in MODEL_PATHS)
    ceiling += sources["cameras"]["size_bytes"]
    print(f"Worst-case transfer: {ceiling} bytes; rate <= {rate_limit_mib:g} MiB/s; deadline 3600s", flush=True)
    kwargs = {} if opener is None else {"opener": opener}
    models = []
    for key, path in MODEL_PATHS:
        spec = AssetSpec.from_dict(_source(sources, key))
        print(f"Verifying/preparing {path}: {spec.size_bytes} bytes", flush=True)
        record = download_verified_asset(spec, destination(root, path), budget=budget, **kwargs)
        models.append({"path": path, "source": _source(sources, key), **record})
    if reused is None:
        print("Streaming pinned camera shard; only the three selected JPEGs will be written", flush=True)
        cameras = stream_cameras(root, sources["cameras"], budget, validator=validator, **kwargs)
        write_manifest(camera_manifest_path, cameras)
    else:
        cameras = previous_cameras
    result = {
        "sources": sources, "scene_token": sources["scene_token"], "frame_index": sources["frame_index"],
        "camera_order": sources["camera_order"], "models": models, "cameras": cameras,
        "bytes_transferred_this_run": budget.received,
        "pickle_deserialized": False, "camera_archive_saved": False,
    }
    write_manifest(manifest_path, result)
    print(f"Prepared official checkpoint, offline encoder and three decoded JPEGs; manifest: {manifest_path}", flush=True)
    return result


def verify_prepared(root, sources, *, validator=validate_jpeg):
    """Verify an existing read-only preparation, without mkdir, downloads or writes."""
    validate_sources(sources)
    root = Path(os.path.abspath(root))
    check_components(root)
    if not root.is_dir():
        raise PreparationError("Prepared asset root is missing")
    manifest = read_manifest(destination(root, "ltf-manifest.json"))
    if manifest is None:
        raise PreparationError("Prepared LTF manifest is missing")
    expected = {"sources": sources, "scene_token": sources["scene_token"],
                "frame_index": sources["frame_index"], "camera_order": sources["camera_order"]}
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise PreparationError("Prepared LTF manifest differs from selected-frame sources")
    models = manifest.get("models")
    if not isinstance(models, list) or len(models) != len(MODEL_PATHS):
        raise PreparationError("Prepared model manifest is incomplete")
    for record, (key, path) in zip(models, MODEL_PATHS):
        spec = AssetSpec.from_dict(_source(sources, key))
        if not isinstance(record, dict) or any(record.get(name) != value for name, value in {
            "path": path, "source": _source(sources, key), "size_bytes": spec.size_bytes, "sha256": spec.sha256,
        }.items()):
            raise PreparationError("Prepared model manifest differs from pinned asset")
        if verify_file(destination(root, path), size_bytes=spec.size_bytes, sha256=spec.sha256) is None:
            raise PreparationError(f"Prepared model file is missing: {path}")
    cameras = read_manifest(destination(root, "ltf-camera-manifest.json"))
    if cameras is None or manifest.get("cameras") != cameras:
        raise PreparationError("Prepared camera manifests are missing or inconsistent")
    if reuse_cameras(root, sources["cameras"], cameras, validator) is None:
        raise PreparationError("Prepared cameras are incomplete")
    return manifest
