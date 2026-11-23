"""Verified tracking labels for existing spatial clips, without reopening any images."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import sys

import numpy as np

from hybrid_drive.adapters.real_camera import OPENSCENE_NATIVE_PROFILE, openscene_camera_calibrations
from hybrid_drive.experiments.bounded_io import canonical_json, decode_json, read_bounded, require_hex, require_keys

from .alignment_manifest import CAMERA, IMAGE_SIZE, digest_json
from .alignment_sample import _frame_calibration_receipt
from .files import AtomicFile, PreparationError, check_components
from .spatial_manifest import BAND_COUNT, read_spatial_targets
from .spatial_sample import _select_spatial_frames, _select_spatial_targets
from .spatial_temporal import TemporalSpatialPack, read_temporal_spatial_pack


TRACK_TRUTH_SCHEMA = "hybrid-drive-spatial-track-truth-v1"
_MAX_TRUTH_BYTES = 256 * 1024


def prepare_track_truth(pack: TemporalSpatialPack, metadata_archive, output_path) -> str:
    """Derive opaque labels from pinned metadata; publish only after verifying every projection."""
    pack = _verified_pack(pack)
    path = _sidecar_path(pack, output_path)
    if path.exists():
        raise FileExistsError(f"Tracking truth already exists: {path}")
    if sys.platform != "linux":
        raise PreparationError("Prepare pinned metadata only in an isolated Linux container without network")

    source = pack.manifest["sources"]
    labels = read_spatial_targets(pack.root, pack.manifest, "holdout")
    selected = _select_spatial_frames(source, metadata_archive)
    rows = []
    partition = pack.manifest["splits"]["holdout"]
    for recording in partition["recordings"]:
        recording_id = recording["recording_id"]
        frames = selected[recording_id]
        if len(frames) != len(recording["samples"]):
            raise PreparationError("Tracking metadata differs from the prepared recording length")
        for sample, (index, frame, member) in zip(recording["samples"], frames):
            _verify_source_frame(recording, sample, index, frame, member)
            calibration = _verified_calibration(sample, frame["cams"][CAMERA])
            aabbs, indices = _select_spatial_targets(frame["anns"], calibration)
            target = labels[len(rows)]
            if target["boxes_2d"] != aabbs or target["presence"] != [box is not None for box in aabbs]:
                raise PreparationError("Tracking projections differ from the prepared target boxes or presence")
            rows.append({
                "sample_id": target["sample_id"],
                "tracks": _tracks_for_bands(recording_id, frame["anns"], indices),
            })
    if len(rows) != len(pack.frames):
        raise PreparationError("Tracking truth must cover every prepared frame")

    encoded = canonical_json({**_header(pack), "rows": rows})
    if len(encoded) > _MAX_TRUTH_BYTES:
        raise PreparationError("Tracking truth exceeds its byte budget")
    with AtomicFile(path) as temporary:
        temporary.out.write(encoded)
        temporary.commit()
    return hashlib.sha256(encoded).hexdigest()


def read_track_truth(pack: TemporalSpatialPack, path, *, expected_sha256) -> dict[str, tuple[str | None, ...]]:
    """Read a verified sidecar and validate its complete temporal target alignment."""
    require_hex(expected_sha256, 64, "tracking truth byte hash")
    pack = _verified_pack(pack)
    data = read_bounded(_sidecar_path(pack, path), _MAX_TRUTH_BYTES)
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise PreparationError("Tracking truth byte hash differs")

    header = _header(pack)
    parsed = decode_json(data)
    require_keys(parsed, set(header) | {"rows"}, "tracking truth")
    if any(parsed[key] != value for key, value in header.items()):
        raise PreparationError("Tracking truth does not bind this exact temporal pack, targets and metadata")
    rows = parsed["rows"]
    if not isinstance(rows, list) or len(rows) != len(pack.frames):
        raise PreparationError("Tracking truth must cover every prepared frame")

    labels = read_spatial_targets(pack.root, pack.manifest, "holdout")
    result, aliases = {}, {}
    for row, frame, target in zip(rows, pack.frames, labels):
        require_keys(row, {"sample_id", "tracks"}, "tracking truth row")
        tracks = row["tracks"]
        if (row["sample_id"] != frame.sample_id or target["sample_id"] != frame.sample_id
                or not isinstance(tracks, list) or len(tracks) != BAND_COUNT):
            raise PreparationError("Tracking truth frames do not preserve prepared target order")
        seen = set()
        for track, present in zip(tracks, target["presence"]):
            if (track is not None) != present:
                raise PreparationError("Tracking truth track presence differs from the prepared target")
            if track is not None:
                require_hex(track, 64, "opaque track alias")
                if track in seen or aliases.setdefault(track, frame.recording_id) != frame.recording_id:
                    raise PreparationError("Duplicate track in one frame or shared across recordings")
                seen.add(track)
        result[frame.sample_id] = tuple(tracks)
    return result


def _verified_pack(pack):
    if not isinstance(pack, TemporalSpatialPack):
        raise ValueError("Tracking truth needs a prepared TemporalSpatialPack")
    fresh = read_temporal_spatial_pack(pack.root, role=pack.manifest["sources"]["temporal_role"])
    if fresh != pack:
        raise PreparationError("Temporal pack has changed since it was read")
    return fresh


def _sidecar_path(pack, path):
    path = Path(os.path.abspath(path))
    check_components(path)
    if path.is_relative_to(pack.root) or any(
        parent.exists() and parent.samefile(pack.root) for parent in path.parents
    ):
        raise PreparationError("Tracking truth sidecar must be outside the prepared pack")
    return path


def _header(pack):
    return {
        "schema": TRACK_TRUTH_SCHEMA,
        "temporal_role": pack.manifest["sources"]["temporal_role"],
        "pack_hash": pack.manifest_hash,
        "target_sha256": pack.manifest["splits"]["holdout"]["targets"]["sha256"],
        "metadata_sha256": pack.manifest["sources"]["metadata_archive"]["sha256"],
    }


def _verify_source_frame(recording, sample, index, frame, member):
    if (
        frame["log_token"] != recording["log_token"] or index != sample["source_list_index"]
        or frame["token"] != sample["frame_token"] or frame["scene_token"] != sample["scene_token"]
        or frame["timestamp"] != sample["timestamp_us"] or frame["frame_idx"] != sample["frame_idx"]
        or member != sample["jpeg"]["member"]
    ):
        raise PreparationError("Tracking metadata frame identity differs from the prepared sample")


def _verified_calibration(sample, camera):
    jpeg = sample["jpeg"]
    native_width, native_height = jpeg["width"], jpeg["height"]
    _, calibration, _ = openscene_camera_calibrations(
        CAMERA, native_width, native_height, camera["cam_intrinsic"],
        camera["sensor2lidar_rotation"], camera["sensor2lidar_translation"], camera["distortion"],
        profile=OPENSCENE_NATIVE_PROFILE, width=IMAGE_SIZE, height=IMAGE_SIZE,
    )
    receipt = _frame_calibration_receipt(camera, native_width, native_height, calibration)
    if digest_json(receipt) != sample["calibration_sha256"]:
        raise PreparationError("Tracking metadata calibration differs from the prepared sample")
    return calibration


def _tracks_for_bands(recording_id, annotations, indices):
    names = annotations["gt_names"]
    tokens = annotations.get("track_tokens")
    if (not isinstance(tokens, (list, tuple, np.ndarray))
            or isinstance(tokens, np.ndarray) and tokens.ndim != 1 or len(tokens) != len(names)):
        raise PreparationError("OpenScene tracking IDs do not align with annotations")
    seen = set()
    for name, token in zip(names, tokens):
        if name != "vehicle":
            continue
        require_hex(token, 16, "OpenScene vehicle track token")
        if token in seen:
            raise PreparationError("Duplicate source vehicle track token in one frame")
        seen.add(token)
    return [
        None if index is None else hashlib.sha256(f"{recording_id}\0{tokens[index]}".encode()).hexdigest()
        for index in indices
    ]
