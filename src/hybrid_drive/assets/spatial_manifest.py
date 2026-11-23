"""Frozen source and file contracts for the four-band OpenScene LOC dataset."""
from __future__ import annotations

import os
from pathlib import Path
import re

import numpy as np

from hybrid_drive.experiments.bounded_io import decode_json, read_bounded, require_hex, require_keys

from .alignment_manifest import (
    ARCHIVE_VERIFICATION, IMAGE_SIZE, MIN_EXTENT_PX, _numeric_vector, _read_verified,
    _validate_file_record, _validate_recording_sample, digest_json, midpoint_indices,
    read_alignment_rgb, recording_id, targets_path,
)
from .cohort import _check_contents
from .files import PreparationError, destination


SPLITS = ("train", "dev", "holdout")
BAND_COUNT = 4
SOURCE_SCHEMA = "hybrid-drive-spatial-loc-sources-v1"
EVALUATION_SOURCE_SCHEMA = "hybrid-drive-spatial-loc-evaluation-sources-v1"
POLICY_EVALUATION_SOURCE_SCHEMA = "hybrid-drive-spatial-policy-evaluation-sources-v1"
TEMPORAL_SOURCE_SCHEMA = "hybrid-drive-spatial-temporal-sources-v1"
TRACKING_HOLDOUT_ROLE = "tracking_holdout"
APPEARANCE_HOLDOUT_ROLE = "appearance_holdout"
ZOOM_GATE_HOLDOUT_ROLE = "zoom_gate_holdout"
TEMPORAL_ROLES = (
    "development", "holdout", "stress", TRACKING_HOLDOUT_ROLE, APPEARANCE_HOLDOUT_ROLE, ZOOM_GATE_HOLDOUT_ROLE,
)
TEMPORAL_CLIPS_PER_RECORDING = 3
TEMPORAL_FRAMES_PER_CLIP = 24
TEMPORAL_DELTA_US = (250_000, 800_000)
MANIFEST_NAME = "spatial-loc-manifest.json"
MANIFEST_SCHEMA = "hybrid-drive-spatial-loc-data-v1"
TARGET_SCHEMA = "hybrid-drive-spatial-loc-targets-v1"
_SOURCE_HASH = "31028c73e338ad3ffb5047af0a9042c36950529bb15736ae0558abd55df84280"
_EVALUATION_SOURCE_HASH = "73c19f6c0e65e2f505d0e03efa61dcda0b5128b59184b34d21aaed827907c949"
_POLICY_EVALUATION_SOURCE_HASH = "88a8c417833e615494906abe92e0bd1b1d7d765aaf25c458b844c9cb09681657"
_TEMPORAL_SOURCE_HASHES = {
    "development": "8123701b1d8e4272c1dd3fd677ca32bc9b6d76d785665006f52b25d3fc44c80e",
    "holdout": "67a84eb97b46566d864ddb23f4cbccef0a16377ceb8e71ffc17d443e6fff45dd",
    "stress": "c1b0e7f4a67ceab9119fb5d8b429ac98b6a48c75a6f7098a0bb72363c7eb56ef",
    TRACKING_HOLDOUT_ROLE: "a8d0ee46059faaf70fac05f161932f5d8e98fdd4e7ad67c008e9222d7cbaaf41",
    APPEARANCE_HOLDOUT_ROLE: "8367f676f334f3c8d3e9b26fd5a99618b01105514e3267e07b42d2ce5d6d6429",
    ZOOM_GATE_HOLDOUT_ROLE: "8e7662e50ccec298059b63fe7333b9975fd59c00d2b79527bea640020c34cf78",
}
_TEMPORAL_SELECTION = {
    "rule": "uniform-midpoint-contiguous-clips-v1",
    "clips_per_recording": TEMPORAL_CLIPS_PER_RECORDING,
    "frames_per_clip": TEMPORAL_FRAMES_PER_CLIP,
    "adjacent_timestamp_delta_us": list(TEMPORAL_DELTA_US),
    "time_reference": "openscene-associated-lidar-frame-no-camera-timestamp",
    "frozen_before_inference": True,
    "substitute_missing_frame": False,
}
_MAX_MANIFEST_BYTES = 512 * 1024
_MAX_TARGET_BYTES = 128 * 1024


def load_spatial_sources(path):
    sources = decode_json(read_bounded(Path(path), 32 * 1024))
    validate_spatial_sources(sources)
    return sources


def validate_spatial_sources(sources):
    """Validate a pinned source document and return its ordered roles."""
    schema = sources.get("schema") if isinstance(sources, dict) else None
    if schema == SOURCE_SCHEMA:
        expected_hash, splits = _SOURCE_HASH, SPLITS
    elif schema == EVALUATION_SOURCE_SCHEMA:
        expected_hash, splits = _EVALUATION_SOURCE_HASH, ("holdout",)
    elif schema == POLICY_EVALUATION_SOURCE_SCHEMA:
        expected_hash, splits = _POLICY_EVALUATION_SOURCE_HASH, ("holdout",)
    elif schema == TEMPORAL_SOURCE_SCHEMA:
        temporal_role = sources.get("temporal_role")
        if temporal_role not in TEMPORAL_ROLES:
            raise PreparationError("Unknown temporal spatial role")
        expected_hash, splits = _TEMPORAL_SOURCE_HASHES.get(temporal_role), ("holdout",)
    else:
        raise PreparationError("Unknown spatial source schema")
    if digest_json(sources) != expected_hash:
        raise PreparationError("Spatial sources differ from the pinned recordings and protocol")

    require_keys(sources["logs"], set(splits), "spatial source roles")
    if schema == TEMPORAL_SOURCE_SCHEMA:
        _validate_temporal_sources(sources)
    recordings, group_owners = set(), {}
    for split in splits:
        for log in sources["logs"][split]:
            name = log["log_name"]
            identity = recording_id(name)
            match = re.fullmatch(r"(\d{4}\.\d{2}\.\d{2})\.\d{2}\.\d{2}\.\d{2}_(veh-\d+)_\d+_\d+", name)
            if match is None or identity != log["recording_id"] or identity in recordings:
                raise PreparationError("Spatial recordings must be unique full recording identities")
            recordings.add(identity)
            if schema == TEMPORAL_SOURCE_SCHEMA and match.groups() in group_owners:
                raise PreparationError("Temporal spatial recordings must use different date and vehicle groups")
            if group_owners.setdefault(match.groups(), split) != split:
                raise PreparationError("A source date+vehicle group cannot cross spatial splits")
            if not 0 < log["sample_count"] <= log["frame_count"]:
                raise PreparationError("Spatial source metadata cannot supply its frozen sample count")
            if schema == TEMPORAL_SOURCE_SCHEMA:
                temporal_clip_indices(log["frame_count"])
    return splits


def _validate_temporal_sources(sources):
    recordings = sources["logs"]["holdout"]
    if sources["selection"] != _TEMPORAL_SELECTION or not isinstance(recordings, list) or len(recordings) != 2:
        raise PreparationError("Temporal spatial sources require two recordings and the frozen clip rule")
    expected_count = TEMPORAL_CLIPS_PER_RECORDING * TEMPORAL_FRAMES_PER_CLIP
    if (any(log["sample_count"] != expected_count for log in recordings)
            or set(sources["camera_archives"]) != {str(log["camera_shard"]) for log in recordings}):
        raise PreparationError("Temporal spatial sources differ from the clip size or required camera shards")


def temporal_clip_indices(frame_count):
    """Partition by source-list anchors, then take one full consecutive clip per anchor."""
    count, length = TEMPORAL_CLIPS_PER_RECORDING, TEMPORAL_FRAMES_PER_CLIP
    if type(frame_count) is not int or frame_count < count * length:
        raise PreparationError("Source recording is too short for the pinned temporal clips")
    starts = ((2 * clip + 1) * frame_count // (2 * count) - length // 2 for clip in range(count))
    return tuple(range(start, start + length) for start in starts)


def spatial_frame_indices(sources, spec):
    if sources["schema"] == TEMPORAL_SOURCE_SCHEMA:
        return tuple(index for clip in temporal_clip_indices(spec["frame_count"]) for index in clip)
    return midpoint_indices(spec["frame_count"], spec["sample_count"])


def validate_temporal_cadence(timestamps):
    """Use associated LiDAR-frame timestamps, never infer camera exposure time."""
    expected = TEMPORAL_CLIPS_PER_RECORDING * TEMPORAL_FRAMES_PER_CLIP
    if len(timestamps) != expected:
        raise PreparationError("Temporal recording differs from the frozen frame count")
    minimum, maximum = TEMPORAL_DELTA_US
    for offset, (left, right) in enumerate(zip(timestamps, timestamps[1:]), start=1):
        if offset % TEMPORAL_FRAMES_PER_CLIP and not minimum <= right - left <= maximum:
            raise PreparationError("Temporal clip has an out-of-range adjacent source timestamp delta")


def sample_id(split, identity, index):
    return f"{split}-{identity}-{index:04d}"


def box_horizontal_band(box):
    # Both the addition and the half multiply use FP32, including at 1/4, 1/2 and 3/4.
    xmin, xmax = np.float32(box[0]), np.float32(box[2])
    center = (xmin + xmax) * np.float32(0.5)
    return int(center * np.float32(BAND_COUNT))


def read_spatial_manifest(root):
    root = Path(os.path.abspath(root))
    manifest = decode_json(read_bounded(destination(root, MANIFEST_NAME), _MAX_MANIFEST_BYTES))
    require_keys(manifest, {"schema", "sources", "archive_verification", "splits", "manifest_hash"}, "spatial manifest")
    sources = manifest["sources"]
    splits = validate_spatial_sources(sources)
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    if (manifest["schema"] != MANIFEST_SCHEMA or manifest["archive_verification"] != ARCHIVE_VERIFICATION
            or manifest["manifest_hash"] != digest_json(unsigned)):
        raise PreparationError("Spatial manifest schema/provenance/hash differs")

    files = {MANIFEST_NAME}
    frame_tokens, jpeg_members, log_tokens = set(), set(), set()
    jpeg_roles, pixel_roles = {}, {}
    require_keys(manifest["splits"], set(splits), "spatial splits")
    for split in splits:
        partition = manifest["splits"][split]
        require_keys(partition, {"recordings", "targets"}, "spatial split")
        recordings, specs = partition["recordings"], sources["logs"][split]
        if not isinstance(recordings, list) or len(recordings) != len(specs):
            raise PreparationError("Spatial split must contain exactly its planned recordings")

        for recording, spec in zip(recordings, specs):
            prefix = sources["camera_archives"][str(spec["camera_shard"])]["archive_prefix"]
            _validate_recording(recording, sources, spec, prefix, split)
            token = recording["log_token"]
            if token in log_tokens:
                raise PreparationError("Duplicate source log token across spatial recordings")
            log_tokens.add(token)
            for sample in recording["samples"]:
                jpeg, rgb = sample["jpeg"], sample["rgb"]
                token, member = sample["frame_token"], jpeg["member"]
                if token in frame_tokens or member in jpeg_members:
                    raise PreparationError("Spatial dataset contains duplicate frame or JPEG identity")
                if jpeg_roles.setdefault(jpeg["sha256"], split) != split:
                    raise PreparationError("Identical source JPEG cannot cross spatial splits")
                if pixel_roles.setdefault(rgb["pixels_sha256"], split) != split:
                    raise PreparationError("Identical decoded RGB pixels cannot cross spatial splits")
                frame_tokens.add(token)
                jpeg_members.add(member)
                files.add(rgb["path"])

        _validate_file_record(partition["targets"], targets_path(split), _MAX_TARGET_BYTES, "spatial targets")
        files.add(partition["targets"]["path"])

    _check_contents(root, files)
    return manifest


def _validate_recording(recording, sources, spec, archive_prefix, split):
    require_keys(recording, {"recording_id", "log_token", "samples"}, "spatial recording")
    require_hex(recording["log_token"], 16, "log token")
    identity, samples = recording["recording_id"], recording["samples"]
    count = spec["sample_count"]
    if identity != spec["recording_id"] or not isinstance(samples, list) or len(samples) != count:
        raise PreparationError("Spatial recording identity or sample count differs from the frozen plan")
    previous_timestamp = 0
    for index, item in zip(spatial_frame_indices(sources, spec), samples):
        _validate_recording_sample(
            item, archive_prefix, spec["log_name"], sample_id(split, identity, index), index, previous_timestamp,
        )
        previous_timestamp = item["timestamp_us"]
    if sources["schema"] == TEMPORAL_SOURCE_SCHEMA:
        validate_temporal_cadence([sample["timestamp_us"] for sample in samples])


def read_spatial_targets(root, manifest, split):
    if split not in SPLITS:
        raise ValueError("Spatial split must be train, dev or holdout")
    if split not in manifest["splits"]:
        raise ValueError(f"Spatial pack does not contain the {split} split")
    partition = manifest["splits"][split]
    targets = decode_json(_read_verified(root, partition["targets"], _MAX_TARGET_BYTES))
    require_keys(targets, {"schema", "split", "rows"}, "spatial targets")
    rows = targets["rows"]
    samples = [sample for recording in partition["recordings"] for sample in recording["samples"]]
    if (targets["schema"] != TARGET_SCHEMA or targets["split"] != split or not isinstance(rows, list)
            or len(rows) != len(samples)):
        raise PreparationError("Spatial targets must match the requested split exactly")
    for row, sample in zip(rows, samples):
        _validate_target(row, sample["sample_id"])
    return rows


def _validate_target(row, identity):
    require_keys(row, {"sample_id", "presence", "boxes_2d"}, "spatial LOC target")
    mask, boxes = row["presence"], row["boxes_2d"]
    if (row["sample_id"] != identity or not isinstance(mask, list) or len(mask) != BAND_COUNT
            or not all(type(value) is bool for value in mask) or not isinstance(boxes, list) or len(boxes) != BAND_COUNT):
        raise PreparationError("Spatial target identity or mask differs")
    for band, (present, box) in enumerate(zip(mask, boxes)):
        if present:
            if not _valid_positive_box(box, band):
                raise PreparationError("Spatial positive box is invalid or belongs in another band")
        elif box is not None:
            raise PreparationError("A negative spatial target must not contain a box")


def _valid_positive_box(box, band):
    if not _numeric_vector(box, 4, 1) or min(box) < 0:
        return False
    xyxy = np.asarray(box, dtype=np.float32)
    # Eligibility is checked before quantizing: only tolerate coordinate rounding.
    minimum = MIN_EXTENT_PX / IMAGE_SIZE - np.finfo(np.float32).eps
    return xyxy.tolist() == box and min(xyxy[2:] - xyxy[:2]) >= minimum and box_horizontal_band(xyxy) == band


def read_spatial_rgb(root, record):
    return read_alignment_rgb(root, record)


def verify_spatial_sample(root, *, expected_sources=None):
    manifest = read_spatial_manifest(root)
    if expected_sources is not None and manifest["sources"] != expected_sources:
        raise PreparationError("Prepared spatial pack uses different pinned sources")
    for split in manifest["splits"]:
        read_spatial_targets(root, manifest, split)
        for recording in manifest["splits"][split]["recordings"]:
            for item in recording["samples"]:
                read_spatial_rgb(root, item["rgb"])
    return manifest


def verify_disjoint_spatial_manifests(fresh, previous, *, role, compare_selection=True):
    """Require the same camera and target protocol on entirely different observations."""
    if previous["manifest_hash"] == fresh["manifest_hash"]:
        raise ValueError(f"Holdout manifest must be different from the {role}")
    scope = ("dataset", "openscene_revision", "navsim_revision", "nuplan_revision", "rgb", "loc")
    if compare_selection:
        scope += ("selection",)
    if any(previous["sources"][key] != fresh["sources"][key] for key in scope):
        raise ValueError(f"Fresh holdout must preserve the source, image and target protocol of the {role}")
    old = _manifest_identities(previous)
    for label, values in _manifest_identities(fresh).items():
        if values.intersection(old[label]):
            raise ValueError(f"Fresh holdout overlaps the {role} by {label}")


def _manifest_identities(manifest):
    recordings = [recording for partition in manifest["splits"].values() for recording in partition["recordings"]]
    recording_ids = {row["recording_id"] for row in recordings}
    samples = [sample for row in recordings for sample in row["samples"]]
    return {
        "recording": recording_ids,
        "date and vehicle": {
            (".".join(identity.split(".")[:3]), identity.rsplit("_", 1)[1]) for identity in recording_ids
        },
        "log token": {row["log_token"] for row in recordings},
        "frame token": {row["frame_token"] for row in samples},
        "JPEG member": {row["jpeg"]["member"] for row in samples},
        "JPEG digest": {row["jpeg"]["sha256"] for row in samples},
        "RGB digest": {row["rgb"]["pixels_sha256"] for row in samples},
    }
