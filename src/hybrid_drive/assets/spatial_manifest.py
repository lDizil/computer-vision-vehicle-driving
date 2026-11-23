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
MANIFEST_NAME = "spatial-loc-manifest.json"
MANIFEST_SCHEMA = "hybrid-drive-spatial-loc-data-v1"
TARGET_SCHEMA = "hybrid-drive-spatial-loc-targets-v1"
_SOURCE_HASH = "31028c73e338ad3ffb5047af0a9042c36950529bb15736ae0558abd55df84280"
_EVALUATION_SOURCE_HASH = "73c19f6c0e65e2f505d0e03efa61dcda0b5128b59184b34d21aaed827907c949"
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
    else:
        raise PreparationError("Unknown spatial source schema")
    if digest_json(sources) != expected_hash:
        raise PreparationError("Spatial sources differ from the pinned recordings and protocol")

    require_keys(sources["logs"], set(splits), "spatial source roles")
    recordings, group_owners = set(), {}
    for split in splits:
        for log in sources["logs"][split]:
            name = log["log_name"]
            identity = recording_id(name)
            match = re.fullmatch(r"(\d{4}\.\d{2}\.\d{2})\.\d{2}\.\d{2}\.\d{2}_(veh-\d+)_\d+_\d+", name)
            if match is None or identity != log["recording_id"] or identity in recordings:
                raise PreparationError("Spatial recordings must be unique full recording identities")
            recordings.add(identity)
            if group_owners.setdefault(match.groups(), split) != split:
                raise PreparationError("A source date+vehicle group cannot cross spatial splits")
            if not 0 < log["sample_count"] <= log["frame_count"]:
                raise PreparationError("Spatial source metadata cannot supply its frozen sample count")
    return splits


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
            _validate_recording(recording, spec, prefix, split)
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


def _validate_recording(recording, spec, archive_prefix, split):
    require_keys(recording, {"recording_id", "log_token", "samples"}, "spatial recording")
    require_hex(recording["log_token"], 16, "log token")
    identity, samples = recording["recording_id"], recording["samples"]
    count = spec["sample_count"]
    if identity != spec["recording_id"] or not isinstance(samples, list) or len(samples) != count:
        raise PreparationError("Spatial recording identity or sample count differs from the frozen plan")
    previous_timestamp = 0
    for index, item in zip(midpoint_indices(spec["frame_count"], count), samples):
        _validate_recording_sample(
            item, archive_prefix, spec["log_name"], sample_id(split, identity, index), index, previous_timestamp,
        )
        previous_timestamp = item["timestamp_us"]


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
