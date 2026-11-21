"""Pinned identities and strict on-disk contract for the OpenScene alignment slice."""
from __future__ import annotations

import hashlib
import io
import math
import os
from pathlib import Path

from .cameras import MAX_JPEG_PIXELS, MAX_MEMBER_BYTES
from .cohort import _check_contents
from .files import PreparationError, destination
from hybrid_drive.experiments.bounded_io import canonical_json, decode_json, read_bounded, require_hex, require_keys


SPLITS = ("train", "holdout")
CAMERA = "CAM_F0"
IMAGE_SIZE = 224
MAX_DISTANCE_M = 40.0
MIN_DEPTH_M = 0.1
MIN_EXTENT_PX = 2.0
CENTER_X = (0.25, 0.75)

MANIFEST_NAME = "alignment-manifest.json"
MANIFEST_SCHEMA = "hybrid-drive-alignment-data-v1"
TARGET_SCHEMA = "hybrid-drive-alignment-loc-targets-v1"
ARCHIVE_VERIFICATION = "whole-archive-sha256-with-allowlisted-members"
_SOURCE_HASH = "4ab186aa0b2f14e8f6212630e97a2d36da2b772c75db19e2ce8de46ae1fda11c"
_MAX_MANIFEST_BYTES = 256 * 1024
_MAX_TARGET_BYTES = 128 * 1024
_MAX_PNG_BYTES = 256 * 1024


def load_alignment_sources(path):
    sources = decode_json(read_bounded(Path(path), 16 * 1024))
    validate_alignment_sources(sources)
    return sources


def validate_alignment_sources(sources):
    if not isinstance(sources, dict) or digest_json(sources) != _SOURCE_HASH:
        raise PreparationError("Alignment sources differ from the pinned OpenScene slice and protocol")


def digest_json(value):
    return hashlib.sha256(canonical_json(value)).hexdigest()


def midpoint_indices(frame_count, sample_count):
    return tuple((2 * index + 1) * frame_count // (2 * sample_count) for index in range(sample_count))


def recording_id(log_name):
    return log_name.rsplit("_", 2)[0]


def sample_id(split, source_list_index):
    return f"{split}-{source_list_index:04d}"


def rgb_path(identity):
    return f"rgb/{identity}.png"


def targets_path(split):
    return f"targets/{split}.json"


def validate_jpeg_member(member, sources, split):
    validate_recording_jpeg_member(member, sources["camera_archive"]["archive_prefix"], sources["logs"][split]["log_name"])


def validate_recording_jpeg_member(member, archive_prefix, log_name):
    prefix = f"{archive_prefix}/{log_name}/{CAMERA}/"
    if not isinstance(member, str) or not member.startswith(prefix) or not member.endswith(".jpg"):
        raise PreparationError("JPEG must identify the matching OpenScene log and CAM_F0")
    require_hex(member.removeprefix(prefix).removesuffix(".jpg"), 16, "JPEG token")


def read_alignment_manifest(root):
    root = Path(os.path.abspath(root))
    manifest = decode_json(read_bounded(destination(root, MANIFEST_NAME), _MAX_MANIFEST_BYTES))
    require_keys(manifest, {"schema", "sources", "archive_verification", "splits", "manifest_hash"}, "alignment manifest")
    validate_alignment_sources(manifest["sources"])
    if (manifest["schema"] != MANIFEST_SCHEMA or manifest["archive_verification"] != ARCHIVE_VERIFICATION
            or manifest["manifest_hash"] != digest_json({k: v for k, v in manifest.items() if k != "manifest_hash"})):
        raise PreparationError("Alignment manifest schema/provenance/hash differs")
    require_keys(manifest["splits"], set(SPLITS), "alignment splits")
    files, frame_tokens, jpeg_members, recordings, log_tokens = {MANIFEST_NAME}, set(), set(), set(), set()
    for split in SPLITS:
        partition = manifest["splits"][split]
        log = manifest["sources"]["logs"][split]
        require_keys(partition, {"recording_id", "log_token", "positive_count", "negative_count", "samples", "targets"},
                     "alignment split")
        require_hex(partition["log_token"], 16, "log token")
        if (partition["recording_id"] != recording_id(log["log_name"]) or partition["recording_id"] in recordings
                or partition["log_token"] in log_tokens):
            raise PreparationError("Alignment splits must be on different pinned recordings")
        recordings.add(partition["recording_id"])
        log_tokens.add(partition["log_token"])
        samples = partition["samples"]
        count = log["sample_count"]
        positive, negative = partition["positive_count"], partition["negative_count"]
        if (not isinstance(samples, list) or len(samples) != count or type(positive) is not int
                or type(negative) is not int or min(positive, negative) < 0 or positive + negative != count):
            raise PreparationError("Alignment split counts do not match the fixed sample size")
        previous_timestamp = 0
        for index, item in zip(midpoint_indices(log["frame_count"], count), samples):
            _validate_sample(item, manifest["sources"], split, index, previous_timestamp)
            token, member = item["frame_token"], item["jpeg"]["member"]
            if token in frame_tokens or member in jpeg_members:
                raise PreparationError("Alignment slice contains duplicate frame or JPEG identity")
            frame_tokens.add(token)
            jpeg_members.add(member)
            previous_timestamp = item["timestamp_us"]
            files.add(item["rgb"]["path"])
        _validate_file_record(partition["targets"], targets_path(split), _MAX_TARGET_BYTES, "targets")
        files.add(partition["targets"]["path"])
    _check_contents(root, files)
    return manifest


def _validate_sample(item, sources, split, index, previous_timestamp):
    _validate_recording_sample(
        item, sources["camera_archive"]["archive_prefix"], sources["logs"][split]["log_name"],
        sample_id(split, index), index, previous_timestamp,
    )


def _validate_recording_sample(item, archive_prefix, log_name, identity, index, previous_timestamp):
    require_keys(item, {"sample_id", "source_list_index", "frame_token", "scene_token", "frame_idx", "timestamp_us",
                        "jpeg", "rgb", "calibration_sha256"}, "alignment sample")
    if (item["sample_id"] != identity or type(item["source_list_index"]) is not int
            or item["source_list_index"] != index or type(item["frame_idx"]) is not int or item["frame_idx"] < 0
            or type(item["timestamp_us"]) is not int or not previous_timestamp < item["timestamp_us"] < 2**63):
        raise PreparationError("Alignment sample differs from its frozen frame order/identity")
    require_hex(item["frame_token"], 16, "frame token")
    require_hex(item["scene_token"], 16, "scene token")
    require_hex(item["calibration_sha256"], 64, "calibration SHA256")

    jpeg = item["jpeg"]
    require_keys(jpeg, {"member", "size_bytes", "sha256", "width", "height"}, "source JPEG")
    validate_recording_jpeg_member(jpeg["member"], archive_prefix, log_name)
    require_hex(jpeg["sha256"], 64, "JPEG SHA256")
    if (type(jpeg["size_bytes"]) is not int or not 0 < jpeg["size_bytes"] <= MAX_MEMBER_BYTES
            or type(jpeg["width"]) is not int or type(jpeg["height"]) is not int
            or min(jpeg["width"], jpeg["height"]) <= 0 or jpeg["width"] * jpeg["height"] > MAX_JPEG_PIXELS):
        raise PreparationError("Source JPEG size or native dimensions differ")

    _validate_file_record(item["rgb"], rgb_path(item["sample_id"]), _MAX_PNG_BYTES, "RGB", pixels=True)


def _validate_file_record(record, path, maximum, name, *, pixels=False):
    keys = {"path", "size_bytes", "sha256"} | ({"pixels_sha256"} if pixels else set())
    require_keys(record, keys, name)
    if record["path"] != path or type(record["size_bytes"]) is not int or not 0 < record["size_bytes"] <= maximum:
        raise PreparationError(f"Invalid alignment {name} file path or size")
    require_hex(record["sha256"], 64, name + " SHA256")
    if pixels:
        require_hex(record["pixels_sha256"], 64, name + " pixel SHA256")


def read_alignment_targets(root, manifest, split):
    if split not in SPLITS:
        raise ValueError("Alignment split must be train or holdout")
    partition = manifest["splits"][split]
    targets = decode_json(_read_verified(root, partition["targets"], _MAX_TARGET_BYTES))
    require_keys(targets, {"schema", "split", "rows"}, "alignment targets")
    rows = targets["rows"]
    if (targets["schema"] != TARGET_SCHEMA or targets["split"] != split or not isinstance(rows, list)
            or len(rows) != len(partition["samples"])):
        raise PreparationError("Alignment targets must match the requested split exactly")
    for row, item in zip(rows, partition["samples"]):
        _validate_target(row, item["sample_id"])
    if sum(row["presence"] for row in rows) != partition["positive_count"]:
        raise PreparationError("Alignment target presence count differs from manifest")
    return rows


def _validate_target(row, identity):
    require_keys(row, {"sample_id", "presence", "boxes_3d", "boxes_2d", "annotation"}, "alignment LOC target")
    if row["sample_id"] != identity or type(row["presence"]) is not bool:
        raise PreparationError("Alignment target identity or validity differs")
    box, aabb, annotation = row["boxes_3d"], row["boxes_2d"], row["annotation"]
    if not row["presence"]:
        if any(value is not None for value in (box, aabb, annotation)):
            raise PreparationError("Negative LOC target must not contain boxes or annotation identity")
        return
    if (not _numeric_vector(box, 7, 1e6) or math.hypot(*box[:2]) > MAX_DISTANCE_M
            or min(box[3:6]) <= 0 or not -math.pi <= box[6] < math.pi):
        raise PreparationError("Invalid positive LOC ego box")
    if (not _numeric_vector(aabb, 4, 1) or min(aabb) < 0
            or min(aabb[2] - aabb[0], aabb[3] - aabb[1]) * IMAGE_SIZE + 1e-9 < MIN_EXTENT_PX):
        raise PreparationError("Invalid positive LOC normalized image AABB")
    require_keys(annotation, {"row_index", "instance_token", "track_token"}, "source annotation")
    if type(annotation["row_index"]) is not int or annotation["row_index"] < 0:
        raise PreparationError("Invalid source annotation row index")
    require_hex(annotation["instance_token"], 16, "annotation instance token")
    require_hex(annotation["track_token"], 16, "annotation track token")


def _numeric_vector(values, length, bound):
    return (isinstance(values, list) and len(values) == length
            and all(type(value) in (int, float) and abs(value) <= bound for value in values))


def _read_verified(root, record, maximum):
    data = read_bounded(destination(root, record["path"]), maximum)
    if len(data) != record["size_bytes"] or hashlib.sha256(data).hexdigest() != record["sha256"]:
        raise PreparationError("Alignment file differs from its frozen size or SHA256")
    return data


def read_alignment_rgb(root, record):
    import numpy as np
    from PIL import Image

    data = _read_verified(root, record, _MAX_PNG_BYTES)
    with Image.open(io.BytesIO(data)) as image:
        if image.format != "PNG" or image.mode != "RGB" or image.size != (IMAGE_SIZE, IMAGE_SIZE):
            raise PreparationError("Alignment image must be the prepared 224×224 RGB PNG")
        rgb = np.array(image, copy=True)
    if hashlib.sha256(rgb.tobytes()).hexdigest() != record["pixels_sha256"]:
        raise PreparationError("Decoded alignment pixels differ from their frozen SHA256")
    return rgb


def verify_alignment_sample(root):
    manifest = read_alignment_manifest(root)
    for split in SPLITS:
        read_alignment_targets(root, manifest, split)
        for item in manifest["splits"][split]["samples"]:
            read_alignment_rgb(root, item["rgb"])
    return manifest
