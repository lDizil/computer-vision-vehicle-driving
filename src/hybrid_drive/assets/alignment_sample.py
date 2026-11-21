"""Prepare the fixed two-recording OpenScene LOC slice in an isolated Linux container."""
from __future__ import annotations

import gzip
import hashlib
import io
from itertools import product
import math
import os
from pathlib import Path
import pickle
import sys
import tarfile
import tempfile
import zlib

import numpy as np

from .alignment_manifest import (
    ARCHIVE_VERIFICATION, CAMERA, CENTER_X, IMAGE_SIZE, MANIFEST_NAME, MANIFEST_SCHEMA,
    MAX_DISTANCE_M, MIN_DEPTH_M, MIN_EXTENT_PX, SPLITS, TARGET_SCHEMA,
    digest_json, midpoint_indices, recording_id, rgb_path, sample_id, targets_path,
    validate_alignment_sources, validate_jpeg_member, verify_alignment_sample,
)
from .cameras import BoundedTarReader, MAX_MEMBERS
from .cohort import _extract_cameras, _read_padding, _verified_archive
from .files import PreparationError, check_components, destination, safe_relative, write_manifest
from hybrid_drive.adapters.real_camera import OPENSCENE_NATIVE_PROFILE, prepare_openscene_camera
from hybrid_drive.experiments.bounded_io import require_hex


_MAX_LOG_BYTES = 32 * 1024 * 1024
_END_OF_TAR = b"\0" * 512
_UNIT_CORNERS = np.array(list(product((-0.5, 0.5), repeat=3)))


def prepare_alignment_sample(root, sources, metadata_archive, camera_archive):
    validate_alignment_sources(sources)
    root = Path(os.path.abspath(root))
    check_components(root)
    if (root / MANIFEST_NAME).exists():
        return verify_alignment_sample(root)
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise PreparationError("Alignment preparation requires a new empty root")

    if sys.platform != "linux":
        raise PreparationError("Prepare pinned metadata only in an isolated Linux container without network")
    with _verified_archive(metadata_archive, sources["metadata_archive"]) as archive:
        logs = _read_metadata_members(archive, sources["logs"])
    selected = {split: _select_frames(logs[split], sources, split) for split in SPLITS}

    root.parent.mkdir(parents=True, exist_ok=True)
    check_components(root.parent)
    with tempfile.TemporaryDirectory(prefix=".alignment-", dir=root.parent) as temporary:
        raw_root = Path(temporary) / "original"
        stage = Path(temporary) / "publish"
        case_sources = _camera_allowlist(selected)
        with _verified_archive(camera_archive, sources["camera_archive"]) as archive:
            originals = _extract_cameras(archive, raw_root, {"cases": case_sources})
        by_sample_id = {item["case_id"]: item for item in originals}

        manifest = {
            "schema": MANIFEST_SCHEMA,
            "sources": sources,
            "archive_verification": ARCHIVE_VERIFICATION,
            "splits": {split: _prepare_split(stage, raw_root, selected[split], by_sample_id, sources, split)
                       for split in SPLITS},
        }
        manifest["manifest_hash"] = digest_json(manifest)
        write_manifest(stage / MANIFEST_NAME, manifest)
        verify_alignment_sample(stage)
        check_components(root)
        os.rename(stage, root)
    return manifest


def _read_metadata_members(archive, specs):
    by_member = {spec["member"]: (split, spec) for split, spec in specs.items()}
    logs, seen = {}, set()
    try:
        with gzip.GzipFile(fileobj=archive, mode="rb") as stream:
            tar = BoundedTarReader(stream)
            while True:
                header = tar.read_exact(512)
                if header == _END_OF_TAR:
                    if tar.read_exact(512) != _END_OF_TAR:
                        raise PreparationError("Invalid metadata tar end markers")
                    _read_padding(tar)
                    break
                member = tarfile.TarInfo.frombuf(header, "utf-8", "strict")
                safe_relative(member.name)
                if member.name in seen or len(seen) >= MAX_MEMBERS:
                    raise PreparationError("Duplicate metadata member or member count exceeds budget")
                seen.add(member.name)
                if not (member.isreg() or member.isdir()) or (member.isdir() and member.size):
                    raise PreparationError("Metadata tar contains a special member")
                if member.name in by_member:
                    split, spec = by_member[member.name]
                    if (not member.isreg() or not 0 < member.size <= _MAX_LOG_BYTES
                            or member.size != spec["size_bytes"]):
                        raise PreparationError("Metadata member differs from pinned size")
                    raw = tar.read_exact(member.size)
                    if hashlib.sha256(raw).hexdigest() != spec["sha256"]:
                        raise PreparationError("Metadata member differs from pinned SHA256")
                    logs[split] = raw
                else:
                    tar.skip(member.size)
                tar.skip((-member.size) % 512)
    except (tarfile.TarError, EOFError, gzip.BadGzipFile, zlib.error, UnicodeError, OSError) as exc:
        raise PreparationError("Invalid or truncated metadata archive") from exc
    if set(logs) != set(specs):
        raise PreparationError("Metadata archive is missing a pinned recording")
    return logs


def _select_frames(verified_log, sources, split):
    spec = sources["logs"][split]
    frames = pickle.loads(verified_log)
    if not isinstance(frames, list):
        raise PreparationError(f"{split} source metadata is not a list of frames")
    if len(frames) != spec["frame_count"]:
        raise PreparationError(
            f"{split} source metadata has {len(frames)} frames, expected {spec['frame_count']}"
        )
    indices = set(midpoint_indices(spec["frame_count"], spec["sample_count"]))
    selected, tokens, image_members = [], set(), set()
    previous_timestamp = 0
    for index, frame in enumerate(frames):
        token, timestamp = frame["token"], frame["timestamp"]
        require_hex(frame["log_token"], 16, "log token")
        require_hex(token, 16, "frame token")
        if (frame["log_name"] != spec["log_name"] or frame["log_token"] != frames[0]["log_token"]
                or type(timestamp) is not int or not previous_timestamp < timestamp < 2**63 or token in tokens):
            raise PreparationError("OpenScene metadata recording/order/identity differs")
        tokens.add(token)
        previous_timestamp = timestamp
        if index not in indices:
            continue
        camera = frame["cams"][CAMERA]
        relative = safe_relative(camera["data_path"])
        member = f"{sources['camera_archive']['archive_prefix']}/{relative}"
        validate_jpeg_member(member, sources, split)
        if member in image_members:
            raise PreparationError("A frozen source JPEG is linked from more than one frame")
        image_members.add(member)
        selected.append((index, frame, member))
    return selected


def _camera_allowlist(selected):
    cases = []
    for split in SPLITS:
        for index, _, member in selected[split]:
            identity = sample_id(split, index)
            cases.append({"case_id": identity, "cameras": [
                {"camera": CAMERA, "member": member, "path": f"jpeg/{identity}.jpg"}
            ]})
    return cases


def _prepare_split(stage, raw_root, selected, originals, sources, split):
    from PIL import Image

    samples, targets = [], []
    for index, frame, _ in selected:
        identity = sample_id(split, index)
        original = originals[identity]
        with Image.open(destination(raw_root, original["path"])) as image:
            native = np.asarray(image)
        camera = frame["cams"][CAMERA]
        view = prepare_openscene_camera(
            CAMERA, native, camera["cam_intrinsic"], camera["sensor2lidar_rotation"],
            camera["sensor2lidar_translation"], camera["distortion"],
            profile=OPENSCENE_NATIVE_PROFILE, width=IMAGE_SIZE, height=IMAGE_SIZE,
        )
        pixels = view.frame.rgb
        encoded = io.BytesIO()
        Image.fromarray(pixels).save(encoded, format="PNG")
        path = destination(stage, rgb_path(identity))
        path.parent.mkdir(parents=True, exist_ok=True)
        data = encoded.getvalue()
        path.write_bytes(data)
        calibration = {
            "camera_to_ego_rotation": np.asarray(camera["sensor2lidar_rotation"]).tolist(),
            "camera_to_ego_translation": np.asarray(camera["sensor2lidar_translation"]).tolist(),
            "native_intrinsic": np.asarray(camera["cam_intrinsic"]).tolist(),
            "distortion": np.asarray(camera["distortion"]).tolist(),
            "native_size": [original["width"], original["height"]],
            "rectified_intrinsic": view.calibration.intrinsic.tolist(),
            "ego_to_camera": view.calibration.ego_to_camera.tolist(),
        }
        samples.append({
            "sample_id": identity, "source_list_index": index, "frame_token": frame["token"],
            "scene_token": frame["scene_token"], "frame_idx": frame["frame_idx"], "timestamp_us": frame["timestamp"],
            "jpeg": {key: original[key] for key in ("member", "size_bytes", "sha256", "width", "height")},
            "rgb": {"path": rgb_path(identity), "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                    "pixels_sha256": hashlib.sha256(pixels.tobytes()).hexdigest()},
            "calibration_sha256": digest_json(calibration),
        })
        targets.append(_loc_target(frame["anns"], view.calibration, identity))

    relative_targets = targets_path(split)
    path = destination(stage, relative_targets)
    write_manifest(path, {"schema": TARGET_SCHEMA, "split": split, "rows": targets})
    data = path.read_bytes()
    positive = sum(row["presence"] for row in targets)
    return {
        "recording_id": recording_id(sources["logs"][split]["log_name"]),
        "log_token": selected[0][1]["log_token"],
        "positive_count": positive, "negative_count": len(targets) - positive,
        "samples": samples,
        "targets": {"path": relative_targets, "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()},
    }


def _loc_target(annotations, calibration, identity):
    names = np.asarray(annotations["gt_names"])
    boxes = np.asarray(annotations["gt_boxes"])
    instance_tokens, track_tokens = annotations["instance_tokens"], annotations["track_tokens"]
    if (names.ndim != 1 or any(not isinstance(name, str) for name in names) or boxes.shape != (len(names), 7)
            or boxes.dtype.kind not in "fiu" or len(instance_tokens) != len(names) or len(track_tokens) != len(names)):
        raise PreparationError("OpenScene annotations do not have aligned categories, cuboids and source tokens")

    target = {"sample_id": identity, "presence": False, "boxes_3d": None, "boxes_2d": None, "annotation": None}
    best_distance = math.inf
    # Source order implements the declared tie-break without using track IDs.
    for index, name in enumerate(names):
        if name != "vehicle":
            continue
        box = boxes[index].astype(np.float64)
        if not np.isfinite(box).all() or np.any(box[3:6] <= 0):
            raise PreparationError("Malformed source vehicle cuboid cannot be labeled as absent")
        distance = math.hypot(*box[:2])
        if distance > MAX_DISTANCE_M or distance >= best_distance:
            continue
        aabb = _project_cuboid(box, calibration)
        if aabb is None:
            continue
        x, y, z, length, width, height, yaw = box.tolist()
        best_distance = distance
        target.update(
            presence=True,
            boxes_3d=[x, y, z, width, length, height, (yaw + math.pi) % math.tau - math.pi],
            boxes_2d=aabb,
            annotation={"row_index": int(index), "instance_token": instance_tokens[index], "track_token": track_tokens[index]},
        )
    return target


def _project_cuboid(box, calibration):
    x, y, z, length, width, height, yaw = box
    cosine, sine = math.cos(yaw), math.sin(yaw)
    yaw_rotation = np.array([[cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1]])
    corners = (_UNIT_CORNERS * (length, width, height)) @ yaw_rotation.T + (x, y, z)
    transform = calibration.ego_to_camera
    camera = np.vstack((corners, (x, y, z))) @ transform[:3, :3].T + transform[:3, 3]
    if np.any(camera[:8, 2] < MIN_DEPTH_M):
        return None
    intrinsic = calibration.intrinsic
    pixels = (camera[:, :2] / camera[:, 2, None]) @ intrinsic[:2, :2].T + intrinsic[:2, 2]
    if not np.isfinite(pixels).all():
        raise PreparationError("OpenScene cuboid projection is nonfinite")
    size = np.array((calibration.width, calibration.height))
    # Convert pixel centers to image-edge fractions, as used by the 2D box head.
    normalized = (pixels + 0.5) / size
    if not CENTER_X[0] <= normalized[8, 0] < CENTER_X[1]:
        return None
    lower, upper = normalized[:8].min(axis=0), normalized[:8].max(axis=0)
    if np.any(lower < 0) or np.any(upper > 1) or np.any((upper - lower) * size < MIN_EXTENT_PX):
        return None
    return np.concatenate((lower, upper)).tolist()
