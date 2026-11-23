"""One fixed six-view smoke snapshot from the OpenScene Mini nuPlan rig."""
from __future__ import annotations

import gzip
import hashlib
import io
from pathlib import Path
import pickle
import sys
import tarfile
import zlib

import numpy as np

from hybrid_drive.adapters.real_camera import _rectify_native_openscene_rgb
from hybrid_drive.assets.cameras import BoundedTarReader, MAX_MEMBERS
from hybrid_drive.assets.cohort import _read_padding, _verified_archive
from hybrid_drive.assets.files import PreparationError, safe_relative
from hybrid_drive.contracts.motion import EgoMotion
from hybrid_drive.contracts.sensors import SURROUND_ORDER, CameraSnapshot, SurroundView
from hybrid_drive.experiments.bounded_io import canonical_json, decode_json, read_bounded
from hybrid_drive.geometry.rigid import pose_from_quaternion


# Allow only the pinned source description; caller-provided archive hashes cannot
# authorize a different metadata pickle. Re-pin only after verifying publisher provenance.
_SOURCE_SHA256 = "129763903e7134693c2ab8dff72fd4a17ee9001c7beb515e1dec0dde1cceffd4"


def load_openscene_surround(metadata_archive, camera_archive, source_manifest) -> CameraSnapshot:
    """Load the frozen snapshot in isolated Linux with both input archives mounted read-only.

    OpenScene associates images to the LiDAR timestamp without recording individual
    exposure timestamps or enabling motion compensation. The selected six of
    eight nuPlan views are not nuScenes and do not establish uninterrupted coverage.
    """
    if sys.platform != "linux":
        raise PreparationError("Read pinned metadata only in an isolated Linux container without network")
    source = _load_source(source_manifest)
    frame = _load_frame(metadata_archive, source)
    images = _read_verified_members(camera_archive, source["camera_archive"], source["cameras"])
    return _snapshot(source, frame, images)


def _load_source(path):
    source = decode_json(read_bounded(Path(path), 8 * 1024))
    if hashlib.sha256(canonical_json(source)).hexdigest() != _SOURCE_SHA256:
        raise PreparationError("OpenScene surround sources must match the pinned manifest")
    return source


def _load_frame(archive_path, source):
    log = source["log"]
    selected = source["frame"]
    payloads = _read_verified_members(archive_path, source["metadata_archive"], (log,))
    frames = pickle.loads(payloads[log["member"]])
    if not isinstance(frames, list) or len(frames) != log["frame_count"]:
        raise PreparationError("OpenScene surround source log does not match the pinned frame count")
    frame = frames[selected["index"]]
    if (not isinstance(frame, dict) or frame.get("log_name") != log["log_name"]
            or frame.get("frame_idx") != selected["index"] or frame.get("token") != selected["token"]
            or frame.get("timestamp") != selected["timestamp_us"] or frame.get("scene_token") != selected["scene_token"]):
        raise PreparationError("OpenScene surround frame does not match the pinned identity")
    return frame


def _read_verified_members(path, archive_spec, entries):
    expected = {entry["member"]: entry for entry in entries}
    found, seen = {}, set()
    try:
        with _verified_archive(path, archive_spec) as archive, gzip.GzipFile(fileobj=archive, mode="rb") as stream:
            reader = BoundedTarReader(stream)
            while True:
                header = reader.read_exact(512)
                if header == b"\0" * 512:
                    if reader.read_exact(512) != b"\0" * 512:
                        raise PreparationError("Invalid OpenScene tar end markers")
                    _read_padding(reader)
                    break
                member = tarfile.TarInfo.frombuf(header, "utf-8", "strict")
                safe_relative(member.name)
                if member.name in seen or len(seen) >= MAX_MEMBERS:
                    raise PreparationError("Duplicate OpenScene tar member or member count exceeds budget")
                seen.add(member.name)
                if not (member.isreg() or member.isdir()) or (member.isdir() and member.size):
                    raise PreparationError("OpenScene tar contains a special member")
                entry = expected.get(member.name)
                if entry is None:
                    reader.skip(member.size)
                else:
                    if not member.isreg() or member.size != entry["size_bytes"]:
                        raise PreparationError("OpenScene member differs from pinned size")
                    payload = reader.read_exact(member.size)
                    if hashlib.sha256(payload).hexdigest() != entry["sha256"]:
                        raise PreparationError("OpenScene member differs from pinned SHA256")
                    found[member.name] = payload
                reader.skip((-member.size) % 512)
    except (OSError, EOFError, tarfile.TarError, UnicodeError, zlib.error) as exc:
        raise PreparationError("Invalid or truncated OpenScene archive") from exc
    if set(found) != set(expected):
        raise PreparationError("OpenScene archive missing a pinned member")
    return found


def _snapshot(source, frame, images):
    from PIL import Image

    width, height = source["rectified_size"]
    archive_prefix = source["camera_archive"]["archive_prefix"]
    views = []
    for direction, entry in zip(SURROUND_ORDER, source["cameras"], strict=True):
        camera_name = entry["camera"]
        camera = frame["cams"][camera_name]
        if entry["member"] != f"{archive_prefix}/{camera['data_path']}":
            raise PreparationError("OpenScene camera path differs from its pinned frame association")
        with Image.open(io.BytesIO(images[entry["member"]])) as image:
            if image.format != "JPEG" or image.mode != "RGB" or list(image.size) != source["native_size"]:
                raise PreparationError("Expected a pinned native-size RGB JPEG")
            image.load()
            rgb = np.asarray(image)

        rectified, calibration = _rectify_native_openscene_rgb(
            camera_name, rgb, camera["cam_intrinsic"], camera["sensor2lidar_rotation"],
            camera["sensor2lidar_translation"], camera["distortion"], width, height,
        )
        views.append(SurroundView(direction, calibration, rectified, capture_us=None))

    ego_to_world = pose_from_quaternion(frame["ego2global_translation"], frame["ego2global_rotation"])
    motion = EgoMotion(*frame["ego_dynamic_state"])
    return CameraSnapshot(
        f"openscene-v1.1-mini:{frame['token']}", frame["timestamp"], ego_to_world, motion, views,
        rig_id=source["rig_id"], max_skew_us=None,
    )
