"""Portable current-camera inputs and one historical Fast output; no runtime clocks."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import io
import json
import math
from numbers import Integral
import os
from pathlib import Path
import re
import stat

import numpy as np
from PIL import Image

from hybrid_drive.adapters.real_camera import CalibratedView, OPENSCENE_NATIVE_PROFILE
from hybrid_drive.assets.files import check_components
from hybrid_drive.contracts.motion import EgoMotion
from hybrid_drive.contracts.trajectory import Trajectory
from hybrid_drive.geometry.camera import CameraCalibration
from hybrid_drive.slow.context import CAMERA_NAMES, MAX_HEIGHT, MAX_POINTS, MAX_WIDTH, CameraFrame
from .protocol import trajectory_hash


SCHEMA = "hybrid-drive-real-scene-bundle-v1"
COORDINATES = "fixed rear-axle ego: x forward, y left, z up; metres, heading radians; future dt..Ndt"
MAX_MANIFEST_BYTES = 64 * 1024
MAX_PNG_BYTES = 2 * 1024 * 1024
PROVENANCE_KEYS = {
    "source_revision", "navsim_revision", "nuplan_revision", "metadata_sha256",
    "source_assets_manifest_hash", "source_experiment_manifest_hash", "raw_candidate_hash",
    "checkpoint_sha256", "source_image_hashes", "profile",
}
MANIFEST_KEYS = {
    "schema", "scene_token", "frame_timestamp_us", "coordinates", "trajectory", "motion",
    "cameras", "provenance", "bundle_hash",
}
CAMERA_KEYS = {
    "name", "path", "sha256", "size_bytes", "width", "height", "state", "effective_distortion",
    "capture_timestamp_us", "intrinsic", "ego_to_camera", "near_m",
}
MOTION_KEYS = {"vx_mps", "vy_mps", "ax_mps2", "ay_mps2"}


@dataclass(frozen=True, eq=False, init=False)
class RealSceneBundle:
    scene_token: str
    frame_timestamp_us: int
    trajectory: Trajectory
    motion: EgoMotion
    views: tuple[CalibratedView, ...]
    _provenance_json: bytes = field(repr=False)
    _png_data: tuple[bytes, ...] = field(repr=False)

    def __init__(self, scene_token: str, frame_timestamp_us: int, trajectory: Trajectory,
                 motion: EgoMotion, views: tuple[CalibratedView, ...], provenance: dict):
        if not isinstance(scene_token, str) or not re.fullmatch(r"[0-9a-f]{16}", scene_token):
            raise ValueError("scene_token must be a 16-character lowercase frame token")
        _integer(frame_timestamp_us, "frame_timestamp_us", 0, 2**63 - 1)
        if not isinstance(trajectory, Trajectory) or trajectory.num_poses > MAX_POINTS:
            raise ValueError("Bundle requires a trajectory with at most 32 poses")
        if not isinstance(motion, EgoMotion):
            raise ValueError("motion must be EgoMotion")
        views = tuple(views)
        if len(views) != 3 or any(not isinstance(view, CalibratedView) for view in views):
            raise ValueError("Exactly three calibrated current views are required")
        if tuple(view.frame.name for view in views) != CAMERA_NAMES:
            raise ValueError("Camera order must be CAM_L0, CAM_F0, CAM_R0")
        provenance = _validate_provenance(provenance, trajectory)
        for name, value in (("scene_token", scene_token), ("frame_timestamp_us", int(frame_timestamp_us)),
                            ("trajectory", trajectory), ("motion", motion), ("views", views)):
            object.__setattr__(self, name, value)
        object.__setattr__(self, "_provenance_json", _canonical(provenance))
        object.__setattr__(self, "_png_data", tuple(_encode_png(view.frame) for view in views))
        if len(_canonical(_manifest(self))) > MAX_MANIFEST_BYTES:
            raise ValueError("Bundle manifest exceeds 64 KiB")

    @property
    def provenance(self) -> dict:
        return json.loads(self._provenance_json)

    @property
    def bundle_hash(self) -> str:
        return _bundle_hash(_metadata(self), self._png_data)


def _validate_provenance(provenance, trajectory):
    _keys(provenance, PROVENANCE_KEYS, "provenance")
    for key in ("source_revision", "navsim_revision", "nuplan_revision"):
        _hex(provenance[key], 40, key)
    for key in PROVENANCE_KEYS - {"source_revision", "navsim_revision", "nuplan_revision",
                                  "source_image_hashes", "profile"}:
        _hex(provenance[key], 64, key)
    _keys(provenance["source_image_hashes"], set(CAMERA_NAMES), "source_image_hashes")
    for digest in provenance["source_image_hashes"].values():
        _hex(digest, 64, "source image hash")
    if provenance["profile"] != OPENSCENE_NATIVE_PROFILE:
        raise ValueError("Unsupported source camera profile")
    if provenance["raw_candidate_hash"] != trajectory_hash(trajectory):
        raise ValueError("Raw Fast candidate hash differs from bundle trajectory")
    return json.loads(_canonical(provenance))


def _encode_png(frame):
    output = io.BytesIO()
    Image.fromarray(frame.rgb).save(output, format="PNG")
    data = output.getvalue()
    if len(data) > MAX_PNG_BYTES:
        raise ValueError("PNG exceeds byte budget")
    return data


def _metadata(bundle):
    cameras = []
    for view, data in zip(bundle.views, bundle._png_data):
        frame, calibration = view.frame, view.calibration
        cameras.append({
            "name": frame.name, "path": frame.name + ".png", "sha256": _sha(data),
            "size_bytes": len(data), "width": frame.width, "height": frame.height,
            "state": "rectified", "effective_distortion": [0, 0, 0, 0, 0],
            "capture_timestamp_us": None,
            "intrinsic": calibration.intrinsic.tolist(),
            "ego_to_camera": calibration.ego_to_camera.tolist(), "near_m": calibration.near_m,
        })
    return {
        "schema": SCHEMA, "scene_token": bundle.scene_token,
        "frame_timestamp_us": bundle.frame_timestamp_us, "coordinates": COORDINATES,
        "trajectory": {"poses": bundle.trajectory.poses.tolist(), "interval_s": bundle.trajectory.interval_s},
        "motion": {key: getattr(bundle.motion, key) for key in sorted(MOTION_KEYS)},
        "cameras": cameras, "provenance": bundle.provenance,
    }


def _manifest(bundle):
    metadata = _metadata(bundle)
    return {**metadata, "bundle_hash": _bundle_hash(metadata, bundle._png_data)}


def _bundle_hash(metadata, images):
    digest = hashlib.sha256(_canonical(metadata))
    for image in images:
        digest.update(image)
    return digest.hexdigest()


def write_real_bundle(bundle: RealSceneBundle, path) -> None:
    if not isinstance(bundle, RealSceneBundle):
        raise ValueError("bundle must be RealSceneBundle")
    encoded = _canonical(_manifest(bundle)) + b"\n"
    if len(encoded) > MAX_MANIFEST_BYTES:
        raise ValueError("Bundle manifest exceeds 64 KiB")
    root = Path(os.path.abspath(path))
    check_components(root)
    root.mkdir(parents=True, exist_ok=False)
    for name, data in zip(CAMERA_NAMES, bundle._png_data):
        _write_new(root / (name + ".png"), data)
    # Publish the manifest last: an interrupted export is never a loadable bundle.
    _write_new(root / "bundle.json", encoded)


def _write_new(path, data):
    check_components(path)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "wb") as output:
        output.write(data)


def load_real_bundle(path) -> RealSceneBundle:
    root = Path(os.path.abspath(path))
    check_components(root)
    if not root.is_dir():
        raise ValueError("Bundle root must be a directory")
    expected_files = {"bundle.json", *(name + ".png" for name in CAMERA_NAMES)}
    if {entry.name for entry in root.iterdir()} != expected_files:
        raise ValueError("Bundle must contain only its manifest and three fixed PNG files")
    manifest = _decode_json(_read_bounded(root / "bundle.json", MAX_MANIFEST_BYTES))
    _keys(manifest, MANIFEST_KEYS, "manifest")
    if manifest["schema"] != SCHEMA or manifest["coordinates"] != COORDINATES:
        raise ValueError("Unsupported bundle schema or coordinate frame")
    _hex(manifest["bundle_hash"], 64, "bundle_hash")
    _keys(manifest["trajectory"], {"poses", "interval_s"}, "trajectory")
    _keys(manifest["motion"], MOTION_KEYS, "motion")
    _numeric(manifest["trajectory"], "trajectory")
    _numeric(manifest["motion"], "motion")
    poses = manifest["trajectory"]["poses"]
    if not isinstance(poses, list) or not 0 < len(poses) <= MAX_POINTS:
        raise ValueError("Bundle trajectory must have 1..32 poses")
    trajectory = Trajectory(poses, manifest["trajectory"]["interval_s"])
    motion = EgoMotion(**manifest["motion"])
    cameras = manifest["cameras"]
    if not isinstance(cameras, list) or len(cameras) != 3:
        raise ValueError("Bundle requires three cameras")
    views, images = [], []
    for name, camera in zip(CAMERA_NAMES, cameras):
        view, data = _load_view(root, name, camera)
        views.append(view)
        images.append(data)
    metadata = {key: value for key, value in manifest.items() if key != "bundle_hash"}
    if _bundle_hash(metadata, images) != manifest["bundle_hash"]:
        raise ValueError("Bundle content hash differs")
    bundle = RealSceneBundle(manifest["scene_token"], manifest["frame_timestamp_us"], trajectory,
                             motion, tuple(views), manifest["provenance"])
    # Keep exact verified PNG bytes rather than re-encoding with this Pillow version.
    object.__setattr__(bundle, "_png_data", tuple(images))
    if bundle.bundle_hash != manifest["bundle_hash"]:
        raise ValueError("Bundle metadata is not in canonical contract form")
    return bundle


def _load_view(root, name, camera):
    _keys(camera, CAMERA_KEYS, "camera")
    if camera["name"] != name or camera["path"] != name + ".png":
        raise ValueError("Camera identity/order/path differs from fixed allowlist")
    if (camera["state"] != "rectified" or camera["effective_distortion"] != [0, 0, 0, 0, 0]
            or camera["capture_timestamp_us"] is not None):
        raise ValueError("Bundle cameras must be rectified with unknown per-camera capture time")
    _numeric(camera["effective_distortion"], "effective_distortion")
    _integer(camera["width"], "width", 1, MAX_WIDTH)
    _integer(camera["height"], "height", 1, MAX_HEIGHT)
    _integer(camera["size_bytes"], "size_bytes", 1, MAX_PNG_BYTES)
    _hex(camera["sha256"], 64, "PNG hash")
    data = _read_bounded(root / camera["path"], MAX_PNG_BYTES)
    if len(data) != camera["size_bytes"] or _sha(data) != camera["sha256"]:
        raise ValueError("PNG size/hash differs")
    with Image.open(io.BytesIO(data)) as image:
        if (image.format != "PNG" or image.mode != "RGB" or getattr(image, "n_frames", 1) != 1
                or image.size != (camera["width"], camera["height"])):
            raise ValueError("Expected bounded RGB PNG with exact manifest dimensions")
        image.load()
        frame = CameraFrame(name, np.asarray(image))
    for key in ("intrinsic", "ego_to_camera", "near_m"):
        _numeric(camera[key], key)
    calibration = CameraCalibration(name, frame.width, frame.height, camera["intrinsic"],
                                    camera["ego_to_camera"], camera["near_m"])
    return CalibratedView(frame, calibration), data


def _read_bounded(path, maximum):
    check_components(path)
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as source:
        info = os.fstat(source.fileno())
        if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= maximum:
            raise ValueError("Bundle file exceeds byte budget or is not regular")
        data = source.read(maximum + 1)
    if not 0 < len(data) <= maximum:
        raise ValueError("Bundle file exceeds byte budget")
    return data


def _decode_json(data):
    try:
        return json.loads(data, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    except (UnicodeError, RecursionError) as exc:
        raise ValueError("Invalid bounded JSON") from exc


def _reject_constant(value):
    raise ValueError("Nonfinite JSON")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _keys(value, expected, name):
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(name + " has missing or unknown keys")


def _hex(value, length, name):
    if not isinstance(value, str) or not re.fullmatch("[0-9a-f]{" + str(length) + "}", value):
        raise ValueError(name + " must be a lowercase hex digest")


def _integer(value, name, minimum, maximum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or not minimum <= value <= maximum:
        raise ValueError(name + " is outside its integer bounds")


def _numeric(value, name):
    if isinstance(value, dict):
        for item in value.values():
            _numeric(item, name)
    elif isinstance(value, list):
        for item in value:
            _numeric(item, name)
    else:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(name + " must contain only finite real numbers")
        try:
            finite = math.isfinite(value)
        except OverflowError as exc:
            raise ValueError(name + " exceeds the finite numeric range") from exc
        if not finite:
            raise ValueError(name + " must contain only finite real numbers")


def _sha(data):
    return hashlib.sha256(data).hexdigest()
