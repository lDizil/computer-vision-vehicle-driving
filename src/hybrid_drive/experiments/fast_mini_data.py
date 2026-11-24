"""Freeze a recording-disjoint nuScenes Mini forecast pilot from the local release."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import stat
import tarfile
import tempfile

import numpy as np

from hybrid_drive.assets.files import check_components, safe_relative
from hybrid_drive.contracts.sensors import timestamp_us
from hybrid_drive.experiments.bounded_io import canonical_json, decode_json
from hybrid_drive.geometry.rigid import invert_pose, matrix, pose_from_quaternion


# Byte identity of the Motional v1.0-mini release already in the project cache.
_ARCHIVE_SIZE = 4_167_696_325
_ARCHIVE_SHA256 = "943037abbb3b26b3070dc76504a43eb440503b00baf9ac2f1538d9c03fc9298f"
_VERSION = "v1.0-mini"
_TEST_SCENES = frozenset(("scene-0103", "scene-0916"))  # nuScenes devkit mini_val.
_DEV_SCENE = "scene-0796"
_CAPS = {"train": 8, "dev": 7, "test": 8}
_TABLES = (
    "scene", "sample", "sample_data", "sensor", "calibrated_sensor", "ego_pose",
    "sample_annotation", "instance", "category",
)
_CAMERAS = (
    "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT", "CAM_BACK_RIGHT", "CAM_BACK",
)
_CHANNELS = (*_CAMERAS, "LIDAR_TOP")
_MAX_SKEW_US = 50_000
_MIN_STEP_US = 350_000
_MAX_STEP_US = 650_000
_MAX_TABLE_BYTES = 64 * 1024 * 1024
_MAX_JPEG_BYTES = 16 * 1024 * 1024
_MOTOR_VEHICLES = frozenset((
    "vehicle.car", "vehicle.truck", "vehicle.bus.bendy", "vehicle.bus.rigid",
    "vehicle.construction", "vehicle.trailer",
))


@dataclass(frozen=True)
class _Selection:
    roles: dict
    observations: dict
    image_roles: dict
    targets: dict
    largest_skew_us: int


def prepare_fast_mini(archive: Path, output: Path) -> dict:
    """Publish only selected camera JPEGs and metadata to inputs, labels separately."""
    archive = Path(archive).expanduser().absolute()
    output = Path(output).expanduser().absolute()
    check_components(archive)
    check_components(output)
    if output.exists():
        raise ValueError("Fast Mini output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)

    descriptor = os.open(archive, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as source:
        signature = _verify_archive(source)
        selection = _select(_read_tables(source))
        with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
            staged = Path(temporary) / "ready"
            inputs = staged / "inputs"
            _write_json(inputs / "index.json", {
                "schema": "hybrid-drive-fast-mini-inputs-v1",
                "source_sha256": _ARCHIVE_SHA256,
                "splits": selection.roles,
            })
            for name, rows in selection.observations.items():
                _write_json(inputs / _VERSION / f"{name}.json", rows)
            image_bytes = _extract_jpegs(source, inputs, selection.image_roles)
            for role, targets in selection.targets.items():
                _write_json(staged / "targets" / f"{role}.json", targets)
            if _archive_signature(source) != signature:
                raise ValueError("Fast Mini archive changed during preparation")
            check_components(output)
            if output.exists():
                raise ValueError("Fast Mini output already exists")
            staged.rename(output)

    return {
        "source_sha256": _ARCHIVE_SHA256,
        "anchors": {role: len(rows) for role, rows in selection.roles.items()},
        "recordings": {role: len({row["recording"] for row in rows}) for role, rows in selection.roles.items()},
        "observed_keyframes": len(selection.observations["sample"]),
        "jpegs": len(selection.image_roles),
        "jpeg_bytes": image_bytes,
        "largest_camera_skew_us": selection.largest_skew_us,
    }


def _verify_archive(source):
    signature = _archive_signature(source)
    if signature[0] != _ARCHIVE_SIZE:
        raise ValueError("Fast Mini archive size differs from pinned v1.0-mini")
    digest = hashlib.sha256()
    for data in iter(lambda: source.read(8 * 1024 * 1024), b""):
        digest.update(data)
    if digest.hexdigest() != _ARCHIVE_SHA256 or _archive_signature(source) != signature:
        raise ValueError("Fast Mini archive SHA256 differs from pinned v1.0-mini")
    return signature


def _archive_signature(source):
    info = os.fstat(source.fileno())
    if not stat.S_ISREG(info.st_mode):
        raise ValueError("Fast Mini archive must be a regular file")
    return info.st_size, info.st_mtime_ns, info.st_ctime_ns


def _read_tables(source):
    names = {f"{_VERSION}/{name}.json": name for name in _TABLES}
    return {
        names[path]: _token_index(decode_json(data), names[path])
        for path, data in _selected_members(source, names, _MAX_TABLE_BYTES)
    }


def _token_index(rows, table):
    if not isinstance(rows, list):
        raise ValueError(f"{table} must be a list")
    indexed = {}
    for row in rows:
        token = row.get("token") if isinstance(row, dict) else None
        if not isinstance(token, str) or not token or token in indexed:
            raise ValueError(f"{table} contains an invalid or duplicate token")
        indexed[token] = row
    return indexed


def _selected_members(source, names, limit):
    # A pinned source is used; no tar paths are passed to an extraction API.
    source.seek(0)
    remaining = set(names)
    with tarfile.open(fileobj=source, mode="r|gz", bufsize=1024 * 1024) as tar:
        for item in tar:
            if item.name not in names:
                continue
            if item.name not in remaining or not item.isfile() or not 0 < item.size <= limit:
                raise ValueError(f"Invalid or duplicate Fast Mini member: {item.name}")
            with tar.extractfile(item) as member:
                data = member.read(item.size + 1)
            if len(data) != item.size:
                raise ValueError(f"Truncated Fast Mini member: {item.name}")
            remaining.remove(item.name)
            yield item.name, data
            if not remaining:
                break
    if remaining:
        raise ValueError(f"Missing Fast Mini members: {sorted(remaining)[:3]}")


def _select(tables):
    scene_roles = _scene_roles(tables["scene"])
    samples = tables["sample"]
    images = _sensor_groups(tables)
    by_scene = defaultdict(list)
    for sample in samples.values():
        if sample["scene_token"] not in scene_roles:
            raise ValueError("Sample references an unknown scene")
        by_scene[sample["scene_token"]].append(sample)

    poses = tables["ego_pose"]
    ref_times = {token: _pose_time(rows["LIDAR_TOP"], poses) for token, rows in images.items() if "LIDAR_TOP" in rows}
    valid_skews = {token: _camera_skew(rows, poses) for token, rows in images.items()}
    selected = {role: [] for role in _CAPS}
    anchors = {}
    observe_roles = {}
    for scene_id, scene in sorted(tables["scene"].items(), key=lambda item: item[1]["name"]):
        role = scene_roles[scene_id]
        eligible = [
            row for row in by_scene[scene_id]
            if _eligible(row, samples, ref_times, valid_skews)
        ]
        eligible.sort(key=lambda row: ref_times[row["token"]])
        if not eligible:
            raise ValueError(f"No valid camera/reference intervals in {scene['name']}")
        for row in _midpoints(eligible, _CAPS[role]):
            token, previous = row["token"], row["prev"]
            selected[role].append({
                "scene": scene["name"], "recording": scene["log_token"],
                "sample_token": token, "previous_token": previous,
            })
            anchors[token] = row["next"]
            for observed in (token, previous):
                if observe_roles.setdefault(observed, role) != role:
                    raise ValueError("Observation appears in multiple roles")
    if any(not rows for rows in selected.values()):
        raise ValueError("All three Fast Mini roles must be represented")

    observations, image_roles = _observation_tables(tables, images, observe_roles)
    boxes = _vehicle_annotations(tables)
    targets = {
        role: {
            row["sample_token"]: _target(
                row["sample_token"], anchors[row["sample_token"]], boxes, images, poses, ref_times,
            )
            for row in rows
        }
        for role, rows in selected.items()
    }
    return _Selection(selected, observations, image_roles, targets, max(valid_skews[token] for token in observe_roles))


def _scene_roles(scenes):
    by_name = {}
    for token, scene in scenes.items():
        name, log = scene["name"], scene["log_token"]
        if not isinstance(name, str) or not name or name in by_name or not isinstance(log, str) or not log:
            raise ValueError("Invalid scene/log grouping")
        by_name[name] = token
    if not {*_TEST_SCENES, _DEV_SCENE}.issubset(by_name):
        raise ValueError("Missing pinned nuScenes Mini dev or official mini_val scenes")
    test_logs = {scenes[by_name[name]]["log_token"] for name in _TEST_SCENES}
    dev_log = scenes[by_name[_DEV_SCENE]]["log_token"]
    if len(test_logs) != len(_TEST_SCENES) or dev_log in test_logs or any(
        scene["log_token"] in test_logs and scene["name"] not in _TEST_SCENES for scene in scenes.values()
    ):
        raise ValueError("Official mini_val recording overlaps another scene/role")
    return {
        token: "test" if scene["log_token"] in test_logs else
        "dev" if scene["log_token"] == dev_log else "train"
        for token, scene in scenes.items()
    }


def _sensor_groups(tables):
    groups = defaultdict(dict)
    sensors, calibrations = tables["sensor"], tables["calibrated_sensor"]
    for row in tables["sample_data"].values():
        if row["is_key_frame"] is False:
            continue
        if row["is_key_frame"] is not True:
            raise ValueError("Invalid keyframe flag")
        sensor = sensors[calibrations[row["calibrated_sensor_token"]]["sensor_token"]]
        channel = sensor["channel"]
        if channel not in _CHANNELS:
            continue
        expected_modality = "lidar" if channel == "LIDAR_TOP" else "camera"
        if sensor["modality"] != expected_modality:
            raise ValueError(f"Wrong modality for {channel}")
        cameras = groups[row["sample_token"]]
        if channel in cameras:
            raise ValueError(f"Duplicate keyframe channel: {channel}")
        cameras[channel] = row
    return groups


def _pose_time(row, poses):
    recorded = timestamp_us(row["timestamp"], "sample_data timestamp")
    if timestamp_us(poses[row["ego_pose_token"]]["timestamp"], "ego_pose timestamp") != recorded:
        raise ValueError("Capture and ego-pose clocks disagree")
    return recorded


def _camera_skew(rows, poses):
    if set(rows) != set(_CHANNELS):
        return None
    times = [_pose_time(row, poses) for row in rows.values()]
    return max(times) - min(times)


def _eligible(sample, samples, ref_times, skews):
    token, previous, following = sample["token"], sample["prev"], sample["next"]
    if not previous or not following:
        return False
    before, after = samples[previous], samples[following]
    if before["scene_token"] != sample["scene_token"] or after["scene_token"] != sample["scene_token"]:
        raise ValueError("Linked keyframes cross a scene boundary")
    if before["next"] != token or after["prev"] != token:
        raise ValueError("Keyframe links are not reciprocal")
    if any(skews.get(key) is None or skews[key] > _MAX_SKEW_US for key in (previous, token)):
        return False
    if any(key not in ref_times for key in (previous, token, following)):
        return False
    return (
        _MIN_STEP_US <= ref_times[token] - ref_times[previous] <= _MAX_STEP_US
        and _MIN_STEP_US <= ref_times[following] - ref_times[token] <= _MAX_STEP_US
    )


def _midpoints(rows, cap):
    count = min(len(rows), cap)
    return [rows[(2 * index + 1) * len(rows) // (2 * count)] for index in range(count)]


def _observation_tables(tables, images, roles):
    results = {"sample": [], "sample_data": []}
    selected_calibrations, selected_poses, selected_sensors = set(), set(), set()
    image_roles = {}
    for token, role in sorted(roles.items()):
        sample = tables["sample"][token]
        results["sample"].append({"token": token, "timestamp": sample["timestamp"]})
        for channel in _CHANNELS:
            row = images[token][channel]
            record = {field: row[field] for field in (
                "token", "sample_token", "is_key_frame", "timestamp", "ego_pose_token", "calibrated_sensor_token",
            )}
            selected_calibrations.add(row["calibrated_sensor_token"])
            selected_poses.add(row["ego_pose_token"])
            if channel != "LIDAR_TOP":
                path = safe_relative(row["filename"])
                if (len(path.parts) != 3 or path.parts[:2] != ("samples", channel)
                        or path.suffix != ".jpg" or row["fileformat"] != "jpg"):
                    raise ValueError(f"Invalid keyframe JPEG path for {channel}")
                name = path.as_posix()
                if name in image_roles:
                    raise ValueError("A source JPEG path describes multiple keyframes")
                image_roles[name] = role
                record.update({field: row[field] for field in ("filename", "fileformat", "width", "height")})
            results["sample_data"].append(record)

    results["calibrated_sensor"] = [tables["calibrated_sensor"][token] for token in sorted(selected_calibrations)]
    results["ego_pose"] = [tables["ego_pose"][token] for token in sorted(selected_poses)]
    selected_sensors.update(row["sensor_token"] for row in results["calibrated_sensor"])
    results["sensor"] = [tables["sensor"][token] for token in sorted(selected_sensors)]
    return results, image_roles


def _vehicle_annotations(tables):
    categories, instances = tables["category"], tables["instance"]
    by_sample = defaultdict(dict)
    for row in tables["sample_annotation"].values():
        instance_id = row["instance_token"]
        if categories[instances[instance_id]["category_token"]]["name"] not in _MOTOR_VEHICLES:
            continue
        per_frame = by_sample[row["sample_token"]]
        if instance_id in per_frame:
            raise ValueError("Duplicate instance annotation in one sample")
        per_frame[instance_id] = row
    return by_sample


def _target(token, following, annotations, images, poses, ref_times):
    pose = poses[images[token]["LIDAR_TOP"]["ego_pose_token"]]
    reference_from_world = invert_pose(pose_from_quaternion(pose["translation"], pose["rotation"]))
    boxes = []
    following_boxes = annotations.get(following, {})
    for instance_id, current in sorted(annotations.get(token, {}).items()):
        # Published visibility bins 2/3/4 mean at least 40% in the surround views.
        if current["visibility_token"] not in ("2", "3", "4"):
            continue
        reference_from_box = reference_from_world @ pose_from_quaternion(current["translation"], current["rotation"])
        size = matrix(current["size"], (3,), "annotation dimensions")
        if np.any(size <= 0):
            raise ValueError("Nonpositive annotation size")
        next_box = following_boxes.get(instance_id)
        next_center = None if next_box is None else (
            reference_from_world[:3, :3] @ matrix(next_box["translation"], (3,), "future translation")
            + reference_from_world[:3, 3]
        ).tolist()
        boxes.append({
            "instance_id": instance_id,
            "center": reference_from_box[:3, 3].tolist(),
            "size_wlh": size.tolist(),
            "yaw": float(np.arctan2(reference_from_box[1, 0], reference_from_box[0, 0])),
            "future_center": next_center,
        })
    return {"next_dt_s": (ref_times[following] - ref_times[token]) / 1_000_000, "boxes": boxes}


def _extract_jpegs(source, inputs, image_roles):
    digest_roles = {}
    count = 0
    for name, data in _selected_members(source, image_roles, _MAX_JPEG_BYTES):
        if not data.startswith(b"\xff\xd8"):
            raise ValueError("Selected keyframe is not JPEG")
        role = image_roles[name]
        digest = hashlib.sha256(data).hexdigest()
        if digest_roles.setdefault(digest, role) != role:
            raise ValueError("Identical JPEG payload crosses recording roles")
        target = inputs / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as output:
            output.write(data)
        count += len(data)
    return count


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(canonical_json(value) + b"\n")
