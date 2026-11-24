"""Local preparation of disjoint OpenScene recordings for the motion pilot."""

from __future__ import annotations

from hashlib import sha256
from math import hypot, pi, tau
from pathlib import Path, PurePosixPath
import pickle
import re
import sys
import tarfile
from tempfile import TemporaryDirectory

import numpy as np

from hybrid_drive.assets.cameras import MAX_MEMBERS
from hybrid_drive.assets.cohort import _verified_archive
from hybrid_drive.assets.files import PreparationError
from hybrid_drive.experiments.bounded_io import canonical_json, decode_json, read_bounded, require_hex
from hybrid_drive.experiments.io import write_json
from hybrid_drive.geometry.rigid import invert_pose, pose_from_quaternion


CAMERAS = ("CAM_F0", "CAM_L0", "CAM_R0", "CAM_L2", "CAM_R2", "CAM_B0")
SOURCES_SHA256 = "a42c1f1756c5e65999ceb4024794a9be419a922743ff42a752c43486314d457b"
_INPUT_SCHEMA = "hybrid-drive-box-motion-inputs-v1"
_LOG_PATTERN = re.compile(r"(\d{4}\.\d{2}\.\d{2})\.\d{2}\.\d{2}\.\d{2}_(veh-\d+)_\d+_\d+")
_METADATA_LIMIT = 48 * 1024**2
_IMAGE_LIMIT = 4 * 1024**2
_ANCHORS_PER_LOG = 20
_STEP_US = (250_000, 1_000_000)
_MIN_OBJECT_MOVEMENT_M = .50


def prepare_motion_holdout(metadata_archive, camera_archive, source_path, output):
    """Extract observation JPEGs and a separate target file from pinned archives."""
    if sys.platform != "linux":
        raise PreparationError("OpenScene pickle preparation requires the isolated Linux image")
    source = decode_json(read_bounded(Path(source_path), 16 * 1024))
    if sha256(canonical_json(source)).hexdigest() != SOURCES_SHA256:
        raise PreparationError("OpenScene sources must match the checked-in digest")
    _check_recording_groups(source)
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise PreparationError("Output must not already exist")
    output.parent.mkdir(parents=True, exist_ok=True)
    meta_by_member = {row["member"]: row for row in source["logs"]}
    observations, anchors, labels = {}, [], {}
    images = {}
    following_tokens = set()
    curation = {}
    for member, data in _read_selected(metadata_archive, source["metadata_archive"], set(meta_by_member), _METADATA_LIMIT):
        row = meta_by_member[member]
        if len(data) != row["size_bytes"] or sha256(data).hexdigest() != row["sha256"]:
            raise PreparationError("Metadata member differs from its pinned SHA256 or size")
        frames = pickle.loads(data)
        if (not isinstance(frames, list) or len(frames) != row["frame_count"]
                or any(f.get("log_name") != row["log_name"] for f in frames)):
            raise PreparationError("Metadata recording does not match the pinned cohort")
        frames = _chronological(frames)
        select, stats = _select_anchors(frames, _ANCHORS_PER_LOG)
        if [i for i,_ in select] != row["anchor_indices"]:
            raise PreparationError("Motion anchor selection differs from the predeclared cohort")
        curation[row["recording"]] = stats
        for i, target in select:
            now = frames[i]["token"]
            observed = frames[i-3:i+1]
            history = [f["token"] for f in observed[:-1]]
            if now in labels:
                raise PreparationError("Duplicated current frame")
            labels[now] = target
            following_tokens.add(frames[i+1]["token"])
            anchors.append({"recording": row["recording"], "sample_token": now, "history_tokens": history})
            for frame in observed:
                token = frame["token"]
                if token not in observations:
                    obs = _observation(frame, row["recording"], row["log_name"],
                                       source["camera_archive"]["archive_prefix"], images)
                    observations[token] = obs
    if following_tokens & observations.keys():
        raise PreparationError("Following-frame input would leak into the observation cohort")
    # Tar ordering does not determine the published anchor order.
    anchors.sort(key=lambda anchor: (anchor["recording"], observations[anchor["sample_token"]]["ref_time_us"]))
    if len(anchors) != len(source["logs"]) * _ANCHORS_PER_LOG:
        raise PreparationError("Wrong total anchor count")

    with TemporaryDirectory(dir=output.parent, prefix=".box-motion-") as temp:
        staging = Path(temp) / "holdout"
        root = staging / "inputs"
        root.mkdir(parents=True)
        seen_bytes = {}
        for member, payload in _read_selected(camera_archive, source["camera_archive"], set(images), _IMAGE_LIMIT):
            rel, view, recording = images[member]
            if not payload.startswith(b"\xff\xd8\xff"):
                raise PreparationError("Pinned image is not a JPEG")
            digest = sha256(payload).hexdigest()
            if seen_bytes.setdefault(digest, recording) != recording:
                raise PreparationError("JPEG bytes are shared by supposedly disjoint recordings")
            view["sha256"] = digest
            dest = root / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(payload)
        write_json(root / "index.json", {
            "schema": _INPUT_SCHEMA,
            "source": {
                "metadata_sha256": source["metadata_archive"]["sha256"],
                "camera_sha256": source["camera_archive"]["sha256"],
                "camera_shard": source["camera_shard"],
                "publisher_mapping_sha256": source["publisher_mapping_sha256"],
                "source_sha256": SOURCES_SHA256,
                "capture_time": "lidar_association_not_camera_exposure",
                "selection": source["selection"],
            },
            "frames": observations,
            "anchors": anchors,
        })
        truth = staging / "targets"
        truth.mkdir()
        write_json(truth / "targets.json", labels)
        staging.rename(output)
    return {"anchors": len(anchors), "observed_frames": len(observations),
            "camera_jpegs": len(images), "curation": curation}


def _check_recording_groups(source):
    def group(log):
        m = _LOG_PATTERN.fullmatch(log)
        if m is None:
            raise PreparationError("Invalid original OpenScene log identity")
        return m.groups()

    exclusion = {
        group(f"{recording}_00000_00000") for recording in source["previous_recordings"]
    }
    marked = set()
    for row in source["logs"]:
        log = row["log_name"]
        recording = log.rsplit("_", 2)[0]
        d = group(log)
        if (d in exclusion or d in marked or row["recording"] != recording
                or row["member"] != f'openscene-v1.1/meta_datas/mini/{log}.pkl'):
            raise PreparationError("Holdout recording intersects old or selected day/vehicle groups")
        marked.add(d)


def _read_selected(path, spec, wanted, max_bytes):
    seen = set()
    with _verified_archive(path, spec) as archive, tarfile.open(fileobj=archive, mode="r|gz") as stream:
        for i, member in enumerate(stream):
            if i >= MAX_MEMBERS:
                raise PreparationError("Archive member count exceeds budget")
            name = member.name
            if name not in wanted:
                continue
            if name in seen or not member.isfile() or not 0 < member.size <= max_bytes:
                raise PreparationError("Unexpected pinned archive member type, size or duplicate")
            reader = stream.extractfile(member)
            if reader is None:
                raise PreparationError("Pinned archive member could not be opened")
            with reader:
                payload = reader.read(member.size + 1)
            if len(payload) != member.size:
                raise PreparationError("Pinned archive member is truncated")
            seen.add(name)
            yield name, payload
    if seen != wanted:
        raise PreparationError(f"Missing {len(wanted-seen)} required members in pinned archive")


def _chronological(frames):
    if len({f["log_token"] for f in frames}) != 1:
        raise PreparationError("Metadata contains multiple physical recordings")
    stamps, tokens = [f["timestamp"] for f in frames], [f["token"] for f in frames]
    if len(set(stamps)) != len(frames) or len(set(tokens)) != len(frames):
        raise PreparationError("Metadata has ambiguous duplicate LiDAR times or frame tokens")
    return sorted(frames, key=lambda f: f["timestamp"])


def _select_anchors(frames, count):
    # First produce a non-overlapping eligible window list; then spread the fixed
    # count uniformly over eligible ranks. No learned predictions are loaded.
    eligible, spaced = 0, []
    for i in range(3, len(frames)-1):
        w = frames[i-3:i+2]
        if not _valid_window(w):
            continue
        target, n = _target(w[3], w[4])
        if n < 2:
            continue
        eligible += 1
        if not spaced or i-spaced[-1][0] >= 5:
            spaced.append((i, target))
    if len(spaced) < count:
        raise PreparationError(f"Need {count} non-overlapping moving windows; got {len(spaced)}")
    positions = [((2*j+1)*len(spaced))//(2*count) for j in range(count)]
    selected = [spaced[j] for j in positions]
    moving = sum(
        hypot(*(np.asarray(b["future_center"])-b["center"])[:2]) >= _MIN_OBJECT_MOVEMENT_M
        for _, target in selected
        for b in target["boxes"]
        if b["future_center"] is not None
    )
    return selected, {"eligible_windows": eligible, "nonoverlapping_eligible_windows": len(spaced),
                      "selected_anchors": count, "selected_moving_targets": int(moving)}


def _valid_window(window):
    for a, b in zip(window, window[1:]):
        # frame_idx resets at publisher scene boundaries; the physical
        # recording and its association clock define temporal continuity.
        if (a["log_token"] != b["log_token"]
                or not _STEP_US[0] <= b["timestamp"]-a["timestamp"] <= _STEP_US[1]):
            return False
    return all(all(c in f.get("cams", {}) and f["cams"][c].get("data_path")
                   for c in CAMERAS) for f in window[:4])


def _vehicle_boxes(frame):
    annotation = frame["anns"]
    found = {}
    for name, token, raw in zip(annotation.get("gt_names", ()), annotation.get("track_tokens", ()),
                                annotation.get("gt_boxes", ()), strict=True):
        if name != "vehicle":
            continue
        require_hex(token, 16, "track token")
        if token in found:
            raise PreparationError("Duplicate vehicle track in a frame")
        box = np.asarray(raw, dtype=np.float64)
        if box.shape != (7,) or not np.isfinite(box).all() or np.any(box[3:6] <= 0):
            raise PreparationError("Malformed source vehicle box")
        found[token] = box
    return found


def _pose(frame):
    return pose_from_quaternion(frame["ego2global_translation"], frame["ego2global_rotation"])


def _target(current, following):
    follow_to_now = invert_pose(_pose(current)) @ _pose(following)
    future = _vehicle_boxes(following)
    scope_moving, boxes = 0, []
    for token, box in sorted(_vehicle_boxes(current).items()):
        if not (-16 <= box[0] <= 40 and abs(box[1]) <= 24):
            continue
        after = None
        if token in future:
            after = follow_to_now[:3, :3] @ future[token][:3] + follow_to_now[:3, 3]
            scope_moving += hypot(*(after-box[:3])[:2]) >= _MIN_OBJECT_MOVEMENT_M
        # Stable track linkage only in target files, never in observation inputs.
        ident = sha256(f'{current["log_name"]}\0{token}'.encode()).hexdigest()
        boxes.append({
            "instance_id": ident, "center": box[:3].tolist(), "size_wlh": box[[4, 3, 5]].tolist(),
            "yaw": (float(box[6])+pi)%tau-pi,
            "future_center": None if after is None else after.tolist(),
        })
    return {"next_dt_s": (following["timestamp"]-current["timestamp"])/1e6, "boxes": boxes}, int(scope_moving)


def _observation(frame, recording, log, archive_prefix, images):
    token = frame["token"]
    require_hex(token, 16, "frame token")
    views = []
    for name in CAMERAS:
        camera = frame["cams"][name]
        raw = camera["data_path"]
        rel = PurePosixPath(raw)
        if rel.parts[:2] != (log, name) or len(rel.parts) != 3 or rel.suffix != ".jpg" or ".." in rel.parts:
            raise PreparationError("Camera member is not the exact recorded camera path")
        require_hex(rel.stem, 16, "camera JPEG")
        member = f"{archive_prefix}/{rel}"
        destination = f"rgb/{token}/{name}.jpg"
        view = {
            "camera": name, "path": destination,
            "intrinsic": np.asarray(camera["cam_intrinsic"]).tolist(),
            "camera_to_ego_rotation": np.asarray(camera["sensor2lidar_rotation"]).tolist(),
            "camera_to_ego_translation": np.asarray(camera["sensor2lidar_translation"]).tolist(),
            "distortion": np.asarray(camera["distortion"]).tolist(),
        }
        if member in images:
            raise PreparationError("Unexpected JPEG member used by multiple observations")
        images[member] = destination, view, recording
        views.append(view)
    return {"recording": recording, "ref_time_us": int(frame["timestamp"]),
            "ego_to_world": _pose(frame).tolist(), "views": views}
