"""Build the frozen spatial LOC sample from verified local OpenScene archives."""
from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

from .alignment_manifest import ARCHIVE_VERIFICATION, CAMERA, MAX_DISTANCE_M, digest_json, targets_path
from .alignment_sample import _prepare_frame, _project_cuboid, _read_metadata_members, _select_recording_frames
from .cohort import _extract_cameras, _verified_archive
from .files import PreparationError, check_components, destination, write_manifest
from .spatial_manifest import (
    BAND_COUNT, MANIFEST_NAME, MANIFEST_SCHEMA, TARGET_SCHEMA, TEMPORAL_SOURCE_SCHEMA, box_horizontal_band,
    sample_id, spatial_frame_indices, validate_spatial_sources, validate_temporal_cadence, verify_spatial_sample,
)


def prepare_spatial_sample(root, sources, metadata_archive, camera_archives):
    splits = validate_spatial_sources(sources)
    if set(camera_archives) != set(sources["camera_archives"]):
        raise PreparationError("Exactly the pinned spatial camera shard paths must be supplied")
    root = Path(os.path.abspath(root))
    check_components(root)
    if (root / MANIFEST_NAME).exists():
        return verify_spatial_sample(root, expected_sources=sources)
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise PreparationError("Spatial preparation requires a new empty root")
    if sys.platform != "linux":
        raise PreparationError("Prepare pinned metadata only in an isolated Linux container without network")

    selected = _select_spatial_frames(sources, metadata_archive)
    root.parent.mkdir(parents=True, exist_ok=True)
    check_components(root.parent)
    with tempfile.TemporaryDirectory(prefix=".spatial-loc-", dir=root.parent) as temporary:
        raw_root = Path(temporary) / "original"
        stage = Path(temporary) / "publish"
        originals = _extract_spatial_cameras(raw_root, camera_archives, sources, selected)
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "sources": sources,
            "archive_verification": ARCHIVE_VERIFICATION,
            "splits": {split: _prepare_split(stage, raw_root, sources, selected, originals, split) for split in splits},
        }
        manifest["manifest_hash"] = digest_json(manifest)
        write_manifest(stage / MANIFEST_NAME, manifest)
        verify_spatial_sample(stage)
        check_components(root)
        os.rename(stage, root)
    return manifest


def _select_spatial_frames(sources, metadata_archive):
    specs = {log["recording_id"]: log for logs in sources["logs"].values() for log in logs}
    with _verified_archive(metadata_archive, sources["metadata_archive"]) as archive:
        metadata = _read_metadata_members(archive, specs)
    selected = {}
    for identity, log in specs.items():
        prefix = sources["camera_archives"][str(log["camera_shard"])]["archive_prefix"]
        frames = _select_recording_frames(
            metadata.pop(identity), log, prefix, identity, source_indices=spatial_frame_indices(sources, log),
        )
        if sources["schema"] == TEMPORAL_SOURCE_SCHEMA:
            validate_temporal_cadence([frame["timestamp"] for _, frame, _ in frames])
        selected[identity] = frames
    return selected


def _extract_spatial_cameras(raw_root, archives, sources, selected):
    by_shard = {shard: [] for shard in archives}
    for split, logs in sources["logs"].items():
        for log in logs:
            identity = log["recording_id"]
            for index, _, member in selected[identity]:
                frame_id = sample_id(split, identity, index)
                by_shard[str(log["camera_shard"])].append({
                    "case_id": frame_id,
                    "cameras": [{"camera": CAMERA, "member": member, "path": f"jpeg/{frame_id}.jpg"}],
                })
    originals = {}
    for shard, spec in sources["camera_archives"].items():
        with _verified_archive(archives[shard], spec) as archive:
            records = _extract_cameras(archive, raw_root, {"cases": by_shard[shard]})
        originals.update((record["case_id"], record) for record in records)
    return originals


def _prepare_split(stage, raw_root, sources, selected, originals, split):
    recordings, rows = [], []
    for log in sources["logs"][split]:
        identity = log["recording_id"]
        frames, samples = selected[identity], []
        for index, frame, _ in frames:
            frame_id = sample_id(split, identity, index)
            sample, calibration = _prepare_frame(stage, raw_root, index, frame, originals[frame_id], frame_id)
            samples.append(sample)
            rows.append(_spatial_target(frame["anns"], calibration, frame_id))
        recordings.append({"recording_id": identity, "log_token": frames[0][1]["log_token"], "samples": samples})

    relative = targets_path(split)
    path = destination(stage, relative)
    write_manifest(path, {"schema": TARGET_SCHEMA, "split": split, "rows": rows})
    data = path.read_bytes()
    return {
        "recordings": recordings,
        "targets": {"path": relative, "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()},
    }


def _spatial_target(annotations, calibration, identity):
    aabbs, _ = _select_spatial_targets(annotations, calibration)
    return {"sample_id": identity, "presence": [aabb is not None for aabb in aabbs], "boxes_2d": aabbs}


def _select_spatial_targets(annotations, calibration):
    names = np.asarray(annotations["gt_names"])
    boxes = np.asarray(annotations["gt_boxes"])
    if (names.ndim != 1 or any(not isinstance(name, str) for name in names) or boxes.shape != (len(names), 7)
            or boxes.dtype.kind not in "fiu"):
        raise PreparationError("OpenScene annotations do not have aligned categories and cuboids")

    nearest = [math.inf] * BAND_COUNT
    aabbs = [None] * BAND_COUNT
    indices = [None] * BAND_COUNT
    for index, (name, cuboid) in enumerate(zip(names, boxes)):
        if name != "vehicle":
            continue
        box = cuboid.astype(np.float64)
        if not np.isfinite(box).all() or np.any(box[3:6] <= 0):
            raise PreparationError("Malformed source vehicle cuboid cannot be labeled as absent")
        distance = math.hypot(*box[:2])
        if distance > MAX_DISTANCE_M:
            continue
        projected = _project_cuboid(box, calibration, center_x_range=None)
        if projected is None:
            continue
        aabb = np.asarray(projected, dtype=np.float32)
        band = box_horizontal_band(aabb)
        # The first annotation wins an equal ground-distance tie.
        if distance < nearest[band]:
            nearest[band], aabbs[band], indices[band] = distance, aabb.tolist(), index
    return aabbs, indices
