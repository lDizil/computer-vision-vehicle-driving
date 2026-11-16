"""Export verified original OpenScene images and an existing successful LTF baseline.

The original-v1 profile rectifies the source JPEGs once with the metadata's
Brown-Conrady coefficients, then resizes RGB and its matching pinhole K. Bundle
loading never invokes this native-image preparation again. No Fast/model runs,
scoring, maps, privileged Scene construction or future labels are required.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import pickle

import numpy as np
from PIL import Image

from hybrid_drive.adapters.real_camera import OPENSCENE_NATIVE_PROFILE, prepare_openscene_camera
from hybrid_drive.assets.files import destination
from hybrid_drive.assets.prepare import verify_prepared
from hybrid_drive.contracts.motion import EgoMotion
from hybrid_drive.contracts.trajectory import Trajectory
from hybrid_drive.experiments.protocol import asset_identity, content_hash, trajectory_hash, verify_manifest
from hybrid_drive.experiments.real_bundle import (
    RealSceneBundle, _decode_json, _read_bounded, write_real_bundle,
)


VERIFIED_METADATA_SHA256 = "d2a9b373da30670aaf327c2ab4ccc332e2b441be6cd5e3f72d77debd3adbcff3"
MAX_SOURCE_JSON_BYTES = 128 * 1024
MAX_METADATA_BYTES = 16 * 1024 * 1024


def export_real_bundle(data_root, asset_root, source_run, output, *, width=384, height=216):
    sources = _json_file(Path(__file__).with_name("navsim-sample-sources.json"))
    fast_sources = _json_file(Path(__file__).with_name("ltf-sample-sources.json"))
    assets = verify_prepared(asset_root, fast_sources)
    trajectory, manifest = _verified_baseline(source_run, sources, fast_sources, assets)
    current = _verified_current_frame(data_root, sources, fast_sources)
    motion = _current_motion(current)
    views = []
    for entry in fast_sources["cameras"]["files"]:
        name = entry["camera"]
        metadata = current["cams"][name]
        if "sensor_blobs/" + metadata["data_path"] != entry["path"]:
            raise ValueError("Current camera identity differs from verified source allowlist")
        record = next(record for record in assets["cameras"]["files"] if record["camera"] == name)
        data = _read_bounded(destination(asset_root, entry["path"]), 16 * 1024 * 1024)
        if len(data) != record["size_bytes"] or hashlib.sha256(data).hexdigest() != record["sha256"]:
            raise ValueError("Source JPEG changed after preparation verification")
        with Image.open(io.BytesIO(data)) as image:
            if image.format != "JPEG" or image.mode != "RGB" or image.size != (record["width"], record["height"]):
                raise ValueError("Current input must be the verified original RGB JPEG")
            image.load()
            rgb = np.asarray(image)
        views.append(prepare_openscene_camera(
            name, rgb, metadata["cam_intrinsic"], metadata["sensor2lidar_rotation"],
            metadata["sensor2lidar_translation"], metadata["distortion"],
            profile=OPENSCENE_NATIVE_PROFILE, width=width, height=height,
        ))
    provenance = {
        "source_revision": manifest["source_revision"],
        "navsim_revision": manifest["navsim_revision"], "nuplan_revision": manifest["nuplan_revision"],
        "metadata_sha256": sources["log"]["sha256"],
        "source_assets_manifest_hash": manifest["assets_manifest_hash"],
        "source_experiment_manifest_hash": manifest["manifest_hash"],
        "raw_candidate_hash": manifest["raw_candidate_hash"],
        "checkpoint_sha256": manifest["model"]["checkpoint_sha256"],
        "source_image_hashes": {record["camera"]: record["sha256"] for record in assets["cameras"]["files"]},
        "profile": OPENSCENE_NATIVE_PROFILE,
    }
    bundle = RealSceneBundle(fast_sources["scene_token"], current["timestamp"], trajectory,
                             motion, tuple(views), provenance)
    write_real_bundle(bundle, output)
    return bundle


def _verified_baseline(source_run, sources, fast_sources, assets):
    result = _json_file(destination(source_run, "result.json"))
    outer = _json_file(destination(source_run, "experiment-manifest.json"))
    manifest = outer.get("fast", outer)
    verify_manifest(manifest)
    if result.get("status") != "success":
        raise ValueError("Source run did not succeed")
    expected = {
        "scene_token": fast_sources["scene_token"], "log_name": Path(sources["log"]["path"]).stem,
        "navsim_revision": sources["navsim_revision"], "nuplan_revision": sources["nuplan_revision"],
        "dataset_source_hash": content_hash(sources),
        "assets_manifest_hash": content_hash(asset_identity(assets)),
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("Historical Fast manifest differs from verified current sources/assets")
    if (result.get("scene_token") != expected["scene_token"]
            or result.get("source_revision") != manifest["source_revision"]):
        raise ValueError("Source result and experiment manifest are not associated")
    model = manifest["model"]
    if (model.get("checkpoint_sha256") != fast_sources["checkpoint"]["sha256"]
            or model.get("navsim_revision") != sources["navsim_revision"]
            or result.get("model", result.get("fast_model")) != model):
        raise ValueError("Historical Fast model provenance differs from verified assets")
    baseline = [row for row in result["records"] if row.get("mode") == "baseline"]
    if (len(baseline) != 1 or baseline[0].get("failed") is not False
            or baseline[0].get("scene_token") != expected["scene_token"]):
        raise ValueError("Exactly one successful raw Fast baseline is required")
    native = baseline[0]["native_trajectory"]
    if isinstance(native, dict):
        if set(native) != {"poses", "interval_s"}:
            raise ValueError("Unsupported historical baseline sampling")
        trajectory = Trajectory(native["poses"], native["interval_s"])
    else:
        trajectory = Trajectory(native, manifest["sampling"]["interval_s"])
    if (trajectory_hash(trajectory) != manifest["raw_candidate_hash"]
            or manifest["sampling"] != {"num_poses": trajectory.num_poses, "interval_s": trajectory.interval_s}):
        raise ValueError("Historical raw Fast candidate differs from its manifest")
    return trajectory, manifest


def _verified_current_frame(data_root, sources, fast_sources):
    spec = sources["log"]
    if spec["sha256"] != VERIFIED_METADATA_SHA256 or not 0 < spec["size_bytes"] <= MAX_METADATA_BYTES:
        raise ValueError("Only the pinned verified OpenScene metadata may be deserialized")
    data = _read_bounded(destination(data_root, spec["path"]), MAX_METADATA_BYTES)
    if len(data) != spec["size_bytes"] or hashlib.sha256(data).hexdigest() != VERIFIED_METADATA_SHA256:
        raise ValueError("Metadata failed size/SHA-256 verification before pickle")
    frames = pickle.loads(data)
    index = fast_sources["frame_index"]
    if index != 3 or not isinstance(frames, list) or len(frames) <= index:
        raise ValueError("Expected the selected OpenScene frame 3")
    current = frames[index]
    if current["token"] != fast_sources["scene_token"] or current["frame_idx"] != index:
        raise ValueError("Metadata frame differs from the selected Fast input")
    return current


def _current_motion(current):
    values = current["ego_dynamic_state"]
    if not isinstance(values, (list, np.ndarray)) or np.asarray(values).shape != (4,):
        raise ValueError("Current ego motion must contain vx, vy, ax, ay")
    return EgoMotion(*values)


def _json_file(path):
    value = _decode_json(_read_bounded(path, MAX_SOURCE_JSON_BYTES))
    if not isinstance(value, dict):
        raise ValueError("Expected source JSON object")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--height", type=int, default=216)
    args = parser.parse_args()
    try:
        bundle = export_real_bundle(args.data_root, args.asset_root, args.source_run, args.output,
                                    width=args.width, height=args.height)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, "Real bundle export failed: " + str(exc) + "\n")
    print(json.dumps({"bundle_hash": bundle.bundle_hash, "output": str(args.output),
                      "historical_fast_source_revision": bundle.provenance["source_revision"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
