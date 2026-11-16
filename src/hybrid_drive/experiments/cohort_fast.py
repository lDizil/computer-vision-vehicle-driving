"""Freeze one actual LTF output per fixed current frame before any evaluator access."""

from datetime import datetime, timezone
import hashlib
import io
import json
from numbers import Integral
from pathlib import Path
import pickle
import re

from hybrid_drive.assets.files import destination
from hybrid_drive.experiments.io import write_json
from hybrid_drive.experiments.protocol import content_hash, make_manifest, trajectory_hash
from hybrid_drive.experiments.bounded_io import decode_json as _decode_json, read_bounded as _read_bounded
from .navsim import configure_environment, load_cohort_inputs, repository_sources


SCHEMA = "hybrid-drive-real-cohort-fast-v1"
SCOPE = "single_recording_debug"
MAX_JSON_BYTES = 128 * 1024


def load_fast_cohort(path):
    path = Path(path)
    if path.is_dir():
        path = path / "cohort.json"
    manifest = _decode_json(_read_bounded(path, MAX_JSON_BYTES))
    keys = {"schema", "scope", "source_revision", "sources_hash", "model", "seed", "planned_cases", "cases", "cohort_hash"}
    if not isinstance(manifest, dict) or set(manifest) != keys:
        raise ValueError("Invalid Fast cohort manifest fields")
    if manifest["schema"] != SCHEMA or manifest["scope"] != SCOPE:
        raise ValueError("Unsupported Fast cohort schema or scope")
    _revision(manifest["source_revision"])
    _digest(manifest["sources_hash"], "sources_hash")
    _seed(manifest["seed"])
    payload = {key: value for key, value in manifest.items() if key != "cohort_hash"}
    if manifest["cohort_hash"] != content_hash(payload):
        raise ValueError("Fast cohort content hash differs")
    cases = manifest["cases"]
    if type(manifest["planned_cases"]) is not int or manifest["planned_cases"] != 6 or not isinstance(cases, list) or len(cases) != 6:
        raise ValueError("Fast cohort must retain all six planned cases")
    tokens, indices = set(), set()
    for index, case in enumerate(cases):
        required = {"case_id", "scene_token", "source_list_index", "bundle_path", "bundle_hash", "raw_candidate_hash", "status"}
        if not isinstance(case, dict) or not required <= set(case) or set(case) - required - {"error"}:
            raise ValueError("Invalid Fast cohort case fields")
        case_id = f"r{index:02d}"
        if case["case_id"] != case_id or case["bundle_path"] != case_id + "/bundle":
            raise ValueError("Fast cohort case identity or bundle path differs")
        token = case["scene_token"]
        if not isinstance(token, str) or not re.fullmatch(r"[0-9a-f]{16}", token) or token in tokens:
            raise ValueError("Fast cohort scene tokens must be unique frame identifiers")
        source_index = case["source_list_index"]
        if type(source_index) is not int or source_index < 3 or source_index in indices:
            raise ValueError("Invalid or duplicate Fast source frame index")
        tokens.add(token)
        indices.add(source_index)
        if case["status"] == "success":
            _digest(case["bundle_hash"], "bundle_hash")
            _digest(case["raw_candidate_hash"], "raw_candidate_hash")
            if "error" in case:
                raise ValueError("Successful Fast case cannot retain a failure")
        elif case["status"] == "failed":
            if case["bundle_hash"] is not None or case["raw_candidate_hash"] is not None or not isinstance(case.get("error"), str) or not case["error"]:
                raise ValueError("Failed Fast case must retain its cause without a fabricated bundle")
        else:
            raise ValueError("Unknown Fast case status")
    if any(case["status"] == "success" for case in cases) and not isinstance(manifest["model"], dict):
        raise ValueError("Successful Fast cohort requires model provenance")
    return manifest


def run_cohort_fast(data_root, asset_root, cohort_root, output_root, *, revision, seed=0, sources_path=None):
    from hybrid_drive.assets.cohort import load_cohort_sources, verify_cohort

    _revision(revision)
    _seed(seed)
    sources = load_cohort_sources(sources_path or repository_sources("real-cohort-sources.json"))
    run_dir = Path(output_root).resolve() / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run_dir.mkdir(parents=True)
    manifest = {
        "schema": SCHEMA, "scope": SCOPE, "source_revision": revision,
        "sources_hash": content_hash(sources), "model": None, "seed": seed, "planned_cases": 6,
        "cases": [{
            "case_id": case["case_id"], "scene_token": case["scene_token"],
            "source_list_index": case["source_list_index"], "bundle_path": case["case_id"] + "/bundle",
            "bundle_hash": None, "raw_candidate_hash": None, "status": "failed", "error": "not_attempted",
        } for case in sources["cases"]],
    }
    try:
        configure_environment(data_root, run_dir)
        models = _verified_models(asset_root)
        assets = verify_cohort(cohort_root, sources)
        frames = _current_frames(data_root, sources)
        fast = _initialize_fast(asset_root, seed)
        manifest["model"] = fast.provenance()
        tokens = [case["scene_token"] for case in sources["cases"]]
        loader, _ = load_cohort_inputs(sources, tokens, sensor_root=Path(cohort_root) / "sensor_blobs",
                                       sensor_config=fast.get_sensor_config())
        from hybrid_drive.adapters.navsim import from_navsim
        from scripts.run_navsim_sample import sample_policy

        for spec, case in zip(sources["cases"], manifest["cases"]):
            case_dir = run_dir / case["case_id"]
            case_dir.mkdir()
            try:
                raw = from_navsim(fast.compute_trajectory(loader.get_agent_input_from_token(spec["scene_token"])))
                camera_records = [record for record in assets["files"] if record["case_id"] == case["case_id"]]
                frame_manifest = make_manifest(
                    revision=revision, sources=sources,
                    assets={"sources": sources, "models": models, "cameras": {"files": camera_records}},
                    model=manifest["model"], scene_token=spec["scene_token"],
                    log_name=Path(sources["log"]["path"]).stem, seed=seed, policy=sample_policy(), candidate=raw,
                    split_audit={"role": "smoke", "scope": SCOPE, "sources_hash": manifest["sources_hash"]},
                )
                frame_manifest.update(
                    execution="current_inputs_fast_freeze_before_scoring", executed_ablations=[],
                    unavailable_ablations={"baseline": "not_scored_at_capture", "checks_only": "not_scored_at_capture",
                                           "combined": "not_scored_at_capture", "slow_only": "mandatory_checks_not_bypassed"},
                    quality_claim="Fixed single-recording debug cohort; no quality or driving-safety claim",
                )
                frame_manifest.pop("manifest_hash")
                frame_manifest["manifest_hash"] = content_hash(frame_manifest)
                bundle = _make_bundle(spec, frames[spec["case_id"]], raw, cohort_root, camera_records, frame_manifest,
                                      metadata_sha256=sources["log"]["sha256"])
                write_json(case_dir / "fast-manifest.json", frame_manifest)
                from hybrid_drive.experiments.real_bundle import write_real_bundle

                write_real_bundle(bundle, case_dir / "bundle")
                case.update(status="success", bundle_hash=bundle.bundle_hash, raw_candidate_hash=trajectory_hash(raw))
                case.pop("error")
            except Exception as exc:
                case["error"] = f"{type(exc).__name__}: {exc}"
            _write_cohort(run_dir, manifest)
    except Exception as exc:
        for case in manifest["cases"]:
            if case.get("error") == "not_attempted":
                case["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        _write_cohort(run_dir, manifest)
    return run_dir, manifest


def _verified_models(asset_root):
    from hybrid_drive.assets.files import verify_file

    sources = json.loads(repository_sources("ltf-sample-sources.json").read_text())
    selected = [("checkpoints/ltf_seed_0.ckpt", sources["checkpoint"]),
                ("encoder/config.json", sources["encoder"]["config"]),
                ("encoder/model.safetensors", sources["encoder"]["weights"])]
    records = []
    for path, spec in selected:
        record = verify_file(destination(asset_root, path), size_bytes=spec["size_bytes"], sha256=spec["sha256"])
        if record is None:
            raise ValueError("Prepared LTF model file is missing: " + path)
        records.append({"path": path, **record})
    return records


def _current_frames(data_root, sources):
    spec = sources["log"]
    data = _read_bounded(destination(data_root, spec["path"]), 16 * 1024 * 1024)
    if len(data) != spec["size_bytes"] or hashlib.sha256(data).hexdigest() != spec["sha256"]:
        raise ValueError("Cohort metadata failed size/SHA256 verification before pickle")
    frames = pickle.loads(data)
    if not isinstance(frames, list):
        raise ValueError("Cohort metadata must contain the pinned frame list")
    current, observed_tokens = {}, set()
    for case in sources["cases"]:
        index = case["source_list_index"]
        if index >= len(frames):
            raise ValueError("Cohort frame is outside verified metadata")
        frame = frames[index]
        expected = {"token": case["scene_token"], "scene_token": case["metadata_scene_token"],
                    "frame_idx": case["metadata_frame_idx"], "timestamp": case["frame_timestamp_us"]}
        if not isinstance(frame, dict) or any(frame.get(key) != value for key, value in expected.items()):
            raise ValueError("Cohort current frame differs from fixed selection")
        _validate_window(frames, index, case, sources["selection"], observed_tokens)
        current[case["case_id"]] = frame
    return current



def _validate_window(frames, index, case, selection, observed_tokens):
    history, future = selection["num_history_frames"], selection["num_future_frames"]
    window = frames[index - history + 1:index + future + 1]
    if len(window) != history + future:
        raise ValueError("Cohort window does not have the full fixed history/future extent")
    previous_timestamp, tokens = None, set()
    minimum_step, maximum_step = selection["timestamp_step_us"]
    for offset, frame in enumerate(window):
        if not isinstance(frame, dict):
            raise ValueError("Cohort temporal window must contain metadata frames")
        token, timestamp, frame_index = frame.get("token"), frame.get("timestamp"), frame.get("frame_idx")
        if (not isinstance(token, str) or not re.fullmatch(r"[0-9a-f]{16}", token)
                or token in tokens or token in observed_tokens
                or not isinstance(timestamp, Integral) or isinstance(timestamp, bool)
                or not isinstance(frame_index, Integral) or isinstance(frame_index, bool)
                or frame_index != case["metadata_frame_idx"] + offset - history + 1
                or frame.get("scene_token") != case["metadata_scene_token"]):
            raise ValueError("Cohort temporal window has inconsistent frame identity or chunk boundaries")
        if previous_timestamp is not None and not minimum_step <= timestamp - previous_timestamp <= maximum_step:
            raise ValueError("Cohort temporal window has a timestamp discontinuity")
        tokens.add(token)
        previous_timestamp = timestamp
    observed_tokens.update(tokens)

def _initialize_fast(asset_root, seed):
    import numpy as np
    import random
    import torch
    from hybrid_drive.fast.latent_transfuser import LatentTransfuserAgent

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    fast = LatentTransfuserAgent(Path(asset_root) / "checkpoints/ltf_seed_0.ckpt", asset_root=asset_root)
    fast.initialize()
    return fast


def _make_bundle(spec, current, raw, cohort_root, camera_records, manifest, *, metadata_sha256):
    import numpy as np
    from PIL import Image
    from hybrid_drive.adapters.real_camera import OPENSCENE_NATIVE_PROFILE, prepare_openscene_camera
    from hybrid_drive.contracts.motion import EgoMotion
    from hybrid_drive.experiments.real_bundle import RealSceneBundle

    views = []
    if [record["camera"] for record in camera_records] != ["CAM_L0", "CAM_F0", "CAM_R0"]:
        raise ValueError("Cohort frame must have exactly three ordered verified camera records")
    for entry, record in zip(spec["cameras"], camera_records):
        name = entry["camera"]
        metadata = current["cams"][name]
        if "sensor_blobs/" + metadata["data_path"] != entry["path"] or record["path"] != entry["path"]:
            raise ValueError("Current image identity differs from fixed cohort camera")
        data = _read_bounded(destination(cohort_root, entry["path"]), 16 * 1024 * 1024)
        if len(data) != record["size_bytes"] or hashlib.sha256(data).hexdigest() != record["sha256"]:
            raise ValueError("Cohort camera changed after verification")
        with Image.open(io.BytesIO(data)) as image:
            if image.format != "JPEG" or image.mode != "RGB" or image.size != (record["width"], record["height"]):
                raise ValueError("Current cohort input must be the verified original RGB JPEG")
            image.load()
            rgb = np.asarray(image)
        views.append(prepare_openscene_camera(
            name, rgb, metadata["cam_intrinsic"], metadata["sensor2lidar_rotation"],
            metadata["sensor2lidar_translation"], metadata["distortion"], profile=OPENSCENE_NATIVE_PROFILE,
        ))
    motion = current["ego_dynamic_state"]
    if np.asarray(motion).shape != (4,):
        raise ValueError("Current ego motion must contain vx, vy, ax, ay")
    provenance = {
        "source_revision": manifest["source_revision"], "navsim_revision": manifest["navsim_revision"],
        "nuplan_revision": manifest["nuplan_revision"],
        "metadata_sha256": metadata_sha256,
        "source_assets_manifest_hash": manifest["assets_manifest_hash"],
        "source_experiment_manifest_hash": manifest["manifest_hash"],
        "raw_candidate_hash": manifest["raw_candidate_hash"],
        "checkpoint_sha256": manifest["model"]["checkpoint_sha256"],
        "source_image_hashes": {record["camera"]: record["sha256"] for record in camera_records},
        "profile": OPENSCENE_NATIVE_PROFILE,
    }
    return RealSceneBundle(spec["scene_token"], spec["frame_timestamp_us"], raw, EgoMotion(*motion), tuple(views), provenance)


def _write_cohort(run_dir, manifest):
    manifest.pop("cohort_hash", None)
    manifest["cohort_hash"] = content_hash(manifest)
    write_json(run_dir / "cohort.json", manifest)


def _revision(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{40}", value):
        raise ValueError("revision must identify the exact source commit")


def _seed(value):
    if type(value) is not int or not 0 <= value < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")


def _digest(value, name):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(name + " must be a lowercase SHA256 digest")
