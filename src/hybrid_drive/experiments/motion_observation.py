"""Independent immutable observations bound to unchanged legacy real bundles."""

from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import pickle

from hybrid_drive.assets.files import check_components, destination
from hybrid_drive.contracts._validation import finite_real, tuple_snapshot
from hybrid_drive.contracts.motion_observation import (
    HEADING_RATE_METHOD, OBSERVATION_METHOD, POSE_RATE_METHOD, MotionObservation, MotionSample,
)
from .bounded_io import canonical_json, decode_json, read_bounded, require_hex, require_keys
from .real_bundle import RealSceneBundle, load_real_bundle


SCHEMA = "hybrid-drive-motion-observation-v1"
INDEX_SCHEMA = "hybrid-drive-motion-observation-index-v1"
MAX_OBSERVATION_BYTES = 32 * 1024
MAX_INDEX_BYTES = 32 * 1024
MAX_METADATA_BYTES = 16 * 1024 * 1024
VERIFIED_METADATA_SHA256 = "d2a9b373da30670aaf327c2ab4ccc332e2b441be6cd5e3f72d77debd3adbcff3"
METHODS = {
    "source": OBSERVATION_METHOD, "associated_heading_rate": HEADING_RATE_METHOD,
    "past_interval_mean": POSE_RATE_METHOD,
    "capture_times": "linked-pose-and-motion-capture-times-not-retained",
    "uncertainty": "not-provided-no-estimated-confidence-bound",
    "coordinates": "nuplan-associated-rear-axle-body-x-forward-y-left-z-up",
    "heading_rate_semantics": "model-from-associated-orientation-and-body-gyro-not-synchronized-measurement",
}
RAW_SAMPLE_KEYS = {
    "frame_token", "metadata_scene_token", "source_list_index", "metadata_frame_idx", "frame_timestamp_us",
    "global_translation_m", "quaternion_wxyz", "body_velocity_mps", "body_acceleration_mps2",
    "body_angular_rate_radps", "pose_capture_timestamp_us", "motion_capture_timestamp_us",
}
PROVENANCE_KEYS = {"source_revision", "navsim_revision", "nuplan_revision", "metadata_sha256", "source_manifest_hash"}
PAYLOAD_KEYS = {"schema", "base_bundle_hash", "provenance", "methods", "samples"}


def load_motion_observation(bundle: RealSceneBundle, path) -> MotionObservation:
    if not isinstance(bundle, RealSceneBundle):
        raise ValueError("bundle must be RealSceneBundle")
    observation = _load_observation(path)
    _validate_for_bundle(bundle, observation)
    return observation


def load_motion_observation_index(root, *, fast_cohort_hash, sources_hash):
    from hybrid_drive.assets.cohort import load_cohort_sources
    from .navsim import repository_sources

    require_hex(fast_cohort_hash, 64, "fast_cohort_hash")
    require_hex(sources_hash, 64, "sources_hash")
    root = Path(os.path.abspath(root))
    check_components(root)
    if not root.is_dir():
        raise ValueError("Motion observation index root must be a directory")
    index = decode_json(read_bounded(root / "index.json", MAX_INDEX_BYTES))
    require_keys(index, {"schema", "source_revision", "sources_hash", "fast_cohort_hash", "metadata_sha256",
                         "methods", "planned_cases", "cases", "index_hash"}, "motion observation index")
    if index["schema"] != INDEX_SCHEMA or index["methods"] != METHODS:
        raise ValueError("Unsupported motion observation index schema or methods")
    require_hex(index["source_revision"], 40, "source_revision")
    require_hex(index["metadata_sha256"], 64, "metadata_sha256")
    require_hex(index["index_hash"], 64, "index_hash")
    if index["index_hash"] != _hash({key: value for key, value in index.items() if key != "index_hash"}):
        raise ValueError("Motion observation index content hash differs")
    if index["fast_cohort_hash"] != fast_cohort_hash or index["sources_hash"] != sources_hash:
        raise ValueError("Motion observation index cohort/source hash differs")
    sources = load_cohort_sources(repository_sources("real-cohort-sources.json"))
    if _hash(sources) != sources_hash or index["metadata_sha256"] != sources["log"]["sha256"]:
        raise ValueError("Motion observation index differs from pinned source selection")
    cases = index["cases"]
    if type(index["planned_cases"]) is not int or index["planned_cases"] != 6 or not isinstance(cases, list) or len(cases) != 6:
        raise ValueError("Motion observation index must retain all six ordered cases")
    for position, (case, spec) in enumerate(zip(cases, sources["cases"])):
        _validate_index_case(root, index, case, spec, position, sources)
    return index


def _validate_index_case(root, index, case, spec, position, sources):
    keys = {"case_id", "scene_token", "source_list_index", "base_bundle_hash", "observation_path", "observation_hash", "status"}
    if not isinstance(case, dict) or not keys <= set(case) or set(case) - keys - {"error"}:
        raise ValueError("Invalid motion observation index case fields")
    if (case["case_id"] != f"r{position:02d}" or case["case_id"] != spec["case_id"]
            or case["scene_token"] != spec["scene_token"]
            or type(case["source_list_index"]) is not int or case["source_list_index"] != spec["source_list_index"]):
        raise ValueError("Motion observation index case identity differs from fixed selection")
    if case["base_bundle_hash"] is not None:
        require_hex(case["base_bundle_hash"], 64, "base_bundle_hash")
    if case["status"] == "failed":
        if (case["observation_path"] is not None or case["observation_hash"] is not None
                or not isinstance(case.get("error"), str) or not case["error"].strip()):
            raise ValueError("Failed observation case must retain its error without fabricated evidence")
        return
    if case["status"] != "success" or "error" in case or case["base_bundle_hash"] is None:
        raise ValueError("Invalid motion observation case status")
    if case["observation_path"] != case["case_id"] + "/motion-observation.json":
        raise ValueError("Motion observation path differs from fixed relative case path")
    require_hex(case["observation_hash"], 64, "observation_hash")
    observation = _load_observation(destination(root, case["observation_path"]))
    current = observation.current
    if (motion_observation_hash(observation) != case["observation_hash"]
            or observation.base_bundle_hash != case["base_bundle_hash"]
            or observation.source_revision != index["source_revision"]
            or observation.source_manifest_hash != index["sources_hash"]
            or observation.metadata_sha256 != index["metadata_sha256"]
            or observation.navsim_revision != sources["navsim_revision"]
            or observation.nuplan_revision != sources["nuplan_revision"]
            or current.frame_token != case["scene_token"] or current.source_list_index != case["source_list_index"]
            or current.frame_timestamp_us != spec["frame_timestamp_us"]
            or current.metadata_scene_token != spec["metadata_scene_token"]
            or current.metadata_frame_idx != spec["metadata_frame_idx"]):
        raise ValueError("Motion observation differs from its index/source binding")


def _load_observation(path):
    manifest = decode_json(read_bounded(Path(path), MAX_OBSERVATION_BYTES))
    require_keys(manifest, PAYLOAD_KEYS | {"observation_hash"}, "motion observation")
    if manifest["schema"] != SCHEMA or manifest["methods"] != METHODS:
        raise ValueError("Unsupported motion observation schema or methods")
    require_hex(manifest["observation_hash"], 64, "observation_hash")
    payload = {key: value for key, value in manifest.items() if key != "observation_hash"}
    if _hash(payload) != manifest["observation_hash"]:
        raise ValueError("Motion observation content hash differs")
    require_keys(manifest["provenance"], PROVENANCE_KEYS, "motion provenance")
    values = manifest["samples"]
    if not isinstance(values, list) or len(values) != 4:
        raise ValueError("Exactly four motion samples are required")
    samples = []
    for value in values:
        require_keys(value, RAW_SAMPLE_KEYS, "motion sample")
        samples.append(MotionSample(**{key: value[key] for key in RAW_SAMPLE_KEYS}))
    observation = MotionObservation(manifest["base_bundle_hash"], samples=tuple(samples), **manifest["provenance"])
    if canonical_json(_payload(observation)) != canonical_json(payload):
        raise ValueError("Motion observation canonical raw fields differ")
    return observation


def motion_observation_hash(observation: MotionObservation) -> str:
    return _hash(_payload(observation))


def write_motion_observation(bundle: RealSceneBundle, observation: MotionObservation, path) -> None:
    _validate_for_bundle(bundle, observation)
    path = Path(os.path.abspath(path))
    _outside_bundle(path)
    encoded = canonical_json({**_payload(observation), "observation_hash": motion_observation_hash(observation)}) + b"\n"
    if len(encoded) > MAX_OBSERVATION_BYTES:
        raise ValueError("Motion observation exceeds byte budget")
    check_components(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_new(path, encoded)


def _validate_for_bundle(bundle, observation):
    if not isinstance(bundle, RealSceneBundle) or not isinstance(observation, MotionObservation):
        raise ValueError("Expected RealSceneBundle and MotionObservation")
    current, provenance = observation.current, bundle.provenance
    motion = bundle.motion
    if (observation.base_bundle_hash != bundle.bundle_hash or current.frame_token != bundle.scene_token
            or current.frame_timestamp_us != bundle.frame_timestamp_us
            or observation.metadata_sha256 != provenance["metadata_sha256"]
            or observation.navsim_revision != provenance["navsim_revision"]
            or observation.nuplan_revision != provenance["nuplan_revision"]
            or current.body_velocity_mps[:2] != (motion.vx_mps, motion.vy_mps)
            or current.body_acceleration_mps2[:2] != (motion.ax_mps2, motion.ay_mps2)):
        raise ValueError("Motion observation identity/source/motion differs from base bundle")


def _payload(observation):
    if not isinstance(observation, MotionObservation):
        raise ValueError("observation must be MotionObservation")
    return {
        "schema": SCHEMA, "base_bundle_hash": observation.base_bundle_hash,
        "provenance": {key: getattr(observation, key) for key in PROVENANCE_KEYS}, "methods": METHODS.copy(),
        "samples": [{key: getattr(sample, key) for key in RAW_SAMPLE_KEYS} for sample in observation.samples],
    }


def export_motion_observations(data_root, fast_run_root, output_root, *, revision):
    from hybrid_drive.assets.cohort import load_cohort_sources
    from .cohort_fast import load_fast_cohort
    from .navsim import repository_sources

    require_hex(revision, 40, "revision")
    fast_root, output_root = Path(os.path.abspath(fast_run_root)), Path(os.path.abspath(output_root))
    check_components(fast_root)
    check_components(output_root)
    if output_root == fast_root or fast_root in output_root.parents:
        raise ValueError("Motion observations must be written outside the historical Fast run")
    _outside_bundle(output_root / "motion-observation.json")
    cohort = load_fast_cohort(fast_root)
    sources = load_cohort_sources(repository_sources("real-cohort-sources.json"))
    if cohort["sources_hash"] != _hash(sources):
        raise ValueError("Fast cohort and motion source selection differ")
    for case, spec in zip(cohort["cases"], sources["cases"]):
        if any(case[key] != spec[key] for key in ("case_id", "scene_token", "source_list_index")):
            raise ValueError("Fast cohort and motion source frame differ")
    frames, source_error = None, None
    try:
        frames = _verified_frames(data_root, sources)
    except Exception as exc:
        source_error = f"{type(exc).__name__}: {exc}"
    run_dir = output_root / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    check_components(run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    for case, spec in zip(cohort["cases"], sources["cases"]):
        row = {
            "case_id": case["case_id"], "scene_token": case["scene_token"],
            "source_list_index": case["source_list_index"], "base_bundle_hash": case["bundle_hash"],
            "observation_path": None, "observation_hash": None, "status": "failed",
        }
        try:
            if frames is None:
                raise ValueError(source_error)
            if case["status"] != "success":
                raise ValueError(case.get("error") or "Historical Fast capture failed")
            bundle = load_real_bundle(destination(fast_root, case["bundle_path"]))
            if bundle.bundle_hash != case["bundle_hash"]:
                raise ValueError("Historical Fast bundle hash differs from cohort")
            observation = _from_frames(bundle, spec, frames, sources, revision=revision)
            relative = case["case_id"] + "/motion-observation.json"
            write_motion_observation(bundle, observation, destination(run_dir, relative))
            row.update(status="success", observation_path=relative, observation_hash=motion_observation_hash(observation))
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    index = {
        "schema": INDEX_SCHEMA, "source_revision": revision, "sources_hash": cohort["sources_hash"],
        "fast_cohort_hash": cohort["cohort_hash"], "metadata_sha256": sources["log"]["sha256"],
        "methods": METHODS.copy(), "planned_cases": len(cohort["cases"]), "cases": rows,
    }
    index["index_hash"] = _hash(index)
    encoded = canonical_json(index) + b"\n"
    if len(encoded) > MAX_INDEX_BYTES:
        raise ValueError("Motion observation index exceeds byte budget")
    _write_new(run_dir / "index.json", encoded)
    return run_dir, index


def _verified_frames(data_root, sources):
    spec = sources["log"]
    if spec["sha256"] != VERIFIED_METADATA_SHA256 or not 0 < spec["size_bytes"] <= MAX_METADATA_BYTES:
        raise ValueError("Only the pinned verified metadata may be deserialized")
    data = read_bounded(destination(data_root, spec["path"]), MAX_METADATA_BYTES)
    if len(data) != spec["size_bytes"] or hashlib.sha256(data).hexdigest() != VERIFIED_METADATA_SHA256:
        raise ValueError("Metadata failed size/SHA256 verification before pickle")
    frames = pickle.loads(data)
    if not isinstance(frames, list) or len(frames) != sources["selection"]["num_source_frames"]:
        raise ValueError("Expected the pinned metadata frame list")
    return frames


def _from_frames(bundle, spec, frames, sources, *, revision):
    index = spec["source_list_index"]
    if type(index) is not int or index < 3 or index >= len(frames):
        raise ValueError("Motion frame is outside verified past/current metadata")
    samples = []
    for source_index in range(index - 3, index + 1):
        frame = frames[source_index]
        if not isinstance(frame, dict):
            raise ValueError("Motion metadata frame must be a dictionary")
        sample = _sample(frame, source_index)
        expected_idx = spec["metadata_frame_idx"] + source_index - index
        if sample.metadata_scene_token != spec["metadata_scene_token"] or sample.metadata_frame_idx != expected_idx:
            raise ValueError("Motion past/current frames cross the selected source chunk")
        samples.append(sample)
    if samples[-1].frame_token != spec["scene_token"] or samples[-1].frame_timestamp_us != spec["frame_timestamp_us"]:
        raise ValueError("Motion current frame differs from fixed source selection")
    minimum, maximum = sources["selection"]["timestamp_step_us"]
    if any(not minimum <= b.frame_timestamp_us - a.frame_timestamp_us <= maximum for a, b in zip(samples, samples[1:])):
        raise ValueError("Motion past/current timestamps are discontinuous")
    observation = MotionObservation(
        base_bundle_hash=bundle.bundle_hash, source_revision=revision,
        navsim_revision=sources["navsim_revision"], nuplan_revision=sources["nuplan_revision"],
        metadata_sha256=sources["log"]["sha256"], source_manifest_hash=_hash(sources), samples=tuple(samples),
    )
    _validate_for_bundle(bundle, observation)
    return observation


def _sample(frame, source_index):
    can_bus = tuple_snapshot(frame["can_bus"], "can_bus")
    if len(can_bus) != 18:
        raise ValueError("Expected the verified OpenScene can_bus layout")
    can_bus = tuple(finite_real(value, "can_bus") for value in can_bus)
    translation = tuple_snapshot(frame["ego2global_translation"], "ego2global_translation")
    quaternion = tuple_snapshot(frame["ego2global_rotation"], "ego2global_rotation")
    motion = tuple(finite_real(value, "ego_dynamic_state")
                   for value in tuple_snapshot(frame["ego_dynamic_state"], "ego_dynamic_state"))
    if (can_bus[:3] != translation or can_bus[3:7] != quaternion
            or (can_bus[10], can_bus[11], can_bus[7], can_bus[8]) != motion):
        raise ValueError("OpenScene can_bus and explicit pose/motion fields differ")
    return MotionSample(
        frame_token=frame["token"], metadata_scene_token=frame["scene_token"], source_list_index=source_index,
        metadata_frame_idx=frame["frame_idx"], frame_timestamp_us=frame["timestamp"],
        global_translation_m=translation, quaternion_wxyz=quaternion, body_velocity_mps=can_bus[10:13],
        body_acceleration_mps2=can_bus[7:10], body_angular_rate_radps=can_bus[13:16],
    )


def _outside_bundle(path):
    if path.name == "bundle.json" or any((parent / "bundle.json").exists() for parent in path.parents):
        raise ValueError("Motion observations must remain outside legacy bundle directories")


def _write_new(path, data):
    check_components(path)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "wb") as output:
        output.write(data)


def _hash(value):
    return hashlib.sha256(canonical_json(value)).hexdigest()
