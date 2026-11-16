"""Freeze twenty actual LTF outputs before any scoring or timed Slow replay."""

from datetime import datetime, timezone
import hashlib
from numbers import Integral
from pathlib import Path
import pickle
import time

from hybrid_drive.assets.files import check_components, destination
from hybrid_drive.assets.temporal import load_temporal_sources, verify_temporal_sample
from .bounded_io import decode_json, read_bounded, require_keys, require_hex
from .cohort_fast import _initialize_fast, _make_bundle, _revision, _seed, _verified_models
from .io import write_json
from .navsim import configure_environment, load_cohort_inputs
from .protocol import content_hash, make_manifest, trajectory_hash, verify_manifest
from .real_bundle import load_real_bundle, write_real_bundle


SCHEMA = "hybrid-drive-temporal-fast-v1"
SCOPE = "single_recording_source_cadence_debug"
EXECUTION = "frozen_before_any_scoring_or_source_cadence_replay"
LATENCY = "LTF compute_trajectory only: process CPU seconds and monotonic wall seconds; input loading separate"


def run_temporal_fast(data_root, asset_root, temporal_root, output_root, *, revision, seed=0, sources_path=None):
    _revision(revision)
    _seed(seed)
    sources = load_temporal_sources(sources_path)
    output_root = Path(output_root).absolute()
    if any(output_root == Path(root).absolute() or Path(root).absolute() in output_root.parents
           for root in (data_root,asset_root,temporal_root)):
        raise ValueError("Fast output must remain outside verified input roots")
    run_dir = Path(output_root) / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    check_components(run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema": SCHEMA, "scope": SCOPE, "source_revision": revision, "sources_hash": content_hash(sources),
        "model": None, "seed": seed, "planned_cases": 20,
        "latency_definition": LATENCY,
        "execution": EXECUTION,
        "cases": [{"case_id": s["case_id"], "scene_token": s["scene_token"], "source_list_index": s["source_list_index"],
                   "frame_timestamp_us": s["frame_timestamp_us"], "bundle_path": s["case_id"] + "/bundle", "bundle_hash": None,
                   "raw_candidate_hash": None, "status": "failed", "error": "not_attempted",
                   "fast_cpu_s": None, "fast_wall_s": None, "input_wall_s": None} for s in sources["cases"]],
    }
    try:
        configure_environment(data_root, run_dir)
        assets = verify_temporal_sample(temporal_root, sources)
        frames = _current_frames(data_root, sources)
        tokens = [s["scene_token"] for s in sources["cases"]]
        load_cohort_inputs(sources, tokens)
        models = _verified_models(asset_root)
        fast = _initialize_fast(asset_root, seed)
        manifest["model"] = fast.provenance()
        loader, _ = load_cohort_inputs(sources, tokens,
                                      sensor_root=Path(temporal_root) / "sensor_blobs", sensor_config=fast.get_sensor_config())
        from hybrid_drive.adapters.navsim import from_navsim
        from scripts.run_navsim_sample import sample_policy

        for spec, case in zip(sources["cases"], manifest["cases"]):
            case_dir = run_dir / case["case_id"]
            case_dir.mkdir()
            try:
                started = time.monotonic()
                agent_input = loader.get_agent_input_from_token(spec["scene_token"])
                case["input_wall_s"] = time.monotonic() - started
                cpu, wall = time.process_time(), time.monotonic()
                try:
                    native = fast.compute_trajectory(agent_input)
                finally:
                    case["fast_cpu_s"], case["fast_wall_s"] = time.process_time()-cpu, time.monotonic()-wall
                raw = from_navsim(native)
                records = [record for record in assets["files"] if record["case_id"] == case["case_id"]]
                frame_manifest = make_manifest(
                    revision=revision, sources=sources, assets={"sources": sources, "models": models, "cameras": {"files": records}},
                    model=manifest["model"], scene_token=spec["scene_token"], log_name=Path(sources["log"]["path"]).stem,
                    seed=seed, policy=sample_policy(), candidate=raw,
                    split_audit={"role": "smoke", "scope": SCOPE, "sources_hash": manifest["sources_hash"]},
                )
                frame_manifest.update(execution=manifest["execution"], executed_ablations=[],
                                      quality_claim="Twenty successive frozen debug frames; no real-time inference or driving-safety claim")
                frame_manifest.pop("manifest_hash")
                frame_manifest["manifest_hash"] = content_hash(frame_manifest)
                bundle = _make_bundle(spec, frames[case["case_id"]], raw, temporal_root, records, frame_manifest,
                                      metadata_sha256=sources["log"]["sha256"])
                write_real_bundle(bundle, case_dir / "bundle")
                write_json(case_dir / "fast-manifest.json", frame_manifest)
                case.update(status="success", bundle_hash=bundle.bundle_hash, raw_candidate_hash=trajectory_hash(raw))
                case.pop("error")
            except Exception as exc:
                case["error"] = f"{type(exc).__name__}: {exc}"
            _save(run_dir, manifest)
    except Exception as exc:
        for case in manifest["cases"]:
            if case.get("error") == "not_attempted":
                case["error"] = f"{type(exc).__name__}: {exc}"
    _save(run_dir, manifest)
    return run_dir, manifest


def _current_frames(data_root, sources):
    spec = sources["log"]
    data = read_bounded(destination(data_root, spec["path"]), 16 * 1024 * 1024)
    if len(data) != spec["size_bytes"] or hashlib.sha256(data).hexdigest() != spec["sha256"]:
        raise ValueError("Temporal metadata failed pinned size/SHA256 verification before pickle")
    frames = pickle.loads(data)
    if not isinstance(frames, list) or len(frames) != 738:
        raise ValueError("Expected the pinned 738-frame recording")
    for identity in sources["selection"]["identity_window"]:
        frame = frames[identity["source_list_index"]]
        if not isinstance(frame, dict) or any(frame[key] != identity[key] for key in ("token", "scene_token", "frame_idx", "timestamp")):
            raise ValueError("Temporal source identity/time differs from frozen pre-inference selection")
        if any(isinstance(frame[key], bool) or not isinstance(frame[key], Integral) for key in ("frame_idx", "timestamp")):
            raise ValueError("Temporal source frame index/time must be integral")
    current = {}
    for case in sources["cases"]:
        frame = frames[case["source_list_index"]]
        if any("sensor_blobs/"+frame["cams"][camera["camera"]]["data_path"] != camera["path"] for camera in case["cameras"]):
            raise ValueError("Temporal current camera differs from the pre-inference allowlist")
        current[case["case_id"]] = frame
    return current


def load_temporal_fast(root, *, require_success=True):
    root = Path(root)
    manifest = decode_json(read_bounded(root / "temporal-fast.json", 128 * 1024))
    require_keys(manifest, {"schema", "scope", "source_revision", "sources_hash", "model", "seed", "planned_cases", "latency_definition",
                            "execution", "cases", "temporal_fast_hash"}, "temporal Fast manifest")
    sources = load_temporal_sources()
    if (manifest["schema"] != SCHEMA or manifest["scope"] != SCOPE or manifest["sources_hash"] != content_hash(sources)
            or manifest["execution"] != EXECUTION or manifest["latency_definition"] != LATENCY
            or manifest["temporal_fast_hash"] != content_hash({k:v for k,v in manifest.items() if k != "temporal_fast_hash"})
            or type(manifest["planned_cases"]) is not int or manifest["planned_cases"] != 20
            or not isinstance(manifest["cases"], list) or len(manifest["cases"]) != 20):
        raise ValueError("Temporal Fast manifest identity/hash/extent differs")
    _revision(manifest["source_revision"])
    _seed(manifest["seed"])
    for case, spec in zip(manifest["cases"], sources["cases"]):
        required = {"case_id", "scene_token", "source_list_index", "frame_timestamp_us", "bundle_path", "bundle_hash",
                    "raw_candidate_hash", "status", "fast_cpu_s", "fast_wall_s", "input_wall_s"}
        if not isinstance(case, dict) or not required <= set(case) or set(case)-required-{"error"}:
            raise ValueError("Invalid temporal Fast case fields")
        if any(type(case[k]) is not type(spec[k]) or case[k] != spec[k] for k in ("case_id", "scene_token", "source_list_index", "frame_timestamp_us")):
            raise ValueError("Temporal Fast case differs from fixed source selection")
        if case["bundle_path"] != case["case_id"] + "/bundle":
            raise ValueError("Temporal Fast bundle path differs")
        for key in ("fast_cpu_s", "fast_wall_s", "input_wall_s"):
            value = case[key]
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int,float)) or not 0 <= value < float("inf")):
                raise ValueError("Invalid measured Fast latency")
        if case["status"] == "success":
            require_hex(case["bundle_hash"], 64, "bundle_hash")
            require_hex(case["raw_candidate_hash"], 64, "raw_candidate_hash")
            if "error" in case or any(case[k] is None for k in ("fast_cpu_s", "fast_wall_s", "input_wall_s")) or not isinstance(manifest["model"], dict):
                raise ValueError("Successful temporal Fast frame lacks measured provenance/latency")
        elif (case["status"] != "failed" or case["bundle_hash"] is not None or case["raw_candidate_hash"] is not None
              or not isinstance(case.get("error"), str) or not case["error"]):
            raise ValueError("Failed temporal Fast frame must retain its cause without fabricated output")
        if require_success and case["status"] != "success":
            raise ValueError("Source cadence replay requires all twenty successful frozen Fast frames")
    return manifest


def load_temporal_bundles(root, manifest):
    sources = load_temporal_sources()
    bundles = []
    for case in manifest["cases"]:
        bundle = load_real_bundle(destination(root, case["bundle_path"]))
        fast = decode_json(read_bounded(destination(root, case["case_id"] + "/fast-manifest.json"), 128 * 1024))
        verify_manifest(fast)
        if (bundle.bundle_hash != case["bundle_hash"] or bundle.scene_token != case["scene_token"]
                or bundle.frame_timestamp_us != case["frame_timestamp_us"] or trajectory_hash(bundle.trajectory) != case["raw_candidate_hash"]
                or fast["raw_candidate_hash"] != case["raw_candidate_hash"] or fast["source_revision"] != manifest["source_revision"]
                or fast["dataset_source_hash"] != manifest["sources_hash"] or fast["model"] != manifest["model"]
                or bundle.provenance["source_experiment_manifest_hash"] != fast["manifest_hash"]
                or bundle.provenance["source_revision"] != manifest["source_revision"]
                or bundle.provenance["metadata_sha256"] != sources["log"]["sha256"]
                or bundle.provenance["navsim_revision"] != sources["navsim_revision"]
                or bundle.provenance["nuplan_revision"] != sources["nuplan_revision"]
                or bundle.provenance["checkpoint_sha256"] != manifest["model"]["checkpoint_sha256"]):
            raise ValueError("Temporal Fast output/bundle provenance differs")
        bundles.append(bundle)
    return tuple(bundles)


def _save(root, manifest):
    manifest.pop("temporal_fast_hash", None)
    manifest["temporal_fast_hash"] = content_hash(manifest)
    write_json(root / "temporal-fast.json", manifest)
