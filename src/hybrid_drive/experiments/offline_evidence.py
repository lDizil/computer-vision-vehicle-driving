"""Verified historical Slow evidence. Never rebases foreign clocks into runtime time."""

from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from hybrid_drive.assets.files import AtomicFile, check_components
from hybrid_drive.contracts._validation import finite_real
from hybrid_drive.contracts.slow import SlowRequest, SlowResult
from hybrid_drive.slow.context import CAMERA_NAMES, MAX_HEIGHT, MAX_WIDTH, CameraFrame, SlowContext
from hybrid_drive.slow.protocol import MAX_RESPONSE_BYTES, bounded_prompt
from hybrid_drive.slow.real_grounding import PROTOCOL_NAME, RealPathGroundingFactory, parse_real_path_trace
from .bounded_io import canonical_json, decode_json, read_bounded, require_hex, require_keys
from .io import write_json
from .protocol import content_hash
from .real_replay import plan_for_bundle


SCHEMA = "hybrid-drive-frozen-slow-evidence-v1"
INDEX_SCHEMA = "hybrid-drive-cohort-evidence-v1"
MAX_JSON_BYTES = 128 * 1024
MAX_PNG_BYTES = 2 * 1024 * 1024
OUTCOMES = {"completed", "invalid_output", "backend_error", "process_exit", "timeout", "cleanup_error"}
KEYS = {"schema", "bundle_hash", "source_revision", "protocol", "model_sources", "model_sources_hash",
        "model_version", "runtime_profile", "request", "context_hash", "prompts", "prompt_sha256",
        "outcome", "raw_response", "received_at_monotonic_s", "decision_checkpoint_monotonic_s",
        "cleanup_confirmed", "cameras", "evidence_hash"}
REQUEST_KEYS = {"request_id", "submitted_at_monotonic_s", "deadline_monotonic_s", "valid_until_sim_s"}
CAMERA_KEYS = {"name", "path", "sha256", "size_bytes", "width", "height"}
INDEX_KEYS = {"schema", "source_revision", "cohort_hash", "model_version", "model_sources_hash",
              "protocol", "cases", "evidence_index_hash"}
CASE_KEYS = {"case_id", "scene_token", "bundle_hash", "status", "evidence_path", "evidence_hash", "error"}


@dataclass(frozen=True)
class FrozenEvidence:
    context: SlowContext
    result: SlowResult | None
    outcome: str
    received_at_monotonic_s: float | None
    decision_checkpoint_monotonic_s: float
    bundle_hash: str
    source_revision: str
    evidence_hash: str
    model_sources_hash: str
    protocol: str
    raw_response: str | None
    cleanup_confirmed: bool
    prompt_sha256: str


def write_frozen_evidence(bundle, replay_dir, destination):
    replay_dir, destination = Path(replay_dir), Path(destination)
    manifest = _json_file(replay_dir / "manifest.json")
    result = _json_file(replay_dir / "result.json")
    rows = _json_file(replay_dir / "records.json")
    if not isinstance(rows, list) or len(rows) != 1 or len(manifest["seeds"]) != 1:
        raise ValueError("Frozen evidence requires exactly one actual replay attempt")
    row = rows[0]
    if type(row["seed"]) is not int or row["seed"] != manifest["seeds"][0]:
        raise ValueError("Replay seed differs from original manifest")
    if (manifest["bundle_hash"] != bundle.bundle_hash or result["bundle_hash"] != bundle.bundle_hash
            or manifest["source_revision"] != result["source_revision"]):
        raise ValueError("Replay result/manifest differs from bundle or revision")
    plan = plan_for_bundle(bundle)
    if row["candidates_hash"] != plan.candidates_hash:
        raise ValueError("Replay candidate differs from frozen Fast plan")
    request = {
        "request_id": row["request_id"], "submitted_at_monotonic_s": row["submitted_at_monotonic_s"],
        "deadline_monotonic_s": row["deadline_monotonic_s"],
        "valid_until_sim_s": plan.t_ref_s + bundle.trajectory.horizon_s,
    }
    cameras, images = [], []
    for name in CAMERA_NAMES:
        data = read_bounded(replay_dir / (name + "-overlay.png"), MAX_PNG_BYTES)
        frame = _decode_png(name, data)
        cameras.append({"name": name, "path": name + ".png", "sha256": _sha(data),
                        "size_bytes": len(data), "width": frame.width, "height": frame.height})
        images.append(data)
    prompts = _json_file(replay_dir / f"prompts-{row['seed']}.json")
    payload = {
        "schema": SCHEMA, "bundle_hash": bundle.bundle_hash, "source_revision": manifest["source_revision"],
        "protocol": manifest["protocol"], "model_sources": manifest["model_sources"],
        "model_sources_hash": content_hash(manifest["model_sources"]), "model_version": manifest["model_version"],
        "runtime_profile": {**manifest["runtime_profile"], "seed": row["seed"], "temperature": 0}, "request": request, "context_hash": row["context_hash"],
        "prompts": prompts, "prompt_sha256": row["prompt_sha256"], "outcome": row["outcome"],
        "raw_response": row["raw_response"], "received_at_monotonic_s": row["received_at_monotonic_s"],
        "decision_checkpoint_monotonic_s": row["decision_checkpoint_monotonic_s"],
        "cleanup_confirmed": row["cleanup_confirmed"], "cameras": cameras,
    }
    payload["evidence_hash"] = _evidence_hash(payload, images)
    evidence = _validate_payload(bundle, payload, images)
    if (row["status"] != (evidence.result.status.value if evidence.result else None)
            or row["vetoed_candidate_ids"] != (list(evidence.result.vetoed_candidate_ids) if evidence.result else [])):
        raise ValueError("Saved replay decision differs from actual raw model trace")
    encoded = canonical_json(payload)
    if len(encoded) > MAX_JSON_BYTES:
        raise ValueError("Evidence metadata exceeds byte budget")
    check_components(destination)
    destination.mkdir(parents=True, exist_ok=False)
    for camera, data in zip(cameras, images):
        with AtomicFile(destination / camera["path"]) as temporary:
            temporary.out.write(data)
            temporary.commit()
    write_json(destination / "evidence.json", payload)
    return evidence


def _json_file(path):
    return decode_json(read_bounded(path, MAX_JSON_BYTES))


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _decode_png(name, data):
    with Image.open(io.BytesIO(data)) as image:
        if (image.format != "PNG" or image.mode != "RGB" or getattr(image, "n_frames", 1) != 1
                or not 0 < image.width <= MAX_WIDTH or not 0 < image.height <= MAX_HEIGHT):
            raise ValueError("Evidence requires bounded single-frame RGB PNG")
        image.load()
        return CameraFrame(name, np.asarray(image))


def _evidence_hash(payload, images):
    digest = hashlib.sha256(canonical_json({k: v for k, v in payload.items() if k != "evidence_hash"}))
    for image in images:
        digest.update(image)
    return digest.hexdigest()


def load_frozen_evidence(bundle, path):
    path = Path(path)
    check_components(path)
    if {p.name for p in path.iterdir()} != {"evidence.json", *(name + ".png" for name in CAMERA_NAMES)}:
        raise ValueError("Evidence directory has missing or unexpected files")
    payload = _json_file(path / "evidence.json")
    # Fixed names are used before interpreting any untrusted metadata path.
    images = [read_bounded(path / (name + ".png"), MAX_PNG_BYTES) for name in CAMERA_NAMES]
    return _validate_payload(bundle, payload, images)


def _validate_payload(bundle, payload, images):
    require_keys(payload, KEYS, "evidence")
    if (payload["schema"] != SCHEMA or payload["bundle_hash"] != bundle.bundle_hash
            or payload["protocol"] != PROTOCOL_NAME):
        raise ValueError("Evidence schema/bundle/protocol differs")
    for key in ("bundle_hash", "context_hash", "model_sources_hash", "prompt_sha256", "evidence_hash"):
        require_hex(payload[key], 64, key)
    require_hex(payload["source_revision"], 40, "source_revision")
    if _evidence_hash(payload, images) != payload["evidence_hash"]:
        raise ValueError("Evidence content hash differs")
    if content_hash(payload["model_sources"]) != payload["model_sources_hash"]:
        raise ValueError("Model source binding differs")
    factory = RealPathGroundingFactory("unused", payload["model_sources"])
    if factory.model_version != payload["model_version"]:
        raise ValueError("Model version differs from source identity")
    require_keys(payload["runtime_profile"], {
        "execution_profile", "threads", "max_tokens", "image_min_tokens", "image_max_tokens",
        "context_size", "cache_prompt", "seed", "temperature",
    }, "runtime profile")
    profile = payload["runtime_profile"]
    if (type(profile["seed"]) is not int or not 0 <= profile["seed"] < 2**32
            or type(profile["cache_prompt"]) is not bool or type(profile["temperature"]) not in (int, float)
            or profile["temperature"] != 0
            or profile["execution_profile"] not in ("cpu", "metal")):
        raise ValueError("Invalid recorded generation profile")
    require_keys(payload["request"], REQUEST_KEYS, "request")
    plan = plan_for_bundle(bundle)
    request = SlowRequest(plan=plan, model_version=payload["model_version"], **payload["request"])
    if request.valid_until_sim_s != plan.t_ref_s + bundle.trajectory.horizon_s:
        raise ValueError("Simulation validity differs from frozen bundle horizon")
    if not isinstance(payload["cameras"], list) or len(payload["cameras"]) != 3:
        raise ValueError("Evidence must contain three ordered overlays")
    frames = []
    for name, camera, data, view in zip(CAMERA_NAMES, payload["cameras"], images, bundle.views):
        require_keys(camera, CAMERA_KEYS, "camera")
        frame = _decode_png(name, data)
        expected = {"name": name, "path": name + ".png", "sha256": _sha(data), "size_bytes": len(data),
                    "width": frame.width, "height": frame.height}
        if (any(type(camera[key]) is not int for key in ("width", "height", "size_bytes"))
                or camera != expected) or (frame.width, frame.height) != (view.frame.width, view.frame.height):
            raise ValueError("Overlay identity/hash/dimensions differ")
        frames.append(frame)
    context = SlowContext(request, bundle.motion, tuple(frames))
    if context.content_hash != payload["context_hash"]:
        raise ValueError("Original Slow context hash differs")
    prompts = payload["prompts"]
    if not isinstance(prompts, list) or prompts != list(factory.prompts_for(context)):
        raise ValueError("Evidence prompt program differs from pinned protocol")
    program = json.dumps([bounded_prompt(p) for p in prompts], ensure_ascii=False, separators=(",", ":"))
    if _sha(program.encode()) != payload["prompt_sha256"]:
        raise ValueError("Prompt hash differs")
    checkpoint = finite_real(payload["decision_checkpoint_monotonic_s"], "decision checkpoint")
    received = payload["received_at_monotonic_s"]
    if received is not None:
        received = finite_real(received, "received time")
        if not request.submitted_at_monotonic_s <= received <= checkpoint:
            raise ValueError("Receive time is outside original clock interval")
    if checkpoint < request.submitted_at_monotonic_s:
        raise ValueError("Decision checkpoint precedes original request")
    outcome = payload["outcome"]
    if not isinstance(outcome, str) or outcome not in OUTCOMES or type(payload["cleanup_confirmed"]) is not bool:
        raise ValueError("Invalid replay outcome/cleanup state")
    if not payload["cleanup_confirmed"] and outcome != "cleanup_error":
        raise ValueError("Unconfirmed cleanup cannot be successful evidence")
    raw = payload["raw_response"]
    if raw is not None and (not isinstance(raw, str) or len(raw.encode("utf-8")) > MAX_RESPONSE_BYTES):
        raise ValueError("Raw response exceeds strict byte budget")
    result = None
    if outcome == "completed":
        if received is None:
            raise ValueError("Completed evidence must contain measured receive time")
        result = parse_real_path_trace(raw, context)
        result.validate_for(request)
    return FrozenEvidence(context, result, outcome, received, checkpoint, bundle.bundle_hash,
                          payload["source_revision"], payload["evidence_hash"], payload["model_sources_hash"],
                          payload["protocol"], raw, payload["cleanup_confirmed"], payload["prompt_sha256"])


def load_cohort_evidence_index(root, *, cohort_hash):
    index = _json_file(Path(root) / "cohort-evidence.json")
    require_keys(index, INDEX_KEYS, "evidence index")
    if (index["schema"] != INDEX_SCHEMA or index["cohort_hash"] != cohort_hash
            or index["protocol"] != PROTOCOL_NAME):
        raise ValueError("Evidence index does not belong to Fast cohort/protocol")
    for key in ("cohort_hash", "model_sources_hash", "evidence_index_hash"):
        require_hex(index[key], 64, key)
    require_hex(index["source_revision"], 40, "source_revision")
    if content_hash({k: v for k, v in index.items() if k != "evidence_index_hash"}) != index["evidence_index_hash"]:
        raise ValueError("Evidence index hash differs")
    if not isinstance(index["cases"], list) or len(index["cases"]) != 6:
        raise ValueError("Evidence index requires six planned cases")
    for i, case in enumerate(index["cases"]):
        require_keys(case, CASE_KEYS, "case")
        if case["case_id"] != f"r{i:02d}":
            raise ValueError("Evidence cases must remain in frozen order")
        require_hex(case["scene_token"], 16, "scene_token")
        if case["bundle_hash"] is not None:
            require_hex(case["bundle_hash"], 64, "bundle_hash")
        if case["status"] == "success":
            if case["evidence_path"] != case["case_id"] + "/evidence" or case["bundle_hash"] is None:
                raise ValueError("Evidence path differs from fixed case layout")
            require_hex(case["evidence_hash"], 64, "evidence_hash")
        elif case["status"] not in ("failed", "not_started") or case["evidence_path"] is not None or case["evidence_hash"] is not None:
            raise ValueError("Failed cases must not claim exported evidence")
        if case["error"] is not None and (not isinstance(case["error"], str) or len(case["error"]) > 512):
            raise ValueError("Invalid evidence case error")
    return index
