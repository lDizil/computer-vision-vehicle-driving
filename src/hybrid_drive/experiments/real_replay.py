"""Offline replay of one verified real frame; no arbiter, evaluator or driving loop."""

from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import time
from uuid import uuid4

from hybrid_drive.adapters.real_camera import render_plan_overlay
from hybrid_drive.contracts.planning import Candidate, CandidateSet
from hybrid_drive.contracts.slow import SlowRequest
from hybrid_drive.contracts.trajectory import Pose2D
from hybrid_drive.slow.context import SlowContext
from hybrid_drive.slow.real_grounding import PROTOCOL_NAME
from hybrid_drive.slow.worker import AsyncSlowWorker, wait_for_completion
from .io import write_json
from .real_bundle import RealSceneBundle
from .real_report import write_real_report


def context_for_bundle(bundle, model_version, *, request_id, started, deadline_s):
    if not isinstance(bundle, RealSceneBundle):
        raise ValueError("bundle must be RealSceneBundle")
    # Coordinates remain local rear-axle SE2; no world reference pose is invented.
    plan = CandidateSet(
        "ltf-" + bundle.provenance["raw_candidate_hash"], "bundle-" + bundle.bundle_hash,
        bundle.frame_timestamp_us / 1e6, Pose2D(0, 0, 0), (Candidate("ltf-0", bundle.trajectory),),
    )
    request = SlowRequest(request_id, plan, started, started + deadline_s,
                          plan.t_ref_s + bundle.trajectory.horizon_s, model_version)
    frames = tuple(render_plan_overlay(view, bundle.trajectory) for view in bundle.views)
    return SlowContext(request, bundle.motion, frames)


def run_real_replay(bundle, factory, output_root, *, revision, seeds=(0,), deadline_s=60):
    if revision != "working-tree" and (not isinstance(revision, str) or len(revision) != 40
                                      or any(c not in "0123456789abcdef" for c in revision)):
        raise ValueError("revision must be an exact commit or explicit working-tree diagnostic")
    if isinstance(deadline_s, bool) or not isinstance(deadline_s, (int, float)) or not math.isfinite(deadline_s) or not 0 < deadline_s <= 120:
        raise ValueError("Deadline must be in (0, 120] seconds")
    seeds = tuple(seeds)
    if not seeds or len(seeds) > 8 or len(set(seeds)) != len(seeds) or any(type(s) is not int or not 0 <= s < 2**32 for s in seeds):
        raise ValueError("Seeds must be 1..8 unique integers in [0, 2**32)")
    if factory.protocol_name != PROTOCOL_NAME:
        raise ValueError("Real replay requires the explicit single-candidate advisory protocol")
    # Validate before creating output or starting inference.
    context_for_bundle(bundle, factory.model_version, request_id="validate", started=0, deadline_s=deadline_s)
    run_dir = Path(output_root) / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run_dir.mkdir(parents=True)
    manifest = {
        "schema": "hybrid-drive-real-advisory-replay-v1", "source_revision": revision,
        "bundle_hash": bundle.bundle_hash, "bundle_provenance": bundle.provenance,
        "scene_token": bundle.scene_token, "frame_timestamp_us": bundle.frame_timestamp_us,
        "camera_capture_timestamps": "unknown", "geometry": "planar_rear_axle_z0_no_motion_compensation",
        "scope": "offline_advisory_single_frame_no_safety_or_quality_claim", "automatic_veto": False,
        "platform": platform.platform(), "architecture": platform.machine(),
        "seeds": list(seeds), "deadline_s": deadline_s, "protocol": factory.protocol_name,
        "model_version": factory.model_version, "model_sources": factory.sources,
        "runtime_profile": {key: getattr(factory, key) for key in (
            "execution_profile", "threads", "max_tokens", "image_min_tokens", "image_max_tokens",
            "context_size", "cache_prompt")},
        "latency_definition": "cold worker startup plus inference until parent completion; not control-loop latency",
    }
    write_json(run_dir / "manifest.json", manifest)
    _write_images(run_dir, bundle)
    records = []
    for seed in seeds:
        seeded = replace(factory, seed=seed)
        row = _attempt(bundle, seeded, run_dir, deadline_s)
        records.append(row)
        write_json(run_dir / "records.json", records)
        print(f"Real scene seed={seed}: {row['outcome']} {row['status']} in {row['latency_s']:.3f}s", flush=True)
        if not row["cleanup_confirmed"]:
            break
    result = {
        "status": "completed" if all(row["outcome"] == "completed" for row in records) else "failed",
        "quality": "unscored_no_ground_truth", "automatic_veto": False, "bundle_hash": bundle.bundle_hash,
        "planned_attempts": len(seeds), "attempts": len(records),
        "completed": sum(row["outcome"] == "completed" for row in records),
        "source_revision": revision,
    }
    write_json(run_dir / "result.json", result)
    write_real_report(run_dir / "report.html", bundle, manifest, records, result)
    return run_dir, result


def _attempt(bundle, factory, run_dir, deadline_s):
    started = time.monotonic()
    context = context_for_bundle(bundle, factory.model_version, request_id="real-" + uuid4().hex,
                                 started=started, deadline_s=deadline_s)
    prompts = factory.prompts_for(context)
    program = json.dumps(prompts, ensure_ascii=False, separators=(",", ":"))
    row = {
        "seed": factory.seed, "outcome": "backend_error", "status": None, "vetoed_candidate_ids": [],
        "raw_response": None, "explanation": None, "context_hash": context.content_hash,
        "request_id": context.request.request_id, "candidates_hash": context.request.plan.candidates_hash,
        "submitted_at_monotonic_s": started, "deadline_monotonic_s": started + deadline_s,
        "prompt_sha256": hashlib.sha256(program.encode()).hexdigest(), "declared_prompt_stages": len(prompts),
    }
    (run_dir / f"prompts-{factory.seed}.json").write_text(program, encoding="utf-8")
    worker = None
    try:
        worker = AsyncSlowWorker(factory, factory.model_version, response_parser=factory.parse_output)
        worker.submit(context)
        event = wait_for_completion(worker, context)
        if event is None:
            row["outcome"] = "process_exit"
        else:
            row.update(outcome=event.outcome, raw_response=event.raw_response)
            if event.result is not None:
                event.result.validate_for(context.request)
                row.update(status=event.result.status.value,
                           vetoed_candidate_ids=list(event.result.vetoed_candidate_ids),
                           explanation=event.result.explanation)
    except Exception as exc:
        row.update(outcome="backend_error", status=None, vetoed_candidate_ids=[], explanation=None,
                   error_type=type(exc).__name__)
    finally:
        # An unconfirmed process group must never overlap another model attempt.
        row["latency_s"] = time.monotonic() - started
        row["cleanup_confirmed"] = worker is None
        if worker is not None:
            try:
                worker.close()
                row["cleanup_confirmed"] = True
            except Exception as exc:
                row.update(outcome="cleanup_error", status=None, vetoed_candidate_ids=[], explanation=None,
                           error_type=type(exc).__name__)
    return row


def _write_images(run_dir, bundle):
    from PIL import Image

    for view in bundle.views:
        Image.fromarray(view.frame.rgb).save(run_dir / (view.frame.name + "-rectified.png"))
        overlay = render_plan_overlay(view, bundle.trajectory)
        Image.fromarray(overlay.rgb).save(run_dir / (view.frame.name + "-overlay.png"))
