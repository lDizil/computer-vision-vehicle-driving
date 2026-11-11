"""Offline deliberation pilot: one real Fast plan, three views and a local VLM."""

import argparse
import ctypes
import gc
import json
from pathlib import Path
import time
import traceback
from uuid import uuid4

from scripts.run_neural_sample import render_comparison, write_comparison_csv, write_json
from scripts.run_navsim_sample import (
    build_hybrid_agent, run_logged, sample_policy, score_hybrid_prediction, write_decision,
)
from scripts.sample_support import load_sample, prepare_sample_paths


MODES = ("baseline", "checks_only", "combined")


def run_slow_sample(data_root, fast_root, slow_root, output_root, *, revision, deadline_s=60, seed=0):
    if not isinstance(revision, str) or len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("revision must identify the exact tested source commit")
    if not 0 < deadline_s <= 120:
        raise ValueError("Offline deadline must be in (0, 120] seconds")
    sources, run_dir = prepare_sample_paths(data_root, output_root)
    fast_sources = json.loads(Path(__file__).with_name("ltf-sample-sources.json").read_text())
    slow_sources = json.loads(Path(__file__).with_name("smolvlm-sources.json").read_text())
    token = fast_sources["scene_token"]
    result = {
        "status": "failed", "scene_token": token, "source_revision": revision,
        "execution": "offline_frozen_scene_deliberation", "real_time_claim": False,
        "driving_safety_claim": False, "reason2drive_reproduction": False,
        "seed": seed, "deadline_seconds": deadline_s, "fast_calls": 0, "records": [],
    }
    try:
        import numpy as np
        import torch
        from hybrid_drive.adapters.navsim import from_navsim, to_navsim
        from hybrid_drive.adapters.slow_context import context_from_navsim
        from hybrid_drive.agents.hybrid import _current_motion
        from hybrid_drive.assets.prepare import verify_prepared
        from hybrid_drive.assets.slow import verify_slow
        from hybrid_drive.contracts.slow import SlowRequest
        from hybrid_drive.experiments.protocol import make_manifest, scene_result
        from hybrid_drive.experiments.splits import audit_official_splits
        from hybrid_drive.fast.latent_transfuser import LatentTransfuserAgent
        from hybrid_drive.safety.arbiter import SafetyArbiter
        from hybrid_drive.safety.fallback import BrakingConfig
        from hybrid_drive.safety.kinematics import KinematicLimits
        from hybrid_drive.slow.llama import LlamaServerFactory
        from hybrid_drive.slow.protocol import make_prompt
        from hybrid_drive.slow.worker import AsyncSlowWorker

        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.use_deterministic_algorithms(True)
        fast_assets = verify_prepared(fast_root, fast_sources)
        slow_assets = verify_slow(slow_root, slow_sources)
        fast = LatentTransfuserAgent(Path(fast_root) / "checkpoints/ltf_seed_0.ckpt", asset_root=fast_root)
        loader, selected, overrides = load_sample(
            sources, token, sensor_config=fast.get_sensor_config(), sensor_root=Path(fast_root) / "sensor_blobs",
        )
        if selected != token:
            raise ValueError("Fast and Slow selected different scenes")
        agent_input = loader.get_agent_input_from_token(token)
        hybrid = build_hybrid_agent(fast)
        started = time.monotonic()
        hybrid.initialize()
        result["fast_calls"] += 1
        checked = hybrid.compute_trajectory(agent_input)
        result["fast_initialization_and_inference_seconds"] = time.monotonic() - started
        original = hybrid.last_decision
        plan = original.selection.plan
        motion = _current_motion(agent_input)
        raw = plan.candidates[0].trajectory
        fast_model = fast.provenance()
        result["fast_model"] = fast_model
        # Keep only the owned plan and RGB inputs while the second model is resident.
        del hybrid, fast
        gc.collect()
        ctypes.CDLL(None).malloc_trim(0)
        started = time.monotonic()
        request = SlowRequest("slow-" + uuid4().hex, plan, started, started + deadline_s,
                              plan.t_ref_s + raw.horizon_s, slow_sources["model_version"])
        context = context_from_navsim(request, motion, agent_input)
        (run_dir / "prompt.txt").write_text(make_prompt(context))
        factory = LlamaServerFactory(str(slow_root), slow_sources, seed=seed)
        completion = None
        with AsyncSlowWorker(factory, factory.model_version) as worker:
            worker.submit(context)
            peak_queue_depth = worker.queue_depth
            while worker.queue_depth:
                completions = worker.poll()
                peak_queue_depth = max(peak_queue_depth, worker.queue_depth)
                for value in completions:
                    if value.context.content_hash == context.content_hash:
                        completion = value
                if completion is not None:
                    break
                time.sleep(0.005)  # Offline harness only; never inside the fast decision path.
            for value in worker.poll():
                if value.context.content_hash == context.content_hash:
                    completion = value
            result["slow_wall_seconds"] = time.monotonic() - started
            result["worker"] = {
                "outcome_counts": dict(worker.outcome_counts), "max_queue_depth": peak_queue_depth,
                "dropped_completion_count": worker.dropped_completion_count,
            }
            policy = sample_policy()
            arbiter = SafetyArbiter(KinematicLimits(**policy["limits"]), BrakingConfig(**policy["braking"]))
            decision = arbiter.decide(
                plan, motion, now_monotonic_s=time.monotonic(), now_sim_s=plan.t_ref_s,
                request=request, result=None if completion is None else completion.result,
                received_at_monotonic_s=None if completion is None else completion.received_at_monotonic_s,
            )
        result["slow"] = {
            "model_version": factory.model_version, "context_hash": context.content_hash,
            "outcome": None if completion is None else completion.outcome,
            "raw_response": None if completion is None else completion.raw_response,
            "valid_output": completion is not None and completion.outcome == "completed",
            "status": decision.slow.status.value, "disposition": decision.slow.disposition.value,
            "used": decision.slow.used,
        }
        combined = to_navsim(decision.trajectory)
        write_decision(run_dir, token, decision, combined)
        predictions = {"baseline": to_navsim(raw), "checks_only": checked, "combined": combined}
        # Privileged labels and scoring are deliberately read only after both agents finish.
        reference = loader.get_scene_from_token(token).get_future_trajectory(num_trajectory_frames=raw.num_poses)
        cache_dir = run_dir / "metric-cache"
        import sys
        command = [sys.executable, "/opt/navsim/navsim/planning/script/run_metric_caching.py", *overrides,
                   "cache.cache_path=" + str(cache_dir)]
        run_logged("caching", command, token, run_dir, {})
        for mode, prediction in predictions.items():
            mode_decision = {"checks_only": original, "combined": decision}.get(mode)
            try:
                if mode_decision is not None and mode_decision.failed:
                    raise ValueError("Safety fallback failed; placeholder is not a valid prediction")
                metrics = score_hybrid_prediction(prediction, token, cache_dir, overrides)
                record = scene_result(mode=mode, token=token, metrics=metrics, prediction=from_navsim(prediction),
                                      reference=from_navsim(reference),
                                      status="candidate" if mode_decision is None else mode_decision.selection.status.value)
            except Exception as exc:
                record = scene_result(mode=mode, token=token, failed=True, error=type(exc).__name__, status="failure")
            record["native_trajectory"] = prediction.poses.tolist()
            result["records"].append(record)
        write_comparison_csv(run_dir / "comparison.csv", result["records"])
        render_comparison(run_dir, predictions, reference, result["records"])
        log_name = Path(sources["log"]["path"]).stem
        manifest = make_manifest(
            revision=revision, sources=sources, assets=fast_assets, model=fast_model, scene_token=token,
            log_name=log_name, seed=seed, policy=sample_policy(), candidate=raw,
            split_audit=audit_official_splits(smoke_token=token, smoke_log=log_name),
        )
        write_json(run_dir / "experiment-manifest.json", {
            "fast": manifest, "slow_assets": slow_assets, "context_hash": context.content_hash,
            "preprocessing": "independent RGB bilinear 384x216, llama image_max_tokens=64",
            "execution": result["execution"], "deadline_seconds": deadline_s,
            "generation": {"temperature": 0, "seed": seed, "max_tokens": factory.max_tokens},
            "modes": list(predictions), "slow_only": "not_implemented_in_this_pilot",
        })
        if not result["slow"]["valid_output"] or any(row["failed"] for row in result["records"]):
            raise ValueError("Slow pilot failed; actual outcomes and attempted modes retained")
        result["status"] = "success"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        (run_dir / "failure.log").write_text(traceback.format_exc())
        if not result["records"]:
            from hybrid_drive.experiments.protocol import scene_result
            result["records"] = [scene_result(mode=mode, token=token, failed=True, error=result["error"])
                                 for mode in MODES]
            write_comparison_csv(run_dir / "comparison.csv", result["records"])
        raise
    finally:
        peak = Path("/sys/fs/cgroup/memory.peak")
        result["container_peak_memory_bytes"] = int(peak.read_text()) if peak.exists() else None
        write_json(run_dir / "result.json", result)
        print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data-root", "fast-root", "slow-root", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--deadline-seconds", type=float, default=60)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    try:
        run_slow_sample(args.data_root, args.fast_root, args.slow_root, args.output_root,
                        revision=args.revision, deadline_s=args.deadline_seconds, seed=args.seed)
    except Exception:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
