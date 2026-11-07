"""Evaluate one real neural candidate, checks-only decision and CV reference offline."""

import argparse
import csv
import json
import os
from pathlib import Path
import platform
import random
import resource
import sys
import time
import traceback

from hybrid_drive.experiments.protocol import (
    METRICS, make_manifest, scene_result, summarize, validate_splits,
)
from scripts.run_navsim_sample import (
    build_hybrid_agent, run_logged, sample_policy, score_hybrid_prediction, write_decision, write_score_rows,
)
from scripts.sample_support import load_sample, prepare_sample_paths


MODES = ("baseline", "checks_only", "cv_reference")


def run_neural_sample(data_root, asset_root, output_root, *, revision, seed=0):
    sources, run_dir = prepare_sample_paths(data_root, output_root)
    asset_root = Path(asset_root).resolve()
    asset_sources = json.loads(Path(__file__).with_name("ltf-sample-sources.json").read_text())
    token = asset_sources["scene_token"]
    result = {
        "status": "failed", "scene_token": token, "source_revision": revision,
        "cpu_only": True, "platform": platform.machine(), "python_version": platform.python_version(),
        "records": [], "summary": None, "durations": {}, "hybrid_calls_attempted": 0,
    }
    predictions = {}
    try:
        # Environment paths above must precede importing NAVSIM/model modules.
        import numpy as np
        import torch
        from navsim.agents.constant_velocity_agent import ConstantVelocityAgent
        from hybrid_drive.adapters.navsim import to_navsim
        from hybrid_drive.assets.prepare import verify_prepared
        from hybrid_drive.experiments.splits import audit_official_splits
        from hybrid_drive.fast.latent_transfuser import LatentTransfuserAgent

        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
            raise ValueError("Seed must be an integer in [0, 2**32)")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
        assets = verify_prepared(asset_root, asset_sources)
        fast = LatentTransfuserAgent(asset_root / "checkpoints/ltf_seed_0.ckpt", asset_root=asset_root)
        loader, selected, overrides = load_sample(
            sources, token, sensor_config=fast.get_sensor_config(), sensor_root=asset_root / "sensor_blobs",
        )
        if selected != token:
            raise ValueError("The camera sample and evaluator selected different scenes")
        log_name = Path(sources["log"]["path"]).stem
        validate_splits({"smoke": [{"scene_token": token, "log_name": log_name}]})
        cache_dir = run_dir / "metric-cache"
        command = [sys.executable, "/opt/navsim/navsim/planning/script/run_metric_caching.py", *overrides,
                   "cache.cache_path=" + str(cache_dir)]
        result["commands"] = [command]
        run_logged("caching", command, token, run_dir, result["durations"])
        hybrid = build_hybrid_agent(fast)
        started = time.monotonic()
        hybrid.initialize()
        result["durations"]["initialization_seconds"] = time.monotonic() - started
        result["model"] = fast.provenance()
        agent_input = loader.get_agent_input_from_token(token)
        started = time.monotonic()
        result["hybrid_calls_attempted"] += 1
        checked = hybrid.compute_trajectory(agent_input)
        elapsed = time.monotonic() - started
        result["durations"]["neural_and_arbitration_seconds"] = elapsed
        decision = hybrid.last_decision
        if decision is None:
            raise ValueError("Hybrid produced no decision")
        write_decision(run_dir, token, decision, checked)
        raw = decision.selection.plan.candidates[0].trajectory
        result["fast_candidates_observed"] = len(decision.selection.plan.candidates)
        result["raw_candidate_reused_for_ablation"] = True
        predictions["baseline"] = to_navsim(raw)
        predictions["checks_only"] = checked
        # Privileged future labels appear only after the agent has produced its output.
        reference = loader.get_scene_from_token(token).get_future_trajectory(num_trajectory_frames=raw.num_poses)
        cv = ConstantVelocityAgent()
        cv.initialize()
        started = time.monotonic()
        predictions["cv_reference"] = cv.compute_trajectory(agent_input)
        cv_elapsed = time.monotonic() - started
        manifest = make_manifest(
            revision=revision, sources=sources, assets=assets, model=fast.provenance(), scene_token=token,
            log_name=log_name, seed=seed, policy=sample_policy(), candidate=raw,
            split_audit=audit_official_splits(smoke_token=token, smoke_log=log_name),
        )
        write_json(run_dir / "experiment-manifest.json", manifest)
        result["manifest_hash"] = manifest["manifest_hash"]
        failures = {}
        if decision.failed:
            fallback = decision.selection.fallback
            failures["checks_only"] = fallback.failure_reason.value if fallback else "arbitration_failed"
        latencies = {"checks_only": elapsed, "cv_reference": cv_elapsed}
        score = lambda value: score_hybrid_prediction(value, token, cache_dir, overrides)
        result["records"] = evaluate_predictions(
            predictions, reference, token, score, failures=failures, latencies=latencies,
            checked_status=decision.selection.status.value,
        )
        score_dir = run_dir / "scores"
        score_dir.mkdir()
        for row in result["records"]:
            write_score_rows(score_dir / (row["mode"] + ".csv"), token, row["metrics"])
        write_comparison_csv(run_dir / "comparison.csv", result["records"])
        result["summary"] = summarize(result["records"], expected_tokens=[token], expected_modes=MODES)
        render_comparison(run_dir, predictions, reference, result["records"])
        if any(row["failed"] for row in result["records"]):
            raise ValueError("At least one requested mode failed; full records retained")
        result["status"] = "success"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        (run_dir / "failure.log").write_text(traceback.format_exc())
        if not result["records"]:
            result["records"] = [scene_result(mode=mode, token=token, failed=True, error=result["error"])
                                 for mode in MODES]
            result["summary"] = summarize(result["records"], expected_tokens=[token], expected_modes=MODES)
            write_comparison_csv(run_dir / "comparison.csv", result["records"])
        raise
    finally:
        result["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        write_json(run_dir / "result.json", result)
        print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    return result


def evaluate_predictions(predictions, reference, token, score, *, failures=None, latencies=None,
                         checked_status="candidate"):
    from hybrid_drive.adapters.navsim import from_navsim

    failures, latencies = failures or {}, latencies or {}
    records = []
    for mode in MODES:
        started = time.monotonic()
        try:
            if mode in failures:
                raise ValueError(failures[mode])
            prediction = predictions[mode]
            metrics = score(prediction)
            record = scene_result(
                mode=mode, token=token, metrics=metrics, prediction=from_navsim(prediction),
                reference=from_navsim(reference), status=checked_status if mode == "checks_only" else "candidate",
                latency_s=latencies.get(mode),
            )
        except Exception as exc:
            record = scene_result(mode=mode, token=token, failed=True, error=f"{type(exc).__name__}: {exc}",
                                  status="failure", latency_s=latencies.get(mode))
        record["scoring_seconds"] = time.monotonic() - started
        if mode in predictions:
            record["native_trajectory"] = {
                "poses": predictions[mode].poses.tolist(),
                "interval_s": predictions[mode].trajectory_sampling.interval_length,
            }
        records.append(record)
    return records


def write_comparison_csv(path, records):
    fields = ["mode", "scene_token", "failed", "status", "error", "ade_m", "fde_m", *METRICS]
    with Path(path).open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record[key] for key in fields if key not in METRICS} | (record["metrics"] or {}))


def render_comparison(run_dir, predictions, reference, records):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 6))
    for record in records:
        if record["failed"]:
            continue
        mode = record["mode"]
        poses = predictions[mode].poses
        ax.plot(poses[:, 0], poses[:, 1], "o-", label=f'{mode} · PDMS {record["metrics"]["score"]:.3f}')
    ax.plot(reference.poses[:, 0], reference.poses[:, 1], "k--", label="Recorded human trajectory")
    ax.scatter([0], [0], color="black", label="Ego at t=0")
    failures = [row["mode"] for row in records if row["failed"]]
    title = "One real scene · same neural Fast candidate before/after Safety"
    if failures:
        title += "\nFailed (not scored): " + ", ".join(failures)
    ax.set(xlabel="Ego x, m", ylabel="Ego y, m", title=title)
    ax.axis("equal")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "trajectory.png", dpi=150)
    plt.close(fig)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--revision", default=os.environ.get("HYBRID_DRIVE_REVISION"), required=False)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.revision is None:
        parser.error("--revision must identify the exact tested source commit")
    try:
        run_neural_sample(args.data_root, args.asset_root, args.output_root, revision=args.revision, seed=args.seed)
    except Exception as exc:
        print(f"Neural sample failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
