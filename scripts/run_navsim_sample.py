"""Run the upstream CPU evaluator on one verified OpenScene scene."""

import argparse
import csv
from dataclasses import asdict
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback


def run_sample(data_root, output_root, token=None, agent="cv"):
    if agent not in ("cv", "hybrid"):
        raise ValueError("Unknown sample agent")
    from scripts.sample_support import load_sample, prepare_sample_paths

    sources, run_dir = prepare_sample_paths(data_root, output_root)
    log_name = Path(sources["log"]["path"]).stem
    loader, selected_token, overrides = load_sample(sources, token)

    cache_dir = run_dir / "metric-cache"
    score_dir = run_dir / "scores"
    source_root = Path("/opt/navsim/navsim/planning/script")
    caching_command = [
        sys.executable, str(source_root / "run_metric_caching.py"), *overrides,
        "cache.cache_path=" + str(cache_dir),
    ]
    commands = [caching_command]
    result = {
        "scene_token": selected_token, "log_name": log_name,
        "agent": "HybridDriveAgent" if agent == "hybrid" else "ConstantVelocityAgent", "scenarios": 1,
        "navsim_revision": sources["navsim_revision"],
        "nuplan_revision": sources["nuplan_revision"],
        "cpu_only": True, "sensor_blobs_used": False,
        "platform": platform.machine(), "python_version": platform.python_version(),
        "status": "failed", "metrics": None, "trajectory": None,
        "durations": {}, "commands": commands, "run_directory": str(run_dir),
    }
    try:
        run_logged("caching", caching_command, selected_token, run_dir, result["durations"])
        prediction = None
        if agent == "hybrid":
            started = time.monotonic()
            hybrid = build_hybrid_agent()
            hybrid.initialize()
            score_dir.mkdir()
            csv_path = score_dir / "hybrid.csv"
            result["csv"] = str(csv_path)
            prediction = compute_hybrid_prediction(
                hybrid, loader.get_agent_input_from_token(selected_token), selected_token, run_dir, csv_path,
            )
            metrics = score_hybrid_prediction(prediction, selected_token, cache_dir, overrides)
            write_score_rows(csv_path, selected_token, metrics)
            result["durations"]["scoring_seconds"] = time.monotonic() - started
        else:
            scoring_command = [
                sys.executable, str(source_root / "run_pdm_score.py"), *overrides,
                "agent=constant_velocity_agent", "metric_cache_path=" + str(cache_dir),
                "experiment_name=constant-velocity-sample", "output_dir=" + str(score_dir),
            ]
            commands.append(scoring_command)
            run_logged("scoring", scoring_command, selected_token, run_dir, result["durations"])
        csv_paths = list(score_dir.glob("*.csv"))
        if len(csv_paths) != 1:
            raise ValueError("Evaluator must produce exactly one CSV in the fresh score directory")
        result["csv"] = str(csv_paths[0])
        result["metrics"] = validate_scores(csv_paths[0], selected_token)
        result["trajectory"] = render_trajectory(
            loader, selected_token, run_dir, result["metrics"]["score"], prediction, result["agent"],
        )
        result["status"] = "success"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        (run_dir / "failure.log").write_text(traceback.format_exc())
        if agent == "hybrid" and result.get("csv") and not Path(result["csv"]).exists():
            write_score_rows(result["csv"], selected_token, None)
        if agent == "hybrid" and not (run_dir / "decision.json").exists():
            write_decision(run_dir, selected_token, None, None, error=result["error"])
        raise
    finally:
        (run_dir / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    return result


def run_logged(name, command, token, run_dir, durations):
    print(f"Running {name} for scene {token}", flush=True)
    started = time.monotonic()
    with (run_dir / (name + ".log")).open("w") as output:
        subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, check=True)
    durations[name + "_seconds"] = time.monotonic() - started


def build_hybrid_agent(fast_agent=None):
    from navsim.agents.constant_velocity_agent import ConstantVelocityAgent
    from hybrid_drive.agents.hybrid import HybridDriveAgent
    from hybrid_drive.safety.fallback import BrakingConfig
    from hybrid_drive.safety.kinematics import KinematicLimits

    policy = sample_policy()
    return HybridDriveAgent(
        fast_agent=ConstantVelocityAgent() if fast_agent is None else fast_agent,
        limits=KinematicLimits(**policy["limits"]),
        braking=BrakingConfig(**policy["braking"]),
    )


def sample_policy():
    return {
        "limits": {"interval_s": 0.5, "max_speed_mps": 20, "max_acceleration_mps2": 8, "max_jerk_mps3": 20},
        "braking": {"deceleration_mps2": 4, "jerk_mps3": 4},
        "safety_calibrated": False,
    }


def compute_hybrid_prediction(agent, agent_input, token, run_dir, csv_path):
    prediction = None
    try:
        prediction = agent.compute_trajectory(agent_input)
        decision = agent.last_decision
        if decision is None:
            raise ValueError("Hybrid agent returned no arbitration decision")
        write_decision(run_dir, token, decision, prediction)
        if decision.failed:
            raise ValueError("Hybrid arbitration failed; native placeholder is not a successful scene")
        return prediction
    except Exception as exc:
        write_decision(run_dir, token, agent.last_decision, prediction, error=f"{type(exc).__name__}: {exc}")
        write_score_rows(csv_path, token, None)
        raise


def write_decision(run_dir, token, decision, prediction, error=None):
    evidence = {
        "scene_token": token, "failed": True, "selection_status": None, "candidate_id": None,
        "slow_status": None, "slow_disposition": None, "fallback_reason": None,
        "candidate_kinematics": [], "native_trajectory": None,
        "policy": sample_policy(),
    }
    if prediction is not None:
        evidence["native_trajectory"] = {
            "poses": prediction.poses.tolist(),
            "interval_seconds": prediction.trajectory_sampling.interval_length,
        }
    if decision is not None:
        selection = decision.selection
        evidence.update(
            failed=decision.failed, selection_status=selection.status.value, candidate_id=selection.candidate_id,
            slow_status=decision.slow.status.value, slow_disposition=decision.slow.disposition.value,
            candidate_kinematics=[{
                "candidate_id": item.candidate_id, "vetoed": item.vetoed,
                **kinematic_evidence(item.report),
            } for item in selection.assessments],
        )
        fallback = selection.fallback
        if fallback is not None:
            evidence["fallback_reason"] = fallback.failure_reason.value if fallback.failure_reason else "no_eligible_candidate"
            evidence["fallback"] = {
                "kind": fallback.kind.value, "succeeded": fallback.succeeded,
                "failure_reason": fallback.failure_reason.value if fallback.failure_reason else None,
                "stop_time_s": fallback.stop_time_s, "stop_distance_m": fallback.stop_distance_m,
                "peak_speed_mps": fallback.peak_speed_mps, "kinematics": kinematic_evidence(fallback.kinematics),
            }
    if error is not None:
        evidence["error"] = error
        evidence["failed"] = True
    (Path(run_dir) / "decision.json").write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n")


def kinematic_evidence(report):
    return {
        "within_limits": report.within_limits, "violations": [value.value for value in report.violations],
        "max_speed_mps": report.max_speed_mps, "max_acceleration_mps2": report.max_acceleration_mps2,
        "max_jerk_mps3": report.max_jerk_mps3,
    }


def score_hybrid_prediction(prediction, token, cache_dir, overrides):
    from hydra import compose, initialize_config_module
    from hydra.utils import instantiate
    from navsim.common.dataloader import MetricCacheLoader
    from navsim.evaluate.pdm_score import pdm_score

    with initialize_config_module(config_module="navsim.planning.script.config.pdm_scoring", version_base=None):
        cfg = compose(config_name="default_run_pdm_score", overrides=overrides)
    simulator = instantiate(cfg.simulator)
    scorer = instantiate(cfg.scorer)
    if simulator.proposal_sampling != scorer.proposal_sampling:
        raise ValueError("Simulator and scorer proposal sampling differ")
    cache = MetricCacheLoader(cache_dir).get_from_token(token)
    return asdict(pdm_score(
        metric_cache=cache, model_trajectory=prediction, future_sampling=simulator.proposal_sampling,
        simulator=simulator, scorer=scorer,
    ))


def write_score_rows(path, token, metrics):
    metrics = {} if metrics is None else metrics
    valid = bool(metrics)
    fields = ["token", "valid", *metrics] if valid else ["token", "valid", "score"]
    with Path(path).open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for row_token in (token, "average"):
            writer.writerow({"token": row_token, "valid": valid, **metrics})


def validate_scores(path, expected_token):
    with Path(path).open(newline="") as source:
        rows = list(csv.DictReader(source))
    scenarios = [row for row in rows if row.get("token") != "average"]
    averages = [row for row in rows if row.get("token") == "average"]
    if len(scenarios) != 1 or len(averages) != 1 or scenarios[0].get("token") != expected_token:
        raise ValueError("CSV does not contain exactly the requested scene and one average")
    parsed = []
    for row in (scenarios[0], averages[0]):
        if row.get("valid", "").strip().lower() != "true":
            raise ValueError("Evaluator reported an invalid scenario")
        metrics = {}
        for key, value in row.items():
            if key in ("", "token", "valid"):
                continue
            number = float(value)
            if not math.isfinite(number):
                raise ValueError(f"Non-finite metric: {key}")
            metrics[key] = number
        score = metrics.get("score", math.nan)
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Evaluator score must be finite and in [0, 1]")
        parsed.append(metrics)
    if parsed[0].keys() != parsed[1].keys() or any(
        not math.isclose(value, parsed[1][key], rel_tol=1e-12, abs_tol=1e-12)
        for key, value in parsed[0].items()
    ):
        raise ValueError("Single-scene averages do not match the scene metrics")
    return parsed[0]


def render_trajectory(loader, token, run_dir, score, prediction=None, agent_name="ConstantVelocityAgent"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from navsim.agents.constant_velocity_agent import ConstantVelocityAgent

    if prediction is None:
        agent = ConstantVelocityAgent()
        agent.initialize()
        prediction = agent.compute_trajectory(loader.get_agent_input_from_token(token))
    scene = loader.get_scene_from_token(token)
    reference = scene.get_future_trajectory(num_trajectory_frames=len(prediction.poses))
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(prediction.poses[:, 0], prediction.poses[:, 1], "o-", label=agent_name)
    ax.plot(reference.poses[:, 0], reference.poses[:, 1], "o--", label="Recorded human trajectory")
    ax.scatter([0], [0], color="black", label="Ego at t=0")
    ax.set(xlabel="Ego x, m", ylabel="Ego y, m", title=f"Real OpenScene scene · PDMS {score:.4f}")
    ax.axis("equal")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "trajectory.png", dpi=150)
    plt.close(fig)
    return {"poses": prediction.poses.tolist(), "reference_poses": reference.poses.tolist(),
            "interval_seconds": prediction.trajectory_sampling.interval_length}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--token")
    parser.add_argument("--agent", choices=("cv", "hybrid"), default="cv")
    args = parser.parse_args()
    try:
        run_sample(args.data_root, args.output_root, args.token, args.agent)
    except Exception as exc:
        print(f"Sample run failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
