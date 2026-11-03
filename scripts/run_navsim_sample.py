"""Run the upstream CPU evaluator on one verified OpenScene scene."""

import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


def run_sample(data_root, output_root, token=None):
    from scripts.prepare_navsim_sample import SOURCES_PATH, destination, validate_sources, verify_file

    data_root = Path(data_root).resolve()
    sources = json.loads(SOURCES_PATH.read_text())
    validate_sources(data_root, sources)
    for spec in [sources["log"], *sources["maps"]["files"]]:
        verify_file(destination(data_root, spec["path"]), spec)

    run_dir = Path(output_root).resolve() / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run_dir.mkdir(parents=True)
    environment = {
        "OPENSCENE_DATA_ROOT": str(data_root),
        "NUPLAN_MAPS_ROOT": str(data_root / "maps"),
        "NUPLAN_MAP_VERSION": "nuplan-maps-v1.0",
        "NUPLAN_DATA_STORE": "local",
        "NAVSIM_EXP_ROOT": str(run_dir),
    }
    os.environ.update(environment)

    from hydra import compose, initialize_config_module
    from hydra.utils import instantiate
    from navsim.common.dataloader import SceneLoader
    from navsim.common.dataclasses import SensorConfig

    log_name = Path(sources["log"]["path"]).stem
    overrides = [
        "train_test_split=mini", "worker=sequential", "gpu=false",
        "train_test_split.scene_filter.max_scenes=1",
        "train_test_split.scene_filter.log_names=" + json.dumps([log_name]),
    ]
    if token:
        overrides.append("train_test_split.scene_filter.tokens=" + json.dumps([token]))
    with initialize_config_module(config_module="navsim.planning.script.config.metric_caching", version_base=None):
        cfg = compose(config_name="default_metric_caching", overrides=overrides)
    loader = SceneLoader(
        data_path=Path(cfg.navsim_log_path), sensor_blobs_path=None,
        scene_filter=instantiate(cfg.train_test_split.scene_filter),
        sensor_config=SensorConfig.build_no_sensors(),
    )
    if len(loader.tokens) != 1:
        raise ValueError("Expected exactly one eligible scene in the verified sample")
    selected_token = loader.tokens[0]
    if token and selected_token != token:
        raise ValueError("Requested scene token was not selected")
    if not token:
        overrides.append("train_test_split.scene_filter.tokens=" + json.dumps([selected_token]))

    cache_dir = run_dir / "metric-cache"
    score_dir = run_dir / "scores"
    source_root = Path("/opt/navsim/navsim/planning/script")
    commands = [
        [sys.executable, str(source_root / "run_metric_caching.py"), *overrides,
         "cache.cache_path=" + str(cache_dir)],
        [sys.executable, str(source_root / "run_pdm_score.py"), *overrides,
         "agent=constant_velocity_agent", "metric_cache_path=" + str(cache_dir),
         "experiment_name=constant-velocity-sample", "output_dir=" + str(score_dir)],
    ]
    durations = {}
    for name, command in zip(("caching", "scoring"), commands):
        print(f"Running {name} for scene {selected_token}", flush=True)
        started = time.monotonic()
        with (run_dir / (name + ".log")).open("w") as output:
            subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, check=True)
        durations[name + "_seconds"] = time.monotonic() - started
    csv_paths = list(score_dir.glob("*.csv"))
    if len(csv_paths) != 1:
        raise ValueError("Evaluator must produce exactly one CSV in the fresh score directory")
    metrics = validate_scores(csv_paths[0], selected_token)
    trajectory = render_trajectory(loader, selected_token, run_dir, metrics["score"])
    result = {
        "scene_token": selected_token, "log_name": log_name,
        "agent": "ConstantVelocityAgent", "scenarios": 1,
        "navsim_revision": sources["navsim_revision"],
        "nuplan_revision": sources["nuplan_revision"],
        "cpu_only": True, "sensor_blobs_used": False,
        "platform": platform.machine(), "python_version": platform.python_version(),
        "metrics": metrics, "trajectory": trajectory,
        "durations": durations, "commands": commands,
        "csv": str(csv_paths[0]), "run_directory": str(run_dir),
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return result


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


def render_trajectory(loader, token, run_dir, score):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from navsim.agents.constant_velocity_agent import ConstantVelocityAgent

    agent = ConstantVelocityAgent()
    agent.initialize()
    prediction = agent.compute_trajectory(loader.get_agent_input_from_token(token))
    scene = loader.get_scene_from_token(token)
    reference = scene.get_future_trajectory(num_trajectory_frames=len(prediction.poses))
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(prediction.poses[:, 0], prediction.poses[:, 1], "o-", label="Constant velocity")
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
    args = parser.parse_args()
    try:
        run_sample(args.data_root, args.output_root, args.token)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"Sample run failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
