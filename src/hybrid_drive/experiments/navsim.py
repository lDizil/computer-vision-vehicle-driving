"""Shared pinned NAVSIM loading and actual PDM evaluation; no model imports on import."""

from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys


def repository_sources(name):
    return Path(__file__).resolve().parents[3] / "scripts" / name


def configure_environment(data_root, run_dir, *, verify_maps=False):
    from hybrid_drive.assets.files import destination
    from scripts.prepare_navsim_sample import validate_sources, verify_file

    data_root = Path(data_root).resolve()
    sources = json.loads(repository_sources("navsim-sample-sources.json").read_text())
    validate_sources(data_root, sources)
    verify_file(destination(data_root, sources["log"]["path"]), sources["log"])
    maps_root = data_root / "maps"
    if verify_maps:
        from scripts.sample_support import writable_map_workspace

        for spec in sources["maps"]["files"]:
            verify_file(destination(data_root, spec["path"]), spec)
        maps_root = writable_map_workspace(data_root, sources["maps"]["files"])
    os.environ.update({
        "OPENSCENE_DATA_ROOT": str(data_root), "NUPLAN_MAPS_ROOT": str(maps_root),
        "NUPLAN_MAP_VERSION": "nuplan-maps-v1.0", "NUPLAN_DATA_STORE": "local",
        "NAVSIM_EXP_ROOT": str(run_dir),
    })
    return sources


def cohort_overrides(sources, tokens):
    tokens = tuple(tokens)
    if not tokens or len(set(tokens)) != len(tokens):
        raise ValueError("Selected scene tokens must be nonempty and unique")
    return [
        "train_test_split=mini", "worker=sequential", "gpu=false",
        "train_test_split.scene_filter.frame_interval=1",
        "train_test_split.scene_filter.max_scenes=" + str(len(tokens)),
        "train_test_split.scene_filter.num_history_frames=4",
        "train_test_split.scene_filter.num_future_frames=10",
        "train_test_split.scene_filter.log_names=" + json.dumps([Path(sources["log"]["path"]).stem]),
        "train_test_split.scene_filter.tokens=" + json.dumps(tokens),
    ]


def load_cohort_inputs(sources, tokens, *, sensor_root=None, sensor_config=None):
    from hydra import compose, initialize_config_module
    from hydra.utils import instantiate
    from navsim.common.dataloader import SceneLoader
    from navsim.common.dataclasses import SensorConfig

    tokens = tuple(tokens)
    overrides = cohort_overrides(sources, tokens)
    with initialize_config_module(config_module="navsim.planning.script.config.metric_caching", version_base=None):
        cfg = compose(config_name="default_metric_caching", overrides=overrides)
    loader = SceneLoader(
        data_path=Path(cfg.navsim_log_path), sensor_blobs_path=sensor_root,
        scene_filter=instantiate(cfg.train_test_split.scene_filter),
        sensor_config=SensorConfig.build_no_sensors() if sensor_config is None else sensor_config,
    )
    if len(loader.tokens) != len(tokens) or set(loader.tokens) != set(tokens):
        raise ValueError("NAVSIM did not select exactly the fixed cohort tokens")
    return loader, overrides


def cache_cohort(cache_dir, overrides, log_path):
    command = [
        sys.executable, "/opt/navsim/navsim/planning/script/run_metric_caching.py", *overrides,
        "cache.cache_path=" + str(cache_dir),
    ]
    with Path(log_path).open("w") as output:
        subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, check=True)
    return command


def sample_arbiter():
    from hybrid_drive.safety.arbiter import SafetyArbiter
    from hybrid_drive.safety.fallback import BrakingConfig
    from hybrid_drive.safety.kinematics import KinematicLimits
    from scripts.run_navsim_sample import sample_policy

    policy = sample_policy()
    return SafetyArbiter(KinematicLimits(**policy["limits"]), BrakingConfig(**policy["braking"]))


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
