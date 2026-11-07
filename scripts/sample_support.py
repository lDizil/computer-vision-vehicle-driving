"""Shared verified-data setup for the CV and neural single-scene runners."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile


def prepare_sample_paths(data_root, output_root):
    from scripts.prepare_navsim_sample import SOURCES_PATH, destination, validate_sources, verify_file

    data_root = Path(data_root).resolve()
    sources = json.loads(SOURCES_PATH.read_text())
    validate_sources(data_root, sources)
    for spec in [sources["log"], *sources["maps"]["files"]]:
        verify_file(destination(data_root, spec["path"]), spec)
    run_dir = Path(output_root).resolve() / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run_dir.mkdir(parents=True)
    maps_runtime = writable_map_workspace(data_root, sources["maps"]["files"])
    # NAVSIM reads some of these at module import time, before any SceneLoader.
    os.environ.update({
        "OPENSCENE_DATA_ROOT": str(data_root), "NUPLAN_MAPS_ROOT": str(maps_runtime),
        "NUPLAN_MAP_VERSION": "nuplan-maps-v1.0", "NUPLAN_DATA_STORE": "local",
        "NAVSIM_EXP_ROOT": str(run_dir),
    })
    return sources, run_dir


def writable_map_workspace(data_root, specs):
    from scripts.prepare_navsim_sample import destination, safe_relative

    # nuPlan creates .maplocks below map_root, even when reading vector maps.
    # Keep the verified source read-only; only links/locks live in this process's
    # disposable container filesystem, outside uploaded experiment artifacts.
    workspace = Path(tempfile.mkdtemp(prefix="hybrid-navsim-maps-"))
    for spec in specs:
        relative = safe_relative(spec["path"]).relative_to("maps")
        source = destination(Path(data_root).resolve(), spec["path"])
        target = workspace.joinpath(*relative.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(source)
    return workspace


def load_sample(sources, token=None, *, sensor_config=None, sensor_root=None):
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
        data_path=Path(cfg.navsim_log_path), sensor_blobs_path=sensor_root,
        scene_filter=instantiate(cfg.train_test_split.scene_filter),
        sensor_config=SensorConfig.build_no_sensors() if sensor_config is None else sensor_config,
    )
    if len(loader.tokens) != 1:
        raise ValueError("Expected exactly one eligible scene in the verified sample")
    selected = loader.tokens[0]
    if token and selected != token:
        raise ValueError("Requested scene token was not selected")
    if not token:
        overrides.append("train_test_split.scene_filter.tokens=" + json.dumps([selected]))
    return loader, selected, overrides
