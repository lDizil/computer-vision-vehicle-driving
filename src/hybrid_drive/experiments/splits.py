"""Log/recording-level split policy, never a random split of adjacent frames."""

import hashlib
import math
from pathlib import Path
import re


SPLIT_SOURCES = {
    "navtrain.yaml": "fcfc95704d532849c7fe1f1ac2bc9e158305fef93a1d5340637ad4b9f440c8cb",
    "navtest.yaml": "f72568e6282e413474c91b69acd889ecb9affc36f3571a25cf10a44c8d3d1804",
    "scene_filter/navtrain.yaml": "c37fea567a0cfdbc29076cca893d4f5dd32db59baec18ae214527206d6b64e6f",
    "scene_filter/navtest.yaml": "61284edf5003c0291f843ce9817c822ba306609a62d54544223adae3fc7fc9cd",
}


def recording_group(log_name):
    match = re.fullmatch(r"(.+_veh-\d+)_\d+_\d+", log_name)
    return match.group(1) if match else log_name


def split_training_logs(logs, *, validation_fraction=0.2, seed=0):
    logs = tuple(logs)
    if not logs or any(not isinstance(log, str) or not log for log in logs) or len(set(logs)) != len(logs):
        raise ValueError("Training logs must be nonempty and unique")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("Split seed must be an integer in [0, 2**32)")
    if isinstance(validation_fraction, bool) or not 0 < validation_fraction < 1:
        raise ValueError("Validation fraction must be between zero and one")
    groups = {recording_group(log) for log in logs}
    ordered = sorted(groups, key=lambda group: hashlib.sha256(
        f"hybrid-drive-navtrain-val-v1:{seed}:{group}".encode(),
    ).hexdigest())
    count = math.ceil(len(groups) * validation_fraction)
    if count >= len(groups):
        raise ValueError("At least one independent recording is required in each role")
    validation = set(ordered[:count])
    return {
        "train": sorted(log for log in logs if recording_group(log) not in validation),
        "validation": sorted(log for log in logs if recording_group(log) in validation),
    }


def audit_official_splits(*, smoke_token, smoke_log,
                          root=Path("/opt/navsim/navsim/planning/script/config/common/train_test_split")):
    import yaml

    configs = {}
    for relative, expected in SPLIT_SOURCES.items():
        payload = (Path(root) / relative).read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError(f"Official NAVSIM split source changed: {relative}")
        configs[relative] = yaml.safe_load(payload)
    train = _partition(configs["scene_filter/navtrain.yaml"])
    test = _partition(configs["scene_filter/navtest.yaml"])
    for field in ("logs", "tokens", "recordings"):
        if train[field] & test[field]:
            raise ValueError(f"Official train/test overlap in {field}")
    assigned = split_training_logs(train["logs"])
    if (smoke_token in train["tokens"]) != (smoke_log in train["logs"]):
        raise ValueError("Smoke token/log disagree on official train membership")
    if (smoke_token in test["tokens"]) != (smoke_log in test["logs"]):
        raise ValueError("Smoke token/log disagree on official test membership")
    return {
        "policy": "hybrid-drive-navtrain-val-v1", "seed": 0, "validation_fraction_recordings": 0.2,
        "source_sha256": SPLIT_SOURCES,
        "official_navtrain": {key: len(value) for key, value in train.items()},
        "official_navtest": {key: len(value) for key, value in test.items()},
        "official_overlap": {key: 0 for key in train},
        "own_log_roles": {key: {
            "logs": values, "recordings": len({recording_group(log) for log in values}),
            "log_manifest_sha256": hashlib.sha256(("\n".join(values) + "\n").encode()).hexdigest(),
        } for key, values in assigned.items()},
        "smoke_scene": {"token": smoke_token, "log_name": smoke_log, "role": "smoke",
                        "in_official_navtrain": smoke_token in train["tokens"],
                        "in_official_navtest": smoke_token in test["tokens"]},
        "token_counts_per_own_role": "Not materialized; token-to-log metadata for the full split is not loaded",
        "pretrained_checkpoint_training_overlap": "Unknown; smoke is not a held-out quality measurement",
        "test_usage": "No threshold/prompt/training tuning on navtest",
    }


def _partition(config):
    logs, tokens = config["log_names"], config["tokens"]
    for name, values in (("log_names", logs), ("tokens", tokens)):
        if not isinstance(values, list) or not values or any(not isinstance(value, str) or not value for value in values):
            raise ValueError(f"Invalid official split {name}")
        if len(set(values)) != len(values):
            raise ValueError(f"Duplicate official split {name}")
    return {"logs": set(logs), "tokens": set(tokens), "recordings": {recording_group(log) for log in logs}}
