"""Offline comparison rules; failures remain in the attempted-scene denominator."""

from collections import defaultdict
import hashlib
import json
import math
from numbers import Real
import re

import numpy as np

from hybrid_drive.contracts.trajectory import Trajectory
from .splits import recording_group


METRICS = (
    "no_at_fault_collisions", "drivable_area_compliance", "ego_progress",
    "time_to_collision_within_bound", "comfort", "driving_direction_compliance", "score",
)
ABLATIONS = {
    "baseline": "Raw Fast output; mandatory format/finite checks, no extra risk checks or Slow.",
    "checks_only": "Same Fast candidate; deterministic checks and shared braking fallback, no Slow.",
    "slow_only": "Same Fast candidate; applicable structured Slow veto and shared fallback; no extra risk checks.",
    "combined": "Same Fast candidate; deterministic checks, applicable Slow veto and shared fallback.",
}


def content_hash(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def trajectory_hash(value: Trajectory):
    return content_hash({"interval_s": value.interval_s, "poses": value.poses.tolist()})


def asset_identity(assets):
    # Transfer counters and cache-hit flags are operational history, not inputs
    # to the model. Re-preparing identical verified files must remain pairable.
    return {
        "sources": assets["sources"],
        "models": [{key: record[key] for key in ("path", "size_bytes", "sha256")}
                   for record in assets["models"]],
        "cameras": [{key: record[key] for key in (
            "camera", "member", "path", "size_bytes", "sha256", "width", "height", "format", "mode",
        )} for record in assets["cameras"]["files"]],
    }


def make_manifest(*, revision, sources, assets, model, scene_token, log_name, seed, policy, candidate,
                  split_audit=None):
    for key, value in (("revision", revision), ("navsim_revision", sources["navsim_revision"]),
                       ("nuplan_revision", sources["nuplan_revision"])):
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{40}", value):
            raise ValueError(f"{key} must identify the exact source commit")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if not scene_token or not log_name:
        raise ValueError("Scene and log identity are required")
    manifest = {
        "schema": "hybrid-drive-experiment-v1", "source_revision": revision,
        "navsim_revision": sources["navsim_revision"], "nuplan_revision": sources["nuplan_revision"],
        "dataset_source_hash": content_hash(sources), "assets_manifest_hash": content_hash(asset_identity(assets)),
        "model": model, "scene_token": scene_token, "log_name": log_name,
        "split": "openscene-mini-debug-selection", "role": "smoke", "seed": seed,
        "sampling": {"num_poses": candidate.num_poses, "interval_s": candidate.interval_s},
        "raw_candidate_hash": trajectory_hash(candidate), "policy": policy,
        "ablations": ABLATIONS, "executed_ablations": ["baseline", "checks_only"],
        "unavailable_ablations": {"slow_only": "Slow model not connected", "combined": "Slow model not connected"},
        "pretrained_training_overlap": "unknown",
        "split_audit": split_audit,
        "metric_protocol": {
            "driving": "NAVSIM v1.1 PDMS (not EPDMS)", "time_grid": "future dt, 2dt, ..., Ndt",
            "coordinates": "rear-axle ego x forward / y left; metres, heading radians",
            "ade_m": "mean future XY Euclidean error; heading excluded",
            "fde_m": "last future XY Euclidean error; heading excluded",
        },
        "quality_claim": "Single debug scene; not a held-out quality or collision-reduction study",
    }
    # Snapshot nested mutable inputs; serialization also rejects NaN/Infinity.
    manifest = json.loads(json.dumps(manifest, allow_nan=False))
    manifest["manifest_hash"] = content_hash(manifest)
    return manifest


def verify_manifest(manifest):
    copy = dict(manifest)
    expected = copy.pop("manifest_hash", None)
    if expected != content_hash(copy):
        raise ValueError("Experiment manifest content changed")


def require_paired(left, right):
    verify_manifest(left)
    verify_manifest(right)
    for key in ("source_revision", "navsim_revision", "nuplan_revision", "dataset_source_hash",
                "assets_manifest_hash", "model", "scene_token", "log_name", "split", "role",
                "seed", "sampling", "raw_candidate_hash", "policy", "split_audit", "metric_protocol"):
        if left[key] != right[key]:
            raise ValueError(f"Not a paired ablation: {key} differs")


def validate_splits(assignments):
    """All windows from a log belong to one role; repeated source frames cannot cross roles."""
    log_roles, frame_roles, scene_roles = {}, {}, {}
    counts = defaultdict(int)
    for role, scenes in assignments.items():
        if role not in ("train", "validation", "test", "smoke"):
            raise ValueError(f"Unknown split role: {role}")
        for scene in scenes:
            token, log = scene["scene_token"], scene["log_name"]
            if not isinstance(token, str) or not token or not isinstance(log, str) or not log:
                raise ValueError("Every scene needs nonempty token and log identity")
            if token in scene_roles:
                raise ValueError(f"Scene appears more than once: {token}")
            scene_roles[token] = role
            recording = recording_group(log)
            if recording in log_roles and log_roles[recording] != role:
                raise ValueError(f"Log crosses split roles: {log}")
            log_roles[recording] = role
            for frame in scene.get("source_tokens", ()):
                if not isinstance(frame, str) or not frame:
                    raise ValueError("Source frame identifiers must be nonempty strings")
                if frame in frame_roles and frame_roles[frame] != role:
                    raise ValueError(f"Source frame crosses split roles: {frame}")
                frame_roles[frame] = role
            counts[role] += 1
    return dict(counts)


def trajectory_errors(prediction: Trajectory, reference: Trajectory):
    if prediction.num_poses != reference.num_poses or not math.isclose(
        prediction.interval_s, reference.interval_s, rel_tol=1e-12, abs_tol=0,
    ):
        raise ValueError("ADE/FDE require the same future time grid; resample explicitly")
    with np.errstate(over="ignore", invalid="ignore"):
        deltas = prediction.poses[:, :2] - reference.poses[:, :2]
        distances = np.hypot(deltas[:, 0], deltas[:, 1])
        ade, fde = float(np.mean(distances)), float(distances[-1])
    if not math.isfinite(ade) or not math.isfinite(fde):
        raise ValueError("Trajectory error exceeds the numeric range")
    return {"ade_m": ade, "fde_m": fde}


def scene_result(*, mode, token, metrics=None, prediction=None, reference=None,
                 failed=False, error=None, status="candidate", latency_s=None, all_veto=False):
    if mode not in (*ABLATIONS, "cv_reference") or not isinstance(token, str) or not token:
        raise ValueError("Unknown experiment mode or missing scene token")
    if not isinstance(failed, bool) or not isinstance(all_veto, bool):
        raise ValueError("Failure/veto flags must be booleans")
    if latency_s is not None:
        _number(latency_s, "latency_s", minimum=0)
    result = {"mode": mode, "scene_token": token, "failed": failed, "error": error,
              "status": status, "all_veto": all_veto, "latency_s": latency_s,
              "metrics": None, "ade_m": None, "fde_m": None}
    if failed:
        if not error:
            raise ValueError("A failed scene must retain its cause")
        return result
    if metrics is None or prediction is None or reference is None:
        raise ValueError("Successful scene needs metrics and evaluated/reference trajectories")
    result["metrics"] = {key: _number(metrics[key], key, minimum=0, maximum=1) for key in METRICS}
    result.update(trajectory_errors(prediction, reference))
    return result


def summarize(records, *, expected_tokens, expected_modes):
    tokens, modes = tuple(expected_tokens), tuple(expected_modes)
    if not tokens or not modes or len(set(tokens)) != len(tokens) or len(set(modes)) != len(modes):
        raise ValueError("Expected tokens and modes must be nonempty and unique")
    grouped = defaultdict(dict)
    for record in records:
        mode, token = record["mode"], record["scene_token"]
        if mode not in modes or token not in tokens or token in grouped[mode]:
            raise ValueError("Unexpected or duplicate scene result")
        grouped[mode][token] = record
    output = {}
    for mode in modes:
        if set(grouped[mode]) != set(tokens):
            raise ValueError(f"Missing attempted scenes for {mode}; failures cannot be dropped")
        rows = list(grouped[mode].values())
        valid = [row for row in rows if not row["failed"]]
        # Failed scenes are penalized only in the explicitly named score mean,
        # not invented as measured collisions or measured zero displacement.
        scores = [_number(row["metrics"]["score"], "score", minimum=0, maximum=1) for row in valid]
        output[mode] = {
            "attempted": len(rows), "valid": len(valid), "failed": len(rows) - len(valid),
            "failure_rate": (len(rows) - len(valid)) / len(rows),
            "score_mean_all_attempts_failure_zero": math.fsum(scores) / len(rows),
            "score_mean_valid_only": math.fsum(scores) / len(scores) if scores else None,
            "metrics_mean_valid_only": {
                key: math.fsum(_number(row["metrics"][key], key, minimum=0, maximum=1) for row in valid) / len(valid)
                if valid else None for key in METRICS
            },
            "ade_m_valid_only": _mean(valid, "ade_m"), "fde_m_valid_only": _mean(valid, "fde_m"),
            "all_veto_count": sum(row["all_veto"] for row in rows),
            "latency_s_mean": _mean([row for row in rows if row["latency_s"] is not None], "latency_s"),
        }
    return output


def _mean(rows, key):
    if not rows:
        return None
    return math.fsum(_number(row[key], key, minimum=0) for row in rows) / len(rows)


def _number(value, name, *, minimum, maximum=None):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite real number")
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{name} is outside the metric range")
    return float(value)
