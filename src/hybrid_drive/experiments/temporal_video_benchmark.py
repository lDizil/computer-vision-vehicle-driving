"""Compare temporal camera recipes on immutable predicted observations."""

from dataclasses import asdict
import hashlib
from pathlib import Path

from hybrid_drive.assets.spatial_manifest import read_spatial_targets
from .bounded_io import decode_json, read_bounded, require_hex, require_keys
from .io import write_json
from .proposal_alignment import _new_result_directory
from .protocol import content_hash
from .spatial_decision_policy import ZoneDecisionPolicy
from .temporal_spatial_filter import DEFAULT_MAX_AGE_US, DEFAULT_MAX_GAP_US, RECIPES
from .temporal_spatial_gallery import write_temporal_spatial_gallery
from .temporal_spatial_metrics import TemporalTarget, evaluate_temporal, select_temporal_recipe
from .temporal_spatial_replay import CLEAN, STRESS_SCENARIOS, replay_temporal, verify_stress_receipts
from .temporal_video_report import write_temporal_video_report


SCOPE = "openscene_camera_spatial_temporal_v1"
_GATE_SCHEMA = "hybrid-drive-spatial-video-dev-gate-v1"
_RESULT_LIMIT = 3 * 1024 * 1024
_GATE_LIMIT = 128 * 1024
_BASELINE = "framewise"


def run_temporal_video_benchmark(
    capture, capture_path, output_path, *, revision, additional_input_roots=(),
):
    require_hex(revision, 40, "benchmark revision")
    root = _new_result_directory(output_path, _inputs(capture, capture_path, additional_input_roots))
    protocol = _protocol(capture)
    result = {
        "scope": SCOPE, "status": "running", "revision": revision,
        "capture_manifest_sha256": capture.manifest_sha256, "protocol": protocol,
    }
    write_json(root / "result.json", result)
    try:
        matrix, gate, holdout, targets = _compute(capture, protocol, lambda x: _persist_gate(root, x))
        result.update(
            matrix=matrix, matrix_hash=content_hash(matrix), development_gate_hash=content_hash(gate),
        )
        gallery = root / "video.html"
        write_temporal_spatial_gallery(
            gallery, capture.packs["holdout"], targets,
            {name: entry.decisions for name, entry in holdout.items()},
            selected_name=matrix["selection"]["name"],
        )
        result.update(gallery=gallery.name, status="passed")
    except KeyboardInterrupt:
        result.update(status="interrupted", error="Interrupted by operator")
    except Exception as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    try:
        write_temporal_video_report(root, result)
    except Exception as exc:
        result["report_error"] = f"{type(exc).__name__}: {exc}"
        if result["status"] == "passed":
            result.update(status="failed", error="Could not save aggregate video report")
    write_json(root / "result.json", result)
    raw = read_bounded(root / "result.json", _RESULT_LIMIT)
    return root, result, hashlib.sha256(raw).hexdigest()


def verify_temporal_video_replay(
    benchmark_path, capture, capture_path, output_path, *,
    expected_benchmark_result_sha256, additional_input_roots=(),
):
    """Recompute the result using the frozen capture; never load a model or RGB."""
    require_hex(expected_benchmark_result_sha256, 64, "expected benchmark byte hash")
    original = Path(benchmark_path).absolute()
    raw = read_bounded(original / "result.json", _RESULT_LIMIT)
    if hashlib.sha256(raw).hexdigest() != expected_benchmark_result_sha256:
        raise ValueError("Original benchmark byte hash differs")
    first = decode_json(raw)
    protocol = _protocol(capture)
    if (
        first.get("scope") != SCOPE or first.get("status") != "passed"
        or first.get("capture_manifest_sha256") != capture.manifest_sha256
        or first.get("protocol") != protocol
        or first.get("matrix_hash") != content_hash(first.get("matrix"))
    ):
        raise ValueError("Original benchmark is not a passed run on this capture and protocol")
    frozen = decode_json(read_bounded(original / "frozen-development.json", _GATE_LIMIT))
    if content_hash(frozen) != first.get("development_gate_hash"):
        raise ValueError("Original dev gate was tampered with")

    def check_gate(submitted):
        if content_hash(submitted) != first["development_gate_hash"]:
            raise ValueError("Computing development again did not match the frozen recipe choice")
        return frozen

    matrix, _, _, _ = _compute(capture, protocol, check_gate)
    if content_hash(matrix) != first["matrix_hash"]:
        raise ValueError("Replayed matrix did not match the complete original")
    root = _new_result_directory(
        output_path, (*_inputs(capture, capture_path, additional_input_roots), original),
    )
    receipt = {
        "schema": "hybrid-drive-temporal-video-replay-receipt-v1", "status": "identical",
        "benchmark_result_sha256": expected_benchmark_result_sha256, "capture_sha256": capture.manifest_sha256,
        "matrix_hash": first["matrix_hash"], "development_gate_hash": first["development_gate_hash"],
        "experiments": matrix["experiment_counts"], "inference_repeated": False, "rgb_loaded": False,
    }
    write_json(root / "replay-proof.json", receipt)
    return root, receipt


def _inputs(capture, capture_path, extra):
    return (capture_path, *(pack.root for pack in capture.packs.values()), *extra)


def _protocol(capture):
    model = capture.manifest["model_receipt"]
    confidence = model["calibrated_policy"]
    require_keys(confidence, {"absent_below", "present_at_or_above"}, "frozen confidence policy")
    return {
        "capture_revision": capture.manifest["revision"],
        "pack_hashes": {role: pack.manifest_hash for role, pack in capture.packs.items()},
        "model_hash": capture.manifest["model_hash"], "calibrated_policy": confidence,
        "recipes": {name: asdict(recipe) for name, recipe in RECIPES.items()},
        "scenarios": list(STRESS_SCENARIOS), "baseline": _BASELINE,
        "max_gap_us": DEFAULT_MAX_GAP_US, "max_age_us": DEFAULT_MAX_AGE_US,
        "temporal_clock": "openscene_associated_source_timestamps",
        "runtime_changed": False, "driving_claim": False,
    }


def _compute(capture, protocol, freeze):
    policy = ZoneDecisionPolicy(**protocol["calibrated_policy"])
    _require_ordered_late_delivery(capture.traces["stress"].frames, protocol["max_age_us"])
    development_targets = _targets(capture.packs["development"])
    development = _replays(capture, "development", policy, CLEAN)
    dev_metrics = _score(development, development_targets)
    proposed = select_temporal_recipe(dev_metrics, baseline_name=_BASELINE)
    gate = freeze({
        "schema": _GATE_SCHEMA, "capture_sha256": capture.manifest_sha256,
        "protocol": protocol, "development_metrics_hash": content_hash(dev_metrics),
        "selection": proposed,
    })
    if gate["selection"]["name"] not in RECIPES:
        raise ValueError("Dev gate contains an unknown recipe")

    # The first access to final and stress annotation bytes is below the
    # persisted/read-back development gate. They cannot select a recipe.
    targets = _targets(capture.packs["holdout"])
    holdout = _replays(capture, "holdout", policy, CLEAN)
    holdout_metrics = _score(holdout, targets)
    stress_targets = _targets(capture.packs["stress"])
    stress = _replays(capture, "stress", policy, CLEAN)
    stress_metrics = _score(stress, stress_targets)
    faults = {scenario: _replays(capture, "stress", policy, scenario) for scenario in STRESS_SCENARIOS}
    verified = {
        recipe: verify_stress_receipts(stress[recipe], {key: arms[recipe] for key, arms in faults.items()})
        for recipe in RECIPES
    }
    fault_reports = {scenario: _score(arms, stress_targets) for scenario, arms in faults.items()}
    clean_count = len(development) + len(holdout) + len(stress)
    stress_count = sum(len(arms) for arms in faults.values())
    matrix = {
        "selection": gate["selection"],
        "holdout_diagnostic": _check_preselected_holdout(gate["selection"]["name"], holdout_metrics),
        "development": dev_metrics, "holdout": holdout_metrics, "stress_clean": stress_metrics,
        "faults": fault_reports, "fault_invariants": verified,
        "experiment_counts": {"clean": clean_count, "stress": stress_count, "total": clean_count + stress_count},
        "source_frames": {name: len(trace.frames) for name, trace in capture.traces.items()},
    }
    return matrix, gate, holdout, targets


def _require_ordered_late_delivery(frames, age_limit):
    # An injected stale event is processed before the next real capture.
    for left, right in zip(frames, frames[1:]):
        if (
            (left.recording_id, left.clip_id) == (right.recording_id, right.clip_id)
            and right.timestamp_us - left.timestamp_us <= age_limit + 1
        ):
            raise ValueError("Stress clip cadence is too fast for the declared late delivery schedule")


def _check_preselected_holdout(name, metrics):
    if name == _BASELINE:
        return {"evaluated_recipe": name, "outcome": "framewise_selected_on_dev", "reasons": []}
    # Check only the preselected mode against the same guards on new records.
    gate = select_temporal_recipe({_BASELINE: metrics[_BASELINE], name: metrics[name]}, baseline_name=_BASELINE)
    candidate = gate["candidates"][name]
    return {
        "evaluated_recipe": name,
        "outcome": "preselected_recipe_passed" if candidate["eligible"] else "keep_framewise",
        "reasons": candidate["reasons"],
    }


def _persist_gate(root, data):
    target = root / "frozen-development.json"
    write_json(target, data)
    replayed = decode_json(read_bounded(target, _GATE_LIMIT))
    if content_hash(replayed) != content_hash(data):
        raise ValueError("The persisted dev selection did not round-trip")
    return replayed


def _targets(pack):
    labels = read_spatial_targets(pack.root, pack.manifest, "holdout")
    if tuple(row["sample_id"] for row in labels) != tuple(frame.sample_id for frame in pack.frames):
        raise ValueError("Clip supervision does not match the captured source frame order")
    return {row["sample_id"]: TemporalTarget(row["presence"], row["boxes_2d"]) for row in labels}


def _replays(capture, role, policy, scenario):
    return {
        name: replay_temporal(capture.traces[role].frames, policy, recipe, scenario=scenario)
        for name, recipe in RECIPES.items()
    }


def _score(arms, targets):
    basis = arms[_BASELINE].decisions
    # Drop and late stress compare on the exact same surviving frames for
    # every recipe, not against a baseline evaluated with extra observations.
    frozen = {entry.observation.sample_id: targets[entry.observation.sample_id] for entry in basis}
    return {
        name: {
            **evaluate_temporal(arm.decisions, frozen, baseline=basis if name != _BASELINE else None),
            "replay": arm.receipt(),
        }
        for name, arm in arms.items()
    }
