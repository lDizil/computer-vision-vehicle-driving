"""Actual NAVSIM scores of fixed, historically timed frozen-scene decisions."""

from collections import Counter
import csv
from dataclasses import asdict, replace
from datetime import datetime, timezone
import math
from pathlib import Path

from hybrid_drive.assets.files import destination
from hybrid_drive.experiments.io import write_json
from hybrid_drive.experiments.protocol import METRICS, content_hash, scene_result, summarize, trajectory_hash, verify_manifest
from hybrid_drive.experiments.bounded_io import decode_json as _decode_json, read_bounded as _read_bounded
from hybrid_drive.experiments.real_bundle import load_real_bundle
from .cohort_fast import MAX_JSON_BYTES, _revision, load_fast_cohort
from .navsim import cache_cohort, configure_environment, load_cohort_inputs, repository_sources, sample_arbiter, score_hybrid_prediction


MODES = ("baseline", "checks_only", "combined")
UNAVAILABLE = {"slow_only": "Mandatory SafetyArbiter checks are intentionally not bypassed"}


def score_frozen_cohort(data_root, fast_run_root, evidence_root, output_root, *, revision,
                        fallback_model="longitudinal", motion_observation_root=None):
    from hybrid_drive.assets.cohort import load_cohort_sources
    from hybrid_drive.experiments.offline_evidence import load_cohort_evidence_index

    _revision(revision)
    cohort = load_fast_cohort(fast_run_root)
    sources = load_cohort_sources(repository_sources("real-cohort-sources.json"))
    _verify_selection(cohort, sources)
    arbiter, motion_index = _fallback_composition(fallback_model, motion_observation_root, cohort)
    run_dir = Path(output_root).resolve() / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run_dir.mkdir(parents=True)
    index, evidence_error = None, None
    try:
        index = load_cohort_evidence_index(evidence_root, cohort_hash=cohort["cohort_hash"])
    except Exception as exc:
        evidence_error = f"{type(exc).__name__}: {exc}"
    frozen = []
    policy = {"limits": asdict(arbiter.limits), "braking": asdict(arbiter.braking)}
    if motion_index is not None:
        policy["fallback_model"] = arbiter.fallback_model.value
    for case in cohort["cases"]:
        case_dir = run_dir / case["case_id"]
        case_dir.mkdir()
        motion_input = {} if motion_index is None else {
            "motion_observation_root": motion_observation_root, "motion_index": motion_index}
        item = _freeze_case(case, cohort, fast_run_root, evidence_root, index, evidence_error, arbiter, **motion_input)
        frozen.append(item)
        write_json(case_dir / "decision.json", _serializable_case(item))
    # This artifact is written before constructing a privileged Scene or metric cache.
    write_json(run_dir / "outputs.json", {
        "schema": "hybrid-drive-frozen-cohort-outputs-v1", "cohort_hash": cohort["cohort_hash"],
        "execution": "offline_replay_original_native_clock", "policy": policy,
        "cases": [_serializable_case(item) for item in frozen],
    })
    loader, overrides, scoring_error = None, None, None
    try:
        configure_environment(data_root, run_dir, verify_maps=True)
        tokens = [case["scene_token"] for case in cohort["cases"]]
        loader, overrides = load_cohort_inputs(sources, tokens)
        cache_cohort(run_dir / "metric-cache", overrides, run_dir / "caching.log")
    except Exception as exc:
        scoring_error = f"{type(exc).__name__}: {exc}"
    records = []
    plot_errors = []
    for item in frozen:
        rows, reference = _score_case(item, loader, run_dir / "metric-cache", overrides, scoring_error)
        records.extend(rows)
        write_json(run_dir / "records.json", records)
        if reference is not None:
            try:
                _render_case(run_dir / item["case_id"] / "trajectory.png", item, reference, rows)
            except Exception as exc:
                plot_errors.append({"case_id": item["case_id"], "error": f"{type(exc).__name__}: {exc}"})
    result = {
        "schema": "hybrid-drive-real-cohort-scores-v1", "source_revision": revision,
        "fast_source_revision": cohort["source_revision"], "cohort_hash": cohort["cohort_hash"],
        "policy": policy, "sources_hash": cohort["sources_hash"],
        "native_evidence": None if index is None else {key: index[key] for key in (
            "evidence_index_hash", "source_revision", "model_version", "model_sources_hash", "protocol")},
        "scope": "single_recording_debug", "execution": "offline_replay_original_native_clock",
        "real_time_claim": False, "driving_safety_claim": False, "automatic_runtime_veto": False,
        "executed_modes": list(MODES), "unavailable_modes": UNAVAILABLE,
        "planned_cases": 6, "planned_mode_attempts": 18, "records": records,
        "summary": summarize(records, expected_tokens=[case["scene_token"] for case in cohort["cases"]], expected_modes=MODES),
        "slow": _slow_summary(records), "fallback": _fallback_summary(records),
        "paired_combined_minus_checks_only": _paired_deltas(records), "plot_errors": plot_errors,
        "quality_claim": "Six fixed windows of one recording; descriptive debug results, not independent quality evidence",
        "metric_protocol": "Actual pinned NAVSIM v1.1 PDMS and components; equal-grid future XY ADE/FDE",
    }
    if motion_index is not None:
        result["motion_observations"] = {key: motion_index[key] for key in (
            "index_hash", "source_revision", "sources_hash", "metadata_sha256", "methods")}
        result["input_errors"] = [
            {"case_id": item["case_id"], "error": item["details"]["checks_only"]["motion_observation_error"]}
            for item in frozen if "motion_observation_error" in item["details"]["checks_only"]]
    result["status"] = "failed" if any(row["failed"] for row in records) or result["slow"]["failed"] or plot_errors or result.get("input_errors") else "success"
    _write_csv(run_dir / "comparison.csv", records)
    write_json(run_dir / "result.json", result)
    try:
        from hybrid_drive.experiments.cohort_report import write_cohort_report

        write_cohort_report(run_dir, result, fast_run_root, evidence_root)
    except Exception as exc:
        result.update(status="failed", report_error=f"{type(exc).__name__}: {exc}")
        write_json(run_dir / "result.json", result)
    return run_dir, result


def _fallback_composition(model, observation_root, cohort):
    arbiter = sample_arbiter()
    if model == "longitudinal":
        if observation_root is not None:
            raise ValueError("Longitudinal control must not consume curved observation sidecars")
        return arbiter, None
    if model != "curved" or observation_root is None:
        raise ValueError("Curved replay requires an explicit model and motion observation root")
    from hybrid_drive.safety.fallback_policy import FallbackModel
    from .motion_observation import load_motion_observation_index

    index = load_motion_observation_index(observation_root, fast_cohort_hash=cohort["cohort_hash"],
                                          sources_hash=cohort["sources_hash"])
    return replace(arbiter, fallback_model=FallbackModel.CURVED), index


def _case_observation(bundle, case, root, index):
    from hybrid_drive.safety.fallback_policy import FallbackObservation
    from .motion_observation import load_motion_observation, motion_observation_hash
    from .real_replay import plan_for_bundle

    row = next(value for value in index["cases"] if value["case_id"] == case["case_id"])
    if row["status"] != "success" or row["base_bundle_hash"] != bundle.bundle_hash:
        raise ValueError("Motion sidecar unavailable or bound to another base bundle")
    value = load_motion_observation(bundle, destination(root, row["observation_path"]))
    if motion_observation_hash(value) != row["observation_hash"]:
        raise ValueError("Motion sidecar differs from validated index")
    rate = value.current.associated_planar_heading_rate_radps
    if rate is None:
        return None
    plan = plan_for_bundle(bundle)
    return FallbackObservation(plan.snapshot_id, plan.t_ref_s, bundle.motion, rate, row["observation_hash"])


def _verify_selection(cohort, sources):
    if cohort["sources_hash"] != content_hash(sources):
        raise ValueError("Fast cohort sources differ from the pinned scoring selection")
    for case, spec in zip(cohort["cases"], sources["cases"]):
        if any(case[key] != spec[key] for key in ("case_id", "scene_token", "source_list_index")):
            raise ValueError("Fast cohort case differs from the pinned scoring selection")


def _freeze_case(case, cohort, fast_root, evidence_root, index, evidence_error, arbiter, *,
                 motion_observation_root=None, motion_index=None):
    item = {"case_id": case["case_id"], "scene_token": case["scene_token"],
            "raw_candidate_hash": case["raw_candidate_hash"], "outputs": {}, "details": {}}
    if case["status"] != "success":
        for mode in MODES:
            item["details"][mode] = _failed_details(case.get("error") or "Fast capture failed", slow_outcome="capture_failed")
        return item
    try:
        bundle = load_real_bundle(destination(fast_root, case["bundle_path"]))
        manifest = _decode_json(_read_bounded(destination(fast_root, case["case_id"] + "/fast-manifest.json"), MAX_JSON_BYTES))
        verify_manifest(manifest)
        if (bundle.bundle_hash != case["bundle_hash"] or bundle.scene_token != case["scene_token"]
                or trajectory_hash(bundle.trajectory) != case["raw_candidate_hash"]
                or manifest["raw_candidate_hash"] != case["raw_candidate_hash"]
                or manifest["model"] != cohort["model"] or manifest["source_revision"] != cohort["source_revision"]
                or manifest["dataset_source_hash"] != cohort["sources_hash"]
                or bundle.provenance["source_experiment_manifest_hash"] != manifest["manifest_hash"]
                or bundle.provenance["source_revision"] != cohort["source_revision"]
                or bundle.provenance["checkpoint_sha256"] != cohort["model"]["checkpoint_sha256"]):
            raise ValueError("Frozen bundle and Fast cohort identity differ")
    except Exception as exc:
        for mode in MODES:
            item["details"][mode] = _failed_details(f"{type(exc).__name__}: {exc}", slow_outcome="capture_invalid")
        return item
    fallback_observation, observation_error = None, None
    if motion_index is not None:
        try:
            fallback_observation = _case_observation(bundle, case, motion_observation_root, motion_index)
        except Exception as exc:
            observation_error = f"{type(exc).__name__}: {exc}"
    evidence = None
    try:
        if index is None:
            raise ValueError(evidence_error or "Missing cohort evidence index")
        evidence = _case_evidence(bundle, case, evidence_root, index)
    except Exception as exc:
        evidence_error = f"{type(exc).__name__}: {exc}"
    try:
        outputs, details = _freeze_predictions(bundle, evidence, arbiter, evidence_error=evidence_error,
                                               fallback_observation=fallback_observation, observation_error=observation_error)
    except Exception as exc:
        outputs = {"baseline": bundle.trajectory}
        details = {"baseline": {"status": "candidate", "failure": None, "all_veto": False},
                   **{mode: _failed_details(f"{type(exc).__name__}: {exc}", slow_outcome="decision_error")
                      for mode in ("checks_only", "combined")}}
    item.update(outputs=outputs, details=details)
    return item


def _case_evidence(bundle, case, root, index):
    from hybrid_drive.experiments.offline_evidence import load_frozen_evidence

    rows = [row for row in index["cases"] if row["case_id"] == case["case_id"]]
    if len(rows) != 1:
        raise ValueError("Expected one evidence case for each fixed Fast frame")
    row = rows[0]
    if row["scene_token"] != case["scene_token"] or row["bundle_hash"] != case["bundle_hash"]:
        raise ValueError("Cohort evidence case identity differs")
    if row["status"] != "success" or not row["evidence_path"]:
        raise ValueError(row.get("error") or "Native evidence was not produced")
    evidence = load_frozen_evidence(bundle, destination(root, row["evidence_path"]))
    if (evidence.evidence_hash != row["evidence_hash"] or evidence.bundle_hash != case["bundle_hash"]
            or evidence.context.request.model_version != index["model_version"]
            or evidence.protocol != index["protocol"] or evidence.model_sources_hash != index["model_sources_hash"]
            or evidence.source_revision != index["source_revision"]):
        raise ValueError("Frozen evidence differs from its cohort index")
    return evidence


def _freeze_predictions(bundle, evidence, arbiter, *, evidence_error=None, fallback_observation=None,
                        observation_error=None):
    from hybrid_drive.experiments.real_replay import plan_for_bundle

    canonical = plan_for_bundle(bundle)
    outputs = {"baseline": bundle.trajectory}
    details = {"baseline": {"status": "candidate", "failure": None, "all_veto": False}}
    plan = canonical if evidence is None else evidence.context.request.plan
    if plan.candidates_hash != canonical.candidates_hash:
        raise ValueError("Evidence plan differs from the frozen raw candidate")
    # These are historical native checkpoints, not timestamps from this Linux process.
    checkpoint = 0 if evidence is None else evidence.decision_checkpoint_monotonic_s
    motion_input = {} if fallback_observation is None else {"fallback_observation": fallback_observation}
    checked = arbiter.decide(plan, bundle.motion, now_monotonic_s=checkpoint, now_sim_s=plan.t_ref_s, **motion_input)
    outputs["checks_only"] = checked.trajectory
    details["checks_only"] = _decision_details(checked)
    motion_details = {}
    if fallback_observation is not None:
        motion_details["motion_observation_hash"] = fallback_observation.source_id
    if observation_error is not None:
        motion_details["motion_observation_error"] = observation_error
    details["checks_only"].update(motion_details)
    if evidence is None:
        details["combined"] = _failed_details(evidence_error or "Missing native Slow evidence", slow_outcome="missing_evidence")
        details["combined"].update(motion_details)
        return outputs, details
    combined = arbiter.decide(
        plan, bundle.motion, now_monotonic_s=evidence.decision_checkpoint_monotonic_s, now_sim_s=plan.t_ref_s,
        request=evidence.context.request, result=evidence.result,
        received_at_monotonic_s=evidence.received_at_monotonic_s, **motion_input,
    )
    outputs["combined"] = combined.trajectory
    details["combined"] = _decision_details(combined)
    details["combined"].update(
        slow_outcome=evidence.outcome, slow_failed=evidence.outcome != "completed" or not evidence.cleanup_confirmed,
        evidence_hash=evidence.evidence_hash, native_source_revision=evidence.source_revision,
        native_elapsed_to_decision_s=evidence.decision_checkpoint_monotonic_s - evidence.context.request.submitted_at_monotonic_s,
        timing="original_native_frozen_clock", execution="complete_slow" if evidence.outcome == "completed" else "degraded_without_slow_result",
    )
    details["combined"].update(motion_details)
    return outputs, details


def _decision_details(decision):
    selection = decision.selection
    fallback = selection.fallback
    reason = None if fallback is None or fallback.failure_reason is None else fallback.failure_reason.value
    return {
        "status": selection.status.value, "failure": reason if decision.failed else None,
        "candidate_id": selection.candidate_id, "all_veto": bool(selection.assessments) and all(value.vetoed for value in selection.assessments),
        "slow_status": decision.slow.status.value, "slow_disposition": decision.slow.disposition.value,
        "slow_used": decision.slow.used, "fallback_invoked": fallback is not None,
        "fallback_succeeded": None if fallback is None else fallback.succeeded, "fallback_reason": reason,
        "fallback_model_report": None if fallback is None or fallback.model_report is None else asdict(fallback.model_report),
        "candidate_kinematics": [{"candidate_id": value.candidate_id, "vetoed": value.vetoed,
                                  "within_limits": value.report.within_limits,
                                  "violations": [violation.value for violation in value.report.violations]}
                                 for value in selection.assessments],
    }


def _failed_details(error, *, slow_outcome):
    return {"status": "failure", "failure": error, "all_veto": False,
            "slow_outcome": slow_outcome, "slow_failed": True}


def _serializable_case(item):
    return {key: value for key, value in item.items() if key != "outputs"} | {
        "outputs": {mode: {"poses": value.poses.tolist(), "interval_s": value.interval_s,
                           "trajectory_hash": trajectory_hash(value)} for mode, value in item["outputs"].items()},
    }


def _score_case(item, loader, cache_dir, overrides, scoring_error):
    from hybrid_drive.adapters.navsim import from_navsim, to_navsim

    records, reference = [], None
    reference_error = scoring_error
    if scoring_error is None and item["outputs"]:
        try:
            raw = item["outputs"]["baseline"]
            reference = from_navsim(loader.get_scene_from_token(item["scene_token"]).get_future_trajectory(num_trajectory_frames=raw.num_poses))
        except Exception as exc:
            reference_error = f"{type(exc).__name__}: {exc}"
    for mode in MODES:
        detail = item["details"][mode]
        prediction = item["outputs"].get(mode)
        try:
            failure = detail["failure"] or reference_error
            if failure:
                raise ValueError(failure)
            native = to_navsim(prediction)
            metrics = score_hybrid_prediction(native, item["scene_token"], cache_dir, overrides)
            record = scene_result(mode=mode, token=item["scene_token"], metrics=metrics, prediction=prediction,
                                  reference=reference, status=detail["status"], all_veto=detail["all_veto"])
        except Exception as exc:
            record = scene_result(mode=mode, token=item["scene_token"], failed=True, error=f"{type(exc).__name__}: {exc}",
                                  status="failure", all_veto=detail["all_veto"])
        record.update({key: value for key, value in detail.items() if key not in ("status", "failure", "all_veto")})
        record.update(case_id=item["case_id"], raw_candidate_hash=item["raw_candidate_hash"])
        if prediction is not None:
            record["native_trajectory"] = {"poses": prediction.poses.tolist(), "interval_s": prediction.interval_s}
            record["selected_trajectory_hash"] = trajectory_hash(prediction)
        records.append(record)
    return records, reference


def _slow_summary(records):
    rows = [row for row in records if row["mode"] == "combined"]
    failed = sum(row.get("slow_failed", True) for row in rows)
    outcomes = Counter(row.get("slow_outcome", "missing_evidence") for row in rows)
    elapsed = [row["native_elapsed_to_decision_s"] for row in rows if "native_elapsed_to_decision_s" in row]
    return {"attempted": len(rows), "failed": failed, "failure_rate": failed / len(rows),
            "outcomes": dict(outcomes), "unknown_count": sum(row.get("slow_outcome") == "completed" and row.get("slow_disposition") == "unknown_result" for row in rows),
            "native_elapsed_to_decision_s_mean": math.fsum(elapsed) / len(elapsed) if elapsed else None,
            "applied_count": sum(row.get("slow_used", False) for row in rows)}


def _fallback_summary(records):
    return {mode: {"invoked": sum(row.get("fallback_invoked", False) for row in records if row["mode"] == mode),
                   "failed": sum(row.get("fallback_succeeded") is False for row in records if row["mode"] == mode),
                   "failure_reasons": dict(Counter(row["fallback_reason"] for row in records
                                                   if row["mode"] == mode and row.get("fallback_reason")))} for mode in MODES}


def _paired_deltas(records):
    grouped = {}
    for row in records:
        grouped.setdefault(row["scene_token"], {})[row["mode"]] = row
    output = []
    for token, rows in grouped.items():
        checked, combined = rows["checks_only"], rows["combined"]
        valid = not checked["failed"] and not combined["failed"]
        output.append({"scene_token": token, "valid_pair": valid, "slow_completed": not combined.get("slow_failed", True),
                       "metrics": {key: combined["metrics"][key] - checked["metrics"][key] for key in METRICS} if valid else None,
                       "ade_m": combined["ade_m"] - checked["ade_m"] if valid else None,
                       "fde_m": combined["fde_m"] - checked["fde_m"] if valid else None})
    return output


def _write_csv(path, records):
    fields = ["case_id", "scene_token", "mode", "failed", "status", "error", "all_veto", "fallback_invoked",
              "fallback_succeeded", "fallback_reason", "slow_outcome", "slow_failed", "slow_status", "slow_disposition",
              "raw_candidate_hash", "selected_trajectory_hash", "ade_m", "fde_m", *METRICS]
    with Path(path).open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for row in records:
            writer.writerow({key: row.get(key) for key in fields if key not in METRICS} | (row["metrics"] or {}))


def _render_case(path, item, reference, records):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(8, 6))
    for row in records:
        if not row["failed"]:
            poses = item["outputs"][row["mode"]].poses
            axis.plot(poses[:, 0], poses[:, 1], "o-", label=f'{row["mode"]}: PDMS {row["metrics"]["score"]:.3f}')
    axis.plot(reference.poses[:, 0], reference.poses[:, 1], "k--", label="Recorded human trajectory")
    axis.set(xlabel="Ego x, m", ylabel="Ego y, m", title=item["case_id"] + " · frozen single-recording debug frame")
    axis.axis("equal")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
