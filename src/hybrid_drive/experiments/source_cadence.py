"""Actual source-cadence delivery of frozen inputs, not real-time Fast inference."""

from collections import Counter
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import time
from uuid import uuid4

from hybrid_drive.adapters.real_camera import render_plan_overlay
from hybrid_drive.assets.files import check_components
from hybrid_drive.contracts._validation import finite_real
from hybrid_drive.contracts.slow import SlowRequest
from hybrid_drive.slow.context import SlowContext
from hybrid_drive.slow.real_grounding import PROTOCOL_NAME
from hybrid_drive.slow.worker import AsyncSlowWorker
from .cohort_fast import _revision
from .io import write_json
from .navsim import sample_arbiter
from .protocol import content_hash
from .real_replay import plan_for_bundle
from .temporal_fast import load_temporal_fast, load_temporal_bundles


def run_source_cadence(fast_run_root, factory, output_root, *, revision, deadline_s=60,
                       clock=time.monotonic, sleep=time.sleep, worker_factory=AsyncSlowWorker):
    _revision(revision)
    deadline_s = finite_real(deadline_s, "deadline_s")
    if not 0 < deadline_s <= 120 or factory.protocol_name != PROTOCOL_NAME:
        raise ValueError("Bounded native real-path Slow protocol is required")
    output_root, fast_run_root = Path(output_root).absolute(), Path(fast_run_root).absolute()
    if output_root == fast_run_root or fast_run_root in output_root.parents:
        raise ValueError("Source cadence output must remain outside frozen Fast input")
    fast = load_temporal_fast(fast_run_root)
    bundles = load_temporal_bundles(fast_run_root, fast)
    plans = tuple(plan_for_bundle(bundle) for bundle in bundles)
    overlays = tuple(tuple(render_plan_overlay(view, bundle.trajectory) for view in bundle.views) for bundle in bundles)
    timestamps = [case["frame_timestamp_us"] for case in fast["cases"]]
    offsets = [(value-timestamps[0])/1e6 for value in timestamps]
    until = offsets[1:] + [offsets[-1] + .5]
    arbiter = sample_arbiter()
    run_dir = Path(output_root) / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    check_components(run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema": "hybrid-drive-source-cadence-v1", "source_revision": revision,
        "temporal_fast_hash": fast["temporal_fast_hash"], "fast_source_revision": fast["source_revision"],
        "sources_hash": fast["sources_hash"], "model_version": factory.model_version, "model_sources": factory.sources,
        "protocol": factory.protocol_name, "deadline_s": deadline_s, "planned_frames": 20,
        "seed": getattr(factory,"seed",None),
        "source_span_s": offsets[-1], "execution": "frozen_fast_delivery_at_original_source_cadence",
        "fast_inference_realtime_claim": False, "control_loop_hz_claim": None, "driving_safety_claim": False,
        "queue_policy": "one_active_one_latest_pending_discard_pending_at_stream_end",
        "applicability": "half-open source frame interval; completion audit never revises a returned Fast decision",
        "decision_latency_definition": {"delivery_to_decision_s":"delivery through submit/poll and mandatory checks",
                                        "fast_decision_wall_s":"arbiter call only, excluding submit/poll"},
        "runtime_profile": {key: getattr(factory, key, None) for key in
                            ("execution_profile", "threads", "max_tokens", "image_min_tokens", "image_max_tokens", "context_size", "cache_prompt")},
    }
    manifest["manifest_hash"] = content_hash(manifest)
    write_json(run_dir / "manifest.json", manifest)
    rows = [{"case_id": case["case_id"], "scene_token": case["scene_token"], "bundle_hash": case["bundle_hash"],
             "frame_timestamp_us": case["frame_timestamp_us"], "source_offset_s": offset,
             "fast_cpu_s": case["fast_cpu_s"], "fast_wall_s": case["fast_wall_s"],
             "status": "not_delivered", "request_id": None} for case,offset in zip(fast["cases"],offsets)]
    completions, matching, request_rows = [], {}, {}
    worker, current, error, stream_end, cleanup, discarded_pending = None, None, None, None, False, None
    origin, maximum_depth = clock(), 0

    def consume():
        for event in worker.poll():
            now = clock()
            incoming = event.context
            request = incoming.request
            owner = request_rows.get(request.request_id)
            if owner is None:
                raise ValueError("Completion has no submitted source frame")
            expired = now >= origin+owner["valid_until_source_offset_s"]
            is_current = current is not None and request.request_id == current.request.request_id and incoming.content_hash == current.content_hash
            now_sim = max(current.request.plan.t_ref_s, plans[0].t_ref_s + now-origin)
            assessment = arbiter.decide(
                current.request.plan, current.motion, now_monotonic_s=now,
                now_sim_s=now_sim,
                request=request, result=event.result, received_at_monotonic_s=event.received_at_monotonic_s,
            )
            applicable = is_current and not expired and assessment.slow.used
            audit = {
                "case_id": owner["case_id"], "request_id": request.request_id,
                "plan_id": request.plan.plan_id, "snapshot_id": request.plan.snapshot_id,
                "candidates_hash": request.plan.candidates_hash,
                "current_request_id": current.request.request_id, "current_plan_id": current.request.plan.plan_id,
                "context_hash": incoming.content_hash, "outcome": event.outcome,
                "submitted_at_monotonic_s": request.submitted_at_monotonic_s,
                "deadline_monotonic_s": request.deadline_monotonic_s,
                "received_at_monotonic_s": event.received_at_monotonic_s, "observed_at_monotonic_s": now,
                "elapsed_to_observation_s": now-request.submitted_at_monotonic_s,
                "now_sim_s": now_sim, "valid_until_sim_s": request.valid_until_sim_s,
                "source_window_expired": expired, "matches_current_plan": is_current,
                "arbitration_disposition": assessment.slow.disposition.value,
                "applicable_at_observation": applicable, "used_for_fast_decision": False,
                "raw_response": event.raw_response,
                "status": None if event.result is None else event.result.status.value,
                "vetoed_candidate_ids": [] if event.result is None else list(event.result.vetoed_candidate_ids),
                "explanation": None if event.result is None else event.result.explanation,
            }
            completions.append(audit)
            if applicable:
                matching[request.request_id] = (event, audit)

    try:
        worker = worker_factory(factory, factory.model_version, clock=clock, response_parser=factory.parse_output)
        for position, (bundle, plan, images, offset, expiry, row) in enumerate(zip(bundles,plans,overlays,offsets,until,rows)):
            target = origin + offset
            while clock() < target:
                consume()
                sleep(min(.01, max(0, target-clock())))
            consume()
            delivered = clock()
            request_id = "cadence-" + uuid4().hex
            valid_until_sim = timestamps[position+1]/1e6 if position+1 < len(timestamps) else plan.t_ref_s+.5
            request = SlowRequest(request_id, plan, delivered, delivered+deadline_s,
                                  valid_until_sim, factory.model_version)
            current = SlowContext(request, bundle.motion, images)
            row.update(status="delivered", request_id=request_id, context_hash=current.content_hash,
                       plan_id=plan.plan_id, snapshot_id=plan.snapshot_id, candidates_hash=plan.candidates_hash,
                       scheduled_at_monotonic_s=target, delivered_at_monotonic_s=delivered,
                       delivery_lag_s=delivered-target, valid_until_source_offset_s=expiry,
                       submitted_at_monotonic_s=delivered, deadline_monotonic_s=delivered+deadline_s)
            row["valid_until_sim_s"] = request.valid_until_sim_s
            request_rows[request_id] = row
            row["submission_disposition"] = worker.submit(current)
            row["submission_returned_at_monotonic_s"] = clock()
            maximum_depth = max(maximum_depth, worker.queue_depth)
            consume()
            found = matching.get(request_id)
            now = clock()
            usable = found is not None and now < origin+expiry
            decision = arbiter.decide(
                plan, bundle.motion, now_monotonic_s=now,
                now_sim_s=max(plan.t_ref_s, plans[0].t_ref_s+now-origin), request=request,
                result=found[0].result if usable else None,
                received_at_monotonic_s=found[0].received_at_monotonic_s if usable else None,
            )
            decision_finished = clock()
            row.update(fast_decision_started_at_monotonic_s=now, fast_decision_at_monotonic_s=decision_finished,
                       fast_decision_wall_s=decision_finished-now,
                       delivery_to_decision_s=decision_finished-delivered,
                       decision_status=decision.selection.status.value, slow_used=decision.slow.used,
                       slow_disposition=decision.slow.disposition.value,
                       decision_trajectory_hash=content_hash({"poses":decision.trajectory.poses.tolist(),"interval_s":decision.trajectory.interval_s}))
            if decision.slow.used:
                found[1]["used_for_fast_decision"] = True
            write_json(run_dir / "frames.json", rows)
            write_json(run_dir / "completions.json", completions)
        stream_end = clock()
        pending = worker.pending_request_id
        if pending is not None:
            if worker.discard_pending(pending):
                discarded_pending = pending
        consume()
        while worker.active_request_id is not None:
            sleep(.01)
            consume()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        if worker is not None:
            try:
                worker.close()
                cleanup = True
            except Exception as exc:
                error = error or f"cleanup: {type(exc).__name__}: {exc}"
        else:
            cleanup = True
    outcomes = Counter(row["outcome"] for row in completions)
    terminal = Counter(row["request_id"] for row in completions)
    audit_complete = all(terminal[request_id] == 1 for request_id in request_rows)
    dropped = 0 if worker is None else worker.dropped_completion_count
    technical = {key:value for key,value in outcomes.items() if key not in {"completed","superseded","discarded_pending"}}
    technical_count = sum(technical.values())
    measurement_complete = (error is None and cleanup and audit_complete and dropped == 0
                            and len(request_rows) == 20 and all(row["status"] == "delivered" for row in rows))
    result = {
        "schema": "hybrid-drive-source-cadence-result-v1", "manifest_hash": manifest["manifest_hash"],
        "status": ("completed_with_slow_failures" if technical_count else "completed") if measurement_complete else "failed",
        "measurement_status": "completed" if measurement_complete else "failed",
        "slow_status": "technical_failures_observed" if technical_count else "no_technical_failures_observed",
        "slow_technical_failure_count": technical_count, "slow_technical_outcomes": technical,
        "native_completed_count": outcomes["completed"],
        "normal_queue_termination_count": outcomes["superseded"]+outcomes["discarded_pending"],
        "error": error, "cleanup_confirmed": cleanup, "origin_monotonic_s": origin,
        "stream_end_monotonic_s": stream_end, "finished_at_monotonic_s": clock(),
        "discarded_pending_request_id": discarded_pending,
        "maximum_queue_depth": maximum_depth, "dropped_completion_count": dropped,
        "outcome_counts": dict(outcomes), "planned_frames": 20, "delivered_frames": sum(row["status"] == "delivered" for row in rows),
        "terminal_audit_complete": audit_complete,
        "fast_decisions_using_slow": sum(row.get("slow_used",False) for row in rows),
        "fast_decision_failures": sum(row.get("decision_status") == "failure" for row in rows),
        "frames_hash": content_hash(rows), "completions_hash": content_hash(completions),
        "source_span_s": offsets[-1], "fast_inference_realtime_claim": False, "driving_safety_claim": False,
    }
    if dropped or not audit_complete:
        result["error"] = error or "worker completion audit incomplete/overflow"
    result["result_hash"] = content_hash(result)
    write_json(run_dir / "frames.json", rows)
    write_json(run_dir / "completions.json", completions)
    write_json(run_dir / "result.json", result)
    _write_report(run_dir / "report.html", manifest, result, rows, completions)
    return run_dir, result


def _write_report(path, manifest, result, rows, completions):
    def cell(value):
        return html.escape(f"{value:.6f}" if isinstance(value,float) else str(value))
    headings = ("Frame", "Source, s", "Delivery lag, s", "Fast CPU, s", "Fast wall, s", "Delivery → decision, s", "Arbiter only, s", "Submission", "Decision", "Slow")
    body = []
    for row in rows:
        values = [row["case_id"],row["source_offset_s"],row.get("delivery_lag_s"),row["fast_cpu_s"],row["fast_wall_s"],
                  row.get("delivery_to_decision_s"),row.get("fast_decision_wall_s"),
                  row.get("submission_disposition"),row.get("decision_status",row["status"]),row.get("slow_disposition")]
        body.append("<tr>"+"".join("<td>"+cell(v)+"</td>" for v in values)+"</tr>")
    events = "".join("<section><h3>"+cell(event["case_id"]+": "+event["outcome"])+"</h3><pre>"+
                     html.escape(json.dumps(event,ensure_ascii=False,indent=2))+"</pre></section>" for event in completions)
    document = """<!doctype html><meta charset="utf-8"><title>Source cadence replay</title>
<style>body{font:16px system-ui;margin:2em;max-width:1500px}table{border-collapse:collapse}td,th{border:1px solid #bbb;padding:.4em}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f4f4;padding:1em}</style>
<h1>Source cadence replay — frozen Fast delivery</h1>
<p>20 successive source frames at approximately 2 Hz. Fast CPU/wall times were measured separately before replay.
Delivery is not real-time model inference, a 10 Hz control loop, a closed-loop drive, or driving-safety evidence.
Late Slow results remain observable but never revise returned Fast decisions. Delivery → decision includes submit/poll and mandatory checks;
Arbiter only excludes submit/poll. Native completions are counted separately from superseded/discarded pending requests.</p>"""
    document += "<pre>"+html.escape(json.dumps(result,indent=2))+"</pre><table><tr>"+"".join("<th>"+cell(v)+"</th>" for v in headings)+"</tr>"+"".join(body)+"</table>"
    document += "<h2>Original-clock completions</h2>"+events+"<h2>Manifest</h2><pre>"+html.escape(json.dumps(manifest,indent=2))+"</pre>"
    path.write_text(document,encoding="utf-8")
