"""Offline-only controlled qualification; no Fast control loop or automatic veto."""

from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import resource
import time
from uuid import uuid4

from hybrid_drive.slow.worker import AsyncSlowWorker
from hybrid_drive.slow.protocol import bounded_prompt
from .metrics import summarize
from .prompt import PROMPT_VERSION
from .report import write_html


def run_qualification(suite, factories, output_root, *, revision, seeds=(0,), deadline_s=60,
                      prompt_version=PROMPT_VERSION):
    if revision != "working-tree" and (not isinstance(revision, str) or len(revision) != 40
                                      or any(c not in "0123456789abcdef" for c in revision)):
        raise ValueError("revision must be an exact commit or explicit working-tree diagnostic")
    if isinstance(deadline_s, bool) or not isinstance(deadline_s, (int, float)) or not math.isfinite(deadline_s) or not 0 < deadline_s <= 120:
        raise ValueError("Deadline must be in (0, 120] seconds")
    seeds = tuple(seeds)
    if not seeds or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be unique integers in [0, 2**32)")
    if not factories or any(not isinstance(model, str) or not model for model in factories):
        raise ValueError("At least one named model factory is required")
    profiles = [_execution_profile(factory) for factory in factories.values()]
    if any(profile != profiles[0] for profile in profiles[1:]):
        raise ValueError("Model comparison requires the same execution/generation profile; use separate runs")
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run_dir = output_root / run_id
    run_dir.mkdir()
    (run_dir / "prompts").mkdir()
    records = []
    provenance = {
        "source_revision": revision, "commit_reference_supplied": revision != "working-tree",
        "platform": platform.platform(), "architecture": platform.machine(),
        "python_version": platform.python_version(), "prompt_version": prompt_version,
        "prompt_hash_definition": "Single prompt: UTF8 bytes; multi-stage: canonical JSON ordered prompt program. "
                                  "Conditional stages may be skipped; actual provider replies remain in raw trace.",
        "deadline_seconds": deadline_s, "seeds": list(seeds),
        "scope": "controlled_schematic_diagnostic_not_road_qualification",
        "models": {model: _factory_metadata(factory) for model, factory in factories.items()},
    }
    write_json(run_dir / "manifest.json", {"provenance": provenance, "suite": suite.manifest()})
    write_json(run_dir / "records.json", records)
    started = time.monotonic()
    execution_error = None
    try:
        for model, base_factory in factories.items():
            for seed in seeds:
                factory = replace(base_factory, seed=seed)
                worker = AsyncSlowWorker(factory, factory.model_version, response_parser=factory.parse_output)
                try:
                    for index, case in enumerate(suite.cases):
                        row = _attempt(worker, factory, case, model, seed, deadline_s, run_dir, index)
                        records.append(row)
                        write_json(run_dir / "records.json", records)
                        print(f"{model} seed={seed} case={case.case_id}: {row['outcome']} "
                              f"{row['status']} in {row['latency_s']:.3f}s", flush=True)
                finally:
                    # Never overlap another model with a group whose death is unconfirmed.
                    worker.close()
    except Exception as exc:
        complete = len(records) == len(factories) * len(seeds) * len(suite.cases)
        execution_error = type(exc).__name__
        write_json(run_dir / "failure.json", {"status": "failed", "error_type": execution_error,
                                              "records_retained": len(records), "matrix_complete": complete})
        if not complete:
            raise
    summary = summarize(suite, records, expected_models=tuple(factories), seeds=seeds)
    # Invalid model text is a measured negative result, not a fabricated valid prediction.
    technical_success = execution_error is None and all(
        row["outcome"] in ("completed", "invalid_output") for row in records)
    peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    result = {
        "status": "completed" if technical_success else "failed",
        "wall_seconds": time.monotonic() - started, "source_revision": revision,
        "summary": summary,
        "execution_error": execution_error,
        "max_reaped_child_rss_bytes": int(peak if platform.system() == "Darwin" else peak * 1024),
        "memory_metric": "maximum_reaped_child_process_RSS_not_aggregate_or_unified_GPU_memory",
    }
    write_json(run_dir / "result.json", result)
    write_html(run_dir / "report.html", suite, records, summary, provenance=provenance,
               execution={key: result[key] for key in ("status", "execution_error", "wall_seconds")})
    print(f"Report: {run_dir / 'report.html'}", flush=True)
    return run_dir, result


def _attempt(worker, factory, case, model, seed, deadline_s, run_dir, index):
    started = time.monotonic()
    request_id = "qualification-" + uuid4().hex
    context = case.to_context(factory.model_version, request_id=request_id,
                              submitted_at_monotonic_s=started, deadline_monotonic_s=started + deadline_s)
    prompts = tuple(bounded_prompt(prompt) for prompt in factory.prompts_for(context))
    if not prompts:
        raise ValueError("At least one declared prompt is required")
    prompt = prompts[0] if len(prompts) == 1 else json.dumps(prompts, ensure_ascii=False, separators=(",", ":"))
    row = {
        "model_id": model, "case_id": case.case_id, "seed": seed, "outcome": "backend_error",
        "status": None, "vetoed_candidate_ids": [], "explanation": None,
        "latency_s": 0.0, "raw_response": None, "input_hash": case.input_hash, "request_id": request_id,
        "context_hash": context.content_hash,
        "submitted_at_monotonic_s": context.request.submitted_at_monotonic_s,
        "deadline_monotonic_s": context.request.deadline_monotonic_s,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "declared_prompt_stages": len(prompts),
    }
    prompt_name = f"{index:03d}-{seed}-{hashlib.sha256(model.encode()).hexdigest()[:12]}.txt"
    (run_dir / "prompts" / prompt_name).write_text(prompt)
    try:
        worker.submit(context)
        completion = wait_for_completion(worker, context)
        if completion is None:
            row["outcome"] = "process_exit"
        else:
            row["raw_response"] = completion.raw_response
            if completion.outcome == "completed" and factory.expected_output_fields is not None:
                _require_decision_first(completion.raw_response, factory.expected_output_fields)
            row["outcome"] = completion.outcome
            if completion.result is not None:
                row.update(status=completion.result.status.value,
                           vetoed_candidate_ids=list(completion.result.vetoed_candidate_ids),
                           explanation=completion.result.explanation)
    except Exception as exc:
        row["error_type"] = type(exc).__name__
    row["latency_s"] = time.monotonic() - started
    return row


class DecisionOrderError(ValueError):
    pass


def _require_decision_first(raw, expected=("status", "vetoed_candidate_ids", "explanation", "context_hash")):
    try:
        payload = json.loads(raw, object_pairs_hook=tuple)
    except (ValueError, TypeError, RecursionError) as exc:
        raise DecisionOrderError("Completed response is not a decision-first JSON object") from exc
    if not isinstance(payload, tuple) or tuple(key for key, _ in payload) != expected:
        raise DecisionOrderError("Completed response violated decision-first JSON key order")


def wait_for_completion(worker, context):
    while True:
        for completion in worker.poll():
            if (completion.context.request.request_id == context.request.request_id
                    and completion.context.content_hash == context.content_hash):
                return completion
        if worker.queue_depth == 0:
            return None
        time.sleep(0.005)


def _factory_metadata(factory):
    return {
        "model_version": factory.model_version, "sources": factory.sources,
        "runtime_revision": factory.sources["runtime_revision"],
        "execution_profile": factory.execution_profile, "threads": factory.threads,
        "temperature": 0, "max_tokens": factory.max_tokens,
        "context_size": factory.context_size, "image_min_tokens": factory.image_min_tokens,
        "image_max_tokens": factory.image_max_tokens, "cache_prompt": factory.cache_prompt,
        "response_protocol": factory.protocol_name,
        "decision_first": factory.decision_first,
        "offline_local_files": True,
        "network_isolation": "OS_process_network_not_disabled; loopback_only_backend; runtime_offline_and_no_CURL"
                             if factory.execution_profile == "metal" else "use_network_none_in_Docker",
    }


def _execution_profile(factory):
    return {"runtime_revision": factory.sources["runtime_revision"],
            "execution_profile": factory.execution_profile, "threads": factory.threads,
            "max_tokens": factory.max_tokens, "decision_first": factory.decision_first,
            "context_size": factory.context_size, "image_min_tokens": factory.image_min_tokens,
            "image_max_tokens": factory.image_max_tokens, "cache_prompt": factory.cache_prompt,
            "response_protocol": factory.protocol_name}


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)
