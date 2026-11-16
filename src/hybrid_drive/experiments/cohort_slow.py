"""Sequential native Slow evidence on a frozen real cohort, never a control loop."""

from datetime import datetime, timezone

from hybrid_drive.contracts._validation import finite_real
from hybrid_drive.slow.real_grounding import PROTOCOL_NAME
from pathlib import Path

from .bounded_io import decode_json, read_bounded
from .cohort_fast import load_fast_cohort
from .io import write_json
from .offline_evidence import INDEX_SCHEMA, write_frozen_evidence
from .protocol import content_hash
from .real_bundle import load_real_bundle
from .real_replay import run_real_replay


def run_cohort_slow(fast_run_root, factory, output_root, *, revision, deadline_s=60):
    deadline_s = finite_real(deadline_s, "deadline_s")
    if not 0 < deadline_s <= 120 or factory.protocol_name != PROTOCOL_NAME:
        raise ValueError("Bounded real-path protocol is required")
    fast_run_root = Path(fast_run_root)
    cohort = load_fast_cohort(fast_run_root)
    # Validate before allocating output or starting any backend.
    if not isinstance(revision, str) or len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("An exact source revision is required")
    run_dir = Path(output_root) / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run_dir.mkdir(parents=True)
    index = {
        "schema": INDEX_SCHEMA, "source_revision": revision, "cohort_hash": cohort["cohort_hash"],
        "model_version": factory.model_version, "model_sources_hash": content_hash(factory.sources),
        "protocol": factory.protocol_name,
        "cases": [{"case_id": c["case_id"], "scene_token": c["scene_token"], "bundle_hash": c["bundle_hash"],
                   "status": "not_started", "evidence_path": None, "evidence_hash": None, "error": None}
                  for c in cohort["cases"]],
    }
    records = []
    stopped = None
    for case, entry in zip(cohort["cases"], index["cases"]):
        if stopped or case["status"] != "success":
            entry["error"] = stopped or "fast_case_failed"
            _save_index(run_dir, index)
            continue
        try:
            bundle = load_real_bundle(fast_run_root / case["bundle_path"])
            if bundle.bundle_hash != case["bundle_hash"] or bundle.scene_token != case["scene_token"]:
                raise ValueError("Fast bundle differs from frozen cohort")
            replay_dir, result = run_real_replay(bundle, factory, run_dir / case["case_id"] / "replay",
                                                 revision=revision, seeds=(factory.seed,), deadline_s=deadline_s)
            # A failed export must not hide an unconfirmed model process group.
            row = decode_json(read_bounded(replay_dir / "records.json", 128 * 1024))[0]
            stopped = "previous_cleanup_unconfirmed" if not row["cleanup_confirmed"] else None
            evidence_path = case["case_id"] + "/evidence"
            evidence = write_frozen_evidence(bundle, replay_dir, run_dir / evidence_path)
            entry.update(status="success", evidence_path=evidence_path, evidence_hash=evidence.evidence_hash)
            records.append({"case_id": case["case_id"], "outcome": evidence.outcome,
                            "status": evidence.result.status.value if evidence.result else None,
                            "latency_s": row["latency_s"], "cleanup_confirmed": evidence.cleanup_confirmed})
        except Exception as exc:
            stopped = stopped or "previous_case_failed_cleanup_unknown"
            entry.update(status="failed", error=type(exc).__name__)
            records.append({"case_id": case["case_id"], "outcome": "execution_error", "status": None,
                            "latency_s": None, "cleanup_confirmed": None})
        _save_index(run_dir, index)
        write_json(run_dir / "records.json", records)
    completed = sum(row["outcome"] == "completed" for row in records)
    result = {"status": "completed" if completed == len(cohort["cases"]) else "failed",
              "planned_cases": len(cohort["cases"]), "attempts": len(records), "completed": completed,
              "source_revision": revision, "cohort_hash": cohort["cohort_hash"],
              "automatic_veto": False, "scope": "single_recording_frozen_offline_debug"}
    _save_index(run_dir, index)
    write_json(run_dir / "result.json", result)
    return run_dir, result


def _save_index(run_dir, index):
    index["evidence_index_hash"] = content_hash({k: v for k, v in index.items() if k != "evidence_index_hash"})
    write_json(run_dir / "cohort-evidence.json", index)
