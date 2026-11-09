"""Compare real local VLMs on matched controlled scenes; never enable driving veto."""

import argparse
import json
import os
from pathlib import Path

from hybrid_drive.assets.slow_registry import MODEL_SPECS, resolve_model_key
from hybrid_drive.qualification.cases import build_suite
from hybrid_drive.qualification.prompt import projected_prompt
from hybrid_drive.qualification.runner import run_qualification
from hybrid_drive.slow.llama import LlamaServerFactory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=tuple(MODEL_SPECS), default=["256m"])
    parser.add_argument("--case-ids", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--execution-profile", choices=("cpu", "metal"), default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--deadline-seconds", type=float, default=60)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--server-path", default=os.environ.get("LLAMA_SERVER_PATH", "/opt/llama/llama-server"))
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    if len(set(args.models)) != len(args.models):
        parser.error("Model IDs must be unique")
    suite = build_suite()
    if args.case_ids:
        suite = suite.select(args.case_ids)
    factories = {}
    for model in args.models:
        spec = resolve_model_key(model)
        sources = json.loads(Path(__file__).with_name(spec.sources_filename).read_text())
        factories[model] = LlamaServerFactory(
            str(args.assets_root / spec.asset_subdirectory), sources, server_path=args.server_path,
            execution_profile=args.execution_profile, threads=args.threads, prompt_builder=projected_prompt,
            max_tokens=args.max_tokens,
            request_timeout_s=args.deadline_seconds,
            decision_first=True,
        )
    _, result = run_qualification(suite, factories, args.output_root, revision=args.revision,
                                  seeds=args.seeds, deadline_s=args.deadline_seconds)
    # Negative quality/format results are valid research; missing execution is not success.
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
