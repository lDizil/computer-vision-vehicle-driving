"""Run single-candidate advisory on a verified rectified real-scene bundle."""

import argparse
import json
from pathlib import Path

from hybrid_drive.assets.slow_registry import resolve_model_key
from hybrid_drive.experiments.real_bundle import load_real_bundle
from hybrid_drive.experiments.real_replay import run_real_replay
from hybrid_drive.slow.real_grounding import RealPathGroundingFactory


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--assets-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model", choices=("256m", "500m", "qwen3vl-2b"), default="qwen3vl-2b")
    parser.add_argument("--server-path", required=True)
    parser.add_argument("--execution-profile", choices=("cpu", "metal"), default="metal")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--deadline-seconds", type=float, default=60)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args(argv)
    spec = resolve_model_key(args.model)
    sources = json.loads(Path(__file__).with_name(spec.sources_filename).read_text())
    factory = RealPathGroundingFactory(
        str(args.assets_root / spec.asset_subdirectory), sources, server_path=args.server_path,
        execution_profile=args.execution_profile, threads=args.threads, max_tokens=256,
        request_timeout_s=args.deadline_seconds, image_min_tokens=1024, image_max_tokens=1024,
        context_size=8192, cache_prompt=False,
    )
    run_dir, result = run_real_replay(load_real_bundle(args.bundle), factory, args.output_root,
                                     revision=args.revision, seeds=args.seeds, deadline_s=args.deadline_seconds)
    print(f"Report: {run_dir / 'report.html'}")
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
