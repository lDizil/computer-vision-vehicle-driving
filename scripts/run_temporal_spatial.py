"""Benchmark causal temporal camera evidence; optionally verify an exact saved replay."""

import argparse
from pathlib import Path

from hybrid_drive.experiments.temporal_spatial_capture import reopen_video_capture
from hybrid_drive.experiments.temporal_video_benchmark import run_temporal_video_benchmark, verify_temporal_video_replay
from scripts.capture_spatial_video import add_video_packs, video_packs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_video_packs(parser)
    for name in ("capture-root", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--capture-sha256", required=True)
    parser.add_argument("--revision", help="tested Git commit for a new benchmark")
    parser.add_argument("--verify-benchmark", type=Path,
                        help="verify a previously completed benchmark instead of making a new gallery")
    parser.add_argument("--benchmark-sha256", help="required exact result.json byte hash when verifying")
    args = parser.parse_args(argv)
    verification = args.verify_benchmark is not None
    if verification != (args.benchmark_sha256 is not None) or verification == (args.revision is not None):
        parser.error("Choose a new --revision or a --verify-benchmark with --benchmark-sha256")
    roles, previous = video_packs(args)
    try:
        capture = reopen_video_capture(args.capture_root, roles, previous, expected_capture_sha256=args.capture_sha256)
        if verification:
            root, proof = verify_temporal_video_replay(
                args.verify_benchmark, capture, args.capture_root, args.output_root,
                expected_benchmark_result_sha256=args.benchmark_sha256, additional_input_roots=previous.values(),
            )
            print(f"{proof['status']}: {root / 'replay-proof.json'}")
            return 0
        root, result, digest = run_temporal_video_benchmark(
            capture, args.capture_root, args.output_root, revision=args.revision, additional_input_roots=previous.values(),
        )
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Temporal spatial benchmark failed: {exc}\n")
    print(f"{result['status']}: {root / 'report.html'}")
    print(f"Benchmark SHA256: {digest}")
    if result.get("error"):
        print(result["error"])
    return 130 if result["status"] == "interrupted" else int(result["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
