"""Evaluate whether visible appearance improves identity association on fixed detections."""

import argparse
from pathlib import Path

from hybrid_drive.experiments.spatial_appearance_benchmark import run_appearance_benchmark
from hybrid_drive.experiments.spatial_appearance_capture import INPUT_ROLES, reopen_appearance_capture
from hybrid_drive.experiments.spatial_tracking_benchmark import TrackingTruthFile
from scripts.capture_spatial_appearance import add_appearance_inputs, appearance_inputs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_appearance_inputs(parser)
    parser.add_argument("--appearance-capture-root", type=Path, required=True)
    parser.add_argument("--appearance-capture-sha256", required=True)
    for role in INPUT_ROLES:
        prefix = role.replace("_", "-")
        parser.add_argument(f"--{prefix}-truth", type=Path, required=True)
        parser.add_argument(f"--{prefix}-truth-sha256", required=True)
    args = parser.parse_args(argv)
    try:
        scope = appearance_inputs(args)
        capture = reopen_appearance_capture(
            args.appearance_capture_root, scope, expected_capture_sha256=args.appearance_capture_sha256,
        )
        truth = {
            role: TrackingTruthFile(getattr(args, role + "_truth"), getattr(args, role + "_truth_sha256"))
            for role in INPUT_ROLES
        }
        run, result, digest = run_appearance_benchmark(
            capture, truth, args.output_root, revision=args.revision, input_roots=(args.appearance_capture_root,),
        )
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Appearance benchmark failed: {exc}\n")
    print(f"{result['status']}: {run / 'report.html'}")
    print(f"Appearance benchmark SHA256: {digest}")
    if result.get("error"):
        print(result["error"])
    return 130 if result["status"] == "interrupted" else int(result["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
