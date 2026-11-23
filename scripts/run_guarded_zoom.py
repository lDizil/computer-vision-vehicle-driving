"""Select a guarded camera-box correction on dev and score it once on fresh clips."""

import argparse
from pathlib import Path

import torch

from hybrid_drive.experiments.spatial_tracking_benchmark import TrackingTruthFile
from hybrid_drive.experiments.spatial_zoom_guard_benchmark import TRUTH_ROLES, run_guarded_zoom
from hybrid_drive.experiments.spatial_zoom_guard_capture import reopen_guard_holdout
from scripts.capture_guarded_zoom import add_guard_inputs, guard_inputs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_guard_inputs(parser)
    parser.add_argument("--fresh-capture-root", type=Path, required=True)
    parser.add_argument("--fresh-capture-sha256", required=True)
    for role in TRUTH_ROLES:
        flag = role.replace("_", "-")
        parser.add_argument(f"--{flag}-truth", type=Path, required=True)
        parser.add_argument(f"--{flag}-truth-sha256", required=True)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    try:
        inputs = guard_inputs(args)
        fresh = reopen_guard_holdout(args.fresh_capture_root, inputs, expected_sha256=args.fresh_capture_sha256)
        truth = {
            role: TrackingTruthFile(getattr(args, f"{role}_truth"), getattr(args, f"{role}_truth_sha256"))
            for role in TRUTH_ROLES
        }
        root, result, digest = run_guarded_zoom(
            inputs, fresh, truth, args.output_root, revision=args.revision, extra_inputs=(args.fresh_capture_root,),
        )
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"Guarded box benchmark failed: {exc}\n")

    print(f"{result['status']}: {root / 'result.json'}")
    if (root / "report.html").is_file():
        print(f"Report: {root / 'report.html'}")
    if "matrix" in result:
        diagnostic = result["matrix"]["holdout_diagnostic"]
        print(f"Dev selection: {diagnostic['evaluated_recipe']}; fresh holdout: {diagnostic['outcome']}")
    print(f"Guarded box benchmark SHA256: {digest}")
    if "error" in result:
        print(result["error"])
    if "report_error" in result:
        print(result["report_error"])
    return 130 if result["status"] == "interrupted" else int(result["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
