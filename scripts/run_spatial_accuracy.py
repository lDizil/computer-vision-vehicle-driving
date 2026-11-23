"""Run the independent OpenScene threshold and box-refinement experiment."""

import argparse
from pathlib import Path

import torch

from hybrid_drive.experiments.spatial_accuracy import SpatialAccuracyRecipe, run_spatial_accuracy


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("reference-root", "calibration-root", "evaluation-root", "model-root", "prior-root", "output-root"):
        parser.add_argument(f"--{argument}", type=Path, required=True)
    parser.add_argument("--prior-result-sha256", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    root, result = run_spatial_accuracy(
        args.reference_root, args.calibration_root, args.evaluation_root, args.model_root, args.prior_root,
        args.output_root, revision=args.revision, expected_prior_result_sha256=args.prior_result_sha256,
        recipe=SpatialAccuracyRecipe(args.epochs, args.batch_size),
    )
    print(f"{result['status']}: {root / 'report.html'}")
    if result.get("error"):
        print(result["error"])
    if result["status"] == "interrupted":
        return 130
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
