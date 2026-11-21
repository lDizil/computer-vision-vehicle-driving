"""Run the fixed recording-disjoint spatial experiment offline on CPU."""

import argparse
from pathlib import Path

import torch

from hybrid_drive.experiments.spatial_experiment import SpatialRecipe, run_spatial_experiment


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seeds", nargs="+", type=int, default=(0, 1))
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    directory, result = run_spatial_experiment(
        args.data_root,
        args.model_root,
        args.output_root,
        revision=args.revision,
        recipe=SpatialRecipe(args.epochs, args.batch_size, tuple(args.seeds)),
    )
    print(f"{result['status']}: {directory / 'report.html'}")
    if result.get("error"):
        print(result["error"])
    if result["status"] == "interrupted":
        return 130
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
