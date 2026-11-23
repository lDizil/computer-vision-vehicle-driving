"""Compare fixed and frozen spatial RoIs offline with or without direct patch memory."""

import argparse
from pathlib import Path

import torch

from hybrid_drive.experiments.proposal_alignment import ProposalRecipe, run_proposal_alignment


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--fresh-holdout-root", type=Path,
        help="pinned separate holdout; use only the four ROI tokens as visual memory and save a local gallery",
    )
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--prior-run-root", type=Path, required=True)
    parser.add_argument("--prior-result-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seeds", nargs="+", type=int, default=(0, 1))
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    directory, result = run_proposal_alignment(
        args.data_root,
        args.model_root,
        args.prior_run_root,
        args.output_root,
        revision=args.revision,
        expected_prior_result_sha256=args.prior_result_sha256,
        recipe=ProposalRecipe(args.epochs, args.batch_size, tuple(args.seeds)),
        fresh_holdout_root=args.fresh_holdout_root,
    )
    print(f"{result['status']}: {directory / 'report.html'}")
    if result.get("error"):
        print(result["error"])
    if result["status"] == "interrupted":
        return 130
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
