"""Capture frozen camera predictions for all three pinned clip packs without opening labels."""

import argparse
from pathlib import Path

import torch

from hybrid_drive.experiments.temporal_spatial_capture import capture_video


def add_video_packs(parser):
    for flag in ("development-root", "holdout-root", "stress-root", "reference-root", "previous-dev-root",
                 "accuracy-holdout-root"):
        parser.add_argument("--" + flag, type=Path, required=True)


def video_packs(args):
    return (
        {"development": args.development_root, "holdout": args.holdout_root, "stress": args.stress_root},
        {"reference": args.reference_root, "previous_final_reused_as_dev": args.previous_dev_root,
         "evaluation": args.accuracy_holdout_root},
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_video_packs(parser)
    for flag in ("model-root", "prior-root", "accuracy-root", "output-root"):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--prior-result-sha256", required=True)
    parser.add_argument("--accuracy-result-sha256", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    roles, previous = video_packs(args)
    try:
        root, digest = capture_video(
            roles, previous, args.model_root, args.prior_root, args.accuracy_root, args.output_root,
            revision=args.revision, prior_result_sha256=args.prior_result_sha256,
            accuracy_result_sha256=args.accuracy_result_sha256, batch_size=args.batch_size,
        )
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f"Video capture failed: {exc}\n")
    print(f"Captured frozen observations: {root}")
    print(f"Capture SHA256: {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
