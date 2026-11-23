"""Capture predictions on new tracking clips, without reading target or track IDs."""

import argparse
from pathlib import Path

import torch

from hybrid_drive.experiments.spatial_tracking_capture import capture_tracking_holdout, open_tracking_packs
from scripts.capture_spatial_video import add_video_packs, video_packs


def add_tracking_inputs(parser):
    add_video_packs(parser)
    for name in ("video-capture-root", "tracking-root", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--video-capture-sha256", required=True)
    parser.add_argument("--revision", required=True)


def tracking_inputs(args):
    clips, previous = video_packs(args)
    prior, final = open_tracking_packs(
        clips, previous, args.video_capture_root, args.tracking_root, video_capture_sha256=args.video_capture_sha256,
    )
    return prior, final, (*clips.values(), *previous.values(), args.video_capture_root)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_tracking_inputs(parser)
    for name in ("model-root", "prior-root", "accuracy-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--prior-result-sha256", required=True)
    parser.add_argument("--accuracy-result-sha256", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    try:
        old, final, protected = tracking_inputs(args)
        root, digest = capture_tracking_holdout(
            old, final, args.model_root, args.prior_root, args.accuracy_root, args.output_root,
            revision=args.revision, prior_result_sha256=args.prior_result_sha256,
            accuracy_result_sha256=args.accuracy_result_sha256, protected_roots=protected, batch_size=args.batch_size,
        )
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f"Tracking capture failed: {exc}\n")
    print(f"Captured {len(final.frames)} holdout frames: {root}")
    print(f"Tracking capture SHA256: {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
