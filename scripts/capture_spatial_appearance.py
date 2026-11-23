"""Capture the same detections and standalone visual descriptors for offline tracking."""

import argparse
from pathlib import Path

import torch

from hybrid_drive.experiments.spatial_appearance_capture import capture_appearance, open_appearance_inputs
from scripts.capture_spatial_tracking import add_tracking_inputs
from scripts.capture_spatial_video import video_packs


def add_appearance_inputs(parser):
    add_tracking_inputs(parser)
    for name in ("tracking-capture-root", "appearance-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--tracking-capture-sha256", required=True)


def appearance_inputs(cfg):
    clips, previous = video_packs(cfg)
    return open_appearance_inputs(
        clips, previous, cfg.video_capture_root, cfg.tracking_root, cfg.tracking_capture_root, cfg.appearance_root,
        video_capture_sha256=cfg.video_capture_sha256, tracking_capture_sha256=cfg.tracking_capture_sha256,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_appearance_inputs(parser)
    for role in ("model-root", "prior-root", "accuracy-root"):
        parser.add_argument("--" + role, type=Path, required=True)
    parser.add_argument("--prior-result-sha256", required=True)
    parser.add_argument("--accuracy-result-sha256", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    try:
        scope = appearance_inputs(args)
        root, sha = capture_appearance(
            scope, args.model_root, args.prior_root, args.accuracy_root, args.output_root,
            revision=args.revision, prior_result_sha256=args.prior_result_sha256,
            accuracy_result_sha256=args.accuracy_result_sha256, batch_size=args.batch_size,
        )
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f"Appearance capture failed: {exc}\n")
    print(f"Appearance observations captured: {root}")
    print(f"Appearance capture SHA256: {sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
