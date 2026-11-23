"""Capture one genuinely new holdout for offline guarded box refinement."""

import argparse
from pathlib import Path

import torch

from hybrid_drive.experiments.spatial_zoom_capture import open_zoom_inputs
from hybrid_drive.experiments.spatial_zoom_guard_capture import capture_guard_holdout, open_guard_inputs
from scripts.capture_spatial_appearance import appearance_inputs
from scripts.capture_spatial_video import video_packs
from scripts.capture_spatial_zoom import add_zoom_inputs


def add_guard_inputs(parser):
    add_zoom_inputs(parser)
    parser.add_argument("--fresh-root", type=Path, required=True)
    parser.add_argument("--known-zoom-root", type=Path, required=True)
    parser.add_argument("--known-zoom-sha256", required=True)


def guard_inputs(cfg):
    scope = appearance_inputs(cfg)
    known = open_zoom_inputs(scope, cfg.appearance_capture_root, expected_capture_sha256=cfg.appearance_capture_sha256)
    clips, earlier = video_packs(cfg)
    return open_guard_inputs(
        known, scope.model_receipt, cfg.known_zoom_root, cfg.fresh_root,
        (*clips.values(), *earlier.values(), cfg.tracking_root, cfg.appearance_root),
        known_zoom_sha256=cfg.known_zoom_sha256,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_guard_inputs(parser)
    for name in ("model-root", "prior-root", "accuracy-root"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--prior-result-sha256", required=True)
    parser.add_argument("--accuracy-result-sha256", required=True)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    try:
        scope = guard_inputs(args)
        root, digest = capture_guard_holdout(
            scope, args.model_root, args.prior_root, args.accuracy_root, args.output_root,
            revision=args.revision, prior_result_sha256=args.prior_result_sha256,
            accuracy_result_sha256=args.accuracy_result_sha256,
        )
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f"Fresh zoom capture failed: {exc}\n")
    print(f"Fresh zoom capture: {root}")
    print(f"Fresh zoom SHA256: {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
