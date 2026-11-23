"""Encode existing box-centered RGB crops with the installed frozen vision model."""

import argparse
from pathlib import Path

import torch

from hybrid_drive.experiments.spatial_zoom_capture import capture_zoom_grids, open_zoom_inputs
from scripts.capture_spatial_appearance import add_appearance_inputs, appearance_inputs


def add_zoom_inputs(parser):
    add_appearance_inputs(parser)
    parser.add_argument("--appearance-capture-root", type=Path, required=True)
    parser.add_argument("--appearance-capture-sha256", required=True)


def zoom_inputs(args):
    earlier = appearance_inputs(args)
    return open_zoom_inputs(earlier, args.appearance_capture_root, expected_capture_sha256=args.appearance_capture_sha256)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_zoom_inputs(parser)
    parser.add_argument("--model-root", type=Path, required=True)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    try:
        inputs = zoom_inputs(args)
        root, digest = capture_zoom_grids(inputs, args.model_root, args.output_root, revision=args.revision)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f"Zoom capture failed: {exc}\n")
    print(f"Zoom features: {root}")
    print(f"Zoom capture SHA256: {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
