"""Capture same-lattice ground-plane features from camera observations only."""

import argparse
from pathlib import Path

import torch

from hybrid_drive.experiments.ground_road_capture import capture_ground_road


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("mini-input-root", "mini-capture", "external-input-root", "model-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--threads", type=int, choices=(1, 2), default=2)
    args = parser.parse_args(argv)

    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    capture_ground_road(
        args.mini_input_root, args.mini_capture, args.external_input_root,
        args.model_root, args.output, revision=args.revision,
    )


if __name__ == "__main__":
    main()
