"""Prepare, capture, and blind-score the heldout VRU detector fusion trial."""

import argparse
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    _paths(prepare, "metadata-archive", "camera-root", "sources", "output")
    capture = commands.add_parser("capture")
    _paths(capture, "inputs", "ssd-weights", "faster-weights", "output")
    capture.add_argument("--revision", required=True)
    score = commands.add_parser("score")
    _paths(score, "inputs", "predictions", "labels", "output")
    score.add_argument("--sealed-sha256", required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        from hybrid_drive.experiments.vru_fusion_holdout_data import prepare_fusion_holdout

        result = prepare_fusion_holdout(args.metadata_archive, args.camera_root, args.sources, args.output)
    else:
        import torch
        from hybrid_drive.experiments.vru_fusion_holdout_trial import capture_fusion_holdout, score_fusion_holdout

        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        if args.command == "capture":
            result = capture_fusion_holdout(args.inputs, args.ssd_weights, args.faster_weights, args.output, revision=args.revision)
        else:
            result = score_fusion_holdout(args.inputs, args.predictions, args.labels, args.output, sealed_sha256=args.sealed_sha256)
    print(result, flush=True)


def _paths(parser, *names):
    for name in names:
        parser.add_argument("--" + name, type=Path, required=True)


if __name__ == "__main__":
    main()
