"""Replay two camera bundles, calibrate only on development, score after the seal."""

import argparse
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    steps = parser.add_subparsers(dest="command", required=True)
    capture = steps.add_parser("capture")
    _paths(capture, "input-root", "weights-root", "output")
    capture.add_argument("--revision", required=True)
    capture.add_argument("--gains", type=Path)
    for name in ("calibrate", "score"):
        stage = steps.add_parser(name)
        _paths(stage, "input-root", "labels", "predictions", "output")
        stage.add_argument("--sealed-sha256", required=True)
        if name == "score":
            _paths(stage, "gains")
    args = parser.parse_args(argv)
    import torch
    from hybrid_drive.experiments.object_vru_trial import capture_object_motion, calibrate_object_motion, score_object_motion

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    if args.command == "capture":
        result = capture_object_motion(args.input_root, args.weights_root, args.output,
                                       revision=args.revision, gain_path=args.gains)
    elif args.command == "calibrate":
        result = calibrate_object_motion(args.input_root, args.labels, args.predictions, args.output,
                                         sealed_sha256=args.sealed_sha256)
    else:
        result = score_object_motion(args.input_root, args.labels, args.predictions, args.gains, args.output,
                                     sealed_sha256=args.sealed_sha256)
    print(result, flush=True)


def _paths(parser, *names):
    for name in names:
        parser.add_argument("--" + name, type=Path, required=True)


if __name__ == "__main__":
    main()
