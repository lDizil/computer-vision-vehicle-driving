"""Prepare observations, repeat frozen prediction, then open outcome annotations."""

import argparse
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    _paths(prep, "metadata-archive", "camera-root", "sources", "output")
    predict = commands.add_parser("predict")
    _paths(predict, "input-root", "model-root", "box-fit", "road-fit", "slow-fit", "output")
    predict.add_argument("--revision", required=True)
    score = commands.add_parser("score")
    _paths(score, "predictions", "input-root", "selection", "sources", "metadata-archive", "output")
    score.add_argument("--sealed-sha256", required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        from hybrid_drive.experiments.frozen_transfer_data import prepare_transfer

        result = prepare_transfer(args.metadata_archive, args.camera_root, args.sources, args.output)
        print(result, flush=True)
    elif args.command == "predict":
        import torch
        from hybrid_drive.experiments.frozen_transfer_predict import predict_transfer

        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        result = predict_transfer(args.input_root, args.model_root, args.box_fit, args.road_fit, args.slow_fit,
                                  args.output, revision=args.revision)
        print(result, flush=True)
    else:
        from hybrid_drive.experiments.frozen_transfer_score import score_transfer

        score_transfer(args.predictions, args.sealed_sha256, args.input_root, args.selection,
                       args.sources, args.metadata_archive, args.output)


def _paths(parser, *names):
    for name in names:
        parser.add_argument("--" + name, type=Path, required=True)


if __name__ == "__main__":
    main()
