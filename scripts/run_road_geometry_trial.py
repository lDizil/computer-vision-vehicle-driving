"""Fit, freeze, rank, then evaluate road layers from the captured Fast features."""

import argparse
from pathlib import Path

import torch

from hybrid_drive.experiments.road_trial import fit_roads, predict_roads, score_roads


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="command", required=True)
    fit = modes.add_parser("fit")
    _paths(fit, "capture", "train-targets", "dev-targets", "output")
    fit.add_argument("--epochs", type=int, default=18)
    fit.add_argument("--revision", required=True)
    infer = modes.add_parser("predict")
    _paths(infer, "capture", "fitted", "box-predictions", "output")
    infer.add_argument("--revision", required=True)
    score = modes.add_parser("score")
    _paths(score, "capture", "predictions", "map-targets", "box-predictions", "output")
    score.add_argument("--sealed-sha256", required=True)
    args = parser.parse_args(argv)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    if args.command=="fit":
        fit_roads(args.capture,args.train_targets,args.dev_targets,args.output,revision=args.revision,epochs=args.epochs)
    elif args.command=="predict":
        predict_roads(args.capture,args.fitted,args.box_predictions,args.output,revision=args.revision)
    else:
        score_roads(args.capture,args.predictions,args.map_targets,args.box_predictions,args.output,
                    sealed_sha256=args.sealed_sha256)


def _paths(parser, *names):
    for name in names:
        parser.add_argument("--" + name, type=Path, required=True)


if __name__=="__main__":
    main()
