"""Compare #44 road evidence with ground projection and historical support."""

import argparse
from pathlib import Path

import torch

from hybrid_drive.experiments.ground_road_trial import fit_ground_road, predict_ground_road, score_ground_road


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("fit")
    _paths(fit, "capture", "mini-capture", "train-targets", "dev-targets", "output")
    fit.add_argument("--revision", required=True)
    infer = sub.add_parser("predict")
    _paths(infer, "capture", "fitted", "baseline", "objects", "output")
    infer.add_argument("--revision", required=True)
    score = sub.add_parser("score")
    _paths(score, "capture", "predictions", "targets", "output")
    score.add_argument("--sealed-sha256", required=True)
    a = p.parse_args(argv)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    if a.command == "fit":
        fit_ground_road(a.capture, a.mini_capture, a.train_targets, a.dev_targets, a.output, revision=a.revision)
    elif a.command == "predict":
        predict_ground_road(a.capture, a.fitted, a.baseline, a.objects, a.output, revision=a.revision)
    else:
        score_ground_road(a.capture,a.predictions,a.targets,a.output,sealed_sha256=a.sealed_sha256)


def _paths(command, *names):
    for name in names:
        command.add_argument("--"+name, type=Path, required=True)


if __name__ == "__main__":
    main()
