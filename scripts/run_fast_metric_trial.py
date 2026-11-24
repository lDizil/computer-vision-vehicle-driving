"""Separate stages of the Fast Mini pilot. Mount targets only where necessary."""

import argparse
from pathlib import Path

import torch

from hybrid_drive.experiments.fast_capture import capture_fast
from hybrid_drive.experiments.fast_trial import fit_fast, predict_fast, score_fast


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, choices=(1,2), default=2)
    modes = parser.add_subparsers(dest="command", required=True)
    capture = modes.add_parser("capture")
    _paths(capture, "input-root", "model-root", "output")
    capture.add_argument("--revision", required=True)
    fit = modes.add_parser("fit")
    _paths(fit, "capture", "train-targets", "dev-targets", "output")
    fit.add_argument("--epochs", type=int, default=18)
    fit.add_argument("--revision", required=True)
    predict = modes.add_parser("predict")
    _paths(predict, "capture", "fit", "output")
    predict.add_argument("--revision", required=True)
    score = modes.add_parser("score")
    _paths(score, "capture", "predictions", "test-targets", "output")
    score.add_argument("--sealed-sha256", required=True)

    a = parser.parse_args(argv)
    torch.set_num_threads(a.threads)
    torch.use_deterministic_algorithms(True)
    if a.command == "capture":
        capture_fast(a.input_root, a.model_root, a.output, revision=a.revision)
    elif a.command == "fit":
        fit_fast(a.capture, a.train_targets, a.dev_targets, a.output, revision=a.revision, epochs=a.epochs)
    elif a.command == "predict":
        predict_fast(a.capture, a.fit, a.output, revision=a.revision)
    else:
        score_fast(a.capture, a.predictions, a.test_targets, a.output, sealed_sha256=a.sealed_sha256)


def _paths(parser, *names):
    for name in names:
        parser.add_argument("--" + name, type=Path, required=True)


if __name__ == "__main__":
    main()
