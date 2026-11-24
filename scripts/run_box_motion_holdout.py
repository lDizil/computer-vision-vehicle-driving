"""Run box motion on cached OpenScene six-view inputs and on separate targets."""

import argparse
from pathlib import Path
import torch

from hybrid_drive.experiments.box_motion_metrics import score_box_motion
from hybrid_drive.experiments.box_motion_trial import predict_box_motion


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="command",required=True)
    predict = modes.add_parser("predict")
    for key in ("input-root","model-root","frozen-fit","output"):
        predict.add_argument("--"+key,required=True,type=Path)
    predict.add_argument("--revision",required=True)
    score = modes.add_parser("score")
    for key in ("predictions","targets","output"):
        score.add_argument("--"+key,required=True,type=Path)
    score.add_argument("--sealed-sha256",required=True)
    ns=parser.parse_args(argv)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    if ns.command=="predict":
        predict_box_motion(ns.input_root,ns.model_root,ns.frozen_fit,ns.output,revision=ns.revision)
    else:
        score_box_motion(ns.predictions,ns.targets,ns.output,sealed_sha256=ns.sealed_sha256)


if __name__=="__main__":
    main()
