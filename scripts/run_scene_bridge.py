"""Replay observed road differences and verify the exact Fast-to-Slow handoff."""

import argparse
from pathlib import Path

import torch

from hybrid_drive.experiments.paired_road_replay import predict_paired_replay, score_paired_replay
from hybrid_drive.experiments.scene_bridge_replay import run_handoff


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="command",required=True)
    replay=sub.add_parser("replay")
    _paths(replay,"ground-predictions","ground-capture","output")
    replay.add_argument("--revision",required=True)
    score=sub.add_parser("score")
    _paths(score,"ground-predictions","ground-capture","replay","targets","output")
    score.add_argument("--sealed-sha256",required=True)
    bridge=sub.add_parser("handoff")
    _paths(bridge,"input-root","model-root","box-fit","road-fit","old-objects","ground-predictions","ground-capture","output")
    bridge.add_argument("--revision",required=True)
    a=p.parse_args(argv)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    if a.command=="replay":
        predict_paired_replay(a.ground_predictions,a.ground_capture,a.output,revision=a.revision)
    elif a.command=="score":
        score_paired_replay(a.ground_predictions,a.ground_capture,a.replay,a.targets,a.output,sealed_sha256=a.sealed_sha256)
    else:
        run_handoff(a.input_root,a.model_root,a.box_fit,a.road_fit,a.old_objects,
                    a.ground_predictions,a.ground_capture,a.output,revision=a.revision)


def _paths(parser,*names):
    for name in names:
        parser.add_argument("--"+name,type=Path,required=True)


if __name__=="__main__":
    main()
