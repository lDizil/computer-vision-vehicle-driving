"""Compare two frozen detector profiles on the same heldout observation bytes."""

import argparse
from pathlib import Path


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    c=p.add_subparsers(dest="command",required=True)
    predict=c.add_parser("capture")
    paths(predict,"base","reference","weights","output")
    predict.add_argument("--revision",required=True)
    score=c.add_parser("score")
    paths(score,"base","reference","candidate","labels","output")
    score.add_argument("--sealed-sha256",required=True)
    args=p.parse_args(argv)
    import torch
    from hybrid_drive.experiments.vru_detector_coverage_trial import capture_coverage, score_coverage

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    if args.command=="capture":
        result=capture_coverage(args.base,args.reference,args.weights,args.output,revision=args.revision)
    else:
        result=score_coverage(args.base,args.reference,args.candidate,args.labels,args.output,sealed_sha256=args.sealed_sha256)
    print(result,flush=True)


def paths(parser,*names):
    for name in names:
        parser.add_argument("--"+name,type=Path,required=True)


if __name__=="__main__":
    main()
