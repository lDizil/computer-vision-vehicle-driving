"""Trace VRU detection, reconstruct annotation envelopes and score the same frames."""

import argparse
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    capture = stages.add_parser("capture")
    _paths(capture, "inputs", "ssd-weights", "faster-weights", "output")
    capture.add_argument("--revision", required=True)
    truth = stages.add_parser("prepare-truth")
    _paths(truth, "metadata-archive", "sources", "inputs", "labels", "output")
    score = stages.add_parser("score")
    _paths(score, "inputs", "trace", "reference", "truth", "labels", "output")
    score.add_argument("--trace-sha256", required=True)
    score.add_argument("--reference-sha256", required=True)
    args = parser.parse_args(argv)
    if args.stage == "capture":
        import torch
        from hybrid_drive.experiments.vru_projection_audit import capture_projection_audit

        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        result = capture_projection_audit(args.inputs, args.ssd_weights, args.faster_weights, args.output,
                                          revision=args.revision)
    elif args.stage == "prepare-truth":
        from hybrid_drive.experiments.vru_projection_truth import prepare_projection_truth

        result = prepare_projection_truth(args.metadata_archive, args.sources, args.inputs, args.labels, args.output)
    else:
        from hybrid_drive.experiments.vru_projection_audit import score_projection_audit

        result = score_projection_audit(args.inputs, args.trace, args.reference, args.truth, args.labels, args.output,
                                        trace_sha256=args.trace_sha256, reference_sha256=args.reference_sha256)
    print(result, flush=True)


def _paths(parser, *names):
    for name in names:
        parser.add_argument("--" + name, type=Path, required=True)


if __name__ == "__main__":
    main()
