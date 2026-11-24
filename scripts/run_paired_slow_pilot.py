"""Fit, freeze and separately score the paired Bridge → Slow pilot."""

import argparse
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("fit", "predict"):
        stage = commands.add_parser(command)
        _paths(stage, "input-root", "model-root", "box-fit", "road-fit", "output")
        stage.add_argument("--revision", required=True)
        if command == "predict":
            _paths(stage, "fit-root")
    natural = commands.add_parser("score-natural")
    _paths(natural, "predictions", "metadata-archive", "sources", "input-index", "output")
    natural.add_argument("--sealed-sha256", required=True)
    controls = commands.add_parser("score-controls")
    _paths(controls, "predictions", "targets", "output")
    controls.add_argument("--sealed-sha256", required=True)
    args = parser.parse_args(argv)
    if args.command == "score-natural":
        from hybrid_drive.experiments.paired_slow_natural import score_natural

        score_natural(args.predictions, args.sealed_sha256, args.metadata_archive, args.sources, args.input_index,
                      args.output)
    elif args.command == "score-controls":
        from hybrid_drive.experiments.paired_slow_metrics import score_controls

        score_controls(args.predictions, args.targets, args.output, sealed_sha256=args.sealed_sha256)
    else:
        import torch
        from hybrid_drive.experiments.paired_slow_pilot import run_fit, run_predictions

        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        common = (args.input_root, args.model_root, args.box_fit, args.road_fit)
        if args.command == "fit":
            run_fit(*common, args.output, revision=args.revision)
        else:
            run_predictions(*common, args.fit_root, args.output, revision=args.revision)


def _paths(parser, *names):
    for name in names:
        parser.add_argument("--" + name, type=Path, required=True)


if __name__ == "__main__":
    main()
