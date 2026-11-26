"""Fit and evaluate an isolated constant-height projection alternative."""

import argparse
from pathlib import Path

from hybrid_drive.experiments.vru_plane_probe import fit_plane, forecast_plane, score_plane


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("fit")
    _paths(fit, "inputs", "labels", "truth", "output")
    forecast = commands.add_parser("forecast")
    _paths(forecast, "inputs", "trace", "fit", "output")
    for option in ("trace-sha256", "fit-sha256", "revision"):
        forecast.add_argument("--" + option, required=True)
    score = commands.add_parser("score")
    _paths(score, "inputs", "predictions", "labels", "output")
    score.add_argument("--sealed-sha256", required=True)
    args = parser.parse_args(argv)
    if args.command == "fit":
        result = fit_plane(args.inputs, args.labels, args.truth, args.output)
    elif args.command == "forecast":
        result = forecast_plane(args.inputs, args.trace, args.fit, args.output, trace_sha256=args.trace_sha256,
                                fit_sha256=args.fit_sha256, revision=args.revision)
    else:
        result = score_plane(args.inputs, args.predictions, args.labels, args.output, sealed_sha256=args.sealed_sha256)
    print(result, flush=True)


def _paths(parser, *names):
    for name in names:
        parser.add_argument("--" + name, type=Path, required=True)


if __name__ == "__main__":
    main()
