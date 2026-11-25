"""Add older observations and check motion only after a two-link confirmation."""

import argparse
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    _paths(prepare, "metadata-archive", "camera-root", "sources", "base-root", "output")
    capture = commands.add_parser("capture")
    _paths(capture, "base", "extension", "baseline", "weights", "output")
    capture.add_argument("--revision", required=True)
    forecast = commands.add_parser("forecast")
    _paths(forecast, "base", "extension", "baseline", "objects", "gains", "output")
    forecast.add_argument("--revision", required=True)
    forecast.add_argument("--objects-sha256", required=True)
    score = commands.add_parser("score")
    _paths(score, "base", "extension", "predictions", "labels", "output")
    score.add_argument("--sealed-sha256", required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        from hybrid_drive.experiments.triple_vru_inputs import prepare_third_observations

        value = prepare_third_observations(args.metadata_archive, args.camera_root, args.sources, args.base_root, args.output)
    else:
        value = _trial(args)
    print(value, flush=True)


def _trial(args):
    import torch
    from hybrid_drive.experiments.triple_vru_trial import capture_triple, forecast_triple, score_triple

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    if args.command == "capture":
        return capture_triple(args.base, args.extension, args.baseline, args.weights, args.output,
                              revision=args.revision)
    if args.command == "forecast":
        return forecast_triple(args.base, args.extension, args.baseline, args.objects, args.gains,
                               args.output, revision=args.revision, evidence_sha256=args.objects_sha256)
    return score_triple(args.base, args.extension, args.predictions, args.labels, args.output,
                        sealed_sha256=args.sealed_sha256)


def _paths(parser, *names):
    for name in names:
        parser.add_argument("--" + name, type=Path, required=True)


if __name__ == "__main__":
    main()
