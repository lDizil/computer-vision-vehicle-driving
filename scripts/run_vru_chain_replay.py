"""Replay and evaluate VRU motion repair on separately mounted artifacts."""

import argparse
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    capture = commands.add_parser("capture")
    _paths(capture, "inputs", "ssd-weights", "faster-weights", "output")
    capture.add_argument("--revision", required=True)
    forecast = commands.add_parser("forecast")
    _paths(forecast, "inputs", "capture", "output")
    forecast.add_argument("--capture-sha256", required=True)
    forecast.add_argument("--revision", required=True)
    truth = commands.add_parser("prepare-truth")
    _paths(truth, "metadata-archive", "sources", "inputs", "labels", "output")
    score = commands.add_parser("score")
    _paths(score, "inputs", "predictions", "labels", "history-truth", "output")
    score.add_argument("--prediction-sha256", required=True)
    args = parser.parse_args(argv)

    if args.command == "capture":
        import torch
        from hybrid_drive.experiments.vru_chain_capture import capture_chain_observations

        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        result = capture_chain_observations(
            args.inputs, args.ssd_weights, args.faster_weights, args.output, revision=args.revision,
        )
    elif args.command == "forecast":
        from hybrid_drive.experiments.vru_chain_replay import forecast_chain_replay

        result = forecast_chain_replay(
            args.inputs, args.capture, args.output, capture_sha256=args.capture_sha256, revision=args.revision,
        )
    elif args.command == "prepare-truth":
        from hybrid_drive.experiments.vru_chain_truth import prepare_vru_chain_truth

        result = prepare_vru_chain_truth(args.metadata_archive, args.sources, args.inputs, args.labels, args.output)
    else:
        from hybrid_drive.experiments.vru_chain_replay import score_chain_replay

        result = score_chain_replay(
            args.inputs, args.predictions, args.labels, args.history_truth, args.output,
            prediction_sha256=args.prediction_sha256,
        )
    print(result, flush=True)


def _paths(parser, *names):
    for name in names:
        parser.add_argument("--" + name, type=Path, required=True)


if __name__ == "__main__":
    main()
