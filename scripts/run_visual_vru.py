"""Prepare, fit and independently score an offline six-camera VRU pilot."""

import argparse
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    _paths(prepare, "metadata-archive", "camera-root", "sources", "output")
    capture = commands.add_parser("capture")
    _paths(capture, "input-root", "model-root", "output")
    capture.add_argument("--revision", required=True)
    fit = commands.add_parser("fit")
    for split in ("train", "development"):
        _paths(fit, f"{split}-inputs", f"{split}-labels", f"{split}-features")
    _paths(fit, "output")
    fit.add_argument("--revision", required=True)
    predict = commands.add_parser("predict")
    _paths(predict, "input-root", "features-root", "fit-root", "output")
    predict.add_argument("--revision", required=True)
    score = commands.add_parser("score")
    _paths(score, "input-root", "labels", "predictions", "output")
    score.add_argument("--sealed-sha256", required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        from hybrid_drive.experiments.visual_vru_data import prepare_vru_data

        result = prepare_vru_data(args.metadata_archive, args.camera_root, args.sources, args.output)
    else:
        result = _run_torch(args)
    if args.command in ("prepare", "predict"):
        print(result, flush=True)


def _run_torch(args):
    import torch

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    if args.command == "capture":
        from hybrid_drive.experiments.visual_vru_capture import capture_vru

        return capture_vru(args.input_root, args.model_root, args.output, revision=args.revision)
    if args.command == "fit":
        from hybrid_drive.experiments.visual_vru_training import fit_vru

        train = (args.train_inputs, args.train_labels, args.train_features)
        development = (args.development_inputs, args.development_labels, args.development_features)
        return fit_vru(train, development, args.output, revision=args.revision)
    if args.command == "predict":
        from hybrid_drive.experiments.visual_vru_eval import predict_vru

        return predict_vru(args.input_root, args.features_root, args.fit_root, args.output, revision=args.revision)
    from hybrid_drive.experiments.visual_vru_eval import score_vru

    return score_vru(args.input_root, args.labels, args.predictions, args.output, sealed_sha256=args.sealed_sha256)


def _paths(parser, *names):
    for name in names:
        parser.add_argument("--" + name, type=Path, required=True)


if __name__ == "__main__":
    main()
