"""Run the offline fixed clip benchmark and a separate development comparison."""

import argparse
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    _paths(plan, "metadata-archive", "sources", "output")
    prepare = commands.add_parser("prepare")
    _paths(prepare, "metadata-archive", "camera-root", "sources", "plan", "output")
    capture = commands.add_parser("capture-candidate")
    _paths(capture, "inputs", "models", "manifest", "output")
    capture.add_argument("--split", choices=("development", "control"), required=True)
    capture.add_argument("--device", choices=("mps", "cpu"), required=True)
    capture.add_argument("--revision", required=True)
    score = commands.add_parser("score-development")
    _paths(score, "inputs", "candidate", "baseline", "labels", "history", "projection", "output")
    score.add_argument("--candidate-sha256", required=True)
    score.add_argument("--baseline-sha256", required=True)
    args = parser.parse_args(argv)
    if args.command == "plan":
        from hybrid_drive.experiments.vru_clip_data import plan_clip_corpus

        result = plan_clip_corpus(args.metadata_archive, args.sources, args.output)
    elif args.command == "prepare":
        from hybrid_drive.experiments.vru_clip_data import prepare_clip_corpus

        result = prepare_clip_corpus(args.metadata_archive, args.camera_root, args.sources, args.plan, args.output)
    elif args.command == "capture-candidate":
        import torch
        from hybrid_drive.experiments.vru_clip_replay import capture_candidate

        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        result = capture_candidate(args.inputs, args.models, args.manifest, args.output,
                                   split=args.split, revision=args.revision, device=args.device)
    else:
        from hybrid_drive.experiments.vru_clip_replay import score_development

        result = score_development(args.inputs, args.candidate, args.baseline, args.labels, args.history,
                                   args.projection, args.output, candidate_sha256=args.candidate_sha256,
                                   baseline_sha256=args.baseline_sha256)
    print(result, flush=True)


def _paths(parser, *names):
    for name in names:
        parser.add_argument("--" + name, type=Path, required=True)


if __name__ == "__main__":
    main()
