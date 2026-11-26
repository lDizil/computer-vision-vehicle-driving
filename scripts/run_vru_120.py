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
    capture.add_argument("--tiling", choices=("full", "front"), default="full")
    capture.add_argument("--revision", required=True)
    depthpro = commands.add_parser("capture-depthpro")
    _paths(depthpro, "inputs", "candidate", "worker", "compiled", "manifest", "output")
    depthpro.add_argument("--split", choices=("development", "control"), required=True)
    depthpro.add_argument("--candidate-sha256", required=True)
    depthpro.add_argument("--revision", required=True)
    depthpro.add_argument("--mask-model", type=Path)
    depthpro.add_argument("--mask-manifest", type=Path)
    depthpro.add_argument("--mask-device", choices=("cpu", "mps"), default="cpu")
    baseline = commands.add_parser("capture-baseline")
    _paths(baseline, "inputs", "ssd-weights", "faster-weights", "output")
    baseline.add_argument("--revision", required=True)
    score = commands.add_parser("score-development")
    _paths(score, "inputs", "candidate", "baseline", "labels", "history", "projection", "output")
    score.add_argument("--candidate-sha256", required=True)
    score.add_argument("--baseline-sha256", required=True)
    score.add_argument("--depthpro", type=Path)
    score.add_argument("--depthpro-sha256")
    release = commands.add_parser("freeze-release")
    _paths(release, "inputs", "development-score", "output")
    for field in ("development-sha256", "arm", "revision"):
        release.add_argument("--" + field, required=True)
    control = commands.add_parser("score-control")
    _paths(control, "inputs", "baseline", "candidate", "release", "labels", "output")
    for field in ("baseline-sha256", "candidate-sha256", "release-sha256"):
        control.add_argument("--" + field, required=True)
    control.add_argument("--depthpro", type=Path)
    control.add_argument("--depthpro-sha256")
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
                                   split=args.split, revision=args.revision, device=args.device, tiling=args.tiling)
    elif args.command == "capture-depthpro":
        from hybrid_drive.experiments.vru_clip_replay import capture_depthpro

        result = capture_depthpro(args.inputs, args.candidate, args.worker, args.compiled, args.manifest, args.output,
                                  split=args.split, candidate_sha256=args.candidate_sha256, revision=args.revision,
                                  mask_model=args.mask_model, mask_manifest=args.mask_manifest, mask_device=args.mask_device)
    elif args.command == "capture-baseline":
        import torch
        from hybrid_drive.experiments.vru_clip_control import capture_baseline

        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        result = capture_baseline(args.inputs, args.ssd_weights, args.faster_weights, args.output, revision=args.revision)
    elif args.command == "score-development":
        from hybrid_drive.experiments.vru_clip_replay import score_development

        result = score_development(args.inputs, args.candidate, args.baseline, args.labels, args.history,
                                   args.projection, args.output, candidate_sha256=args.candidate_sha256,
                                   baseline_sha256=args.baseline_sha256, depthpro_path=args.depthpro,
                                   depthpro_sha256=args.depthpro_sha256)
    elif args.command == "freeze-release":
        from hybrid_drive.experiments.vru_clip_control import freeze_release

        result = freeze_release(args.development_score, args.inputs, args.output,
                                dev_sha256=args.development_sha256, arm=args.arm, revision=args.revision)
    else:
        from hybrid_drive.experiments.vru_clip_control import score_control

        result = score_control(args.inputs, args.baseline, args.candidate, args.release, args.labels, args.output,
                               baseline_sha256=args.baseline_sha256, candidate_sha256=args.candidate_sha256,
                               release_sha256=args.release_sha256, depthpro_path=args.depthpro, depthpro_sha256=args.depthpro_sha256)
    print(result, flush=True)


def _paths(parser, *names):
    for name in names:
        parser.add_argument("--" + name, type=Path, required=True)


if __name__ == "__main__":
    main()
