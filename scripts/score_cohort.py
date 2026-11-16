"""Score baseline/checks/combined using original frozen native advisory evidence."""

import argparse
from pathlib import Path

from hybrid_drive.experiments.cohort_score import score_frozen_cohort


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data-root", "fast-run-root", "evidence-root", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--fallback-model", choices=("longitudinal", "curved"), default="longitudinal")
    parser.add_argument("--motion-observation-root", type=Path)
    args = parser.parse_args(argv)
    run_dir, result = score_frozen_cohort(args.data_root, args.fast_run_root, args.evidence_root, args.output_root,
                                          revision=args.revision, fallback_model=args.fallback_model,
                                          motion_observation_root=args.motion_observation_root)
    print(f"Cohort scores: {run_dir / 'result.json'}")
    return 0 if result["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
