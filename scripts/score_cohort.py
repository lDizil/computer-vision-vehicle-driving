"""Score baseline/checks/combined using original frozen native advisory evidence."""

import argparse
from pathlib import Path

from hybrid_drive.experiments.cohort_score import score_frozen_cohort


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data-root", "fast-run-root", "evidence-root", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args(argv)
    run_dir, result = score_frozen_cohort(args.data_root, args.fast_run_root, args.evidence_root, args.output_root,
                                          revision=args.revision)
    print(f"Cohort scores: {run_dir / 'result.json'}")
    return 0 if result["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
