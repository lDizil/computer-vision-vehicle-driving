"""Freeze six real LTF candidates and current-camera bundles without scoring."""

import argparse
from pathlib import Path

from hybrid_drive.experiments.cohort_fast import run_cohort_fast


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data-root", "asset-root", "cohort-root", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sources", type=Path)
    args = parser.parse_args(argv)
    run_dir, manifest = run_cohort_fast(args.data_root, args.asset_root, args.cohort_root, args.output_root,
                                        revision=args.revision, seed=args.seed, sources_path=args.sources)
    print(f"Fast cohort: {run_dir / 'cohort.json'}")
    return 0 if all(case["status"] == "success" for case in manifest["cases"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
