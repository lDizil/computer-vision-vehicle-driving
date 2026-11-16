"""Freeze the twenty temporally fixed LTF outputs and measured inference latencies."""

import argparse
from pathlib import Path

from hybrid_drive.experiments.temporal_fast import run_temporal_fast


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data-root", "asset-root", "temporal-root", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    run_dir, manifest = run_temporal_fast(args.data_root,args.asset_root,args.temporal_root,args.output_root,
                                           revision=args.revision,seed=args.seed)
    print(f"Frozen temporal Fast: {run_dir}")
    return 0 if all(case["status"] == "success" for case in manifest["cases"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
