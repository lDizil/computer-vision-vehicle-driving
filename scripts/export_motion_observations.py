"""Export current/past-only motion sidecars without changing frozen Fast or Slow evidence."""

import argparse
from pathlib import Path

from hybrid_drive.experiments.motion_observation import export_motion_observations


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data-root", "fast-run-root", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args(argv)
    run_dir, index = export_motion_observations(args.data_root, args.fast_run_root, args.output_root, revision=args.revision)
    print(f"Motion observations: {run_dir / 'index.json'}")
    return 0 if all(row["status"] == "success" for row in index["cases"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
