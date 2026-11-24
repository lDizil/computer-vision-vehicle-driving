"""Prepare the recording-disjoint Fast forecast pilot from cached nuScenes Mini."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tarfile

from hybrid_drive.experiments.fast_mini_data import prepare_fast_mini


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        receipt = prepare_fast_mini(args.archive, args.output)
    except (OSError, ValueError, tarfile.TarError) as exc:
        parser.exit(1, f"Fast Mini preparation failed: {exc}\n")
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
