"""Prepare only the frozen cohort's 18 original JPEGs from a verified local archive."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from hybrid_drive.assets.cohort import load_cohort_sources, prepare_cohort
from hybrid_drive.assets.files import PreparationError


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, default=Path(__file__).with_name("real-cohort-sources.json"))
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    try:
        manifest = prepare_cohort(args.root, load_cohort_sources(args.sources), args.archive)
    except (OSError, PreparationError) as exc:
        parser.exit(1, "Cohort preparation failed: " + str(exc) + "\n")
    print(json.dumps({"manifest_hash": manifest["manifest_hash"], "files": len(manifest["files"]), "root": str(args.root)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
