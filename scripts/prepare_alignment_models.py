"""Download the pinned DINOv2 / Pythia files for an offline alignment probe."""

import argparse
import json
from pathlib import Path

from hybrid_drive.assets.alignment_models import SOURCES_FILENAME, prepare_alignment_models
from hybrid_drive.assets.files import PreparationError


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--rate-limit-mib", type=float, default=4)
    args = parser.parse_args()
    try:
        result = prepare_alignment_models(
            args.root, Path(__file__).with_name(SOURCES_FILENAME), rate_limit_mib=args.rate_limit_mib
        )
    except PreparationError as exc:
        parser.exit(1, f"{exc}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
