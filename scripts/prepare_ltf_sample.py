#!/usr/bin/env python3
"""Prepare pinned real LTF weights and one three-camera OpenScene frame.

This does not download maps/logs: run prepare_navsim_sample.py for those assets.
Model files remain opaque here; torch checkpoint loading belongs in the container.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

# Also allow checkout invocation when project dependencies are installed.
# The reproducible entry point runs inside the pinned CPU image.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hybrid_drive.assets.cameras import docker_jpeg_validator, validate_jpeg
from hybrid_drive.assets.files import PreparationError
from hybrid_drive.assets.prepare import prepare


SOURCES_PATH = Path(__file__).with_name("ltf-sample-sources.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Isolated writable data/cache root")
    parser.add_argument("--rate-limit-mib", type=float, default=4, help="Maximum aggregate network rate, MiB/s (default4)")
    parser.add_argument("--jpeg-validator-image", help="Existing Pillow-equipped Docker image for host-only preparation")
    args = parser.parse_args()
    try:
        sources = json.loads(SOURCES_PATH.read_text())
        validator = docker_jpeg_validator(args.jpeg_validator_image) if args.jpeg_validator_image else validate_jpeg
        prepare(args.root, sources, rate_limit_mib=args.rate_limit_mib, validator=validator)
    except (PreparationError, OSError, KeyError, ValueError) as exc:
        parser.exit(1, f"Preparation failed: {exc}\n")


if __name__ == "__main__":
    main()
