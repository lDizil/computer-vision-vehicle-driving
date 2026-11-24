"""Prepare an isolated label / six-camera holdout from pinned local archives."""

from argparse import ArgumentParser
from pathlib import Path

from hybrid_drive.experiments.box_motion_data import prepare_motion_holdout


def main():
    parser = ArgumentParser(description=__doc__)
    for key in ("metadata-archive", "camera-archive", "output"):
        parser.add_argument(f"--{key}", required=True, type=Path)
    args = parser.parse_args()
    sources = Path(__file__).with_name("box-motion-sources.json")
    result = prepare_motion_holdout(args.metadata_archive, args.camera_archive, sources, args.output)
    print(f"Prepared box-motion holdout: {result}")


if __name__ == "__main__":
    main()
