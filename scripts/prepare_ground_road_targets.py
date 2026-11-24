"""Prepare external map targets; run only after sealing the blind predictions."""

from argparse import ArgumentParser
import json
from pathlib import Path

from hybrid_drive.experiments.ground_road_targets import prepare_ground_road_targets


def main():
    parser = ArgumentParser(description=__doc__)
    for key in ("input-root", "metadata-archive", "maps-root", "output"):
        parser.add_argument(f"--{key}", required=True, type=Path)
    args = parser.parse_args()
    scripts = Path(__file__).parent
    summary = prepare_ground_road_targets(
        args.input_root, args.metadata_archive, args.maps_root, args.output,
        motion_sources=scripts / "box-motion-sources.json", navsim_sources=scripts / "navsim-sample-sources.json",
    )
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
