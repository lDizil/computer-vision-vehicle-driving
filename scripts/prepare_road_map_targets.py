"""Build separate road label packs from cached nuScenes Mini and vector-map ZIP."""

import argparse
import json
from pathlib import Path

from hybrid_drive.experiments.road_map_targets import prepare_road_map_targets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--metadata-archive", type=Path, required=True)
    parser.add_argument("--map-archive", type=Path, required=True)
    parser.add_argument("--map-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = prepare_road_map_targets(
        args.capture, args.metadata_archive, args.map_archive, args.map_sha256, args.output,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
