"""Verify cached OpenScene archives and publish the frozen CAM_F0/LOC alignment slice."""

import argparse
import json
from pathlib import Path
import pickle

from hybrid_drive.assets.alignment_manifest import load_alignment_sources
from hybrid_drive.assets.alignment_sample import prepare_alignment_sample


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Run in an isolated Linux container with no network and read-only source archives.",
    )
    parser.add_argument("--root", type=Path, required=True, help="new directory under a writable parent")
    parser.add_argument("--metadata-archive", type=Path, required=True)
    parser.add_argument("--camera-archive", type=Path, required=True)
    parser.add_argument("--sources", type=Path, default=Path(__file__).with_name("alignment-data-sources.json"))
    args = parser.parse_args()
    try:
        sources = load_alignment_sources(args.sources)
        manifest = prepare_alignment_sample(args.root, sources, args.metadata_archive, args.camera_archive)
    except (OSError, ValueError, KeyError, pickle.UnpicklingError) as exc:
        parser.exit(1, f"Alignment preparation failed: {exc}\n")
    summary = {
        "dataset_hash": manifest["manifest_hash"],
        "splits": {
            split: {key: partition[key] for key in ("recording_id", "positive_count", "negative_count")}
            for split, partition in manifest["splits"].items()
        },
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
