"""Verify and export separate opaque tracking labels for a pinned temporal spatial pack."""

import argparse
import json
from pathlib import Path

from hybrid_drive.assets.spatial_manifest import TEMPORAL_ROLES
from hybrid_drive.assets.spatial_temporal import read_temporal_spatial_pack
from hybrid_drive.assets.spatial_track_truth import prepare_track_truth


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="existing prepared temporal pack")
    parser.add_argument("--role", choices=TEMPORAL_ROLES, required=True)
    parser.add_argument("--metadata-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new sidecar path outside the pack")
    args = parser.parse_args()
    try:
        pack = read_temporal_spatial_pack(args.root, role=args.role)
        digest = prepare_track_truth(pack, args.metadata_archive, args.output)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"Track truth preparation failed: {exc}\n")
    print(json.dumps({"pack_hash": pack.manifest_hash, "sidecar_sha256": digest, "frames": len(pack.frames)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
