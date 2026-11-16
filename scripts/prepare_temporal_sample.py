"""Extract only the fixed sixty JPEGs from an already cached pinned archive; no downloads."""

import argparse
from pathlib import Path

from hybrid_drive.assets.temporal import load_temporal_sources, prepare_temporal_sample


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--archive-path", type=Path, required=True)
    parser.add_argument("--sources", type=Path)
    args = parser.parse_args(argv)
    manifest = prepare_temporal_sample(args.output_root, load_temporal_sources(args.sources), args.archive_path)
    print(f"Temporal sample: {len(manifest['files'])} verified JPEGs in {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
