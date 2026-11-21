"""Verify cached OpenScene archives and publish the frozen four-band CAM_F0/LOC dataset."""

import argparse
import json
from pathlib import Path
import pickle

from hybrid_drive.assets.spatial_manifest import BAND_COUNT, SPLITS, load_spatial_sources, read_spatial_targets
from hybrid_drive.assets.spatial_sample import prepare_spatial_sample


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Run in an isolated Linux container with no network and read-only source archives.",
    )
    parser.add_argument("--root", type=Path, required=True, help="new directory under a writable parent")
    parser.add_argument("--metadata-archive", type=Path, required=True)
    parser.add_argument("--camera-archive", required=True, action="append", metavar="SHARD=PATH")
    parser.add_argument("--sources", type=Path, default=Path(__file__).with_name("spatial-loc-sources.json"))
    parser.add_argument("--report-split", action="append", choices=SPLITS,
                        help="include label counts for this split (repeatable; default omits all label counts)")
    args = parser.parse_args()
    archives = {}
    for archive in args.camera_archive:
        shard, separator, path = archive.partition("=")
        if not separator or not path or shard in archives:
            parser.error("--camera-archive must be a unique SHARD=PATH")
        archives[shard] = Path(path)
    try:
        sources = load_spatial_sources(args.sources)
        manifest = prepare_spatial_sample(args.root, sources, args.metadata_archive, archives)
        summary = _summary(args.root, manifest, args.report_split or ())
    except (OSError, ValueError, KeyError, pickle.UnpicklingError) as exc:
        parser.exit(1, f"Spatial preparation failed: {exc}\n")
    print(json.dumps(summary, sort_keys=True))
    return 0


def _summary(root, manifest, report_splits):
    splits = {}
    for split, partition in manifest["splits"].items():
        recordings = [{"recording_id": item["recording_id"], "sample_count": len(item["samples"])}
                      for item in partition["recordings"]]
        count = sum(item["sample_count"] for item in recordings)
        entry = {"recordings": recordings, "sample_count": count}
        if split in report_splits:
            rows = read_spatial_targets(root, manifest, split)
            offset = 0
            total_positives = [0] * BAND_COUNT
            for recording in recordings:
                size = recording["sample_count"]
                positives = [sum(row["presence"][band] for row in rows[offset:offset + size])
                             for band in range(BAND_COUNT)]
                recording.update(band_positive_counts=positives, band_negative_counts=[size - n for n in positives])
                total_positives = [a + b for a, b in zip(total_positives, positives)]
                offset += size
            entry.update(band_positive_counts=total_positives, band_negative_counts=[count - n for n in total_positives])
        splits[split] = entry
    return {"dataset_hash": manifest["manifest_hash"], "splits": splits}


if __name__ == "__main__":
    raise SystemExit(main())
