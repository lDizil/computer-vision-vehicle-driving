"""Verify the six-camera encoder handoff from local data and weights, without downloads."""

import argparse
from hashlib import sha256
from pathlib import Path

import torch

from hybrid_drive.adapters.nuscenes_surround import load_nuscenes_surround
from hybrid_drive.adapters.openscene_surround import load_openscene_surround
from hybrid_drive.experiments.surround_probe import run_surround_probe
from hybrid_drive.neural.pretrained_alignment import load_alignment_vision
from hybrid_drive.neural.surround_features import SharedSurroundEncoder


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    parser.add_argument("--revision", required=True)
    sub = parser.add_subparsers(dest="source", required=True)
    op = sub.add_parser("openscene", help="One fixed verified real OpenScene snapshot")
    op.add_argument("--metadata-archive", type=Path, required=True)
    op.add_argument("--camera-archive", type=Path, required=True)
    ns = sub.add_parser("nuscenes", help="An extracted nuScenes v1.0-mini keyframe")
    ns.add_argument("--root", type=Path, required=True)
    ns.add_argument("--sample-token", required=True)
    ns.add_argument("--max-skew-us", type=int, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("--output must not exist")
    if len(args.revision) != 40 or any(c not in "0123456789abcdef" for c in args.revision):
        parser.error("--revision must be a full commit SHA")
    if args.source == "openscene":
        manifest = Path(__file__).with_name("surround-openscene-sources.json")
        snapshot = load_openscene_surround(args.metadata_archive, args.camera_archive, manifest)
        source = {"kind": args.source, "verified_manifest_sha256": sha256(manifest.read_bytes()).hexdigest()}
    else:
        snapshot = load_nuscenes_surround(args.root, args.sample_token, max_skew_us=args.max_skew_us)
        source = {"kind": args.source, "sample_token": args.sample_token}
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    vision, provenance = load_alignment_vision(args.model_root)
    identity = f'{provenance["source"]["repository"]}@{provenance["source"]["revision"]}'
    result = run_surround_probe(
        snapshot, SharedSurroundEncoder(vision, encoder_id=identity), args.output,
        revision=args.revision, encoder_provenance=provenance, source=source,
    )
    print(f"{result['snapshot_id']}: {result['feature_shape']} / {result['timing_basis']}")
    print(f"Shared Fast/Bridge input: {result['fast_input_sha256']}")
    print(f"Report: {args.output / 'result.json'}")


if __name__ == "__main__":
    main()
