"""Admit a separate, pinned final pack without opening its targets."""

from hybrid_drive.assets.spatial_manifest import (
    EVALUATION_SOURCE_SCHEMA,
    SPLITS,
    read_spatial_manifest,
    verify_disjoint_spatial_manifests,
)


def read_fresh_spatial_holdout(
    reference_root, fresh_root, *, expected_schema=EVALUATION_SOURCE_SCHEMA, previously_seen_roots=(),
):
    """Match the requested pinned source and check its identities against every earlier pack."""
    reference = read_spatial_manifest(reference_root)
    fresh = read_spatial_manifest(fresh_root)
    if (
        set(reference["splits"]) != set(SPLITS)
        or set(fresh["splits"]) != {"holdout"}
        or fresh["sources"]["schema"] != expected_schema
    ):
        raise ValueError("Expected the complete original pack and a separate pinned holdout-only pack")

    current = fresh["splits"]["holdout"]["recordings"]
    if len(current) != 2 or any(len(recording["samples"]) != 32 for recording in current):
        raise ValueError("Fresh holdout must keep the full two-recording, 64-frame selection")

    verify_disjoint_spatial_manifests(fresh, reference, role="original pack")
    for root in previously_seen_roots:
        verify_disjoint_spatial_manifests(fresh, read_spatial_manifest(root), role="previously seen pack")
    return fresh
