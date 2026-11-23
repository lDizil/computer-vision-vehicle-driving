"""Admit a separate, pinned final pack without opening its targets."""

from hybrid_drive.assets.spatial_manifest import (
    EVALUATION_SOURCE_SCHEMA,
    SPLITS,
    read_spatial_manifest,
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

    present = _identities(current)
    _verify_distinct(reference, fresh, present, "original pack")
    for root in previously_seen_roots:
        _verify_distinct(read_spatial_manifest(root), fresh, present, "previously seen pack")
    return fresh


def _verify_distinct(previous, fresh, present, role):
    if previous["manifest_hash"] == fresh["manifest_hash"]:
        raise ValueError(f"Holdout manifest must be different from the {role}")
    scope = ("dataset", "openscene_revision", "navsim_revision", "nuplan_revision", "rgb", "loc", "selection")
    if any(previous["sources"][key] != fresh["sources"][key] for key in scope):
        raise ValueError(f"Fresh holdout must preserve the source, image and target protocol of the {role}")

    earlier = [row for partition in previous["splits"].values() for row in partition["recordings"]]
    original = _identities(earlier)
    for label, values in present.items():
        if values.intersection(original[label]):
            raise ValueError(f"Fresh holdout overlaps the {role} by {label}")


def _identities(recordings):
    recording_ids = {row["recording_id"] for row in recordings}
    samples = [sample for row in recordings for sample in row["samples"]]
    return {
        "recording": recording_ids,
        "date and vehicle": {
            (".".join(identity.split(".")[:3]), identity.rsplit("_", 1)[1]) for identity in recording_ids
        },
        "log token": {row["log_token"] for row in recordings},
        "frame token": {row["frame_token"] for row in samples},
        "JPEG member": {row["jpeg"]["member"] for row in samples},
        "JPEG digest": {row["jpeg"]["sha256"] for row in samples},
        "RGB digest": {row["rgb"]["pixels_sha256"] for row in samples},
    }
