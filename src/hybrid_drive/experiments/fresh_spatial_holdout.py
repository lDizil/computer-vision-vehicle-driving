"""Admit a separate, pinned final pack without opening its targets."""

from hybrid_drive.assets.spatial_manifest import (
    EVALUATION_SOURCE_SCHEMA,
    SPLITS,
    read_spatial_manifest,
)


def read_fresh_spatial_holdout(reference_root, fresh_root):
    reference = read_spatial_manifest(reference_root)
    fresh = read_spatial_manifest(fresh_root)
    _verify_distinct(reference, fresh)
    return fresh


def _verify_distinct(reference, fresh):
    old, new = reference["sources"], fresh["sources"]
    if (
        set(reference["splits"]) != set(SPLITS)
        or set(fresh["splits"]) != {"holdout"}
        or new["schema"] != EVALUATION_SOURCE_SCHEMA
    ):
        raise ValueError("Expected the complete original pack and a separate pinned holdout-only pack")
    if reference["manifest_hash"] == fresh["manifest_hash"]:
        raise ValueError("Holdout manifest must be different from the original pack")
    scope = ("dataset", "openscene_revision", "navsim_revision", "nuplan_revision", "rgb", "loc", "selection")
    if any(old[key] != new[key] for key in scope):
        raise ValueError("Fresh holdout must preserve the source, image and target protocol")

    current = fresh["splits"]["holdout"]["recordings"]
    if len(current) != 2 or any(len(recording["samples"]) != 32 for recording in current):
        raise ValueError("Fresh holdout must keep the full two-recording, 64-frame selection")
    earlier = [row for partition in reference["splits"].values() for row in partition["recordings"]]
    present = _identities(current)
    original = _identities(earlier)
    for label, values in present.items():
        if values.intersection(original[label]):
            raise ValueError(f"Fresh holdout overlaps the original pack by {label}")


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
