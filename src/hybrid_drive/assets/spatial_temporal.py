"""Read the pinned source timeline without reading target or image bytes."""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from .files import PreparationError
from .spatial_manifest import (
    TEMPORAL_FRAMES_PER_CLIP,
    TEMPORAL_ROLES,
    TEMPORAL_SOURCE_SCHEMA,
    read_spatial_manifest,
    verify_disjoint_spatial_manifests,
)


@dataclass(frozen=True)
class TemporalFrameRef:
    recording_id: str
    clip_id: str
    sample_id: str
    source_index: int
    timestamp_us: int
    rgb: dict


@dataclass(frozen=True)
class TemporalSpatialPack:
    root: Path
    manifest: dict
    frames: tuple[TemporalFrameRef, ...]

    @property
    def manifest_hash(self):
        return self.manifest["manifest_hash"]


def read_temporal_spatial_pack(root, *, role):
    """Read frames in manifest order; use clip IDs to delimit causal history."""
    if role not in TEMPORAL_ROLES:
        raise ValueError("Unknown requested temporal spatial role")
    root = Path(os.path.abspath(root))
    manifest = read_spatial_manifest(root)
    sources = manifest["sources"]
    if sources["schema"] != TEMPORAL_SOURCE_SCHEMA or sources["temporal_role"] != role:
        raise PreparationError("Expected the pinned temporal spatial role")
    frames = []
    for recording in manifest["splits"]["holdout"]["recordings"]:
        identity = recording["recording_id"]
        for offset, sample in enumerate(recording["samples"]):
            frames.append(TemporalFrameRef(
                recording_id=identity,
                clip_id=f"{identity}-clip-{offset // TEMPORAL_FRAMES_PER_CLIP:02d}",
                sample_id=sample["sample_id"],
                source_index=sample["source_list_index"],
                timestamp_us=sample["timestamp_us"],
                rgb=sample["rgb"],
            ))
    return TemporalSpatialPack(root, manifest, tuple(frames))


def verify_independent_temporal_final(final_pack, previous_roots):
    """Fail closed if an earlier spatial pack shares the final observation or its group."""
    sources = final_pack.manifest["sources"]
    if sources["schema"] != TEMPORAL_SOURCE_SCHEMA or sources["temporal_role"] != "holdout":
        raise PreparationError("Independence requires the pinned temporal holdout")
    roots = tuple(previous_roots)
    if not roots:
        raise ValueError("At least one earlier spatial pack is required to check temporal independence")
    for root in roots:
        verify_disjoint_spatial_manifests(
            final_pack.manifest, read_spatial_manifest(root), role="previously seen pack", compare_selection=False,
        )
