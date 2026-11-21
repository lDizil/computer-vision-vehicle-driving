"""Read one split of the spatial LOC dataset without using targets as input."""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import Dataset

from hybrid_drive.assets.spatial_manifest import SPLITS, read_spatial_manifest, read_spatial_rgb, read_spatial_targets


@dataclass(frozen=True)
class SpatialAlignmentExample:
    sample_id: str
    recording_id: str
    image: torch.Tensor
    presence: torch.Tensor
    boxes_2d: torch.Tensor


@dataclass(frozen=True)
class SpatialAlignmentBatch:
    sample_ids: tuple[str, ...]
    recording_ids: tuple[str, ...]
    images: torch.Tensor
    presence: torch.Tensor
    boxes_2d: torch.Tensor


class SpatialAlignmentDataset(Dataset[SpatialAlignmentExample]):
    def __init__(self, root: str | Path, split: str):
        if split not in SPLITS:
            raise ValueError("Spatial split must be train, dev or holdout")
        self._root = Path(os.path.abspath(root))
        manifest = read_spatial_manifest(self._root)
        recordings = manifest["splits"][split]["recordings"]
        self._samples = tuple(sample for recording in recordings for sample in recording["samples"])
        self._targets = read_spatial_targets(self._root, manifest, split)
        self.dataset_hash = manifest["manifest_hash"]
        self.sample_ids = tuple(sample["sample_id"] for sample in self._samples)
        self.recording_ids = tuple(recording["recording_id"] for recording in recordings for _ in recording["samples"])

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> SpatialAlignmentExample:
        sample, target = self._samples[index], self._targets[index]
        rgb = read_spatial_rgb(self._root, sample["rgb"])
        image = torch.from_numpy(rgb).permute(2, 0, 1).to(dtype=torch.float32).div_(255)
        boxes = [box if box is not None else [float("nan")] * 4 for box in target["boxes_2d"]]
        return SpatialAlignmentExample(
            sample["sample_id"], self.recording_ids[index], image,
            torch.tensor(target["presence"], dtype=torch.bool), torch.tensor(boxes, dtype=torch.float32),
        )


def collate_spatial_alignment(examples: Sequence[SpatialAlignmentExample]) -> SpatialAlignmentBatch:
    if not examples:
        raise ValueError("Cannot collate an empty spatial alignment batch")
    return SpatialAlignmentBatch(
        sample_ids=tuple(example.sample_id for example in examples),
        recording_ids=tuple(example.recording_id for example in examples),
        images=torch.stack([example.image for example in examples]).unsqueeze(1),
        presence=torch.stack([example.presence for example in examples]),
        boxes_2d=torch.stack([example.boxes_2d for example in examples]),
    )
