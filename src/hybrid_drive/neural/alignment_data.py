"""Read the frozen alignment slice without opening raw logs or generating priors."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import Dataset

from hybrid_drive.assets.alignment_manifest import (
    SPLITS, read_alignment_manifest, read_alignment_rgb, read_alignment_targets,
)


@dataclass(frozen=True)
class AlignmentExample:
    sample_id: str
    image: torch.Tensor
    presence: bool
    box_3d: torch.Tensor
    box_2d: torch.Tensor


@dataclass(frozen=True)
class AlignmentBatch:
    sample_ids: tuple[str, ...]
    images: torch.Tensor
    presence: torch.Tensor
    boxes_3d: torch.Tensor
    boxes_2d: torch.Tensor


class AlignmentDataset(Dataset[AlignmentExample]):
    def __init__(self, root: str | Path, split: str):
        if split not in SPLITS:
            raise ValueError("Alignment split must be train or holdout")
        self._root = Path(os.path.abspath(root))
        self._manifest = read_alignment_manifest(self._root)
        partition = self._manifest["splits"][split]
        self._samples = partition["samples"]
        self._targets = read_alignment_targets(self._root, self._manifest, split)
        self.dataset_hash = self._manifest["manifest_hash"]
        self.recording_id = partition["recording_id"]
        self.sample_ids = tuple(sample["sample_id"] for sample in self._samples)

    @property
    def manifest(self) -> dict:
        return deepcopy(self._manifest)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> AlignmentExample:
        sample, target = self._samples[index], self._targets[index]
        rgb = read_alignment_rgb(self._root, sample["rgb"])
        image = torch.from_numpy(rgb).permute(2, 0, 1).to(dtype=torch.float32).div_(255)
        return AlignmentExample(
            sample["sample_id"], image, target["presence"],
            _box_tensor(target["boxes_3d"], 7), _box_tensor(target["boxes_2d"], 4),
        )


def _box_tensor(value, length):
    if value is None:
        return torch.full((length,), float("nan"), dtype=torch.float32)
    return torch.tensor(value, dtype=torch.float32)


def collate_alignment(examples: Sequence[AlignmentExample]) -> AlignmentBatch:
    if not examples:
        raise ValueError("Cannot collate an empty alignment batch")
    return AlignmentBatch(
        sample_ids=tuple(example.sample_id for example in examples),
        images=torch.stack([example.image for example in examples]).unsqueeze(1),
        presence=torch.tensor([example.presence for example in examples], dtype=torch.bool),
        boxes_3d=torch.stack([example.box_3d for example in examples]),
        boxes_2d=torch.stack([example.box_2d for example in examples]),
    )
