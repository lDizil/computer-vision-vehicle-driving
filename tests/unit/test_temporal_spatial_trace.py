from dataclasses import replace
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from hybrid_drive.experiments import temporal_spatial_trace as traces
from hybrid_drive.experiments.io import write_json
from hybrid_drive.experiments.protocol import content_hash


class _Vision(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1), requires_grad=False)

    def forward(self, images):
        assert images.ndim == 5 and images.shape[1:3] == (1, 3)
        return images * self.scale


class _Model:
    receipt = {"recipe": "fixed-weights"}
    verified = False

    def predict(self, features):
        offsets = torch.arange(4, dtype=torch.float32).reshape(4, 1) * torch.tensor([0.25, 0, 0.25, 0])
        boxes = torch.tensor([0.02, 0.25, 0.2, 0.55]) + offsets
        scores = features.mean(dim=(1, 2, 3, 4))[:, None].expand(-1, 4)
        return scores, boxes[None].expand(len(scores), -1, -1)

    def verify_unchanged(self):
        self.verified = True


def _pack(tmp_path):
    return SimpleNamespace(root=tmp_path, manifest_hash="a" * 64,
        manifest={"sources": {"temporal_role": "holdout"}},
        frames=tuple(
            SimpleNamespace(recording_id="recording", clip_id="clip", sample_id=f"frame-{i}",
                            source_index=i, timestamp_us=1 + i * 500000, rgb={"pixel": 16 * i})
            for i in range(4)
        ))


def _capture(tmp_path, monkeypatch):
    pack, model = _pack(tmp_path), _Model()
    pixels = []

    def rgb(_, record):
        pixels.append(record["pixel"])
        return np.full((4, 8, 3), record["pixel"], dtype=np.uint8)

    monkeypatch.setattr(traces, "read_spatial_rgb", rgb)
    trace = traces.capture_spatial_trace(pack, _Vision().eval(), model, batch_size=3)
    assert pixels == [0, 16, 32, 48] and model.verified
    return pack, trace


def test_roundtrip_preserves_exact_frame_time_order_and_model_digest(tmp_path, monkeypatch):
    pack, trace = _capture(tmp_path, monkeypatch)
    assert [row.sample_id for row in trace.frames] == [x.sample_id for x in pack.frames]
    assert trace.frames[2].scores[1] == pytest.approx(32 / 255)
    assert trace.role == "holdout" and trace.model_hash == content_hash(_Model.receipt)
    path = tmp_path / "trace.json"
    digest = traces.write_spatial_trace(path, trace)
    restored = traces.read_spatial_trace(path, pack, expected_sha256=digest, expected_model_hash=trace.model_hash)
    assert restored == trace


def test_bound_trace_rejects_wrong_model_role_or_reordered_frames(tmp_path, monkeypatch):
    pack, trace = _capture(tmp_path, monkeypatch)
    path = tmp_path / "trace.json"
    digest = traces.write_spatial_trace(path, replace(trace, frames=trace.frames[::-1]))
    with pytest.raises(ValueError, match="identity or order"):
        traces.read_spatial_trace(path, pack, expected_sha256=digest, expected_model_hash=trace.model_hash)
    with pytest.raises(ValueError, match="bound role, pack or model"):
        traces.read_spatial_trace(path, pack, expected_sha256=digest, expected_model_hash="b" * 64)


def test_added_labels_are_not_part_of_the_observation_contract(tmp_path, monkeypatch):
    pack, trace = _capture(tmp_path, monkeypatch)
    from dataclasses import asdict
    data = {"schema": "hybrid-drive-temporal-spatial-observations-v1", **asdict(trace)}
    data["frames"][0]["targets"] = [1, 0, 0, 0]
    path = tmp_path / "trace.json"
    write_json(path, data)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="observation frame"):
        traces.read_spatial_trace(path, pack, expected_sha256=digest, expected_model_hash=trace.model_hash)


def test_modified_trace_byte_hash_stops_before_decoding(tmp_path, monkeypatch):
    pack, trace = _capture(tmp_path, monkeypatch)
    path = tmp_path / "trace.json"
    digest = traces.write_spatial_trace(path, trace)
    with path.open("ab") as f:
        f.write(b" ")
    with pytest.raises(ValueError, match="trace byte hash"):
        traces.read_spatial_trace(path, pack, expected_sha256=digest, expected_model_hash=trace.model_hash)


def test_train_mode_vision_is_not_used_for_capture(tmp_path):
    with pytest.raises(ValueError, match="frozen eval-mode vision"):
        traces.capture_spatial_trace(_pack(tmp_path), _Vision(), _Model())
