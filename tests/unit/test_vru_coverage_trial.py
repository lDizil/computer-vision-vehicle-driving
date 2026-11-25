from types import SimpleNamespace

import numpy as np
import pytest
import torch

from hybrid_drive.experiments import vru_coverage_detector as model
from hybrid_drive.experiments.vru_detector_coverage_trial import _bind_observation, recheck_indices, score_coverage


def test_recheck_windows_are_chosen_by_fixed_position_without_labels():
    assert recheck_indices(115) == (0, 28, 57, 85, 114)
    with pytest.raises(ValueError, match="enough windows"):
        recheck_indices(4)


def test_camera_comparison_cannot_silently_use_another_frame_or_history():
    anchor = {"sample_token": "now", "previous_token": "ago", "recording": "north"}
    reference = {**anchor, "current_sha256": "a" * 64, "ref_time_us": 450_000}
    _bind_observation(anchor, reference, "a" * 64, 450_000)
    for record, digest, stamp in (({**anchor, "previous_token": "another"}, "a"*64, 450_000),
                                  (anchor, "b"*64, 450_000), (anchor, "a"*64, 450_001)):
        with pytest.raises(ValueError, match="same rectified current frame"):
            _bind_observation(record, reference, digest, stamp)


def test_weights_are_checked_before_deserialization(tmp_path, monkeypatch):
    monkeypatch.setattr(model, "version", lambda name: model.TORCHVISION_VERSION)
    monkeypatch.setattr(model, "read_bounded", lambda path, limit: b"wrong bytes")
    def should_not_load(*args,**kw):
        pytest.fail("deserialized without the pinned checksum")
    monkeypatch.setattr(torch, "load", should_not_load)
    with pytest.raises(ValueError, match="pinned publisher"):
        model.CoverageDetector(tmp_path)


def test_current_replay_preserves_all_six_views_while_counting_unprojectable_boxes(monkeypatch):
    w,h=model.RECTIFIED_WH
    raster=np.zeros((h,w,3),dtype=np.uint8)
    views=[SimpleNamespace(calibration=SimpleNamespace(width=w,height=h,name=f"camera-{i}"), rgb=raster) for i in range(6)]
    detector=model.CoverageDetector.__new__(model.CoverageDetector)
    def net(images):
        assert len(images)==1 and images[0].shape==(3,h,w)
        return [{"labels":torch.tensor([1,3]),"scores":torch.tensor([.9,.8]),
                 "boxes":torch.tensor([[20.,20.,60.,110.],[12.,24.,30.,60.]])}]
    detector._model=net
    calls=[]
    def ground(calib,image,box,kind,confidence,*,detection_id):
        calls.append((calib.name,box,kind,detection_id)); return None
    monkeypatch.setattr(model, "project_detection", ground)
    projected,cameras=detector(SimpleNamespace(views=views))
    assert projected==() and len(cameras)==len(calls)==6
    assert [c['camera_name'] for c in cameras]==[f"camera-{i}" for i in range(6)]
    assert all(len(c["boxes"])==1 and c["boxes"][0]["ground_projection"] is False for c in cameras)
    assert all(call[1:3]==((20.,20.,60.,110.),"person") for call in calls)


def test_wrong_seal_is_rejected_without_touching_labels(tmp_path):
    raw=tmp_path/"predictions.json"; raw.write_bytes(b"{}")
    with pytest.raises(ValueError,match="pre-label seal"):
        score_coverage(tmp_path/"absent-input",tmp_path/"absent-reference",raw,tmp_path/"absent-labels",
                       tmp_path/"score",sealed_sha256="0"*64)
