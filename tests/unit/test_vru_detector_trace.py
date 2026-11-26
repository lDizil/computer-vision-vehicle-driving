from types import SimpleNamespace

import numpy as np
import pytest
import torch

from hybrid_drive.contracts.sensors import SURROUND_ORDER
from hybrid_drive.experiments import object_vru_detector as ssd
from hybrid_drive.experiments import vru_coverage_detector as faster
from hybrid_drive.experiments.object_vru_geometry import project_detection, trace_projection
from hybrid_drive.geometry.camera import CameraCalibration


def _snapshot():
    width, height = ssd.RECTIFIED_WH
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    pose = [[0., -1., 0., 0.], [0., 0., -1., 1.6], [1., 0., 0., 0.], [0., 0., 0., 1.]]
    intrinsic = [[400, 0, 384], [0, 400, 216], [0, 0, 1]]
    return SimpleNamespace(views=[
        SimpleNamespace(direction=direction, rgb=rgb,
                        calibration=CameraCalibration(f"camera-{i}", width, height, intrinsic, pose))
        for i, direction in enumerate(SURROUND_ORDER)
    ])


def _fake_result(image):
    if image.shape[1:] != ssd.RECTIFIED_WH[::-1]:
        # A real detector emits this in crop-local coordinates.
        return {"labels": torch.tensor([1]), "scores": torch.tensor([.91]),
                "boxes": torch.tensor([[1., 22., 26., 56.]])}
    return {"labels": torch.tensor([1, 1, 1, 1, 3]),
            "scores": torch.tensor([.96, .80, .70, .21, .99]),
            "boxes": torch.tensor([[366., 210., 402., 280.], [366., 210., 402., 280.],
                                   [520., 127., 548., 180.], [180., 110., 215., 148.],
                                   [60., 30., 110., 180.]])}


@pytest.mark.parametrize("detector_type,batches", [
    (ssd.LocalVruDetector, [4, 4, 4]),
    (faster.CoverageDetector, [1] * 6),
])
def test_audit_preserves_projected_objects_and_original_inference_schedule(detector_type, batches):
    model = detector_type.__new__(detector_type)
    snapshot = _snapshot()
    calls = []

    def fake_model(images):
        assert not torch.is_grad_enabled()
        calls.append(len(images))
        return [_fake_result(image) for image in images]

    model._model = fake_model
    original, info = model(snapshot)
    traced, cameras = model.audit(snapshot)
    assert original == traced and len(traced) == 1
    assert calls == batches * 2
    assert [camera["camera_name"] for camera in cameras] == [v.calibration.name for v in snapshot.views]
    hits = {hit.detection_id: hit for obj in traced for hit in obj.hits}
    for index, camera in enumerate(cameras):
        boxes = [box for box in camera["boxes"] if not box["tile"]]
        assert sorted(box["state"] for box in boxes) == [
            "deduplicated", "projected", "projection_rejected", "score_floor",
        ]
        projected, = (box for box in boxes if box["state"] == "projected")
        assert projected["detection_id"] == f"v{index}-b0"
        assert projected["xy_m"] == hits[projected["detection_id"]].xy
        assert projected["projection_reason"] is None
        rejected, = (box for box in boxes if box["state"] == "projection_rejected")
        assert rejected["projection_reason"] == "invalid_ray"
        assert all(box["xy_m"] is None and box["detection_id"] is None for box in boxes if box is not projected)
    if detector_type is ssd.LocalVruDetector:
        assert info == {"tiles": 12, "detector_boxes": 12, "projected_camera_hits": 6}
        crop = [box for box in cameras[0]["boxes"] if box["tile"]]
        assert [box["state"] for box in crop] == ["crop_edge", "crop_edge"]
        assert crop[0]["xyxy"] == (1., 108., 26., 142.)
        assert crop[1]["xyxy"] == (289., 108., 314., 142.)
    else:
        assert len(info) == 6
        assert all([row["ground_projection"] for row in camera["boxes"]] == [True, False] for camera in info)


def test_audit_separates_nms_suppression_from_budget_truncation(monkeypatch):
    monkeypatch.setattr(ssd, "_MAX_IMAGE_BOXES", 2)
    boxes = (
        ssd.DetectedBox("person", .95, (10., 10., 25., 35.), 0),
        ssd.DetectedBox("person", .90, (10., 10., 25., 35.), 0),
        ssd.DetectedBox("person", .85, (55., 10., 80., 35.), 0),
        ssd.DetectedBox("bicycle", .80, (10., 10., 25., 35.), 0),
    )
    assert tuple(ssd._box_states(reversed(boxes), audit=True)) == (
        (boxes[0], None), (boxes[1], "deduplicated"), (boxes[2], None), (boxes[3], "box_budget"),
    )
    assert ssd.deduplicate_boxes(reversed(boxes)) == (boxes[0], boxes[2])


def test_projection_trace_does_not_turn_geometric_rejections_into_ground_hits():
    pose = [[0., -1., 0., 0.], [0., 0., -1., 1.], [1., 0., 0., 0.], [0., 0., 0., 1.]]
    calibration = CameraCalibration("front", 201, 101, [[100, 0, 100], [0, 100, 50], [0, 0, 1]], pose)
    rgb = np.zeros((101, 201, 3), dtype=np.uint8)
    examples = [
        ((74., 26., 86., 60.), None),
        ((74., 26., 77., 60.), "size"),
        ((0., 26., 86., 60.), "edge"),
        ((74., 24., 86., 48.), "invalid_ray"),
        ((4., 26., 16., 56.), "outside_roi"),
        ((94., 26., 106., 54.), "unstable"),
    ]
    for box, expected in examples:
        params = calibration, rgb, box, "person", .9
        hit, reason = trace_projection(*params, detection_id="case")
        assert hit == project_detection(*params, detection_id="case")
        assert reason == expected and (hit is None) == (expected is not None)
