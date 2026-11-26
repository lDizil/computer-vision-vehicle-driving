from types import SimpleNamespace

import numpy as np
import pytest

from hybrid_drive.experiments.object_vru_detector import DetectedBox, _Tile
from hybrid_drive.experiments.vru_metric_detector import DepthSample, depth_sample, metric_hit, place_tile_box
from hybrid_drive.experiments.vru_clip_replay import _capture_window, _validate_comparable_cameras
from hybrid_drive.geometry.camera import CameraCalibration
from hybrid_drive.geometry.rigid import invert_pose


def test_depth_samples_foreground_and_rejects_mixed_or_missing_surfaces():
    rect = (330., 200., 350., 240.)
    depth = np.full((432, 768), 40., dtype=np.float32)
    depth[205:232, 335:345] = 11.
    assert depth_sample(depth, rect) == DepthSample(11., 0.)
    depth[211:222, 335:345] = 23.
    assert depth_sample(depth, rect) is None
    depth[205:232, 335:345] = np.nan
    assert depth_sample(depth, rect) is None
    with pytest.raises(ValueError, match="same pixel grid"):
        depth_sample(depth[:15], rect)


def test_axial_depth_is_not_euclidean_distance_or_an_intersection_with_z_zero():
    k = [[400., 0., 384.], [0., 400., 216.], [0., 0., 1.]]
    camera = np.eye(4)
    camera[:3, :3] = [[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]]
    camera[:3, 3] = (1., -.5, 1.6)
    cal = CameraCalibration("CAM_F0", 768, 432, k, invert_pose(camera))
    view = SimpleNamespace(calibration=cal, rgb=np.zeros((432, 768, 3), dtype=np.uint8))
    box = SimpleNamespace(xyxy=(464., 216., 504., 256.), kind="person", confidence=.8)
    result = metric_hit(view, box, DepthSample(12., .2), detection_id="v0-b0")
    assert result.xy == pytest.approx((13., -3.5))
    assert result.camera_xyz == (1., -.5, 1.6)
    assert result.detection_id == "v0-b0"
    assert metric_hit(view, box, DepthSample(40., .2), detection_id="v0-b0") is None


def test_capture_produces_the_groups_required_by_the_comparability_guard():
    anchor = {"sample_token": "now", "previous_token": "past", "older_token": "old", "recording": "r"}
    index = {"frames": dict.fromkeys(("now", "past", "old"), {})}
    shot = SimpleNamespace(content_sha256="ab"*32)
    fresh = _capture_window(anchor, index, None, lambda _: ({"plane": (), "metric": ()}, []),
                            lambda *args, **kwargs: shot)
    previous = {"observations": {when: {"sha256": shot.content_sha256} for when in ("older", "previous", "current")}}
    _validate_comparable_cameras(fresh, previous)


def test_overlap_tile_remaps_local_coordinates_and_rejects_a_cut_box():
    tile = _Tile(2, 1, (288, 86, 768, 410))
    label = DetectedBox("person", .8, (15, 8, 70, 152), 0)
    result = place_tile_box(tile, label)
    assert result.tile == 1 and result.xyxy == (303, 94, 358, 238)
    assert place_tile_box(tile, DetectedBox("person", .8, (15, 1, 70, 160), 0)) is None
