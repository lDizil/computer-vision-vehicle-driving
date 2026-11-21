import numpy as np
import pytest

from hybrid_drive.assets.alignment_sample import _loc_target, _project_cuboid
from hybrid_drive.assets.files import PreparationError
from hybrid_drive.assets.spatial_manifest import box_horizontal_band
from hybrid_drive.assets.spatial_sample import _spatial_target
from hybrid_drive.geometry.camera import CameraCalibration


def camera():
    intrinsic = [[112, 0, 111.5], [0, 112, 111.5], [0, 0, 1]]
    ego_to_camera = [[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]]
    return CameraCalibration("CAM_F0", 224, 224, intrinsic, ego_to_camera)


def annotations(boxes, names=None):
    return {
        "gt_boxes": np.array(boxes, dtype=np.float64).reshape(-1, 7),
        "gt_names": np.array(names if names is not None else ["vehicle"] * len(boxes), dtype=str),
        "instance_tokens": [f"{index+1:016x}" for index in range(len(boxes))],
        "track_tokens": [f"{index+100:016x}" for index in range(len(boxes))],
    }


def test_four_bands_choose_nearest_eligible_with_stable_source_order():
    cuboids = [
        [10, 5, 0, 4, 2, 2, 0],  # Band 0: centre of the 2D AABB is < .25, 3D centre is not.
        [12, 2, 0, 4, 2, 2, 0],
        [10, 2, 0, 4, 2, 2, 0],  # Band 1: nearest in band.
        [10, 2, 1, 4, 2, 2, 0],  # Same ground distance, later row, different projected y.
        [10, 0, 0, 4, 2, 2, 0],  # Band 2: .50 is included.
        [10, -5, 0, 4, 2, 2, 0],  # Band 3: eligible despite old centre crop.
        [6, 1, 0, 4, 2, 2, 0],  # Pedestrian, not a candidate.
        [5, 1, 0, 4, 2, 20, 0],  # Closer but clips the image.
    ]
    row = _spatial_target(annotations(cuboids, ["vehicle"] * 6 + ["pedestrian", "vehicle"]), camera(), "s")
    assert row["presence"] == [True] * 4
    assert row["boxes_2d"] == pytest.approx(np.array([
        [0.125, 0.4375, 1/3, 0.5625],
        [0.3125, 0.4375, 0.5 - 1/24, 0.5625],
        [0.4375, 0.4375, 0.5625, 0.5625],
        [2/3, 0.4375, 0.875, 0.5625],
    ], dtype=np.float32))
    legacy = _loc_target(annotations([cuboids[0]]), camera(), "s")
    assert legacy["presence"] and legacy["boxes_2d"] == pytest.approx(row["boxes_2d"][0])
    assert _project_cuboid(np.array(cuboids[5]), camera()) is None


@pytest.mark.parametrize(("boundary", "band"), [(0.25, 1), (0.5, 2), (0.75, 3)])
def test_box_band_uses_fp32_arithmetic_at_exact_boundary(boundary, band):
    lower = np.nextafter(np.float32(boundary - 0.0625), np.float32(-np.inf))
    upper = np.float32(boundary + 0.0625)
    assert (float(lower) + float(upper)) / 2 < boundary
    assert box_horizontal_band(np.array([lower, 0.2, upper, 0.8], dtype=np.float32)) == band
    below = np.nextafter(lower, np.float32(-np.inf))
    assert box_horizontal_band([below, 0.2, upper, 0.8]) == band - 1


@pytest.mark.parametrize("box", [
    [41, 0, 0, 4, 2, 2, 0],
    [39, 10, 0, 4, 2, 2, 0],
    [2, 0, 0, 4, 2, 2, 0],
    [10, 0, 0, 4, 2, 20, 0],
    [10, -10, 0, 4, 2, 2, 0],
    [20, 0, 0, 4, .1, 2, 0],
])
def test_invalid_range_depth_size_or_clipping_produces_no_box(box):
    row = _spatial_target(annotations([box]), camera(), "s")
    assert row == {"sample_id": "s", "presence": [False] * 4, "boxes_2d": [None] * 4}


def test_legacy_and_spatial_share_projection_in_central_view_at_distance_limit():
    boxes = annotations([[40, 0, 0, 4, 2, 2, 0]])
    legacy = _loc_target(boxes, camera(), "s")
    spatial = _spatial_target(boxes, camera(), "s")
    assert legacy["presence"]
    assert spatial["presence"] == [False, False, True, False]
    assert spatial["boxes_2d"][2] == np.asarray(legacy["boxes_2d"], dtype=np.float32).tolist()


def test_empty_annotations_are_negative_but_malformed_vehicle_is_not():
    assert _spatial_target(annotations([]), camera(), "s")["presence"] == [False] * 4
    with pytest.raises(PreparationError, match="Malformed source vehicle"):
        _spatial_target(annotations([[10, float("nan"), 0, 4, 2, 2, 0]]), camera(), "s")
