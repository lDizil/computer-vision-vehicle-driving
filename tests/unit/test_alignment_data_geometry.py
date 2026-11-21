import math

import numpy as np
import pytest

from hybrid_drive.assets.alignment_manifest import midpoint_indices
from hybrid_drive.assets.alignment_sample import _loc_target
from hybrid_drive.assets.files import PreparationError
from hybrid_drive.geometry.camera import CameraCalibration


def camera():
    k = [[112, 0, 111.5], [0, 112, 111.5], [0, 0, 1]]
    ego_to_camera = [[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]]
    return CameraCalibration("CAM_F0", 224, 224, k, ego_to_camera)


def annotations(boxes, names=None):
    return {
        "gt_boxes": np.array(boxes, dtype=np.float64).reshape(-1, 7),
        "gt_names": np.array(names if names is not None else ["vehicle"] * len(boxes), dtype=str),
        "instance_tokens": [f"{index+1:016x}" for index in range(len(boxes))],
        "track_tokens": [f"{index+100:016x}" for index in range(len(boxes))],
    }


def test_frozen_bins_choose_midpoints_not_endpoints():
    train = midpoint_indices(738, 64)
    holdout = midpoint_indices(909, 32)
    assert len(train) == len(set(train)) == 64
    assert len(holdout) == len(set(holdout)) == 32
    assert (train[0], train[-1], holdout[0], holdout[-1]) == (5, 732, 14, 894)


def test_nearest_eligible_order_and_edge_normalized_aabb():
    targets = annotations([
        [5, 3, 0, 2, 1, 1, 0],  # closer but center outside the central half
        [12, 0, 0, 4, 2, 2, 0],
        [10, 0, 0, 4, 2, 2, 0],
        [10, 0, 1, 4, 2, 2, 0],  # equal ground distance, later row
        [7, 0, 0, 4, 2, 2, 0],
    ], ["vehicle", "vehicle", "vehicle", "vehicle", "pedestrian"])
    row = _loc_target(targets, camera(), "train-0001")
    assert row["presence"] is True
    assert row["annotation"]["row_index"] == 2
    assert row["boxes_3d"] == [10, 0, 0, 2, 4, 2, 0]
    # Closest corners are at depth 8 m; ±112/8 pixels around the 111.5 pixel center.
    assert row["boxes_2d"] == pytest.approx([0.4375, 0.4375, 0.5625, 0.5625])


def test_yaw_rotates_ego_corners_before_projection():
    row = _loc_target(annotations([[10, 0, 0, 4, 2, 2, -3 * math.pi / 2]]), camera(), "train-0001")
    assert row["boxes_3d"][-1] == pytest.approx(math.pi / 2)
    assert row["boxes_2d"] == pytest.approx([0.5 - 1/9, 0.5 - 1/18, 0.5 + 1/9, 0.5 + 1/18])


@pytest.mark.parametrize("box", [
    [41, 0, 0, 4, 2, 2, 0],
    [2, 0, 0, 4, 2, 2, 0],
    [10, -5, 0, 4, 2, 2, 0],  # projected center at excluded right boundary 0.75
    [10, 0, 0, 4, 2, 20, 0],  # corners would need clipping
    [20, 0, 0, 4, 0.1, 2, 0],
])
def test_out_of_scope_cuboids_do_not_get_coordinates(box):
    row = _loc_target(annotations([box]), camera(), "holdout-0001")
    assert row == {"sample_id": "holdout-0001", "presence": False,
                   "boxes_3d": None, "boxes_2d": None, "annotation": None}


def test_included_left_boundary_and_euclidean_distance():
    row = _loc_target(annotations([[10, 5, 0, 4, 2, 2, 0]]), camera(), "train-0001")
    assert row["presence"] is True
    # x alone is below 40 m; the horizontal distance is not.
    row = _loc_target(annotations([[39, 10, 0, 4, 2, 2, 0]]), camera(), "train-0003")
    assert row["presence"] is False


def test_invalid_source_vehicle_is_not_a_negative():
    with pytest.raises(PreparationError, match="Malformed source vehicle"):
        _loc_target(annotations([[10, float("nan"), 0, 4, 2, 2, 0]]), camera(), "train-0001")



def test_empty_annotation_list_is_a_labeled_negative():
    empty = annotations([])
    empty["gt_names"] = np.array([])
    row = _loc_target(empty, camera(), "train-0001")
    assert row["presence"] is False
    assert row["boxes_2d"] is None and row["boxes_3d"] is None
