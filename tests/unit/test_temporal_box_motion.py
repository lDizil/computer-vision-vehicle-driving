from dataclasses import asdict
import json
import math

import numpy as np
import pytest

from hybrid_drive.fast.forecast_proposals import ForecastObject
from hybrid_drive.fast.temporal_box_motion import MotionFrame, associate_box_motion


def _box(name, center, *, yaw=0.0, size=(1.9, 4.5, 1.7)):
    return ForecastObject(name, center, size, yaw, (12.0, -1.0), .85)


def _pose(rotation=None, translation=(0.0, 0.0, 0.0)):
    pose = np.eye(4)
    if rotation is not None:
        pose[:3, :3] = rotation
    pose[:3, 3] = translation
    return pose


def _frame(time_us, *boxes, pose=None, recording="recording"):
    return MotionFrame(recording, time_us, np.eye(4) if pose is None else pose, boxes)


def test_moving_target_with_different_ego_poses_and_per_frame_ids():
    # The object moves in world consistently along +x at 4 m/s. Current ego
    # is translated and rotated +90 deg, so current ego sees (0, -4) m/s.
    turn = ((0, -1, 0), (1, 0, 0), (0, 0, 1))
    earlier = _frame(0, _box("cell-1", (8.0, 2.0, .7)))
    previous = _frame(500_000, _box("cell-7", (7.0, 1.0, .7), yaw=math.pi),
                      pose=_pose(translation=(3, 1, 0)))
    current_box = _box("cell-4", (1.0, -7.0, .7), yaw=-math.pi / 2)
    current = _frame(1_000_000, current_box, pose=_pose(turn, (5, 1, 0)))

    result = associate_box_motion(current, previous, earlier)
    new_box, = result.objects
    assert new_box.velocity_xy == pytest.approx((0.0, -4.0))
    assert {k: v for k, v in asdict(new_box).items() if k != "velocity_xy"} == {
        k: v for k, v in asdict(current_box).items() if k != "velocity_xy"
    }
    assert current_box.velocity_xy == (12.0, -1.0)
    diagnostic, = result.diagnostics
    assert diagnostic.status == "estimated"
    assert diagnostic.previous_object_id == "cell-7"
    assert diagnostic.earlier_object_id == "cell-1"
    assert diagnostic.recent_velocity_xy == pytest.approx((0.0, -4.0))
    assert diagnostic.prior_velocity_xy == pytest.approx((0.0, -4.0))
    assert json.loads(json.dumps(asdict(diagnostic)))["object_id"] == "cell-4"


def test_stationary_target_survives_translation_and_full_3d_ego_rotation():
    # All three inputs represent world center (10, 2, .8). The previous ego
    # roll intentionally mixes y with z to catch SE(2)-only compensation.
    roll = ((1, 0, 0), (0, 0, -1), (0, 1, 0))
    turn = ((0, -1, 0), (1, 0, 0), (0, 0, 1))
    earlier = _frame(1_000_000, _box("old", (10.0, 2.0, .8)))
    previous = _frame(1_500_000, _box("mid", (6.0, .5, -2.0)),
                      pose=_pose(roll, (4, 0, .3)))
    current = _frame(2_000_000, _box("now", (1.0, -4.0, .55), yaw=-math.pi / 2),
                     pose=_pose(turn, (6, 1, .25)))
    result = associate_box_motion(current, previous, earlier)
    assert result.objects[0].velocity_xy == (0.0, 0.0)
    assert result.diagnostics[0].status == "quiet"
    assert result.diagnostics[0].recent_velocity_xy == pytest.approx((0.0, 0.0))


def test_approaching_target_with_unequal_sample_intervals_uses_actual_time():
    earlier = _frame(1_000_000, _box("old", (17.0, 0.0, 1.0)))
    previous = _frame(1_750_000, _box("mid", (14.0, 0.0, 1.0)))
    current = _frame(2_000_000, _box("now", (13.0, 0.0, 1.0)))
    result = associate_box_motion(current, previous, earlier)
    assert result.diagnostics[0].status == "estimated"
    assert result.objects[0].velocity_xy == (-4.0, 0.0)


def test_weighted_velocity_uses_full_time_baseline_when_intervals_differ():
    earlier = _frame(1_000_000, _box("old", (17.0, 0.0, 1.0)))
    previous = _frame(1_750_000, _box("mid", (14.0, 0.0, 1.0)))
    current = _frame(2_000_000, _box("now", (12.5, 0.0, 1.0)))
    result = associate_box_motion(current, previous, earlier)
    assert result.diagnostics[0].status == "estimated"
    assert result.objects[0].velocity_xy == (-4.5, 0.0)


@pytest.mark.parametrize("change", ["size", "height", "yaw"])
def test_nearer_shape_incompatible_neighbor_does_not_steal_motion(change):
    center = (11.9, .0, .8)
    kwargs = {}
    if change == "size":
        kwargs["size"] = (2.0, 10.0, 1.7)
    elif change == "height":
        center = (11.9, .0, 3.0)
    else:
        kwargs["yaw"] = math.pi / 2
    fake = _box("closer", center, **kwargs)
    current = _frame(1_000_000, _box("now", (12.0, .0, .8)))
    previous = _frame(500_000, fake, _box("match", (10.0, .0, .8)))
    earlier = _frame(0, _box("old", (8.0, .0, .8)))
    result = associate_box_motion(current, previous, earlier)
    assert result.diagnostics[0].previous_object_id == "match"
    assert result.diagnostics[0].status == "estimated"
    assert result.objects[0].velocity_xy == (4.0, 0.0)


def test_nearly_equal_forward_matches_abstain_independently_of_input_order():
    current = _frame(1_000_000, _box("now", (12.0, .0, 1.0)))
    earlier = _frame(0, _box("old", (8.0, .0, 1.0)))
    neighbors = (_box("left", (10.0, .25, 1.0)), _box("right", (10.0, -.25, 1.0)))
    for boxes in (neighbors, neighbors[::-1]):
        result = associate_box_motion(current, _frame(500_000, *boxes), earlier)
        assert result.diagnostics[0].status == "ambiguous_previous"
        assert result.objects[0].velocity_xy == (0.0, 0.0)


def test_ambiguous_reverse_match_preserves_both_current_detections():
    current = _frame(1_000_000, _box("a", (12.0, .1, 1.)), _box("b", (12.0, -.1, 1.)))
    previous = _frame(500_000, _box("one", (10.0, .0, 1.0)))
    earlier = _frame(0, _box("old", (8.0, .0, 1.0)))
    result = associate_box_motion(current, previous, earlier)
    assert [x.object_id for x in result.objects] == ["a", "b"]
    assert [x.velocity_xy for x in result.objects] == [(0.0, 0.0), (0.0, 0.0)]
    assert [d.status for d in result.diagnostics] == ["ambiguous_previous", "ambiguous_previous"]


def test_nonmutual_match_cannot_use_a_previous_box_twice():
    current = _frame(1_000_000, _box("farther", (15.0, 0.0, 1.0)), _box("nearer", (12.0, 0.0, 1.0)))
    previous = _frame(500_000, _box("one", (10.0, 0.0, 1.0)))
    earlier = _frame(0, _box("old", (8.0, 0.0, 1.0)))
    result = associate_box_motion(current, previous, earlier)
    assert [x.object_id for x in result.objects] == ["farther", "nearer"]
    assert [d.status for d in result.diagnostics] == ["nonmutual_previous", "estimated"]
    assert [x.velocity_xy for x in result.objects] == [(0.0, 0.0), (4.0, 0.0)]


def test_direction_gate_abstains_even_when_vector_difference_is_small():
    earlier = _frame(0, _box("old", (8.0, 0.0, 1.0)))
    previous = _frame(500_000, _box("mid", (8.8, 0.0, 1.0)))
    current = _frame(1_000_000, _box("now", (8.8, .8, 1.0)))
    result = associate_box_motion(current, previous, earlier)
    assert result.diagnostics[0].status == "inconsistent_motion"
    assert result.diagnostics[0].recent_velocity_xy == pytest.approx((0, 1.6))
    assert result.diagnostics[0].prior_velocity_xy == pytest.approx((1.6, 0))
    assert result.objects[0].velocity_xy == (0.0, 0.0)


def test_inconsistent_motion_replaces_nonzero_learned_velocity_with_exact_zero():
    current = _frame(1_000_000, _box("now", (9.5, .0, 1.0)))
    previous = _frame(500_000, _box("mid", (12.0, .0, 1.0)))
    earlier = _frame(0, _box("old", (10.0, .0, 1.0)))
    result = associate_box_motion(current, previous, earlier)
    assert result.diagnostics[0].status == "inconsistent_motion"
    assert result.diagnostics[0].recent_velocity_xy == (-5.0, 0.0)
    assert result.diagnostics[0].prior_velocity_xy == (4.0, 0.0)
    assert result.objects[0].velocity_xy == (0.0, 0.0)


def test_unrealistic_speed_at_distance_gate_requires_abstention():
    current = _frame(1_000_000, _box("now", (29.0, .0, 1.0)))
    previous = _frame(500_000, _box("mid", (14.5, .0, 1.0)))
    earlier = _frame(0, _box("old", (0.0, .0, 1.0)))
    result = associate_box_motion(current, previous, earlier)
    assert result.diagnostics[0].status == "implausible_speed"
    assert result.objects[0].velocity_xy == (0.0, 0.0)


@pytest.mark.parametrize("previous_only", [False, True])
def test_history_without_two_matches_abstains_instead_of_fabricating_support(previous_only):
    current = _frame(1_000_000, _box("now", (12.0, .0, 1.0)))
    support = (_box("mid", (10.0, .0, 1.0)),) if previous_only else ()
    result = associate_box_motion(current, _frame(500_000, *support), _frame(0))
    assert result.objects[0].velocity_xy == (0.0, 0.0)
    assert result.diagnostics[0].status == ("missing_earlier" if previous_only else "missing_previous")
    assert result.diagnostics[0].previous_object_id == ("mid" if previous_only else None)


def test_no_detections_means_no_synthetic_tracks():
    obj = _box("history", (10.0, 2.0, 1.0))
    result = associate_box_motion(_frame(1_000_000), _frame(500_000, obj), _frame(0, obj))
    assert result.objects == result.diagnostics == ()


@pytest.mark.parametrize("prev_us, earlier_us", [(2_000_000, 1_500_000), (1_800_000, 1_300_000),
                                                   (999_999, 500_000), (1_500_000, 1_500_000),
                                                   (1_500_000, 1_600_000)])
def test_wrong_adjacent_clocks_are_explicit_failures(prev_us, earlier_us):
    with pytest.raises(ValueError, match="timestamps"):
        associate_box_motion(_frame(2_000_000), _frame(prev_us), _frame(earlier_us))


def test_distinct_recording_and_duplicate_boxes_are_explicit_failures():
    with pytest.raises(ValueError, match="different recordings"):
        associate_box_motion(_frame(1_000_000), _frame(500_000), _frame(0, recording="other"))
    obj = _box("duplicate", (12.0, .0, 1.0))
    with pytest.raises(ValueError, match="unique box IDs"):
        _frame(1_000_000, obj, obj)


def test_frame_owns_an_immutable_snapshot_of_pose_and_input_box_sequence():
    pose = _pose(translation=(1.0, 2.0, 3.0))
    boxes = [_box("first", (12.0, .0, 1.0))]
    frame = MotionFrame("test", 0, pose, boxes)
    pose[0, 3] = 9.0
    boxes.clear()
    assert frame.ego_to_world[0, 3] == 1.0
    assert frame.boxes[0].object_id == "first"
    assert not frame.ego_to_world.flags.writeable
