from dataclasses import FrozenInstanceError, asdict, replace
import json
import math

import numpy as np
import pytest

from hybrid_drive.contracts.planning import CandidateSet
from hybrid_drive.contracts.trajectory import Pose2D
from hybrid_drive.fast.forecast_proposals import ForecastObject, propose_from_forecast


POSE = Pose2D(12, -2, 0.25)
REF_US = 10_250_003


def _object(x, y, *, key="car", yaw=0.0, velocity=(0.0, 0.0), size=(2.0, 4.0, 1.5), confidence=0.6):
    return ForecastObject(key, (x, y, 0.9), size, yaw, velocity, confidence)


def _propose(*objects, speed=4.0):
    return propose_from_forecast("keyframe", REF_US, POSE, speed, objects)


def _diagnostic(proposals, name):
    return next(value for value in proposals.diagnostics if value.candidate_id == name)


def _ids(proposals):
    return [candidate.candidate_id for candidate in proposals.plan.candidates]


@pytest.mark.parametrize("speed", [0.0, 5e-7, 4.0, 60.0])
def test_past_speed_yields_distinct_forward_profiles_same_native_grid_even_when_stopped_or_saturated(speed):
    proposals = _propose(speed=speed)
    assert isinstance(proposals.plan, CandidateSet)
    assert proposals.plan.snapshot_id == "keyframe"
    assert proposals.plan.reference_pose is POSE
    assert proposals.plan.t_ref_s == REF_US / 1e6
    assert proposals.plan.plan_id.endswith(proposals.forecast_hash)
    assert len(proposals.plan.candidates) >= 2
    assert _ids(proposals)[0] == "maintain"
    assert [item.candidate_id for item in proposals.diagnostics] == _ids(proposals)
    seen = []
    for candidate in proposals.plan.candidates:
        trajectory = candidate.trajectory
        assert trajectory.interval_s == 0.5
        assert trajectory.horizon_s == 2.5
        assert trajectory.poses.shape == (5, 3)
        np.testing.assert_array_equal(trajectory.poses[:, 1:], np.zeros((5, 2)))
        distances = trajectory.poses[:, 0]
        assert np.all(np.diff(np.r_[0, distances]) >= 0.0)
        assert all(np.max(np.abs(other - distances)) > 1e-5 for other in seen)
        seen.append(distances)
    hold = proposals.plan.candidates[0].trajectory
    np.testing.assert_array_equal(hold.poses[:, 0], speed * hold.times_s)
    if speed == 0.0:
        assert "brake" not in _ids(proposals)
        assert any(rows[-1] > 0 for rows in seen)


def test_profile_stops_at_analytic_braking_distance_without_reversing():
    trajectories = {entry.candidate_id: entry.trajectory for entry in _propose(speed=3.0).plan.candidates}
    np.testing.assert_array_equal(trajectories["brake"].poses[:, 0], [1.125, 1.5, 1.5, 1.5, 1.5])


def test_blocker_changes_priority_without_hiding_affected_profiles_or_using_object_labels():
    in_lane = _propose(_object(10, 0))
    side_lane = _propose(_object(10, 7))
    assert _ids(side_lane)[0] == "maintain"
    assert _ids(in_lane)[0] == "brake"
    assert set(_ids(in_lane)) == set(_ids(side_lane))
    maintain = _diagnostic(in_lane, "maintain")
    assert maintain.first_overlap_s == 1.125  # Ego front is 8 m, object rear is 8 m.
    assert maintain.overlapping_object_ids == ("car",)
    assert maintain.overlap_sample_count > 0
    brake = _diagnostic(in_lane, "brake")
    assert brake.min_axis_gap_m == pytest.approx(10 - 2 - 3.5 - 4**2 / (2 * 3))
    assert brake.first_overlap_s is None
    assert not in_lane.all_candidates_overlap
    assert in_lane.plan.candidates_hash != side_lane.plan.candidates_hash


def test_future_motion_changes_ranking_with_same_current_box():
    stationary = _object(9.5, 4)
    moving = replace(stationary, velocity_xy=(0, -3))
    assert _ids(_propose(stationary))[0] == "maintain"
    result = _propose(moving)
    assert _ids(result)[0] == "brake"
    assert _diagnostic(result, "maintain").first_overlap_s == 1.0


def test_footprint_scores_wrap_oriented_boxes_and_sample_crossing_between_output_poses():
    across = _object(2, 2.9, yaw=math.pi / 2, size=(1, 4, 1.5))
    assert _diagnostic(_propose(across, speed=0), "maintain").first_overlap_s == 0.0
    straight = replace(across, yaw=math.tau)
    assert _diagnostic(_propose(straight, speed=0), "maintain").min_axis_gap_m == pytest.approx(1.45)
    crossed = _object(1, 10, velocity=(0, -40), size=(1, 2, 1.5))
    assert _diagnostic(_propose(crossed, speed=0), "maintain").first_overlap_s == 0.25


def test_empty_and_all_contact_forecasts_are_distinct_and_keep_reasons_serializable():
    empty = _propose()
    assert empty.forecast_object_count == 0 and not empty.all_candidates_overlap
    assert all(item.min_axis_gap_m is None and item.first_overlap_s is None for item in empty.diagnostics)
    blocker = _object(0.5, 0, confidence=0.0)
    covered = _propose(blocker)
    assert covered.forecast_object_count == 1 and covered.all_candidates_overlap
    assert all(item.first_overlap_s == 0.0 and item.overlapping_object_ids == ("car",) for item in covered.diagnostics)
    encoded = json.dumps([asdict(value) for value in covered.diagnostics], allow_nan=False)
    assert json.loads(encoded)[0]["first_overlap_s"] == 0.0


def test_plan_identity_depends_on_exact_time_pose_and_forecast_and_not_object_order():
    a = _object(10, 0)
    b = _object(8, 4, key="other")
    baseline = _propose(a, b)
    reordered = _propose(b, a)
    assert baseline.forecast_hash == reordered.forecast_hash
    assert baseline.diagnostics == reordered.diagnostics
    assert baseline.plan.candidates_hash == reordered.plan.candidates_hash
    assert baseline.plan.candidates_hash != _propose(a, replace(b, confidence=0.75)).plan.candidates_hash
    assert baseline.plan.candidates_hash != propose_from_forecast("keyframe", REF_US + 1, POSE, 4, (a, b)).plan.candidates_hash
    assert baseline.plan.candidates_hash != propose_from_forecast("keyframe", REF_US, Pose2D(0, 0, 0), 4, (a, b)).plan.candidates_hash


def test_object_normalizes_untrusted_sequence_storage_and_rejects_duplicates():
    center = [4, -2, 0.9]
    item = ForecastObject("a", center, (1, 2, 1), 3 * math.pi, (0, 0), 0.9)
    center[0] = 100
    assert item.center_xyz == (4.0, -2.0, 0.9)
    assert item.yaw == -math.pi
    with pytest.raises(FrozenInstanceError):
        item.confidence = 0.1
    with pytest.raises(ValueError, match="IDs must be unique"):
        _propose(item, item)
    with pytest.raises(ValueError, match="ForecastObject"):
        _propose("car")


@pytest.mark.parametrize("speed", [-1, 61, True, math.nan, math.inf])
def test_invalid_or_unsupported_ego_speed_does_not_produce_paths(speed):
    with pytest.raises(ValueError, match="forward_speed_mps"):
        _propose(speed=speed)


@pytest.mark.parametrize("time", [-1, 1.1, True])
def test_invalid_reference_timestamp_does_not_produce_paths(time):
    with pytest.raises(ValueError, match="ref_time_us"):
        propose_from_forecast("keyframe", time, POSE, 4.0, ())


@pytest.mark.parametrize(("invalid", "name"), [
    ({"object_id": ""}, "object_id"),
    ({"center_xyz": (1, 2)}, "center_xyz"),
    ({"center_xyz": (1, False, 3)}, "center_xyz"),
    ({"center_xyz": (1e200, 0, 1)}, "center_xyz"),
    ({"size_wlh": (0, 4, 1)}, "size_wlh"),
    ({"size_wlh": (1, 120, 1)}, "size_wlh"),
    ({"yaw": math.nan}, "yaw"),
    ({"velocity_xy": (0, 101)}, "velocity_xy"),
    ({"velocity_xy": (math.inf, 0)}, "velocity_xy"),
    ({"confidence": 1.1}, "confidence"),
])
def test_invalid_object_geometry_or_confidence_fails_instead_of_silently_clearing_space(invalid, name):
    with pytest.raises(ValueError, match=name):
        replace(_object(1, 0), **invalid)
