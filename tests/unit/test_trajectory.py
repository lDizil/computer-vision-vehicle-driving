import math

import numpy as np
import pytest

from hybrid_drive import Pose2D, Trajectory


def assert_angles_equal(actual, expected):
    delta = np.asarray(actual) - np.asarray(expected)
    np.testing.assert_allclose(np.arctan2(np.sin(delta), np.cos(delta)), 0, atol=1e-12)


@pytest.mark.parametrize(
    "heading, expected",
    [(0, 0), (math.pi, -math.pi), (-math.pi, -math.pi),
     (3 * math.pi, -math.pi), (-3 * math.pi, -math.pi),
     (5 * math.pi / 2, math.pi / 2), (-5 * math.pi / 2, -math.pi / 2)],
)
def test_pose_normalizes_heading(heading, expected):
    pose = Pose2D(2, -3, heading)
    assert pose.x == 2
    assert pose.y == -3
    assert -math.pi <= pose.heading < math.pi
    assert pose.heading == pytest.approx(expected)


@pytest.mark.parametrize("component", [0, 1, 2])
@pytest.mark.parametrize("value", [True, np.bool_(False), "1", 1 + 0j, None, np.nan, np.inf, -np.inf])
def test_pose_rejects_nonfinite_or_nonreal_components(component, value):
    values = [0, 0, 0]
    values[component] = value
    with pytest.raises(ValueError):
        Pose2D(*values)


def test_pose_accepts_numpy_real_scalars_and_large_finite_heading():
    pose = Pose2D(np.int64(3), np.float32(-2), np.float64(1e300))
    assert pose.x == 3
    assert pose.y == -2
    assert -math.pi <= pose.heading < math.pi


def test_future_only_sampling_and_float64_snapshot():
    poses = np.array([[2, 1, 0.25], [4, -2, -0.5]], dtype=np.float32)
    trajectory = Trajectory(poses, np.float64(0.25))
    assert trajectory.num_poses == 2
    assert trajectory.horizon_s == pytest.approx(0.5)
    np.testing.assert_array_equal(trajectory.times_s, [0.25, 0.5])
    assert trajectory.poses.dtype == np.dtype(np.float64)
    poses[:] = 99
    np.testing.assert_array_equal(trajectory.poses, [[2, 1, 0.25], [4, -2, -0.5]])


def test_exposed_array_cannot_mutate_trajectory_even_after_setflags():
    expected = np.array([[2, 1, 0.25], [4, -2, -0.5]], dtype=np.float64)
    trajectory = Trajectory(expected.copy(), 0.5)
    exposed = np.asarray(trajectory.poses)
    try:
        exposed.setflags(write=True)
    except ValueError:
        pass
    else:
        exposed[:] = 123
    np.testing.assert_array_equal(trajectory.poses, expected)


@pytest.mark.parametrize(
    "poses",
    [[], np.empty((0, 3)), 1, [1, 2, 3], [[1, 2]], [[1, 2, 3, 4]],
     np.zeros((1, 1, 3)), [[1, 2, 3], [4, 5]], None,
     [["1", "2", "3"]], [[1, 2, "bad"]], [[1, 2, 0j]]],
)
def test_trajectory_rejects_invalid_shape_or_nonnumeric_poses(poses):
    with pytest.raises(ValueError):
        Trajectory(poses, 0.5)


@pytest.mark.parametrize("component", [0, 1, 2])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_trajectory_rejects_nonfinite_pose_components(component, value):
    poses = np.zeros((2, 3))
    poses[1, component] = value
    with pytest.raises(ValueError):
        Trajectory(poses, 0.5)


@pytest.mark.parametrize("interval_s", [0, -1, True, np.bool_(True), "0.5", 1j, None, np.nan, np.inf, -np.inf])
def test_trajectory_rejects_invalid_interval(interval_s):
    with pytest.raises(ValueError):
        Trajectory([[1, 0, 0]], interval_s)


def test_trajectory_rejects_overflowing_horizon():
    with pytest.raises(ValueError):
        Trajectory([[1, 0, 0], [2, 0, 0]], np.finfo(np.float64).max)


def test_world_to_local_known_ninety_degree_rotation_and_translation():
    reference = Pose2D(10, -3, math.pi / 2)
    world = [[9, -1, 2 * math.pi / 3], [12, 1, math.pi / 6]]
    trajectory = Trajectory.from_world_poses(world, reference, 0.5)
    np.testing.assert_allclose(trajectory.poses[:, :2], [[2, 1], [4, -2]], atol=1e-12)
    assert_angles_equal(trajectory.poses[:, 2], [math.pi / 6, -math.pi / 3])
    assert trajectory.num_poses == 2
    np.testing.assert_array_equal(trajectory.times_s, [0.5, 1.0])


def test_local_to_world_known_ninety_degree_rotation_and_translation():
    reference = Pose2D(10, -3, math.pi / 2)
    trajectory = Trajectory([[2, 1, math.pi / 6], [4, -2, -math.pi / 3]], 0.5)
    world = trajectory.to_world_poses(reference)
    assert world.shape == (2, 3)
    np.testing.assert_allclose(world[:, :2], [[9, -1], [12, 1]], atol=1e-12)
    assert_angles_equal(world[:, 2], [2 * math.pi / 3, math.pi / 6])
    np.testing.assert_allclose(trajectory.poses[:, :2], [[2, 1], [4, -2]], atol=1e-12)


@pytest.mark.parametrize("poses", [[], [[1, 2]], [["1", "2", "3"]], [[1, np.nan, 0]]])
def test_world_constructor_validates_poses(poses):
    with pytest.raises(ValueError):
        Trajectory.from_world_poses(poses, Pose2D(0, 0, 0), 0.5)


def test_resample_interpolates_from_local_origin_without_emitting_t0():
    source = Trajectory([[2, 1, 0.2], [4, 2, 0.4], [6, 3, 0.6]], 1)
    result = source.resample(interval_s=0.5, num_poses=6)
    expected = [[1, 0.5, 0.1], [2, 1, 0.2], [3, 1.5, 0.3],
                [4, 2, 0.4], [5, 2.5, 0.5], [6, 3, 0.6]]
    np.testing.assert_allclose(result.poses, expected, atol=1e-12)
    np.testing.assert_array_equal(result.times_s, [0.5, 1, 1.5, 2, 2.5, 3])
    assert result.num_poses == 6
    assert result.horizon_s == pytest.approx(3)


@pytest.mark.parametrize("direction", [1, -1])
def test_resample_takes_shortest_arc_across_pi(direction):
    source = Trajectory([[2, 0, direction * math.radians(170)],
                         [4, 0, -direction * math.radians(170)]], 1)
    result = source.resample(0.5, 4)
    np.testing.assert_allclose(result.poses[:, :2], [[1, 0], [2, 0], [3, 0], [4, 0]], atol=1e-12)
    expected = np.radians([85, 170, 180, -170]) * direction
    assert_angles_equal(result.poses[:, 2], expected)


def test_resample_accepts_numpy_integer_count_and_shorter_horizon():
    result = Trajectory([[2, 0, 0], [4, 0, 0]], 0.5).resample(0.25, np.int64(3))
    np.testing.assert_allclose(result.poses, [[1, 0, 0], [2, 0, 0], [3, 0, 0]], atol=1e-12)
    assert result.num_poses == 3
    assert result.horizon_s == pytest.approx(0.75)


@pytest.mark.parametrize("count", [0, -1, 1.5, 2.0, True, np.bool_(False), "2", None, np.nan, np.inf])
def test_resample_rejects_invalid_count(count):
    with pytest.raises(ValueError):
        Trajectory([[1, 0, 0]], 1).resample(0.25, count)


@pytest.mark.parametrize("interval_s", [0, -1, True, np.bool_(True), "0.5", 1j, None, np.nan, np.inf, -np.inf])
def test_resample_rejects_invalid_interval(interval_s):
    with pytest.raises(ValueError):
        Trajectory([[1, 0, 0]], 1).resample(interval_s, 1)


def test_resample_rejects_overflowing_target_horizon():
    with pytest.raises(ValueError):
        Trajectory([[1, 0, 0]], 1).resample(np.finfo(np.float64).max, 2)


def test_resample_rejects_extrapolation():
    with pytest.raises(ValueError):
        Trajectory([[2, 0, 0]], 1).resample(0.25, 5)


def test_resample_allows_one_ulp_rounding_at_horizon():
    source = Trajectory([[3, 0, 0]], 0.3)
    assert 3 * 0.1 > source.horizon_s
    result = source.resample(0.1, 3)
    np.testing.assert_allclose(result.poses, [[1, 0, 0], [2, 0, 0], [3, 0, 0]], atol=1e-12)


def test_resample_rejects_more_than_one_ulp_past_horizon():
    interval_s = np.nextafter(np.nextafter(np.float64(0.3), np.inf), np.inf)
    with pytest.raises(ValueError):
        Trajectory([[3, 0, 0]], 0.3).resample(interval_s, 1)


def test_heading_normalization_preserves_small_angles():
    assert Pose2D(0, 0, 1e-17).heading == 1e-17
    assert Trajectory([[0, 0, -1e-17]], 1).poses[0, 2] == -1e-17


def test_python_integer_overflow_is_a_validation_error():
    with pytest.raises(ValueError):
        Pose2D(10 ** 1000, 0, 0)
    with pytest.raises(ValueError):
        Trajectory([[1, 0, 0]], 10 ** 1000)
    with pytest.raises(ValueError):
        Trajectory([[1, 0, 0]], 1).resample(1, 10 ** 1000)


def test_coordinate_conversion_rejects_overflow():
    with pytest.raises(ValueError):
        Trajectory([[1e308, 0, 0]], 1).to_world_poses(Pose2D(1e308, 0, 0))
    with pytest.raises(ValueError):
        Trajectory.from_world_poses([[1e308, 0, 0]], Pose2D(-1e308, 0, 0), 1)


def test_array_metadata_changes_cannot_alter_trajectory():
    source = Trajectory([[1, 0, 0], [2, 0, 0]], 0.5)
    exposed = source.poses
    exposed.shape = (1, 6)
    exposed.dtype = np.uint8
    exposed.base.shape = (2, 3)
    exposed.base.dtype = np.uint8
    assert source.num_poses == 2
    assert source.horizon_s == 1
    assert source.poses.dtype == np.float64
    np.testing.assert_array_equal(source.poses, [[1, 0, 0], [2, 0, 0]])


def test_resample_avoids_intermediate_slope_overflow():
    source = Trajectory([[2, 0, 0], [4, 0, 0]], 1e-308)
    np.testing.assert_allclose(source.resample(0.5e-308, 4).poses[:, 0], [1, 2, 3, 4])


def test_resample_avoids_intermediate_coordinate_delta_overflow():
    source = Trajectory([[1e308, 0, 0], [-1e308, 0, 0]], 1)
    np.testing.assert_allclose(source.resample(0.5, 4).poses[:, 0], [0.5e308, 1e308, 0, -1e308])


@pytest.mark.parametrize("heading", [np.nextafter(-math.pi, -np.inf), np.nextafter(math.pi, np.inf)])
def test_normalized_heading_stays_in_half_open_range_near_pi(heading):
    assert -math.pi <= Pose2D(0, 0, heading).heading < math.pi


def test_world_heading_is_normalized_before_reference_rotation():
    world_heading = Pose2D(0, 0, 1e300).heading
    actual = Trajectory.from_world_poses([[0, 0, 1e300]], Pose2D(0, 0, 0.5), 1)
    assert_angles_equal(actual.poses[0, 2], world_heading - 0.5)
