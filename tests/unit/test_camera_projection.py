from dataclasses import FrozenInstanceError
import pickle

import numpy as np
import pytest

from hybrid_drive import Trajectory
from hybrid_drive.geometry import CameraCalibration, project_segments, project_trajectory


def calibration(*, width=201, height=101, intrinsic=None, transform=None, near_m=0.1):
    if intrinsic is None:
        intrinsic = [[100, 0, 100], [0, 100, 50], [0, 0, 1]]
    if transform is None:
        transform = np.eye(4)
    return CameraCalibration("front", width, height, intrinsic, transform, near_m)


def test_forward_ego_axes_and_camera_height_have_analytic_pixels():
    # Camera at ego z=1: camera coordinates are (-ego_y, 1-ego_z, ego_x).
    transform = [[0, -1, 0, 0], [0, 0, -1, 1], [1, 0, 0, 0], [0, 0, 0, 1]]
    camera = calibration(transform=transform)
    result = project_segments([[10, 2, 0], [20, -2, 1]], camera)
    np.testing.assert_allclose(result, [[[80, 60], [110, 50]]], atol=1e-12)


def test_rotated_translated_camera_and_skew_use_full_transform():
    # 90-degree rotation about camera z, followed by camera translation (1, 2, 3).
    transform = [[0, -1, 0, 1], [1, 0, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]]
    camera = calibration(intrinsic=[[100, 10, 100], [0, 100, 50], [0, 0, 1]], transform=transform)
    result = project_segments([[2, 1, 7], [0, -1, 7]], camera)
    np.testing.assert_allclose(result, [[[104, 90], [122, 70]]], atol=1e-12)


def test_near_plane_clipping_can_reduce_a_camera_origin_ray_to_no_image_edge():
    camera = calibration(near_m=1)
    # Near intersection is (0.25, 0, 1), not a 2D interpolation of projected endpoints.
    assert project_segments([[0, 0, 0], [0.5, 0, 2]], camera) == ()


def test_near_plane_crossing_keeps_correct_varying_pixel_segment():
    camera = calibration(near_m=1)
    # Intersection: (0.25, 0.5, 1), front endpoint: (0.5, 0, 2).
    result = project_segments([[0, 1, 0], [0.5, 0, 2]], camera)
    np.testing.assert_allclose(result, [[[125, 100], [125, 50]]], atol=1e-12)
    reverse = project_segments([[0.5, 0, 2], [0, 1, 0]], camera)
    np.testing.assert_allclose(reverse, [[[125, 50], [125, 100]]], atol=1e-12)


def test_entirely_behind_camera_or_near_plane_has_no_projection():
    camera = calibration(near_m=1)
    assert project_segments([[-1, 0, -2], [1, 0, -1]], camera) == ()
    assert project_segments([[-1, 0, 0], [1, 0, 0.5]], camera) == ()


@pytest.mark.parametrize(
    "points, expected",
    [([[-4, 0, 2], [4, 0, 2]], [[[0, 50], [200, 50]]]),
     ([[0, -4, 2], [0, 4, 2]], [[[100, 0], [100, 100]]]),
     ([[-4, -4, 2], [4, 4, 2]], [[[50, 0], [150, 100]]]),
     ([[4, 4, 2], [-4, -4, 2]], [[[150, 100], [50, 0]]])],
)
def test_viewport_clips_crossing_segments_even_when_both_endpoints_are_outside(points, expected):
    np.testing.assert_allclose(project_segments(points, calibration()), expected, atol=1e-12)


def test_offscreen_segments_are_omitted_without_joining_visible_pieces():
    # Two visible horizontal pieces leave/re-enter at different heights; no bridge at u=200.
    points = [[0, -0.5, 2], [4, -0.5, 2], [4, 0.5, 2], [0, 0.5, 2]]
    result = project_segments(points, calibration())
    np.testing.assert_allclose(result, [[[100, 25], [200, 25]], [[200, 75], [100, 75]]], atol=1e-12)


def test_behind_sections_are_omitted_without_connecting_visible_sides():
    points = [[-0.5, 0, 2], [-0.5, 0, -2], [0.5, 0, -2], [0.5, 0, 2]]
    result = project_segments(points, calibration(near_m=1))
    np.testing.assert_allclose(result, [[[75, 50], [50, 50]], [[150, 50], [125, 50]]], atol=1e-12)


def test_calibration_resize_uses_pil_pixel_centres_and_scales_skew():
    source = CameraCalibration("wide", 200, 100, [[100, 10, 80], [0, 120, 40], [0, 0, 1]], np.eye(4))
    resized = source.resized(100, 25)
    np.testing.assert_allclose(resized.intrinsic, [[50, 5, 39.75], [0, 30, 9.625], [0, 0, 1]])
    np.testing.assert_array_equal(resized.ego_to_camera, source.ego_to_camera)
    assert (resized.name, resized.width, resized.height, resized.near_m) == ("wide", 100, 25, 0.1)
    original_pixels = project_segments([[0, 0, 10], [1, 1, 10]], source)
    resized_pixels = project_segments([[0, 0, 10], [1, 1, 10]], resized)
    np.testing.assert_allclose(original_pixels, [[[80, 40], [91, 52]]], atol=1e-12)
    np.testing.assert_allclose(resized_pixels, [[[39.75, 9.625], [45.25, 12.625]]], atol=1e-12)


def test_offaxis_principal_point_is_valid_and_resize_can_move_it_outside_pixel_centres():
    source = calibration(intrinsic=[[100, 0, -50], [0, 100, 50], [0, 0, 1]])
    np.testing.assert_allclose(project_segments([[2, 0, 2], [4, 0, 2]], source),
                               [[[50, 50], [150, 50]]], atol=1e-12)
    boundary = CameraCalibration("edge", 200, 100, [[100, 0, 0], [0, 100, 0], [0, 0, 1]], np.eye(4))
    np.testing.assert_allclose(boundary.resized(100, 50).intrinsic,
                               [[50, 0, -0.25], [0, 50, -0.25], [0, 0, 1]])


@pytest.mark.parametrize("roundtrip", [False, True])
def test_calibration_owns_immutable_matrix_bytes_even_after_pickle(roundtrip):
    intrinsic = np.array([[100, 0, 100], [0, 100, 50], [0, 0, 1]], dtype=np.float64)
    transform = np.eye(4)
    camera = CameraCalibration("front", 201, 101, intrinsic, transform)
    intrinsic[:] = 12
    transform[:] = 34
    if roundtrip:
        camera = pickle.loads(pickle.dumps(camera))
    for exposed in (camera.intrinsic, camera.ego_to_camera):
        assert not exposed.flags.writeable
        with pytest.raises(ValueError):
            exposed.setflags(write=True)
        with pytest.raises(ValueError):
            exposed.flat[0] = 99
        exposed.shape = (exposed.size,)
    np.testing.assert_array_equal(camera.intrinsic, [[100, 0, 100], [0, 100, 50], [0, 0, 1]])
    np.testing.assert_array_equal(camera.ego_to_camera, np.eye(4))
    with pytest.raises(FrozenInstanceError):
        camera.width = 7


@pytest.mark.parametrize("field", ["width", "height"])
@pytest.mark.parametrize("value", [0, -1, 65537, 1.5, 2.0, True, np.bool_(False), "2", None, np.inf, np.nan])
def test_calibration_rejects_invalid_image_dimensions(field, value):
    with pytest.raises(ValueError):
        calibration(**{field: value})


@pytest.mark.parametrize("value", [0, -1, True, np.bool_(False), "0.1", None, np.inf, np.nan])
def test_calibration_rejects_invalid_near_plane(value):
    with pytest.raises(ValueError):
        calibration(near_m=value)


@pytest.mark.parametrize("name", ["", "  ", None, 1])
def test_calibration_rejects_blank_or_nonstring_name(name):
    with pytest.raises(ValueError):
        CameraCalibration(name, 201, 101, np.eye(3), np.eye(4))


@pytest.mark.parametrize(
    "intrinsic",
    [[[0, 0, 100], [0, 100, 50], [0, 0, 1]],
     [[100, 0, 100], [0, -1, 50], [0, 0, 1]],
     [[100, 0, 100], [1, 100, 50], [0, 0, 1]],
     [[100, 0, 100], [0, 100, 50], [0, 0.1, 1]],
     [[100, 0, 100], [0, 100, 50], [0, 0, 2]],
     [[100, 0, 100], [0, 100, 50]],
     [[100, 0, 100], [0, 100, 50], [False, 0, 1]],
     [[100, 0, "100"], [0, 100, 50], [0, 0, 1]],
     [[100, 0, 100], [0, np.inf, 50], [0, 0, 1]],
     [[100, 0, 100], [0, 100, np.nan], [0, 0, 1]]],
)
def test_calibration_rejects_invalid_intrinsic(intrinsic):
    with pytest.raises(ValueError):
        calibration(intrinsic=intrinsic)


@pytest.mark.parametrize("kind", ["reflection", "scale", "shear", "homogeneous", "nan", "bool", "shape"])
def test_calibration_requires_proper_finite_se3(kind):
    transform = np.eye(4).tolist()
    if kind == "reflection":
        transform[0][0] = -1
    elif kind == "scale":
        transform[0][0] = 2
    elif kind == "shear":
        transform[0][1] = 0.1
    elif kind == "homogeneous":
        transform[3][0] = 0.01
    elif kind == "nan":
        transform[0][3] = np.nan
    elif kind == "bool":
        transform[0][3] = False
    else:
        transform = transform[:3]
    with pytest.raises(ValueError):
        calibration(transform=transform)


@pytest.mark.parametrize(
    "points",
    [[[0, 0]], [[0, 0, 1, 2]], [0, 0, 1], [[[0, 0, 1]]],
     [[0, False, 1], [1, 0, 1]], [[0, "0", 1], [1, 0, 1]],
     [[0, 0j, 1], [1, 0, 1]], [[0, np.nan, 1], [1, 0, 1]],
     [[0, 0, 1], [np.inf, 0, 1]]],
)
def test_projection_rejects_malformed_nonreal_nonfinite_points(points):
    with pytest.raises(ValueError):
        project_segments(points, calibration())


def test_projection_rejects_overflow_rather_than_emitting_nonfinite_pixels():
    camera = calibration(near_m=1e-300)
    with pytest.raises(ValueError, match="finite pixel"):
        project_segments([[1e300, 0, 1e-300], [0, 0, 1]], camera)
    transform = np.eye(4)
    transform[0, 3] = 1e308
    with pytest.raises(ValueError, match="finite coordinate"):
        project_segments([[1e308, 0, 1], [0, 0, 1]], calibration(transform=transform))


def test_finite_extreme_viewport_crossings_are_rejected_instead_of_inventing_a_line():
    camera = calibration(intrinsic=np.eye(3))
    with pytest.raises(ValueError, match="clipping limit"):
        project_segments([[-1e300, -1e300, 1], [1e300, 1e300, 1]], camera)


def test_extreme_near_clipping_is_rejected_instead_of_inventing_a_line():
    camera = calibration(intrinsic=np.eye(3), near_m=1)
    # x=z throughout the 3D line, so every visible point projects to (1, 0).
    with pytest.raises(ValueError, match="depth-to-near clipping limit"):
        project_segments([[-1e300, 0, -1e300], [1e300, 0, 1e300]], camera)


def test_degenerate_polylines_and_one_pixel_viewport_do_not_invent_lines():
    camera = calibration()
    assert project_segments(np.empty((0, 3)), camera) == ()
    assert project_segments([[0, 0, 1]], camera) == ()
    assert project_segments([[0, 0, 1], [0, 0, 1]], camera) == ()
    assert project_segments([[0, 0, 1], [0, 0, 2]], camera) == ()
    one_pixel = CameraCalibration("pixel", 1, 1, np.eye(3), np.eye(4))
    assert project_segments([[-1, -1, 1], [1, 1, 1]], one_pixel) == ()


def test_trajectory_uses_ground_height_ignores_heading_and_prepends_ego_origin():
    # Ground ego origin is camera (1, 1, 2), so its pixel is (150, 100).
    transform = [[0, -1, 0, 1], [0, 0, -1, 1], [1, 0, 0, 2], [0, 0, 0, 1]]
    camera = calibration(transform=transform)
    trajectory = Trajectory([[2, 1, 1.2], [3, -1, -2.5]], 0.5)
    result = project_trajectory(trajectory, camera)
    np.testing.assert_allclose(result, [[[150, 100], [100, 75]], [[100, 75], [140, 70]]], atol=1e-12)


def test_projection_requires_typed_calibration_and_trajectory():
    with pytest.raises(ValueError):
        project_segments([[0, 0, 1], [1, 0, 1]], None)
    with pytest.raises(ValueError):
        project_trajectory([[2, 1, 0]], calibration())
