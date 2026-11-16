from dataclasses import FrozenInstanceError
import builtins
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from hybrid_drive import Trajectory
from hybrid_drive.adapters.real_camera import (
    OPENSCENE_NATIVE_PROFILE, CalibratedView, prepare_openscene_camera, render_plan_overlay,
)
from hybrid_drive.geometry import CameraCalibration, project_segments
from hybrid_drive.slow.context import CameraFrame


def arguments():
    return dict(name="CAM_F0", rgb=np.arange(8 * 12 * 3, dtype=np.uint8).reshape(8, 12, 3),
                intrinsic=[[8, 0, 5], [0, 10, 3], [0, 0, 1]],
                camera_to_ego_rotation=np.eye(3), camera_to_ego_translation=[0, 0, 0],
                distortion=[0, 0, 0, 0, 0], profile=OPENSCENE_NATIVE_PROFILE, width=6, height=4)


@pytest.fixture
def stub_opencv(monkeypatch):
    calls = []

    def maps(intrinsic, distortion, rectification, new_intrinsic, dimensions, map_type):
        calls.append((intrinsic.copy(), distortion.copy(), rectification, new_intrinsic.copy(), dimensions, map_type))
        return "map-x", "map-y"

    def remap(image, map_x, map_y, **options):
        calls.append((map_x, map_y, options))
        return image.copy()

    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace(
        initUndistortRectifyMap=maps, remap=remap, CV_32FC1=5, INTER_LINEAR=1, BORDER_CONSTANT=0,
    ))
    return calls


def test_native_extrinsic_inverse_has_analytic_sign_and_direction(stub_opencv):
    values = arguments()
    # Camera axes in ego: right=-left, down=-up, forward=forward.
    values.update(camera_to_ego_rotation=[[0, 0, 1], [-1, 0, 0], [0, -1, 0]],
                  camera_to_ego_translation=[1, 2, 3])
    view = prepare_openscene_camera(**values)
    expected = [[0, -1, 0, 2], [0, 0, -1, 3], [1, 0, 0, -1], [0, 0, 0, 1]]
    np.testing.assert_array_equal(view.calibration.ego_to_camera, expected)
    # Ego (11,2,3) lies on the optical axis; (11,0,1) is camera (2,2,10).
    np.testing.assert_allclose(project_segments([[11, 2, 3], [11, 0, 1]], view.calibration),
                               [[[2.25, 1.25], [3.05, 2.25]]], atol=1e-12)


def test_rectification_keeps_native_intrinsic_before_half_pixel_resize(stub_opencv):
    values = arguments()
    view = prepare_openscene_camera(**values)
    intrinsic, distortion, rectification, new_intrinsic, dimensions, map_type = stub_opencv[0]
    np.testing.assert_array_equal(intrinsic, values["intrinsic"])
    np.testing.assert_array_equal(new_intrinsic, values["intrinsic"])
    np.testing.assert_array_equal(distortion, values["distortion"])
    assert rectification is None
    assert (dimensions, map_type) == ((12, 8), 5)
    assert stub_opencv[1] == ("map-x", "map-y", {
        "interpolation": 1, "borderMode": 0, "borderValue": (0, 0, 0),
    })
    np.testing.assert_array_equal(view.calibration.intrinsic, [[4, 0, 2.25], [0, 5, 1.25], [0, 0, 1]])
    expected = Image.fromarray(values["rgb"]).resize((6, 4), resample=Image.Resampling.BILINEAR)
    np.testing.assert_array_equal(view.frame.rgb, expected)
    assert (view.frame.name, view.frame.width, view.frame.height) == ("CAM_F0", 6, 4)


@pytest.mark.parametrize("change", [
    {"profile": None}, {"profile": "rectified"}, {"name": "CAM_B0"}, {"name": None},
    {"width": True}, {"height": np.bool_(True)}, {"width": 6.0}, {"height": 4.0},
    {"width": 0}, {"height": -1}, {"width": 641}, {"height": 481},
    {"rgb": None}, {"rgb": np.zeros((2, 2, 3), np.float32)},
    {"rgb": np.zeros((0, 2, 3), np.uint8)}, {"rgb": np.zeros((2, 2, 4), np.uint8)},
    {"intrinsic": [[8, 1, 5], [0, 10, 3], [0, 0, 1]]},
    {"intrinsic": [[0, 0, 5], [0, 10, 3], [0, 0, 1]]},
    {"intrinsic": [[8, 0, 5], [0, np.nan, 3], [0, 0, 1]]},
    {"camera_to_ego_rotation": np.eye(4)},
    {"camera_to_ego_rotation": [[-1, 0, 0], [0, 1, 0], [0, 0, 1]]},
    {"camera_to_ego_rotation": [[2, 0, 0], [0, 1, 0], [0, 0, 1]]},
    {"camera_to_ego_rotation": [[1, 0.1, 0], [0, 1, 0], [0, 0, 1]]},
    {"camera_to_ego_rotation": [[True, 0, 0], [0, 1, 0], [0, 0, 1]]},
    {"camera_to_ego_translation": [0, 0]}, {"camera_to_ego_translation": [[0, 0, 0]]},
    {"camera_to_ego_translation": [0, np.inf, 0]}, {"camera_to_ego_translation": [0, False, 0]},
    {"distortion": None}, {"distortion": [0, 0, 0, 0]}, {"distortion": [[0, 0, 0, 0, 0]]},
    {"distortion": [0, 0, np.nan, 0, 0]}, {"distortion": [0, 0, "0", 0, 0]},
    {"distortion": [0, 0, False, 0, 0]},
])
def test_invalid_native_inputs_fail_before_importing_opencv(monkeypatch, change):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "cv2":
            raise AssertionError("Invalid calibration reached OpenCV")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    values = arguments()
    values.update(change)
    with pytest.raises(ValueError):
        prepare_openscene_camera(**values)


def test_native_pixel_limit_is_checked_before_allocating_maps(monkeypatch):
    values = arguments()
    values["rgb"] = np.broadcast_to(np.zeros((1, 1, 3), np.uint8), (4001, 4000, 3))
    monkeypatch.setitem(sys.modules, "cv2", None)
    with pytest.raises(ValueError, match="16 megapixels"):
        prepare_openscene_camera(**values)


def test_rectification_owns_pixels_and_calibration(stub_opencv):
    values = arguments()
    values["intrinsic"] = np.asarray(values["intrinsic"], dtype=np.float64)
    view = prepare_openscene_camera(**values)
    expected = view.frame.rgb.copy()
    values["rgb"][:] = 255
    values["intrinsic"][:] = 99
    np.testing.assert_array_equal(view.frame.rgb, expected)
    assert view.calibration.intrinsic[0, 0] == 4
    with pytest.raises(ValueError):
        view.frame.rgb.setflags(write=True)
    with pytest.raises(FrozenInstanceError):
        view.frame = None


@pytest.mark.parametrize("name,width,height", [("CAM_L0", 6, 4), ("CAM_F0", 5, 4), ("CAM_F0", 6, 3)])
def test_calibrated_view_rejects_mismatched_identity_or_dimensions(name, width, height):
    frame = CameraFrame("CAM_F0", np.zeros((4, 6, 3), np.uint8))
    calibration = CameraCalibration(name, width, height, np.eye(3), np.eye(4))
    with pytest.raises(ValueError, match="matching"):
        CalibratedView(frame, calibration)


@pytest.mark.parametrize("frame,calibration", [(None, None), ("image", np.eye(4))])
def test_calibrated_view_requires_typed_values(frame, calibration):
    with pytest.raises(ValueError):
        CalibratedView(frame, calibration)


def test_zero_distortion_rectification_preserves_rgb_channel_order():
    pytest.importorskip("cv2", reason="Native rectification runs in the Linux OpenCV environment")
    values = arguments()
    values.update(width=12, height=8)
    view = prepare_openscene_camera(**values)
    np.testing.assert_array_equal(view.frame.rgb, values["rgb"])
    np.testing.assert_array_equal(view.calibration.intrinsic, values["intrinsic"])


def test_nonzero_brown_distortion_matches_independent_analytic_source_pixel():
    pytest.importorskip("cv2", reason="Native rectification runs in the Linux OpenCV environment")
    rows, columns = np.indices((9, 9), dtype=np.uint8)
    rgb = np.stack((16 * columns, 16 * rows, np.full_like(rows, 7)), axis=-1)
    view = prepare_openscene_camera(
        "CAM_F0", rgb, [[4, 0, 4], [0, 4, 4], [0, 0, 1]], np.eye(3), [0, 0, 0],
        [0.25, 0.125, 0.0625, -0.03125, 0.0625], profile=OPENSCENE_NATIVE_PROFILE, width=9, height=9,
    )
    # Output (6,4) is normalized (.5,0). Brown forward distortion maps it
    # to raw (6.048828125,4.0625); remap's 1/32 interpolation grid gives (6.0625,4.0625).
    np.testing.assert_array_equal(view.frame.rgb[4, 6], [97, 65, 7])
    np.testing.assert_array_equal(view.frame.rgb[8, 8], [0, 0, 0])


def overlay_view():
    frame = CameraFrame("CAM_F0", np.full((31, 41, 3), (17, 23, 31), np.uint8))
    transform = [[0, -1, 0, 0], [0, 0, -1, 1], [1, 0, 0, 0], [0, 0, 0, 1]]
    calibration = CameraCalibration("CAM_F0", 41, 31, [[20, 0, 20], [0, 20, 10], [0, 0, 1]], transform)
    return CalibratedView(frame, calibration)


def test_overlay_uses_actual_plan_projection_and_preserves_source_frame():
    view = overlay_view()
    result = render_plan_overlay(view, Trajectory([[5, 0, 2], [10, 0, -1]], 0.5))
    np.testing.assert_array_equal(result.rgb[20, 20], [0, 200, 220])
    np.testing.assert_array_equal(result.rgb[12, 20], [0, 200, 220])
    assert np.count_nonzero(np.all(result.rgb[20] == [0, 200, 220], axis=1)) == 4
    np.testing.assert_array_equal(result.rgb[0, 0], [17, 23, 31])
    np.testing.assert_array_equal(view.frame.rgb, np.full((31, 41, 3), (17, 23, 31), np.uint8))
    assert result.name == view.frame.name


def test_invisible_plan_does_not_invent_a_candidate_line():
    view = overlay_view()
    result = render_plan_overlay(view, Trajectory([[-5, 1, 0], [-10, -1, 0]], 0.5))
    np.testing.assert_array_equal(result.rgb, view.frame.rgb)


@pytest.mark.parametrize("view,trajectory", [(None, Trajectory([[1, 0, 0]], 0.5)), (overlay_view(), None)])
def test_overlay_rejects_untyped_inputs(view, trajectory):
    with pytest.raises(ValueError):
        render_plan_overlay(view, trajectory)


def test_import_and_rectified_overlay_do_not_require_opencv():
    source = Path(__file__).resolve().parents[2] / "src"
    environment = dict(os.environ, PYTHONPATH=str(source), PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([sys.executable, "-c", """
import sys
sys.modules['cv2'] = None
import numpy as np
from hybrid_drive import Trajectory
from hybrid_drive.adapters.real_camera import CalibratedView, render_plan_overlay
from hybrid_drive.geometry import CameraCalibration
from hybrid_drive.slow.context import CameraFrame
view = CalibratedView(CameraFrame('CAM_F0', np.zeros((2, 2, 3), np.uint8)),
                      CameraCalibration('CAM_F0', 2, 2, np.eye(3), np.eye(4)))
render_plan_overlay(view, Trajectory([[1, 0, 0]], .5))
"""], capture_output=True, text=True, env=environment, timeout=30)
    assert result.returncode == 0, result.stderr
