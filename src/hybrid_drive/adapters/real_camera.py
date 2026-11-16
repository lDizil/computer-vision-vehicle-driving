"""Native OpenScene RGB preparation; exported rectified bundles bypass this adapter."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike, NDArray

from hybrid_drive.contracts.trajectory import Trajectory
from hybrid_drive.geometry.camera import CameraCalibration, project_trajectory
from hybrid_drive.slow.context import CAMERA_NAMES, MAX_HEIGHT, MAX_WIDTH, CameraFrame, _validate_rgb


# Producer contract: camera-to-ego fields are copied without a LiDAR transform.
# https://github.com/OpenDriveLab/OpenScene/blob/72860746787a67946bef07aa1f78bbbc6b20e445/DriveEngine/process_data/helpers/nuplan_cameras_utils.py#L73-L94
# Same-K rectification of original nuPlan JPEG follows:
# https://github.com/OpenDriveLab/MTGS/blob/7ab67a3e386e5a4830017819324922a7bb9f26f7/nuplan_scripts/utils/camera_utils.py#L122-L138
OPENSCENE_NATIVE_PROFILE = "openscene-nuplan-original-v1"
_MAX_NATIVE_PIXELS = 16_000_000


@dataclass(frozen=True, eq=False)
class CalibratedView:
    frame: CameraFrame
    calibration: CameraCalibration

    def __post_init__(self):
        if not isinstance(self.frame, CameraFrame) or not isinstance(self.calibration, CameraCalibration):
            raise ValueError("CalibratedView requires CameraFrame and CameraCalibration")
        if (self.frame.name, self.frame.width, self.frame.height) != (
            self.calibration.name, self.calibration.width, self.calibration.height
        ):
            raise ValueError("Frame and calibration must have matching name and dimensions")


def prepare_openscene_camera(
    name: str, rgb: NDArray[np.uint8], intrinsic: ArrayLike,
    camera_to_ego_rotation: ArrayLike, camera_to_ego_translation: ArrayLike, distortion: ArrayLike,
    *, profile: str, width: int = 384, height: int = 216,
) -> CalibratedView:
    """Rectify original OpenScene/nuPlan RGB once, then resize its matching pinhole calibration.

    The explicit profile excludes already-rectified images. OpenScene's misleading
    sensor2lidar fields copy camera-to-ego extrinsics from nuPlan, in metres;
    they are not camera-to-physical-LiDAR transforms. No capture timestamp or
    motion compensation is inferred from these calibration parameters.
    """
    if not isinstance(profile, str) or profile != OPENSCENE_NATIVE_PROFILE:
        raise ValueError("Unknown original OpenScene camera profile")
    if not isinstance(name, str) or name not in CAMERA_NAMES:
        raise ValueError("Camera name must be CAM_L0, CAM_F0 or CAM_R0")
    width = _output_dimension(width, "width", MAX_WIDTH)
    height = _output_dimension(height, "height", MAX_HEIGHT)
    _validate_rgb(rgb)
    native_height, native_width = rgb.shape[:2]
    if native_width * native_height > _MAX_NATIVE_PIXELS:
        raise ValueError("Native camera image exceeds 16 megapixels")
    rotation = _finite_array(camera_to_ego_rotation, (3, 3), "camera_to_ego_rotation")
    translation = _finite_array(camera_to_ego_translation, (3,), "camera_to_ego_translation")
    distortion = _finite_array(distortion, (5,), "distortion")
    transform = np.eye(4)
    transform[:3, :3] = rotation.T
    with np.errstate(over="ignore", invalid="ignore"):
        transform[:3, 3] = -rotation.T @ translation
    native_calibration = CameraCalibration(name, native_width, native_height, intrinsic, transform)
    if native_calibration.intrinsic[0, 1] != 0:
        raise ValueError("OpenScene native rectification requires zero-skew intrinsics")
    calibration = native_calibration.resized(width, height)

    import cv2
    from PIL import Image

    maps = cv2.initUndistortRectifyMap(
        native_calibration.intrinsic, distortion, None, native_calibration.intrinsic,
        (native_width, native_height), cv2.CV_32FC1,
    )
    rectified = cv2.remap(rgb, maps[0], maps[1], interpolation=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    resized = Image.fromarray(rectified).resize((width, height), resample=Image.Resampling.BILINEAR)
    return CalibratedView(CameraFrame(name, np.asarray(resized)), calibration)


def _output_dimension(value: int, name: str, maximum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or not 0 < value <= maximum:
        raise ValueError(f"{name} must be a positive integer no greater than {maximum}")
    return int(value)


def _finite_array(value: ArrayLike, shape: tuple[int, ...], name: str) -> NDArray[np.float64]:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain finite real numeric values") from exc
    if array.shape != shape or array.dtype.kind not in "fiu":
        raise ValueError(f"{name} must have shape {shape} and contain real numeric values")
    if any(isinstance(item, (bool, np.bool_)) for item in np.asarray(value, dtype=object).flat):
        raise ValueError(f"{name} must contain real numeric values, not booleans")
    with np.errstate(over="ignore", invalid="ignore"):
        result = array.astype(np.float64, copy=True)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain finite real numeric values")
    return result


def render_plan_overlay(view: CalibratedView, trajectory: Trajectory) -> CameraFrame:
    """Overlay planar rear-axle XY lifted to z=0; road elevation and roll/pitch are not reconstructed."""
    if not isinstance(view, CalibratedView):
        raise ValueError("view must be CalibratedView")
    segments = project_trajectory(trajectory, view.calibration)

    from PIL import Image, ImageDraw

    image = Image.fromarray(view.frame.rgb)
    draw = ImageDraw.Draw(image)
    for start, end in segments:
        draw.line((start, end), fill=(0, 200, 220), width=4)
    return CameraFrame(view.frame.name, np.asarray(image))
