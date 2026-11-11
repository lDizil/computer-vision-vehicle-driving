from dataclasses import dataclass
import math

import numpy as np
from PIL import Image, ImageDraw

from hybrid_drive.contracts.planning import CandidateSet
from hybrid_drive.geometry.camera import CameraCalibration, project_segments, project_trajectory
from hybrid_drive.slow.context import CAMERA_NAMES, CameraFrame


WIDTH, HEIGHT = 384, 216
CANDIDATE_COLORS = ((0, 200, 220), (220, 0, 180))


@dataclass(frozen=True)
class Box:
    x: float
    y: float
    length: float = 1.3
    width: float = 1.1
    height: float = 1.4

    def __post_init__(self):
        for name in ("x", "y", "length", "width", "height"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("Box coordinates and dimensions must be finite real numbers")
        if min(self.length, self.width, self.height) <= 0:
            raise ValueError("Box dimensions must be positive")


def schematic_calibrations() -> tuple[CameraCalibration, ...]:
    """Explicit synthetic pinhole rig, not NAVSIM or real-street calibration."""
    intrinsic = np.array([[220., 0., 191.5], [0., 220., 107.5], [0., 0., 1.]])
    values = []
    for name, yaw in zip(CAMERA_NAMES, (25., 0., -25.)):
        c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
        rotation = np.array([[s, -c, 0.], [0., 0., -1.], [c, s, 0.]])
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = rotation
        extrinsic[:3, 3] = -rotation @ np.array([0., 0., 1.6])
        values.append(CameraCalibration(name, WIDTH, HEIGHT, intrinsic, extrinsic))
    return tuple(values)


def render_cameras(plan: CandidateSet, boxes: tuple[Box, ...], *, blank: bool = False):
    if len(plan.candidates) != 2:
        raise ValueError("The controlled renderer requires exactly two candidates")
    raw, overlaid = [], []
    for calibration in schematic_calibrations():
        image = _scene(calibration, boxes, blank)
        raw.append(CameraFrame(calibration.name, np.asarray(image)))
        image = image.copy()
        draw = ImageDraw.Draw(image)
        for candidate, color in zip(plan.candidates, CANDIDATE_COLORS):
            for segment in project_trajectory(candidate.trajectory, calibration):
                draw.line(segment, fill=color, width=4)
        overlaid.append(CameraFrame(calibration.name, np.asarray(image)))
    return tuple(raw), tuple(overlaid)


def _scene(calibration: CameraCalibration, boxes: tuple[Box, ...], blank: bool) -> Image.Image:
    image = Image.new("RGB", (calibration.width, calibration.height), (0, 0, 0))
    if blank:
        return image
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, WIDTH - 1, 107), fill=(149, 173, 185))
    draw.rectangle((0, 108, WIDTH - 1, HEIGHT - 1), fill=(65, 70, 72))
    for x in range(2, 25, 2):
        _line(draw, ((x, -8, 0), (x, 8, 0)), calibration, (92, 96, 96))
    for y in range(-8, 9):
        _line(draw, ((1, y, 0), (25, y, 0)), calibration, (92, 96, 96))
    faces = []
    for box in boxes:
        vertices = np.array([
            (box.x + dx * box.length / 2, box.y + dy * box.width / 2, z * box.height)
            for z in (0, 1) for dy in (-1, 1) for dx in (-1, 1)
        ])
        for indices, color in (
            ((0, 1, 3, 2), (143, 127, 100)), ((4, 5, 7, 6), (197, 177, 140)),
            ((0, 1, 5, 4), (168, 148, 114)), ((2, 3, 7, 6), (168, 148, 114)),
            ((0, 2, 6, 4), (181, 159, 122)), ((1, 3, 7, 5), (181, 159, 122)),
        ):
            points = vertices[list(indices)]
            depth = (calibration.ego_to_camera @ np.append(points.mean(axis=0), 1.))[2]
            faces.append((depth, points, color))
    for _, points, color in sorted(faces, key=lambda face: -face[0]):
        segments = project_segments(np.vstack((points, points[0])), calibration)
        if segments:
            polygon = [point for segment in segments for point in segment]
            draw.polygon(polygon, fill=color, outline=(44, 42, 36))
    return image


def _line(draw, points, calibration, color):
    for segment in project_segments(points, calibration):
        draw.line(segment, fill=color, width=1)
