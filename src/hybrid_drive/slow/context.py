from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json

import numpy as np
from numpy.typing import NDArray

from hybrid_drive.contracts.motion import EgoMotion
from hybrid_drive.contracts.slow import SlowRequest


CAMERA_NAMES = ("CAM_L0", "CAM_F0", "CAM_R0")
MAX_WIDTH = 640
MAX_HEIGHT = 480
MAX_CANDIDATES = 8
MAX_POINTS = 32
MAX_METADATA_BYTES = 32 * 1024


@dataclass(frozen=True, eq=False, init=False)
class CameraFrame:
    """Owned RGB pixels; bytes backing remains immutable after process spawn."""

    name: str
    _data: bytes = field(repr=False)
    width: int
    height: int

    def __init__(self, name: str, rgb: NDArray[np.uint8]):
        if name not in CAMERA_NAMES:
            raise ValueError("Camera name must be CAM_L0, CAM_F0 or CAM_R0")
        _validate_rgb(rgb)
        height, width, _ = rgb.shape
        if width > MAX_WIDTH or height > MAX_HEIGHT:
            raise ValueError("Camera frame exceeds 640x480; resize explicitly first")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "_data", rgb.tobytes())
        object.__setattr__(self, "width", width)
        object.__setattr__(self, "height", height)

    @property
    def rgb(self) -> NDArray[np.uint8]:
        return np.frombuffer(self._data, dtype=np.uint8).reshape(self.height, self.width, 3)


def _validate_rgb(rgb: NDArray[np.uint8]) -> None:
    if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8:
        raise ValueError("Camera image must be an RGB uint8 array")
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.shape[0] <= 0 or rgb.shape[1] <= 0:
        raise ValueError("Camera image must have nonempty HxWx3 shape")


@dataclass(frozen=True, eq=False)
class SlowContext:
    request: SlowRequest
    motion: EgoMotion
    cameras: tuple[CameraFrame, ...]
    content_hash: str = field(init=False)

    def __post_init__(self):
        if not isinstance(self.request, SlowRequest):
            raise ValueError("request must be SlowRequest")
        if not isinstance(self.motion, EgoMotion):
            raise ValueError("motion must be EgoMotion")
        if isinstance(self.cameras, (str, bytes)):
            raise ValueError("cameras must be ordered CameraFrame values")
        try:
            cameras = tuple(self.cameras)
        except TypeError as exc:
            raise ValueError("cameras must be ordered CameraFrame values") from exc
        if len(cameras) != 3 or any(not isinstance(frame, CameraFrame) for frame in cameras):
            raise ValueError("Exactly three CameraFrame values are required")
        if tuple(frame.name for frame in cameras) != CAMERA_NAMES:
            raise ValueError("Cameras must be ordered CAM_L0, CAM_F0, CAM_R0")
        candidates = self.request.plan.candidates
        if len(candidates) > MAX_CANDIDATES:
            raise ValueError("Slow context supports at most 8 candidates")
        if any(candidate.trajectory.num_poses > MAX_POINTS for candidate in candidates):
            raise ValueError("Slow context supports at most 32 points per candidate")
        object.__setattr__(self, "cameras", cameras)
        metadata = _json(_metadata(self))
        # _json escapes non-ASCII: this length is also its UTF-8 byte count.
        if len(metadata) > MAX_METADATA_BYTES:
            raise ValueError("Slow context metadata exceeds 32 KiB")
        digest = hashlib.sha256(metadata.encode("utf-8"))
        for frame in cameras:
            digest.update(frame._data)
        object.__setattr__(self, "content_hash", digest.hexdigest())


def _metadata(context: SlowContext) -> dict:
    request = context.request
    plan = request.plan
    return {
        "schema": "hybrid-drive-slow-context-v1",
        "request": {
            "request_id": request.request_id,
            "model_version": request.model_version,
            "submitted_at_monotonic_s": request.submitted_at_monotonic_s,
            "deadline_monotonic_s": request.deadline_monotonic_s,
            "valid_until_sim_s": request.valid_until_sim_s,
            "plan": {
                "plan_id": plan.plan_id,
                "snapshot_id": plan.snapshot_id,
                "candidates_hash": plan.candidates_hash,
                "t_ref_s": plan.t_ref_s,
                "reference_pose": [plan.reference_pose.x, plan.reference_pose.y, plan.reference_pose.heading],
                "candidates": [
                    {
                        "candidate_id": candidate.candidate_id,
                        "interval_s": candidate.trajectory.interval_s,
                        "times_s": candidate.trajectory.times_s.tolist(),
                        "poses": candidate.trajectory.poses.tolist(),
                    }
                    for candidate in plan.candidates
                ],
            },
        },
        "motion": {name: getattr(context.motion, name) for name in ("vx_mps", "vy_mps", "ax_mps2", "ay_mps2")},
        "cameras": [{"name": frame.name, "width": frame.width, "height": frame.height} for frame in context.cameras],
    }


def _json(value: dict) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
