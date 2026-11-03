from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from hybrid_drive.contracts.trajectory import Trajectory

if TYPE_CHECKING:
    from navsim.common.dataclasses import Trajectory as NavsimTrajectory
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling


def from_navsim(value: NavsimTrajectory) -> Trajectory:
    sampling = value.trajectory_sampling
    trajectory = Trajectory(value.poses, sampling.interval_length)
    if trajectory.num_poses != sampling.num_poses:
        raise ValueError("NAVSIM poses do not match their sampling metadata")
    return trajectory


def to_navsim(value: Trajectory, sampling: TrajectorySampling | None = None) -> NavsimTrajectory:
    """Convert without implicit resampling, extrapolation, or coordinate changes."""
    from navsim.common.dataclasses import Trajectory as NavsimTrajectory
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

    if sampling is None:
        sampling = TrajectorySampling(num_poses=value.num_poses, interval_length=value.interval_s)
    if value.num_poses != sampling.num_poses or not math.isclose(
        value.interval_s, sampling.interval_length, rel_tol=1e-12, abs_tol=0.0
    ):
        raise ValueError("Sampling differs; resample the trajectory explicitly first")
    with np.errstate(over="ignore"):
        poses = value.poses.astype(np.float32)
    if not np.isfinite(poses).all():
        raise ValueError("Coordinates exceed the NAVSIM float32 range")
    return NavsimTrajectory(poses=poses, trajectory_sampling=sampling)
