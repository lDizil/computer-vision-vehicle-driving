import numpy as np
import pytest

from navsim.common.dataclasses import Trajectory as NavsimTrajectory
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from hybrid_drive import Trajectory
from hybrid_drive.adapters.navsim import from_navsim, to_navsim


def test_navsim_round_trip_preserves_sampling_and_owns_data():
    native = NavsimTrajectory(
        np.array([[1, 0, 0], [2, 0, 0]], dtype=np.float32),
        TrajectorySampling(num_poses=2, interval_length=0.5),
    )
    internal = from_navsim(native)
    native.poses[0, 0] = 100
    np.testing.assert_array_equal(internal.poses[:, 0], [1, 2])
    result = to_navsim(internal)
    assert result.poses.dtype == np.float32
    np.testing.assert_array_equal(result.poses, [[1, 0, 0], [2, 0, 0]])
    assert result.trajectory_sampling.num_poses == 2
    assert result.trajectory_sampling.interval_length == 0.5
    result.poses[0, 0] = 200
    assert internal.poses[0, 0] == 1


def test_requires_explicit_resampling_for_native_sampling():
    value = Trajectory([[1, 0, 0], [2, 0, 0]], interval_s=0.5)
    target = TrajectorySampling(num_poses=4, interval_length=0.25)
    with pytest.raises(ValueError, match="resample"):
        to_navsim(value, target)
    actual = to_navsim(value.resample(0.25, 4), target)
    np.testing.assert_allclose(actual.poses[:, 0], [0.5, 1, 1.5, 2])


def test_rejects_mutated_native_sampling_metadata():
    value = NavsimTrajectory(np.zeros((2, 3)), TrajectorySampling(num_poses=2, interval_length=0.5))
    value.poses = np.zeros((3, 3))
    with pytest.raises(ValueError, match="metadata"):
        from_navsim(value)


def test_rejects_float32_overflow():
    value = Trajectory([[1e100, 0, 0]], interval_s=0.5)
    with pytest.raises(ValueError, match="float32"):
        to_navsim(value)
