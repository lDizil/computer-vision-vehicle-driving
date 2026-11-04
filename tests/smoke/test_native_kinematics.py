import numpy as np
import pytest

from navsim.agents.constant_velocity_agent import ConstantVelocityAgent
from navsim.common.dataclasses import AgentInput, EgoStatus

from hybrid_drive.adapters.navsim import from_navsim
from hybrid_drive.contracts.motion import EgoMotion
from hybrid_drive.safety.kinematics import KinematicLimits, KinematicViolation, check_kinematics


@pytest.mark.parametrize("speed,bound,expected", [(0, 1, ()), (5, 5, ()), (5, 4, (KinematicViolation.SPEED,))])
def test_real_agent_output_has_expected_kinematic_report(speed, bound, expected):
    agent_input = AgentInput(
        ego_statuses=[EgoStatus(
            ego_pose=np.zeros(3),
            ego_velocity=np.array([speed, 0], dtype=np.float32),
            ego_acceleration=np.zeros(2),
            driving_command=np.array([1, 0, 0]),
        )],
        cameras=[],
        lidars=[],
    )
    agent = ConstantVelocityAgent()
    agent.initialize()
    trajectory = from_navsim(agent.compute_trajectory(agent_input))
    report = check_kinematics(trajectory, EgoMotion(speed, 0, 0, 0), KinematicLimits(0.5, bound, 3, 5))
    assert report.violations == expected
    assert report.within_limits is (not expected)
    assert report.max_speed_mps == speed
    assert report.max_acceleration_mps2 == 0
    assert report.max_jerk_mps3 == 0
