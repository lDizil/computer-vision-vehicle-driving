import json

import numpy as np

from navsim.agents.constant_velocity_agent import ConstantVelocityAgent
from navsim.common.dataclasses import AgentInput, EgoStatus

from hybrid_drive import Pose2D
from hybrid_drive.adapters.navsim import from_navsim, to_navsim
from hybrid_drive.contracts.motion import EgoMotion
from hybrid_drive.contracts.planning import Candidate, CandidateSet
from hybrid_drive.safety.fallback import BrakingConfig
from hybrid_drive.safety.kinematics import KinematicLimits
from hybrid_drive.safety.selection import DecisionStatus, select_trajectory


def test_all_decision_outcomes_keep_a_native_trajectory_and_failure_accounting():
    agent_input = AgentInput(
        ego_statuses=[EgoStatus(
            ego_pose=np.zeros(3), ego_velocity=np.array([2, 0], dtype=np.float32),
            ego_acceleration=np.zeros(2), driving_command=np.array([1, 0, 0]),
        )],
        cameras=[], lidars=[],
    )
    agent = ConstantVelocityAgent()
    agent.initialize()
    trajectory = from_navsim(agent.compute_trajectory(agent_input))
    plan = CandidateSet("plan", "snapshot", 0, Pose2D(0, 0, 0), (Candidate("fast", trajectory),))
    records = []
    for motion, vetoed, status in (
        (EgoMotion(2, 0, 0, 0), (), DecisionStatus.CANDIDATE),
        (EgoMotion(2, 0, 0, 0), ("fast",), DecisionStatus.FALLBACK),
        (EgoMotion(2, 1, 0, 0), ("fast",), DecisionStatus.FAILURE),
    ):
        decision = select_trajectory(plan, motion, KinematicLimits(0.5, 10, 3, 3), BrakingConfig(2, 2), vetoed)
        native = to_navsim(decision.trajectory)
        assert decision.status is status
        assert decision.failed is (status is DecisionStatus.FAILURE)
        assert native.poses.shape == (8, 3)
        assert native.trajectory_sampling.interval_length == 0.5
        assert np.isfinite(native.poses).all()
        records.append({"status": decision.status.value, "failed": decision.failed})
    assert len(records) == 3
    assert sum(value["failed"] for value in records) == 1
    print(json.dumps({"synthetic_decision_accounting": records}))
