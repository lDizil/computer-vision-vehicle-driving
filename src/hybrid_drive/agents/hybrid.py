import time
from uuid import uuid4

import numpy as np
from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import AgentInput, Trajectory as NavsimTrajectory

from hybrid_drive.adapters.navsim import from_navsim, to_navsim
from hybrid_drive.contracts.motion import EgoMotion
from hybrid_drive.contracts.planning import Candidate, CandidateSet
from hybrid_drive.contracts.trajectory import Pose2D
from hybrid_drive.safety.arbiter import ArbitrationDecision, SafetyArbiter
from hybrid_drive.safety.fallback import BrakingConfig
from hybrid_drive.safety.kinematics import KinematicLimits


class HybridDriveAgent(AbstractAgent):
    """Inference-only Fast → Safety composition for independent NAVSIM scenes.

    Each call uses a new scene-local origin and t=0, not a global timestamp.
    Slow is not connected: the arbiter explicitly records UNKNOWN/not_requested.
    Calls on one instance must be sequential; only the latest decision is kept.
    Backend/input errors propagate instead of returning a fabricated prediction.
    """

    def __init__(self, fast_agent: AbstractAgent, limits: KinematicLimits, braking: BrakingConfig):
        super().__init__()
        if not isinstance(fast_agent, AbstractAgent) or fast_agent.requires_scene:
            raise ValueError("Fast must be a NAVSIM agent that does not require a privileged Scene")
        # Upstream ConstantVelocityAgent skips nn.Module.__init__. Inference is
        # delegated, not registered as a trainable child (which would break eval()).
        object.__setattr__(self, "_fast_agent", fast_agent)
        self._arbiter = SafetyArbiter(limits, braking)
        self._last_decision: ArbitrationDecision | None = None

    @property
    def last_decision(self) -> ArbitrationDecision | None:
        return self._last_decision

    def name(self) -> str:
        return type(self).__name__

    def get_sensor_config(self):
        return self._fast_agent.get_sensor_config()

    def initialize(self) -> None:
        self._last_decision = None
        self._fast_agent.initialize()

    def compute_trajectory(self, agent_input: AgentInput) -> NavsimTrajectory:
        self._last_decision = None
        motion = _current_motion(agent_input)
        native = self._fast_agent.compute_trajectory(agent_input)
        if not isinstance(native, NavsimTrajectory):
            raise ValueError("Fast must return a native NAVSIM Trajectory")
        candidate = Candidate("fast", from_navsim(native))
        call_id = uuid4().hex
        plan = CandidateSet(
            plan_id="plan-" + call_id, snapshot_id="input-" + call_id,
            t_ref_s=0, reference_pose=Pose2D(0, 0, 0), candidates=(candidate,),
        )
        decision = self._arbiter.decide(plan, motion, now_monotonic_s=time.monotonic(), now_sim_s=0)
        result = to_navsim(decision.trajectory, native.trajectory_sampling)
        self._last_decision = decision
        return result


def _current_motion(agent_input: AgentInput) -> EgoMotion:
    if not isinstance(agent_input, AgentInput) or not agent_input.ego_statuses:
        raise ValueError("Expected AgentInput with current ego status")
    current = agent_input.ego_statuses[-1]
    if current.in_global_frame is not False:
        raise ValueError("Current ego status must use the local ego frame")
    pose = _vector(current.ego_pose, 3, "ego_pose")
    if not np.allclose(pose, 0, rtol=0, atol=1e-5):
        raise ValueError("Current ego pose must be the scene-local origin")
    velocity = _vector(current.ego_velocity, 2, "ego_velocity")
    acceleration = _vector(current.ego_acceleration, 2, "ego_acceleration")
    return EgoMotion(*velocity, *acceleration)


def _vector(value, size, name):
    array = np.asarray(value)
    if array.shape != (size,) or array.dtype.kind not in "iuf" or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain {size} finite real values")
    return array.astype(np.float64)
