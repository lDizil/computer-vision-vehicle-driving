import numpy as np
import pytest
from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.constant_velocity_agent import ConstantVelocityAgent
from navsim.common.dataclasses import AgentInput, EgoStatus, SensorConfig, Trajectory
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from hybrid_drive.agents.hybrid import HybridDriveAgent
from hybrid_drive.contracts.slow import RiskStatus
from hybrid_drive.safety.arbiter import SlowDisposition
from hybrid_drive.safety.fallback import BrakingConfig, FallbackFailure
from hybrid_drive.safety.kinematics import KinematicLimits, KinematicViolation
from hybrid_drive.safety.selection import DecisionStatus


def inputs(velocity=(2, 0), acceleration=(0, 0), **changes):
    fields = dict(ego_pose=np.zeros(3), ego_velocity=np.array(velocity),
                  ego_acceleration=np.array(acceleration), driving_command=np.array([1, 0, 0]))
    fields.update(changes)
    return AgentInput([EgoStatus(**fields)], [], [])


class ScriptedFast(AbstractAgent):
    def __init__(self, output):
        super().__init__()
        self.output = output
        self.calls = 0

    def name(self):
        return "ScriptedFast"

    def initialize(self):
        pass

    def get_sensor_config(self):
        return SensorConfig.build_no_sensors()

    def compute_trajectory(self, agent_input):
        self.calls += 1
        if isinstance(self.output, Exception):
            raise self.output
        return self.output


def agent(fast=None):
    return HybridDriveAgent(ConstantVelocityAgent() if fast is None else fast,
                            KinematicLimits(0.5, 10, 2, 4), BrakingConfig(2, 4))


def test_real_upstream_fast_passes_through_arbiter_and_native_boundary():
    hybrid = agent()
    hybrid.initialize()
    hybrid.eval()
    output = hybrid.compute_trajectory(inputs())
    decision = hybrid.last_decision
    assert isinstance(output, Trajectory)
    np.testing.assert_array_equal(output.poses, [[x, 0, 0] for x in range(1, 9)])
    assert output.trajectory_sampling == TrajectorySampling(time_horizon=4, interval_length=0.5)
    assert decision.selection.status is DecisionStatus.CANDIDATE
    assert decision.selection.candidate_id == "fast"
    assert decision.slow.status is RiskStatus.UNKNOWN
    assert decision.slow.disposition is SlowDisposition.NOT_REQUESTED
    assert decision.failed is False
    assert output.poses.dtype == np.float32
    output.poses[:] = 999
    np.testing.assert_array_equal(decision.trajectory.poses, [[x, 0, 0] for x in range(1, 9)])


def test_rejected_fast_really_changes_returned_path_to_braking():
    fast = ScriptedFast(Trajectory(np.array([[50 * t, 0, 0] for t in range(1, 9)])))
    hybrid = agent(fast)
    output = hybrid.compute_trajectory(inputs())
    decision = hybrid.last_decision
    assert fast.calls == 1
    assert decision.selection.status is DecisionStatus.FALLBACK
    assert KinematicViolation.SPEED in decision.selection.assessments[0].report.violations
    assert decision.failed is False
    assert output.poses[-1, 0] == pytest.approx(1.5)
    assert np.all(np.diff(output.poses[:, 0]) >= 0)
    assert np.all(output.poses[:, 1:] == 0)
    np.testing.assert_allclose(output.poses, decision.trajectory.poses)


def test_failed_fallback_is_not_a_success_just_because_native_shape_is_valid():
    hybrid = agent()
    output = hybrid.compute_trajectory(inputs(velocity=(2, 1)))
    assert isinstance(output, Trajectory)
    assert output.poses.shape == (8, 3)
    assert np.isfinite(output.poses).all()
    assert hybrid.last_decision.failed is True
    assert hybrid.last_decision.selection.status is DecisionStatus.FAILURE
    assert hybrid.last_decision.selection.fallback.failure_reason is FallbackFailure.UNSUPPORTED_MOTION


def test_calls_have_independent_snapshot_identity_and_only_one_retained_decision():
    hybrid = agent()
    hybrid.compute_trajectory(inputs())
    first = hybrid.last_decision
    hybrid.compute_trajectory(inputs(velocity=(3, 0)))
    second = hybrid.last_decision
    assert first.selection.plan.snapshot_id != second.selection.plan.snapshot_id
    assert first.selection.plan.plan_id != second.selection.plan.plan_id
    assert first.selection.plan.t_ref_s == second.selection.plan.t_ref_s == 0
    np.testing.assert_array_equal(first.trajectory.poses[:, 0], np.arange(1, 9))
    np.testing.assert_array_equal(second.trajectory.poses[:, 0], np.arange(1, 9) * 1.5)
    hybrid.initialize()
    assert hybrid.last_decision is None


@pytest.mark.parametrize("output", [RuntimeError("backend unavailable"), None, np.zeros((8, 3)),
                                    Trajectory(np.full((8, 3), np.nan)),
                                    Trajectory(np.zeros((8, 3)), TrajectorySampling(num_poses=8, interval_length=1))])
def test_bad_fast_output_propagates_and_clears_previous_success(output):
    fast = ScriptedFast(Trajectory(np.array([[t, 0, 0] for t in range(1, 9)])))
    hybrid = agent(fast)
    hybrid.compute_trajectory(inputs())
    assert hybrid.last_decision is not None
    fast.output = output
    with pytest.raises((ValueError, RuntimeError)):
        hybrid.compute_trajectory(inputs())
    assert hybrid.last_decision is None


@pytest.mark.parametrize("bad_input", [
    AgentInput([], [], []), None,
    inputs(in_global_frame=True), inputs(ego_pose=[1, 0, 0]),
    inputs(ego_velocity=[2]), inputs(ego_velocity=[True, False]), inputs(ego_velocity=["2", "0"]),
    inputs(ego_velocity=[np.inf, 0]), inputs(ego_acceleration=[0, np.nan]),
    inputs(ego_acceleration=[[0, 0]]),
])
def test_invalid_input_is_rejected_before_inference_without_stale_decision(bad_input):
    fast = ScriptedFast(Trajectory(np.array([[t, 0, 0] for t in range(1, 9)])))
    hybrid = agent(fast)
    hybrid.compute_trajectory(inputs())
    with pytest.raises(ValueError):
        hybrid.compute_trajectory(bad_input)
    assert fast.calls == 1
    assert hybrid.last_decision is None


def test_current_measured_acceleration_is_checked_not_replaced_with_zero():
    hybrid = agent()
    hybrid.compute_trajectory(inputs(acceleration=(0, 3)))
    decision = hybrid.last_decision
    assert KinematicViolation.ACCELERATION in decision.selection.assessments[0].report.violations
    assert decision.failed
    assert decision.selection.fallback.failure_reason is FallbackFailure.UNSUPPORTED_MOTION


def test_privileged_scene_backend_is_rejected():
    fast = ScriptedFast(None)
    fast.requires_scene = True
    with pytest.raises(ValueError, match="privileged"):
        agent(fast)
