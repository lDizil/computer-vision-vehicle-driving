from dataclasses import replace
import json

import numpy as np

from navsim.agents.constant_velocity_agent import ConstantVelocityAgent
from navsim.common.dataclasses import AgentInput, EgoStatus

from hybrid_drive import Pose2D
from hybrid_drive.adapters.navsim import from_navsim, to_navsim
from hybrid_drive.contracts.motion import EgoMotion
from hybrid_drive.contracts.planning import Candidate, CandidateSet
from hybrid_drive.contracts.slow import RiskStatus, SlowRequest, SlowResult
from hybrid_drive.safety.arbiter import SafetyArbiter, SlowDisposition
from hybrid_drive.safety.fallback import BrakingConfig
from hybrid_drive.safety.kinematics import KinematicLimits
from hybrid_drive.safety.selection import DecisionStatus


def test_real_agent_arbiter_and_native_output_keep_all_synthetic_outcomes():
    agent_input = AgentInput(
        ego_statuses=[EgoStatus(
            ego_pose=np.zeros(3), ego_velocity=np.array([2, 0], dtype=np.float32),
            ego_acceleration=np.zeros(2), driving_command=np.array([1, 0, 0]),
        )], cameras=[], lidars=[],
    )
    agent = ConstantVelocityAgent()
    agent.initialize()
    path = from_navsim(agent.compute_trajectory(agent_input))
    plan = CandidateSet("plan", "snapshot", 100, Pose2D(0, 0, 0), (Candidate("fast", path),))
    request = SlowRequest("request", plan, 10, 11, 100.5, "fixture-v1")
    veto = SlowResult("request", "snapshot", "plan", plan.candidates_hash, "fixture-v1", RiskStatus.VETO, ("fast",))
    no_veto = replace(veto, status=RiskStatus.NO_VETO, vetoed_candidate_ids=())
    arbiter = SafetyArbiter(KinematicLimits(0.5, 10, 3, 3), BrakingConfig(2, 2))
    records = []
    for active_request, result, received, now, sim, motion, disposition, status in (
        (None, None, None, 10.5, 100, EgoMotion(2, 0, 0, 0), SlowDisposition.NOT_REQUESTED, DecisionStatus.CANDIDATE),
        (request, None, None, 10.5, 100, EgoMotion(2, 0, 0, 0), SlowDisposition.MISSING_RESULT, DecisionStatus.CANDIDATE),
        (request, no_veto, 10.25, 10.5, 100, EgoMotion(2, 0, 0, 0), SlowDisposition.APPLIED, DecisionStatus.CANDIDATE),
        (request, veto, 10.25, 10.5, 100, EgoMotion(2, 0, 0, 0), SlowDisposition.APPLIED, DecisionStatus.FALLBACK),
        (request, veto, 10.25, 11.01, 100, EgoMotion(2, 0, 0, 0), SlowDisposition.DEADLINE_EXCEEDED, DecisionStatus.CANDIDATE),
        (request, veto, 10.25, 10.5, 100.5001, EgoMotion(2, 0, 0, 0), SlowDisposition.SIM_EXPIRED, DecisionStatus.CANDIDATE),
        (request, replace(veto, plan_id="old-plan"), 10.25, 10.5, 100, EgoMotion(2, 0, 0, 0), SlowDisposition.RESPONSE_MISMATCH, DecisionStatus.CANDIDATE),
        (request, veto, 10.25, 10.5, 100, EgoMotion(2, 1, 0, 0), SlowDisposition.APPLIED, DecisionStatus.FAILURE),
    ):
        decision = arbiter.decide(
            plan, motion, now_monotonic_s=now, now_sim_s=sim,
            request=active_request, result=result, received_at_monotonic_s=received,
        )
        native = to_navsim(decision.trajectory)
        assert native.poses.shape == (8, 3)
        assert np.isfinite(native.poses).all()
        assert decision.slow.disposition is disposition
        assert decision.selection.status is status
        assert decision.failed is (status is DecisionStatus.FAILURE)
        records.append({"slow": disposition.value, "selection": status.value, "failed": decision.failed})
    assert len(records) == 8 and sum(value["failed"] for value in records) == 1
    print(json.dumps({"synthetic_arbiter_accounting": records}))
