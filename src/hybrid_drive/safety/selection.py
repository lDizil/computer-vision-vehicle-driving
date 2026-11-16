from dataclasses import dataclass, replace
from enum import Enum

from hybrid_drive.contracts._validation import identifier, tuple_snapshot
from hybrid_drive.contracts.motion import EgoMotion
from hybrid_drive.contracts.planning import CandidateSet
from hybrid_drive.contracts.trajectory import Trajectory

from ._limits import same_interval
from .fallback import BrakingConfig, FallbackResult
from .fallback_policy import FallbackModel, FallbackObservation, plan_fallback
from .kinematics import KinematicLimits, KinematicReport, check_kinematics


class DecisionStatus(str, Enum):
    CANDIDATE = "candidate"
    FALLBACK = "fallback"
    FAILURE = "failure"


@dataclass(frozen=True)
class CandidateAssessment:
    candidate_id: str
    report: KinematicReport
    vetoed: bool


@dataclass(frozen=True)
class TrajectoryDecision:
    status: DecisionStatus
    trajectory: Trajectory
    candidate_id: str | None
    assessments: tuple[CandidateAssessment, ...]
    fallback: FallbackResult | None
    plan: CandidateSet

    @property
    def failed(self) -> bool:
        return self.status is DecisionStatus.FAILURE


def select_trajectory(
    plan: CandidateSet, motion: EgoMotion, limits: KinematicLimits, config: BrakingConfig,
    vetoed_candidate_ids=(), *, fallback_model: FallbackModel = FallbackModel.LONGITUDINAL,
    fallback_observation: FallbackObservation | None = None,
) -> TrajectoryDecision:
    """Resolve already-authorized veto IDs; freshness and model parsing live upstream.

    Plan order is preference order. Even failure carries a shape-valid
    trajectory and an explicit status; callers must retain that status in results.
    """
    if not isinstance(plan, CandidateSet) or not isinstance(motion, EgoMotion):
        raise ValueError("Expected CandidateSet and EgoMotion")
    if not isinstance(config, BrakingConfig):
        raise ValueError("config must be BrakingConfig")
    config.validate_for(limits)
    if not isinstance(fallback_model, FallbackModel):
        raise ValueError("fallback_model must be FallbackModel")
    if fallback_observation is not None and not isinstance(fallback_observation, FallbackObservation):
        raise ValueError("fallback_observation must be FallbackObservation or None")
    sampling = plan.candidates[0].trajectory
    if not same_interval(sampling.interval_s, limits.interval_s):
        raise ValueError("Plan sampling differs from the configured limits")
    vetoed = tuple_snapshot(vetoed_candidate_ids, "vetoed_candidate_ids")
    for value in vetoed:
        identifier(value, "vetoed candidate_id")
    known = {value.candidate_id for value in plan.candidates}
    if len(set(vetoed)) != len(vetoed) or not set(vetoed) <= known:
        raise ValueError("Veto IDs must be unique and belong to this plan")
    assessments = tuple(
        CandidateAssessment(value.candidate_id, check_kinematics(value.trajectory, motion, limits), value.candidate_id in vetoed)
        for value in plan.candidates
    )
    for candidate, assessment in zip(plan.candidates, assessments):
        if not assessment.vetoed and assessment.report.within_limits:
            return TrajectoryDecision(DecisionStatus.CANDIDATE, candidate.trajectory, candidate.candidate_id, assessments, None, plan)
    fallback_limits = replace(limits, interval_s=sampling.interval_s)
    fallback = plan_fallback(fallback_model, plan, motion, fallback_limits, config, fallback_observation)
    status = DecisionStatus.FALLBACK if fallback.succeeded else DecisionStatus.FAILURE
    return TrajectoryDecision(status, fallback.trajectory, None, assessments, fallback, plan)
