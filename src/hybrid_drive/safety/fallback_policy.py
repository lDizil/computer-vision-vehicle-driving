"""Fallback composition and snapshot binding; generators own their model rules."""

from dataclasses import dataclass
from enum import Enum

from hybrid_drive.contracts._validation import finite_real, identifier
from hybrid_drive.contracts.motion import EgoMotion
from hybrid_drive.contracts.planning import CandidateSet
from .fallback import BrakingConfig, FallbackFailure, FallbackResult, braking_fallback, unproven_fallback
from .kinematics import KinematicLimits


class FallbackModel(str, Enum):
    LONGITUDINAL = "longitudinal"
    CURVED = "curved_snapshot_hypothesis"


@dataclass(frozen=True)
class FallbackObservation:
    """Associated heading-rate input, not a synchronized sensor guarantee.

    A composition root supplies this separately for every current snapshot.
    The source ID identifies its immutable observation, not a model confidence.
    """

    snapshot_id: str
    t_ref_s: float
    motion: EgoMotion
    heading_rate_radps: float
    source_id: str

    def __post_init__(self):
        identifier(self.snapshot_id, "snapshot_id")
        identifier(self.source_id, "source_id")
        if not isinstance(self.motion, EgoMotion):
            raise ValueError("motion must be EgoMotion")
        for name in ("t_ref_s", "heading_rate_radps"):
            object.__setattr__(self, name, finite_real(getattr(self, name), name))

    def matches(self, plan: CandidateSet, motion: EgoMotion) -> bool:
        return (self.snapshot_id == plan.snapshot_id and self.t_ref_s == plan.t_ref_s
                and self.motion == motion)


def plan_fallback(
    model: FallbackModel, plan: CandidateSet, motion: EgoMotion,
    limits: KinematicLimits, config: BrakingConfig, observation: FallbackObservation | None,
) -> FallbackResult:
    if not isinstance(model, FallbackModel):
        raise ValueError("model must be FallbackModel")
    count = plan.candidates[0].trajectory.num_poses
    if model is FallbackModel.LONGITUDINAL:
        return braking_fallback(motion, limits, config, count)
    if observation is None:
        return unproven_fallback(motion, limits, count, FallbackFailure.MISSING_HEADING_RATE)
    if not isinstance(observation, FallbackObservation):
        raise ValueError("observation must be FallbackObservation or None")
    if not observation.matches(plan, motion):
        return unproven_fallback(motion, limits, count, FallbackFailure.OBSERVATION_MISMATCH)
    from .curved_braking import curved_braking_fallback

    return curved_braking_fallback(motion, observation.heading_rate_radps, limits, config, count)
