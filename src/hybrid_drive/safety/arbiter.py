from dataclasses import dataclass
from enum import Enum

from hybrid_drive.contracts._validation import finite_real
from hybrid_drive.contracts.motion import EgoMotion
from hybrid_drive.contracts.planning import CandidateSet
from hybrid_drive.contracts.slow import RiskStatus, SlowRequest, SlowResult
from hybrid_drive.contracts.trajectory import Trajectory

from .fallback import BrakingConfig
from .kinematics import KinematicLimits
from .selection import TrajectoryDecision, select_trajectory


class SlowDisposition(str, Enum):
    NOT_REQUESTED = "not_requested"
    MISSING_RESULT = "missing_result"
    INVALID_RESULT = "invalid_result"
    PLAN_MISMATCH = "plan_mismatch"
    RESPONSE_MISMATCH = "response_mismatch"
    DEADLINE_EXCEEDED = "deadline_exceeded"
    SIM_EXPIRED = "sim_expired"
    INVALID_TIMING = "invalid_timing"
    UNKNOWN_RESULT = "unknown_result"
    APPLIED = "applied"


@dataclass(frozen=True)
class SlowAssessment:
    status: RiskStatus
    disposition: SlowDisposition
    request: SlowRequest | None
    applied_result: SlowResult | None
    response_age_s: float | None

    @property
    def used(self) -> bool:
        return self.disposition is SlowDisposition.APPLIED


@dataclass(frozen=True)
class ArbitrationDecision:
    selection: TrajectoryDecision
    slow: SlowAssessment
    decided_at_monotonic_s: float
    simulation_time_s: float

    @property
    def trajectory(self) -> Trajectory:
        return self.selection.trajectory

    @property
    def failed(self) -> bool:
        return self.selection.failed


@dataclass(frozen=True)
class SafetyArbiter:
    limits: KinematicLimits
    braking: BrakingConfig

    def __post_init__(self):
        if not isinstance(self.braking, BrakingConfig):
            raise ValueError("braking must be BrakingConfig")
        self.braking.validate_for(self.limits)

    def decide(
        self, plan: CandidateSet, motion: EgoMotion, *,
        now_monotonic_s: float, now_sim_s: float,
        request: SlowRequest | None = None, result: SlowResult | None = None,
        received_at_monotonic_s: float | None = None,
    ) -> ArbitrationDecision:
        """Apply only a matching, still-applicable reply; never wait for the model.

        Monotonic timestamps must come from one trusted clock domain. The
        worker/parser supplies typed results and trusted receive times; text
        explanations, model-supplied timestamps and queues do not drive control.
        The caller supplies the current fast plan; this method does not rebase
        its coordinates or future timestamps as simulation time advances.
        """
        if not isinstance(plan, CandidateSet) or not isinstance(motion, EgoMotion):
            raise ValueError("Expected CandidateSet and EgoMotion")
        if request is not None and not isinstance(request, SlowRequest):
            raise ValueError("request must be SlowRequest or None")
        now_monotonic_s = _monotonic_time(now_monotonic_s, "now_monotonic_s")
        now_sim_s = finite_real(now_sim_s, "now_sim_s")
        if now_sim_s < plan.t_ref_s:
            raise ValueError("Simulation time cannot precede the current plan")
        if received_at_monotonic_s is not None:
            received_at_monotonic_s = _monotonic_time(received_at_monotonic_s, "received_at_monotonic_s")
        slow = _assess_slow(plan, request, result, now_monotonic_s, now_sim_s, received_at_monotonic_s)
        vetoed = slow.applied_result.vetoed_candidate_ids if slow.used else ()
        selection = select_trajectory(plan, motion, self.limits, self.braking, vetoed)
        return ArbitrationDecision(selection, slow, now_monotonic_s, now_sim_s)


def _monotonic_time(value: float, name: str) -> float:
    value = finite_real(value, name)
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _assess_slow(plan, request, result, now_monotonic_s, now_sim_s, received_at_monotonic_s) -> SlowAssessment:
    if request is None:
        return _unknown(SlowDisposition.NOT_REQUESTED, None)
    expected_plan = request.plan
    if (
        expected_plan.plan_id != plan.plan_id
        or expected_plan.snapshot_id != plan.snapshot_id
        or expected_plan.candidates_hash != plan.candidates_hash
    ):
        return _unknown(SlowDisposition.PLAN_MISMATCH, request)
    if now_monotonic_s < request.submitted_at_monotonic_s:
        return _unknown(SlowDisposition.INVALID_TIMING, request)
    # Deadline bounds applicability at decision time, not only arrival time.
    if now_monotonic_s > request.deadline_monotonic_s:
        return _unknown(SlowDisposition.DEADLINE_EXCEEDED, request)
    if now_sim_s > request.valid_until_sim_s:
        return _unknown(SlowDisposition.SIM_EXPIRED, request)
    if result is None:
        return _unknown(SlowDisposition.MISSING_RESULT, request)
    if not isinstance(result, SlowResult):
        return _unknown(SlowDisposition.INVALID_RESULT, request)
    if (
        received_at_monotonic_s is None
        or received_at_monotonic_s < request.submitted_at_monotonic_s
        or received_at_monotonic_s > now_monotonic_s
    ):
        return _unknown(SlowDisposition.INVALID_TIMING, request)
    try:
        result.validate_for(request)
    except ValueError:
        return _unknown(SlowDisposition.RESPONSE_MISMATCH, request)
    if result.status is RiskStatus.UNKNOWN:
        return _unknown(SlowDisposition.UNKNOWN_RESULT, request)
    return SlowAssessment(
        result.status, SlowDisposition.APPLIED, request, result,
        now_monotonic_s - received_at_monotonic_s,
    )


def _unknown(disposition: SlowDisposition, request: SlowRequest | None) -> SlowAssessment:
    return SlowAssessment(RiskStatus.UNKNOWN, disposition, request, None, None)
