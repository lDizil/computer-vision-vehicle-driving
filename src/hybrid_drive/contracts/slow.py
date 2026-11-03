from dataclasses import dataclass
from enum import Enum

from ._validation import finite_real, identifier, tuple_snapshot
from .planning import CandidateSet


class RiskStatus(str, Enum):
    UNKNOWN = "UNKNOWN"
    NO_VETO = "NO_VETO"
    VETO = "VETO"


@dataclass(frozen=True)
class SlowRequest:
    """Monotonic process deadlines and simulation validity use separate clocks.

    The plan includes the actual trajectories, not only their IDs or hash.
    """

    request_id: str
    plan: CandidateSet
    submitted_at_monotonic_s: float
    deadline_monotonic_s: float
    valid_until_sim_s: float
    model_version: str

    def __post_init__(self):
        identifier(self.request_id, "request_id")
        identifier(self.model_version, "model_version")
        if not isinstance(self.plan, CandidateSet):
            raise ValueError("plan must be CandidateSet")
        for name in ("submitted_at_monotonic_s", "deadline_monotonic_s", "valid_until_sim_s"):
            object.__setattr__(self, name, finite_real(getattr(self, name), name))
        if self.submitted_at_monotonic_s < 0 or self.deadline_monotonic_s <= self.submitted_at_monotonic_s:
            raise ValueError("Monotonic deadline must be after a nonnegative submission time")
        if self.valid_until_sim_s < self.plan.t_ref_s:
            raise ValueError("Simulation validity cannot precede the plan reference time")


@dataclass(frozen=True)
class SlowResult:
    """Structured veto protocol; explanation text never determines the status.

    Correspondence validation does not check freshness or establish road safety.
    The runtime must supply trusted receive times to the future safety arbiter.
    """

    request_id: str
    snapshot_id: str
    plan_id: str
    candidates_hash: str
    model_version: str
    status: RiskStatus
    vetoed_candidate_ids: tuple[str, ...] = ()
    explanation: str | None = None

    def __post_init__(self):
        for name in ("request_id", "snapshot_id", "plan_id", "model_version"):
            identifier(getattr(self, name), name)
        digest = self.candidates_hash
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("candidates_hash must be a lowercase SHA256 hex digest")
        if not isinstance(self.status, RiskStatus):
            raise ValueError("status must be RiskStatus")
        vetoed = tuple_snapshot(self.vetoed_candidate_ids, "vetoed_candidate_ids")
        for value in vetoed:
            identifier(value, "vetoed candidate_id")
        if len(set(vetoed)) != len(vetoed):
            raise ValueError("Vetoed candidate IDs must be unique")
        if (self.status is RiskStatus.VETO) != bool(vetoed):
            raise ValueError("Only VETO results must name one or more vetoed candidates")
        if self.explanation is not None and not isinstance(self.explanation, str):
            raise ValueError("explanation must be a string or None")
        object.__setattr__(self, "vetoed_candidate_ids", vetoed)

    def validate_for(self, request: SlowRequest) -> None:
        if not isinstance(request, SlowRequest):
            raise ValueError("request must be SlowRequest")
        expected = {
            "request_id": request.request_id,
            "snapshot_id": request.plan.snapshot_id,
            "plan_id": request.plan.plan_id,
            "candidates_hash": request.plan.candidates_hash,
            "model_version": request.model_version,
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise ValueError(f"{name} does not match the request")
        known_ids = {value.candidate_id for value in request.plan.candidates}
        if any(value not in known_ids for value in self.vetoed_candidate_ids):
            raise ValueError("Result vetoes an unknown candidate_id")
