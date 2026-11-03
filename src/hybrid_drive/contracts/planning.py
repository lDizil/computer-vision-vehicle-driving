from dataclasses import dataclass, field
import hashlib
import json

from ._validation import finite_real, identifier, tuple_snapshot
from .trajectory import Pose2D, Trajectory


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    trajectory: Trajectory

    def __post_init__(self):
        identifier(self.candidate_id, "candidate_id")
        if not isinstance(self.trajectory, Trajectory):
            raise ValueError("trajectory must be Trajectory")


@dataclass(frozen=True, eq=False)
class CandidateSet:
    """One immutable plan in a shared rear-axle ego frame at simulation t_ref_s."""

    plan_id: str
    snapshot_id: str
    t_ref_s: float
    reference_pose: Pose2D
    candidates: tuple[Candidate, ...]
    candidates_hash: str = field(init=False)

    def __post_init__(self):
        identifier(self.plan_id, "plan_id")
        identifier(self.snapshot_id, "snapshot_id")
        object.__setattr__(self, "t_ref_s", finite_real(self.t_ref_s, "t_ref_s"))
        if not isinstance(self.reference_pose, Pose2D):
            raise ValueError("reference_pose must be Pose2D")
        candidates = tuple_snapshot(self.candidates, "candidates")
        if not candidates or any(not isinstance(value, Candidate) for value in candidates):
            raise ValueError("candidates must be a nonempty sequence of Candidate")
        if len({value.candidate_id for value in candidates}) != len(candidates):
            raise ValueError("candidate_id must be unique within a plan")
        sampling = candidates[0].trajectory
        if any(
            value.trajectory.num_poses != sampling.num_poses
            or value.trajectory.interval_s != sampling.interval_s
            for value in candidates
        ):
            raise ValueError("Candidates must share a time grid; resample explicitly first")
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "candidates_hash", _content_hash(self))


def _content_hash(plan: CandidateSet) -> str:
    payload = {
        "schema": "hybrid-drive-candidates-v1",
        "plan_id": plan.plan_id,
        "snapshot_id": plan.snapshot_id,
        "t_ref_s": plan.t_ref_s,
        "reference_pose": [plan.reference_pose.x, plan.reference_pose.y, plan.reference_pose.heading],
        "candidates": [
            {
                "candidate_id": value.candidate_id,
                "interval_s": value.trajectory.interval_s,
                "poses": value.trajectory.poses.tolist(),
            }
            for value in plan.candidates
        ],
    }
    content = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()
