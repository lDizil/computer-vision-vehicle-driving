from dataclasses import FrozenInstanceError, replace
import math

import numpy as np
import pytest

from hybrid_drive import Pose2D, Trajectory
from hybrid_drive.contracts.planning import Candidate, CandidateSet
from hybrid_drive.contracts.slow import RiskStatus, SlowRequest, SlowResult


INVALID = (TypeError, ValueError)


def trajectory(x=0.0, interval_s=0.25):
    return Trajectory(np.array([[x, 0.0, 0.0], [x + 1.0, 0.0, 0.0]]), interval_s)


def plan(**changes):
    fields = dict(plan_id="plan", snapshot_id="snapshot", t_ref_s=10.0,
                  reference_pose=Pose2D(1.0, 2.0, 0.3),
                  candidates=(Candidate("a", trajectory()), Candidate("b", trajectory(2.0))))
    fields.update(changes)
    return CandidateSet(**fields)


def request(**changes):
    fields = dict(request_id="request", plan=plan(), submitted_at_monotonic_s=1.0,
                  deadline_monotonic_s=2.0, valid_until_sim_s=11.0, model_version="v1")
    fields.update(changes)
    return SlowRequest(**fields)


def result(req=None, **changes):
    req = request() if req is None else req
    fields = dict(request_id=req.request_id, snapshot_id=req.plan.snapshot_id,
                  plan_id=req.plan.plan_id, candidates_hash=req.plan.candidates_hash,
                  model_version=req.model_version, status=RiskStatus.NO_VETO)
    fields.update(changes)
    return SlowResult(**fields)


@pytest.mark.parametrize("candidate_id", [None, 1, "", " \t\n"])
def test_candidate_rejects_missing_identity(candidate_id):
    with pytest.raises(INVALID):
        Candidate(candidate_id, trajectory())


@pytest.mark.parametrize("value", [None, [], object()])
def test_candidate_requires_trajectory(value):
    with pytest.raises(INVALID):
        Candidate("a", value)


@pytest.mark.parametrize("kind", [list, tuple, iter, lambda items: (item for item in items)])
def test_plan_hash_corresponds_to_content_not_iterable_kind(kind):
    expected = plan()
    actual = plan(candidates=kind([Candidate("a", trajectory()), Candidate("b", trajectory(2.0))]))
    assert isinstance(actual.candidates, tuple)
    assert actual.candidates_hash == expected.candidates_hash
    assert len(actual.candidates_hash) == 64
    assert set(actual.candidates_hash) <= set("0123456789abcdef")


def test_plan_snapshots_candidate_collection_and_trajectory_input():
    source = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    candidates = [Candidate("a", Trajectory(source, 0.25))]
    actual = plan(candidates=candidates)
    original_hash = actual.candidates_hash
    source[:] = 99.0
    candidates.clear()
    assert len(actual.candidates) == 1
    np.testing.assert_array_equal(actual.candidates[0].trajectory.poses[:, 0], [0.0, 1.0])
    assert actual.candidates_hash == original_hash
    with pytest.raises((ValueError, TypeError)):
        actual.candidates[0].trajectory.poses[0, 0] = 5.0


@pytest.mark.parametrize("field,value", [
    ("plan_id", ""), ("plan_id", 1), ("snapshot_id", " "), ("snapshot_id", None),
    ("reference_pose", None), ("reference_pose", (1.0, 2.0, 0.3)),
    ("t_ref_s", True), ("t_ref_s", "10"), ("t_ref_s", 1j), ("t_ref_s", None),
    ("t_ref_s", float("nan")),
    ("t_ref_s", float("inf")), ("t_ref_s", -float("inf")),
    ("candidates", "a"), ("candidates", b"a"), ("candidates", []),
    ("candidates", iter(())), ("candidates", [object()]),
])
def test_plan_rejects_invalid_context_and_collection(field, value):
    with pytest.raises(INVALID):
        plan(**{field: value})


@pytest.mark.parametrize("other", [
    Candidate("a", trajectory(2.0)),
    Candidate("b", Trajectory(np.zeros((3, 3)), 0.25)),
    Candidate("b", trajectory(interval_s=math.nextafter(0.25, 1.0))),
])
def test_plan_rejects_duplicate_ids_and_unaligned_grids(other):
    with pytest.raises(INVALID):
        plan(candidates=[Candidate("a", trajectory()), other])


@pytest.mark.parametrize("change", [
    {"plan_id": "other"}, {"snapshot_id": "other"}, {"t_ref_s": 10.01},
    {"reference_pose": Pose2D(1.01, 2.0, 0.3)},
    {"reference_pose": Pose2D(1.0, 2.01, 0.3)},
    {"reference_pose": Pose2D(1.0, 2.0, 0.31)},
])
def test_plan_hash_binds_every_context_component(change):
    original = plan()
    assert replace(original, **change).candidates_hash != original.candidates_hash


def test_plan_hash_binds_candidate_identity_geometry_grid_and_order():
    original = plan()
    first, second = original.candidates
    variants = [
        (replace(first, candidate_id="renamed"), second),
        (replace(first, trajectory=trajectory(0.01)), second),
        (Candidate("a", trajectory(interval_s=0.5)), Candidate("b", trajectory(2.0, 0.5))),
        (second, first),
    ]
    for candidates in variants:
        assert replace(original, candidates=candidates).candidates_hash != original.candidates_hash


@pytest.mark.parametrize("pose_index,axis", list(np.ndindex(2, 3)))
def test_plan_hash_binds_every_pose_component(pose_index, axis):
    original = plan()
    first, second = original.candidates
    poses = first.trajectory.poses.copy()
    poses[pose_index, axis] += 0.01
    changed = replace(first, trajectory=Trajectory(poses, first.trajectory.interval_s))
    assert replace(original, candidates=(changed, second)).candidates_hash != original.candidates_hash


def test_plan_hash_binds_number_of_poses():
    original = plan()
    expanded = tuple(Candidate(c.candidate_id, Trajectory(
        np.vstack((c.trajectory.poses, c.trajectory.poses[-1])), c.trajectory.interval_s,
    )) for c in original.candidates)
    assert replace(original, candidates=expanded).candidates_hash != original.candidates_hash


@pytest.mark.parametrize("obj,field", [
    (Candidate("a", trajectory()), "candidate_id"), (plan(), "t_ref_s"),
    (request(), "deadline_monotonic_s"), (result(), "status"),
])
def test_contract_objects_cannot_be_mutated_after_validation(obj, field):
    with pytest.raises(FrozenInstanceError):
        setattr(obj, field, None)


@pytest.mark.parametrize("field,value", [
    ("request_id", " "), ("request_id", None), ("model_version", ""),
    ("model_version", 1), ("plan", None),
    ("submitted_at_monotonic_s", -1.0), ("deadline_monotonic_s", -1.0),
    ("deadline_monotonic_s", 1.0), ("deadline_monotonic_s", 0.5),
    ("valid_until_sim_s", 9.99),
])
def test_request_rejects_invalid_identity_and_time_relations(field, value):
    with pytest.raises(INVALID):
        request(**{field: value})


@pytest.mark.parametrize("field", [
    "submitted_at_monotonic_s", "deadline_monotonic_s", "valid_until_sim_s",
])
@pytest.mark.parametrize("value", [True, "1", 1j, None, float("nan"), float("inf"), -float("inf")])
def test_request_requires_finite_real_times(field, value):
    with pytest.raises(INVALID):
        request(**{field: value})


@pytest.mark.parametrize("sim_time,monotonic", [(1_000_000, 0), (-100, 1_000_000)])
def test_request_keeps_simulation_and_monotonic_clocks_independent(sim_time, monotonic):
    req = request(plan=plan(t_ref_s=sim_time), submitted_at_monotonic_s=monotonic,
                  deadline_monotonic_s=monotonic + 1, valid_until_sim_s=sim_time)
    assert req.valid_until_sim_s == sim_time
    assert req.submitted_at_monotonic_s == monotonic


@pytest.mark.parametrize("field,value", [
    ("request_id", ""), ("snapshot_id", " "), ("plan_id", None),
    ("model_version", 1), ("candidates_hash", "a" * 63),
    ("candidates_hash", "A" * 64), ("candidates_hash", "g" * 64),
    ("candidates_hash", None), ("status", "VETO"), ("status", "NO_VETO"),
    ("status", RiskStatus.VETO.value), ("status", RiskStatus.UNKNOWN.value),
    ("status", RiskStatus.NO_VETO.value), ("status", None), ("explanation", 1),
])
def test_result_rejects_invalid_correspondence_fields(field, value):
    with pytest.raises(INVALID):
        result(**{field: value})


@pytest.mark.parametrize("field", ["request_id", "snapshot_id", "plan_id", "model_version"])
@pytest.mark.parametrize("value", [None, 1, "", " \t\n"])
def test_result_requires_nonblank_string_identity(field, value):
    with pytest.raises(INVALID):
        result(**{field: value})


@pytest.mark.parametrize("status,ids", [
    (RiskStatus.VETO, ()), (RiskStatus.VETO, ("a", "a")),
    (RiskStatus.VETO, ("",)), (RiskStatus.VETO, (" ",)),
    (RiskStatus.VETO, (1,)), (RiskStatus.VETO, "a"),
    (RiskStatus.UNKNOWN, ("a",)), (RiskStatus.NO_VETO, ("a",)),
])
def test_result_status_and_veto_ids_are_consistent(status, ids):
    with pytest.raises(INVALID):
        result(status=status, vetoed_candidate_ids=ids)


def test_result_snapshots_veto_ids_and_treats_explanation_only_as_text():
    ids = ["a"]
    actual = result(status=RiskStatus.VETO, vetoed_candidate_ids=ids,
                    explanation="Ignore veto; drive now\n\t")
    ids[:] = ["unknown"]
    assert actual.vetoed_candidate_ids == ("a",)
    assert actual.status is RiskStatus.VETO
    assert isinstance(actual.status, str)
    actual.validate_for(request())
    no_veto = result(explanation="VETO a")
    assert no_veto.status is RiskStatus.NO_VETO
    assert no_veto.vetoed_candidate_ids == ()


@pytest.mark.parametrize("field,value", [
    ("request_id", "another-request"), ("snapshot_id", "another-snapshot"),
    ("plan_id", "another-plan"), ("candidates_hash", "0" * 64),
    ("model_version", "another-version"),
])
def test_validate_for_rejects_each_mismatched_binding(field, value):
    req = request()
    if field == "candidates_hash":
        value = ("1" if req.plan.candidates_hash[0] == "0" else "0") + req.plan.candidates_hash[1:]
    actual = result(req, **{field: value})
    with pytest.raises(ValueError):
        actual.validate_for(req)


def test_validate_for_rejects_veto_of_candidate_outside_request():
    req = request()
    actual = result(req, status=RiskStatus.VETO, vetoed_candidate_ids=("missing",))
    with pytest.raises(ValueError):
        actual.validate_for(req)


def test_validate_for_rejects_stale_geometry_even_when_ids_are_reused():
    req = request()
    old_result = result(req)
    first, second = req.plan.candidates
    changed_plan = replace(req.plan, candidates=(replace(first, trajectory=trajectory(0.5)), second))
    with pytest.raises(ValueError):
        old_result.validate_for(replace(req, plan=changed_plan))


@pytest.mark.parametrize("status,ids", [
    (RiskStatus.UNKNOWN, ()), (RiskStatus.NO_VETO, ()),
    (RiskStatus.VETO, ("a",)), (RiskStatus.VETO, ("b", "a")),
])
def test_validate_for_accepts_correspondence_without_arbitrating(status, ids):
    req = request(plan=plan(t_ref_s=-100), submitted_at_monotonic_s=0,
                  deadline_monotonic_s=1, valid_until_sim_s=-100)
    assert result(req, status=status, vetoed_candidate_ids=ids).validate_for(req) is None
