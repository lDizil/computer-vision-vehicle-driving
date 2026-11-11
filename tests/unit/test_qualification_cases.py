from dataclasses import fields, replace
import json
import pickle

import numpy as np
import pytest

from hybrid_drive.contracts.motion import EgoMotion
from hybrid_drive.contracts.planning import Candidate, CandidateSet
from hybrid_drive.contracts.slow import RiskStatus
from hybrid_drive.contracts.trajectory import Trajectory
from hybrid_drive.qualification import CounterfactualPair, QualificationSuite, build_suite
from hybrid_drive.qualification.render import CANDIDATE_COLORS
from hybrid_drive.slow.context import CameraFrame, _metadata


@pytest.fixture(scope="module")
def suite():
    return build_suite()


def test_deterministic_balanced_fixtures_and_complete_provenance(suite):
    repeated = build_suite()
    assert [case.case_id for case in suite.cases] == [f"q{i:02d}" for i in range(10)]
    assert [case.input_hash for case in repeated.cases] == [case.input_hash for case in suite.cases]
    manifest = suite.manifest()
    assert manifest["scope"] == "controlled_suite"
    assert (manifest["known_candidate_units"], manifest["risk_units"], manifest["clear_units"]) == (16, 8, 8)
    assert manifest["unknown_controls"] == 2
    assert manifest["label_source"] == "analytic_controlled_scene"
    assert manifest["real_street_or_holdout"] is False
    assert "not NAVSIM" in manifest["rig"]["provenance"]
    assert len(manifest["rig"]["cameras"]) == 3
    assert json.loads(json.dumps(manifest))["case_count"] == 10
    for case, repeated_case in zip(suite.cases, repeated.cases):
        assert len(case.input_hash) == len(case.scene_hash) == len(case.candidate_hash) == 64
        for frame, duplicate in zip(case.cameras, repeated_case.cameras):
            np.testing.assert_array_equal(frame.rgb, duplicate.rgb)
            assert (frame.width, frame.height) == (384, 216)


def test_expected_labels_are_independently_checked_against_numeric_geometry(suite):
    # All boxes have x footprint [7.35, 8.65]. In that whole interval each
    # piecewise-linear centreline remains inside the matching y footprint.
    box_y_bounds = (((.95, 2.05),), ((-2.05, -.95),), (), ((.95, 2.05), (-2.05, -.95)))
    expected_slots = ((0,), (1,), (), (0, 1), (1,), (0,), (), (0, 1))
    for index, case in enumerate(suite.cases[:8]):
        independently_unsafe = []
        for slot, candidate in enumerate(case.plan.candidates):
            xy = candidate.trajectory.poses[:, :2]
            np.testing.assert_array_equal(xy[:, 0], [2, 4, 6, 8, 10])
            np.testing.assert_array_equal(np.abs(xy[:, 1]), [.2, .5, 1, 1.5, 2])
            low, centre, high = np.interp([7.35, 8, 8.65], xy[:, 0], xy[:, 1])
            assert centre in (1.5, -1.5)
            hits = any(ymin <= min(low, high) <= max(low, high) <= ymax
                       for ymin, ymax in box_y_bounds[index % 4])
            if hits:
                independently_unsafe.append(slot)
        assert tuple(independently_unsafe) == expected_slots[index]
        assert set(case.expected_veto_ids) == {
            case.plan.candidates[slot].candidate_id for slot in independently_unsafe
        }
        assert case.expected_status is (RiskStatus.VETO if independently_unsafe else RiskStatus.NO_VETO)


def test_bindings_change_colors_and_labels_but_not_raw_scene(suite):
    for a, b in ((0, 4), (1, 5), (2, 6), (3, 7), (8, 9)):
        normal, swapped = suite.cases[a], suite.cases[b]
        assert normal.scene_hash == swapped.scene_hash
        assert normal.candidate_hash != swapped.candidate_hash
        assert normal.input_hash != swapped.input_hash
        for first, second in zip(normal.raw_cameras, swapped.raw_cameras):
            assert first.rgb.tobytes() == second.rgb.tobytes()
        np.testing.assert_array_equal(normal.plan.candidates[0].trajectory.poses,
                                      swapped.plan.candidates[1].trajectory.poses)
    normal, swapped = suite.cases[0], suite.cases[4]
    assert normal.expected_veto_ids == (normal.plan.candidates[0].candidate_id,)
    assert swapped.expected_veto_ids == (swapped.plan.candidates[1].candidate_id,)
    # Independent F0 pinhole arithmetic at (x=6,y=1,z=0):
    # u=191.5-220/6=154.833; v=107.5+220*1.6/6=166.167.
    for case, color in ((normal, CANDIDATE_COLORS[0]), (swapped, CANDIDATE_COLORS[1])):
        patch = case.cameras[1].rgb[164:169, 153:158]
        assert np.any(np.all(patch == color, axis=2))
        assert not np.any(np.all(case.raw_cameras[1].rgb == color, axis=2))


def test_blank_controls_keep_only_paths_and_require_unknown(suite):
    for case in suite.cases[8:]:
        assert case.expected_status is RiskStatus.UNKNOWN
        assert case.expected_veto_ids == ()
        for raw, overlaid in zip(case.raw_cameras, case.cameras):
            assert not raw.rgb.any()
            assert overlaid.rgb.any()
            colors = {tuple(value) for value in np.unique(overlaid.rgb.reshape(-1, 3), axis=0)}
            assert colors <= {(0, 0, 0), *CANDIDATE_COLORS}


def test_context_has_no_ground_truth_titles_case_ids_or_layout_metadata(suite):
    case = suite.cases[0]
    context = case.to_context("model-test", request_id="opaque-request",
                              submitted_at_monotonic_s=10, deadline_monotonic_s=20)
    assert {field.name for field in fields(context)} == {"request", "motion", "cameras", "content_hash"}
    metadata = json.dumps(_metadata(context), ensure_ascii=False)
    for forbidden in (case.title, case.case_id, "expected_status", "expected_veto", "layout", "label_source"):
        assert forbidden not in metadata
    for identifier in (case.plan.plan_id, case.plan.snapshot_id, *(c.candidate_id for c in case.plan.candidates)):
        assert identifier.split("-", 1)[1].isalnum()
        assert not any(word in identifier for word in ("left", "right", "clear", "risk", "blank"))
    assert context.request.valid_until_sim_s == 2.5
    assert context.request.submitted_at_monotonic_s == 10
    assert context.request.deadline_monotonic_s == 20
    assert context.cameras is case.cameras


def test_input_fingerprint_ignores_only_non_input_labels_and_runtime_envelope(suite):
    case = suite.cases[0]
    relabelled = replace(case, case_id="different-id", title="Different title",
                        expected_status=RiskStatus.NO_VETO, expected_veto_ids=())
    assert relabelled.input_hash == case.input_hash
    a = case.to_context("model-a", request_id="first", submitted_at_monotonic_s=10, deadline_monotonic_s=20)
    b = case.to_context("model-b", request_id="second", submitted_at_monotonic_s=30, deadline_monotonic_s=40)
    assert a.content_hash != b.content_hash
    assert replace(case, motion=EgoMotion(5, 0, 0, 0)).input_hash != case.input_hash
    pixels = case.cameras[0].rgb.copy()
    pixels[0, 0, 0] ^= 1
    changed = replace(case, cameras=(CameraFrame(case.cameras[0].name, pixels), *case.cameras[1:]))
    assert changed.input_hash != case.input_hash
    assert changed.scene_hash == case.scene_hash
    assert changed.candidate_hash == case.candidate_hash
    poses = case.plan.candidates[0].trajectory.poses.copy()
    poses[-1, 0] += .01
    candidate = Candidate(case.plan.candidates[0].candidate_id, Trajectory(poses, .5))
    plan = replace(case.plan, candidates=(candidate, case.plan.candidates[1]))
    assert replace(case, plan=plan).candidate_hash != case.candidate_hash
    assert replace(case, plan=replace(case.plan, snapshot_id="another-snapshot")).input_hash != case.input_hash


def test_owned_pixels_and_trajectory_survive_spawn_serialization(suite):
    case = pickle.loads(pickle.dumps(suite.cases[0]))
    assert case.input_hash == suite.cases[0].input_hash
    with pytest.raises(ValueError):
        case.cameras[0].rgb.setflags(write=True)
    with pytest.raises(ValueError):
        case.plan.candidates[0].trajectory.poses.setflags(write=True)


@pytest.mark.parametrize("change", [
    {"expected_status": RiskStatus.NO_VETO},
    {"expected_veto_ids": ()},
    {"expected_veto_ids": ("unknown",)},
    {"expected_veto_ids": (["bad"],)},
    {"expected_status": "VETO"},
])
def test_invalid_expected_labels_are_rejected(suite, change):
    with pytest.raises(ValueError):
        replace(suite.cases[0], **change)


def test_select_keeps_only_complete_pairs_and_marks_smoke(suite):
    selected = suite.select(("q00", "q01", "q08"))
    assert [case.case_id for case in selected.cases] == ["q00", "q01", "q08"]
    assert len(selected.pairs) == 1
    assert selected.pairs[0].kind == "image_change"
    assert selected.manifest()["scope"] == "smoke_subset"
    assert selected.manifest()["full_suite_case_count"] == 10
    for ids in (("q00", "missing"), ("q00", "q00"), (), "q00"):
        with pytest.raises(ValueError):
            suite.select(ids)


@pytest.mark.parametrize("kind,a,b", [
    ("image_change", 0, 4), ("image_change", 0, 0),
    ("candidate_change", 0, 1), ("candidate_change", 0, 2),
])
def test_counterfactuals_cannot_claim_wrong_intervention(suite, kind, a, b):
    with pytest.raises(ValueError):
        pair = CounterfactualPair("bad", kind, suite.cases[a].case_id, suite.cases[b].case_id)
        QualificationSuite(suite.cases, (pair,), suite.version)


def test_counterfactual_labels_follow_geometry_and_reject_other_drift(suite):
    first, swapped = suite.cases[0], suite.cases[4]
    pair = CounterfactualPair("binding", "candidate_change", first.case_id, swapped.case_id)
    wrong_label = replace(swapped, expected_veto_ids=first.expected_veto_ids)
    with pytest.raises(ValueError, match="labels must follow"):
        QualificationSuite((first, wrong_label), (pair,), suite.version)
    changed_motion = replace(swapped, motion=EgoMotion(3, 0, 0, 0))
    with pytest.raises(ValueError, match="motion"):
        QualificationSuite((first, changed_motion), (pair,), suite.version)
    wrong_frame = replace(swapped, plan=replace(swapped.plan, t_ref_s=1))
    with pytest.raises(ValueError, match="reference frame"):
        QualificationSuite((first, wrong_frame), (pair,), suite.version)
    reordered = replace(swapped, plan=replace(first.plan, candidates=first.plan.candidates[::-1]))
    with pytest.raises(ValueError, match="permute"):
        QualificationSuite((first, reordered), (pair,), suite.version)
    with pytest.raises(ValueError, match="unique"):
        QualificationSuite((first, first), (), suite.version)
    with pytest.raises(ValueError, match="missing"):
        QualificationSuite((first,), (pair,), suite.version)
    with pytest.raises(ValueError, match="Duplicate"):
        QualificationSuite((first, swapped), (pair, replace(pair, pair_id="other")), suite.version)
