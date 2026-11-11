from dataclasses import replace
import hashlib
import json

import numpy as np
import pytest

from hybrid_drive.contracts.slow import RiskStatus
from hybrid_drive.qualification import build_suite, build_validation_suite
from hybrid_drive.qualification.metrics import summarize
from hybrid_drive.qualification.prompt import projected_prompt


# Independent numeric fixtures: literal paths and footprint bounds, not renderer/projection output.
_ORACLES = (
    ((1.5, 3, 4.5, 6, 7.5), (.35, .8, 1.45, 2, 2.45), (-.25, -.7, -1.3, -1.85, -2.4),
     (5.5, 6.5), (1.6, 2.4), (-2.25, -1.45)),
    ((2, 4.5, 7, 9.5, 12), (.15, .6, 1.4, 2.6, 3.2), (-.45, -1.0, -1.55, -2.1, -2.8),
     (8.6, 10.4), (1.95, 3.25), (-2.75, -1.45)),
    ((3, 6, 9, 12, 15), (.5, .95, 1.8, 3.1, 4.0), (-.2, -.65, -1.1, -1.9, -3.0),
     (10.9, 13.1), (2.3, 3.9), (-2.7, -1.1)),
)
_VALIDATION_MANIFEST_SHA256 = "dff10c09a268877ff2c9d01af2180da0f9e71e025ddb7655b6ea47dbf5641fc9"
_LEGACY_MANIFEST_SHA256 = "3110fea995a3b32e24e654a54169c1058abe0ce9d0f8f3949d2bd657a60a1cc7"


@pytest.fixture(scope="module")
def suite():
    return build_validation_suite()


def _manifest_hash(suite):
    payload = json.dumps(suite.manifest(), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def test_frozen_validation_and_legacy_manifests(suite):
    assert _manifest_hash(suite) == _VALIDATION_MANIFEST_SHA256
    assert _manifest_hash(build_validation_suite()) == _VALIDATION_MANIFEST_SHA256
    assert _manifest_hash(build_suite()) == _LEGACY_MANIFEST_SHA256
    manifest = suite.manifest()
    assert [case.case_id for case in suite.cases] == [f"v{i:02d}" for i in range(30)]
    assert len(suite.pairs) == 27
    assert manifest["suite_role"] == "synthetic_validation"
    assert manifest["scope"] == "controlled_suite"
    assert manifest["real_street_or_holdout"] is False
    assert (manifest["known_candidate_units"], manifest["risk_units"], manifest["clear_units"]) == (48, 24, 24)
    assert manifest["unknown_controls"] == 6
    assert manifest["rig"] == build_suite().manifest()["rig"]


def test_literal_labels_match_independent_geometry_with_clear_separation(suite):
    for family, (xs, left_y, right_y, x_bounds, left_bounds, right_bounds) in enumerate(_ORACLES):
        for index, case in enumerate(suite.cases[10 * family:10 * family + 8]):
            swapped = index >= 4
            box_bounds = ((left_bounds,), (right_bounds,), (), (left_bounds, right_bounds))[index % 4]
            risky = []
            for slot, candidate in enumerate(case.plan.candidates):
                xy = candidate.trajectory.poses[:, :2]
                ys = (left_y, right_y)[slot ^ swapped]
                np.testing.assert_array_equal(xy[:, 0], xs)
                np.testing.assert_array_equal(xy[:, 1], ys)
                # Matching paths pass exactly through the corresponding box centre.
                assert xy[3, 0] == sum(x_bounds) / 2
                expected_centre = sum((left_bounds, right_bounds)[slot ^ swapped]) / 2
                assert xy[3, 1] == pytest.approx(expected_centre)
                probe_x = [x_bounds[0], *(x for x in xs if x_bounds[0] < x < x_bounds[1]), x_bounds[1]]
                probe_y = np.interp(probe_x, xy[:, 0], xy[:, 1])
                hits = False
                for low, high in box_bounds:
                    if low < min(probe_y) <= max(probe_y) < high:
                        hits = True
                    else:
                        # Entire opposite path segment is clearly outside, never grazing the box.
                        assert min(probe_y) - high > 1 or low - max(probe_y) > 1
                if hits:
                    risky.append(candidate.candidate_id)
            assert set(case.expected_veto_ids) == set(risky)
            assert case.expected_status is (RiskStatus.VETO if risky else RiskStatus.NO_VETO)


def test_binding_interventions_and_insufficient_observations(suite):
    for offset in (0, 10, 20):
        for a, b in ((0, 4), (1, 5), (2, 6), (3, 7), (8, 9)):
            first, second = suite.cases[offset + a], suite.cases[offset + b]
            assert first.scene_hash == second.scene_hash
            assert first.candidate_hash != second.candidate_hash
            assert first.input_hash != second.input_hash
            np.testing.assert_array_equal(first.plan.candidates[0].trajectory.poses,
                                          second.plan.candidates[1].trajectory.poses)
            assert {c.candidate_id for c in first.plan.candidates} == {c.candidate_id for c in second.plan.candidates}
        for case in suite.cases[offset + 8:offset + 10]:
            assert case.expected_status is RiskStatus.UNKNOWN
            assert case.expected_veto_ids == ()
            for raw, overlaid in zip(case.raw_cameras, case.cameras):
                assert not raw.rgb.any()
                assert overlaid.rgb.any()
                colors = {tuple(value) for value in np.unique(overlaid.rgb.reshape(-1, 3), axis=0)}
                assert colors <= {(0, 0, 0), (0, 200, 220), (220, 0, 180)}


def test_definition_is_canonical_owned_and_preserved_by_selection(suite):
    original = suite.definition_json
    indented = replace(suite, definition_json=json.dumps(json.loads(original), indent=2))
    assert indented.definition_json == original
    manifest = suite.manifest()
    manifest["hand_checked_geometry"]["families"][0]["left_path_xy_m"][0][0] = -999
    manifest["rig"]["cameras"][0]["intrinsic"][0][0] = -999
    assert suite.definition_json == original
    assert _manifest_hash(suite) == _VALIDATION_MANIFEST_SHA256
    selected = suite.select(("v00", "v01", "v08"))
    assert selected.definition_json == original
    assert selected.manifest()["scope"] == "smoke_subset"
    assert selected.manifest()["full_suite_case_count"] == 30
    assert len(selected.pairs) == 1
    assert suite.select(tuple(case.case_id for case in reversed(suite.cases))).manifest()["scope"] == "controlled_suite"


@pytest.mark.parametrize("change", [
    {"full_case_ids": ["v00", "v00"]}, {"full_case_ids": []}, {"full_case_ids": "v00"},
    {"full_case_ids": [1]}, {"full_case_ids": ["foreign"]},
    {"suite_role": "heldout"}, {"suite_role": None}, {"hand_checked_geometry": {}},
    {"rig": []}, {"rig": {"bad": float("nan")}}, {"rig": {"bad": float("inf")}},
    {"unexpected": True},
])
def test_invalid_definition_schema_roles_ids_and_nonfinite_values_are_rejected(suite, change):
    definition = json.loads(suite.definition_json)
    definition.update(change)
    with pytest.raises(ValueError):
        replace(suite, definition_json=json.dumps(definition))


@pytest.mark.parametrize("definition", ["null", "[]", "{}", "not-json", 3,
                                        '{"rig":{},"rig":{}}'])
def test_malformed_or_duplicate_key_definition_is_rejected(suite, definition):
    with pytest.raises(ValueError):
        replace(suite, definition_json=definition)


def test_unknown_custom_suite_does_not_claim_old_geometry_or_rig(suite):
    custom = replace(suite, version="custom", definition_json=None).manifest()
    assert custom["scope"] == "smoke_subset"
    assert custom["full_suite_case_count"] == 0
    assert custom["label_source"] == "unspecified"
    assert "hand_checked_geometry" not in custom
    assert "rig" not in custom


def test_validation_metadata_and_labels_do_not_leak_into_prompt(suite):
    case = suite.cases[0]
    changed = replace(case, case_id="HIDDEN_CASE", title="HIDDEN_TITLE",
                      expected_status=RiskStatus.NO_VETO, expected_veto_ids=())
    kwargs = dict(request_id="opaque", submitted_at_monotonic_s=10, deadline_monotonic_s=20)
    text = projected_prompt(case.to_context("fixture", **kwargs))
    assert text == projected_prompt(changed.to_context("fixture", **kwargs))
    for hidden in (case.case_id, case.title, "HIDDEN_CASE", "HIDDEN_TITLE", suite.version,
                   "suite_role", "expected_status", "hand_checked_geometry", "synthetic_validation"):
        assert hidden not in text


def test_full_validation_uses_existing_metrics_without_becoming_smoke_or_road_qualified(suite):
    records = [dict(model_id="fixture", case_id=case.case_id, seed=0, outcome="completed",
                    status=case.expected_status.value, vetoed_candidate_ids=list(case.expected_veto_ids),
                    explanation="Fixture decision", latency_s=1.0, raw_response="{}", input_hash=case.input_hash,
                    prompt_sha256="a" * 64, request_id=f"request-{case.case_id}") for case in suite.cases]
    summary = summarize(suite, records, expected_models=("fixture",), seeds=(0,))
    result = summary["models"]["fixture"]
    assert summary["scope"] == "controlled_suite"
    assert result["diagnostic_gate"]["status"] == "pass"
    assert result["confusion"]["tp"] == result["confusion"]["tn"] == 24
    assert result["unknown_controls"]["correct"] == 6
    assert result["counterfactuals"]["both_correct"] == 27
    assert summary["verdict"] == "not_road_qualified"
    assert summary["automatic_veto_allowed"] is False
    subset = suite.select(("v00", "v01", "v08"))
    rows = [row for row in records if row["case_id"] in {case.case_id for case in subset.cases}]
    subset_summary = summarize(subset, rows, expected_models=("fixture",), seeds=(0,))
    assert subset_summary["models"]["fixture"]["diagnostic_gate"]["status"] == "fail"
