import math

import numpy as np
import pytest

from hybrid_drive.experiments.spatial_decision_policy import (
    ABSENT,
    PRESENT,
    UNCERTAIN,
    ZoneDecisionPolicy,
    calibrate_zone_policy,
    measure_zone_policy,
)


def test_rare_recording_cannot_hide_its_false_positives_in_pooled_data():
    scores = np.array([[1, 2, 3, 4]] * 5 + [[100, 110, 120, 130]], dtype=np.float64)
    labels = np.array([[False, False, False, True]] * 5 + [[False, True, True, True]])
    recordings = ["common"] * 5 + ["rare"]
    fitted = calibrate_zone_policy(scores, labels, recordings)

    # Pooled calibration could afford one FP, but the rare log has one negative.
    assert fitted.present_at_or_above == math.nextafter(100, math.inf)
    assert fitted.absent_below == 4
    measured = measure_zone_policy(scores, labels, recordings, fitted)
    assert measured["per_recording"]["rare"]["false_positive_rate"] == 0
    assert measured["per_recording"]["common"]["false_negative_rate"] == 0
    assert measured["uncertain_count"] == 6


def test_ties_force_nextafter_and_keep_boundary_positives_uncertain():
    scores = np.array([[0.5, 0.5, 0.25, 0.25], [0.5, 0.5, 0.25, 0.7]], dtype=np.float32)
    labels = np.array([[False, False, True, True], [False, False, True, True]])
    fitted = calibrate_zone_policy(
        scores, labels, ["a", "b"], max_false_positive_rate=0.5, max_false_negative_rate=0.5
    )
    assert fitted.present_at_or_above == math.nextafter(0.5, math.inf)
    assert fitted.absent_below == 0.25
    actual = fitted.decide(scores)
    assert actual.dtype == np.int8
    # No comparison rounds the saved float64 threshold back to float32.
    np.testing.assert_array_equal(actual, [[UNCERTAIN] * 4, [UNCERTAIN] * 3 + [PRESENT]])


def test_confident_absence_cutoff_is_maximal_under_the_per_recording_budget():
    scores = np.array([[0.10, 0.20, 0.30, 0.40], [0.25, 0.26, 0.31, 0.42]])
    labels = np.array([[True, True, True, False], [True, True, True, False]])
    fitted = calibrate_zone_policy(
        scores, labels, ["a", "b"], max_false_positive_rate=0.0, max_false_negative_rate=1 / 3
    )
    assert fitted.absent_below == 0.20
    assert fitted.present_at_or_above == math.nextafter(0.42, math.inf)

    def first_recording(used):
        return measure_zone_policy(scores, labels, ["a", "b"], used)["per_recording"]["a"]

    assert first_recording(fitted)["false_negative_rate"] == 1 / 3
    bumped = ZoneDecisionPolicy(math.nextafter(fitted.absent_below, math.inf), fitted.present_at_or_above)
    assert first_recording(bumped)["false_negative_rate"] > 1 / 3


def test_identical_scores_can_produce_all_uncertain_and_no_imputed_negatives():
    scores = np.ones((2, 4))
    labels = np.array([[True, True, False, False]] * 2)
    fitted = calibrate_zone_policy(scores, labels, ["a", "b"])
    info = measure_zone_policy(scores, labels, ["a", "b"], fitted)

    assert info["uncertain_count"] == 8
    assert info["absent_count"] == info["present_count"] == 0
    assert info["true_absent"] == info["false_absent"] == 0
    assert info["precision"] is None
    assert info["positive_recall"] == 0
    assert info["coverage"] == 0
    assert info["false_positive_rate"] == info["false_negative_rate"] == 0


def test_missing_outcome_only_omits_the_relevant_constraint_and_metric():
    scores = np.array([[0.9, 0.85, 0.9, 0.93], [0.1, 0.3, 0.2, 0.1]])
    labels = np.array([[True] * 4, [False] * 4])
    fitted = calibrate_zone_policy(scores, labels, ["all positive", "all negative"])
    info = measure_zone_policy(scores, labels, ["all positive", "all negative"], fitted)

    assert fitted.absent_below == fitted.present_at_or_above == math.nextafter(0.3, math.inf)
    assert (info["present_count"], info["absent_count"], info["uncertain_count"]) == (4, 4, 0)
    assert info["false_present"] == info["false_absent"] == 0
    assert info["per_recording"]["all positive"]["false_positive_rate"] is None
    assert info["per_recording"]["all negative"]["false_negative_rate"] is None
    assert info["per_recording"]["all negative"]["positive_recall"] is None


def test_coverage_and_error_rates_use_all_labels_not_just_confident_decisions():
    scores = np.array([[0.1, 0.5, 0.9, 0.9], [0.1, 0.5, 0.9, 0.1]])
    labels = np.array([[True, True, True, False], [False, False, False, False]])
    d = measure_zone_policy(scores, labels, ["a", "b"], ZoneDecisionPolicy(0.25, 0.75))
    assert (d["true_present"], d["false_present"], d["true_absent"], d["false_absent"]) == (1, 2, 2, 1)
    assert d["uncertain_count"] == 2
    assert d["positive_recall"] == 1 / 3
    assert d["false_negative_rate"] == 1 / 3
    assert d["false_positive_rate"] == 2 / 5
    assert d["precision"] == 1 / 3
    assert d["coverage"] == 6 / 8


def test_float_rounding_does_not_reject_an_exact_count_allowed_by_reported_rate():
    # 0.58 * 50 rounds down below 29, but by computed rates 29/50 == 0.58.
    scores = np.vstack((np.arange(52, dtype=np.float64).reshape(13, 4), [[-4, -3, -2, -1]]))
    labels = np.zeros_like(scores, dtype=bool)
    labels[-2:, -2:] = True
    recordings = ["many"] * 13 + ["small"]
    fitted = calibrate_zone_policy(
        scores, labels, recordings, max_false_positive_rate=0.58, max_false_negative_rate=1,
    )
    assert fitted.present_at_or_above == math.nextafter(20, math.inf)
    assert measure_zone_policy(scores, labels, recordings, fitted)["per_recording"]["many"][
        "false_positive_rate"
    ] == 0.58


def test_calibration_is_invariant_to_dev_row_order():
    scores = np.arange(16, dtype=np.float64).reshape(4, 4) / 3
    labels = (scores % 2) < 1
    recordings = np.array(["a", "b", "a", "b"])
    order = [3, 0, 1, 2]
    assert calibrate_zone_policy(scores, labels, recordings) == calibrate_zone_policy(
        scores[order], labels[order], recordings[order]
    )


def test_equal_cutoffs_reproduce_binary_gating_at_the_exact_boundary():
    scores = np.array([[0.49999, 0.5, 0.50001, 1.0]], dtype=np.float32)
    np.testing.assert_array_equal(
        ZoneDecisionPolicy(0.5, 0.5).decide(scores),
        [[ABSENT, PRESENT, PRESENT, PRESENT]],
    )


@pytest.mark.parametrize("lower,upper", [(1, 0), (0, math.inf), (math.nan, 1), (True, 1), (0, "1")])
def test_reject_invalid_thresholds(lower, upper):
    with pytest.raises(ValueError):
        ZoneDecisionPolicy(lower, upper)


@pytest.mark.parametrize(
    "values",
    [np.empty((0, 4)), np.ones(4), np.ones((1, 3)), np.zeros((1, 4), dtype=int), np.full((1, 4), math.nan)],
)
def test_decisions_reject_unavailable_or_invalid_scores(values):
    with pytest.raises(ValueError, match="scores"):
        ZoneDecisionPolicy(0, 1).decide(values)


@pytest.mark.parametrize(
    "labels,recordings,budget",
    [
        (np.zeros((2, 4), dtype=bool), ["a", "b"], 0.2),
        (np.eye(2, 4, dtype=bool), ["a", "a"], 0.2),
        (np.eye(2, 4, dtype=bool), ["a"], 0.2),
        (np.eye(2, 4, dtype=bool), ["a", ""], 0.2),
        (np.eye(2, 4, dtype=bool), "ab", 0.2),
        (np.zeros((2, 4), dtype=float), ["a", "b"], 0.2),
        (np.eye(2, 4, dtype=bool), ["a", "b"], -0.1),
        (np.eye(2, 4, dtype=bool), ["a", "b"], True),
    ],
)
def test_calibration_rejects_invalid_or_insufficient_dev_input(labels, recordings, budget):
    with pytest.raises(ValueError):
        calibrate_zone_policy(np.ones((2, 4)), labels, recordings, max_false_positive_rate=budget)


def test_calibration_rejects_if_a_finite_cutoff_cannot_exclude_extreme_negatives():
    scores = np.full((2, 4), np.finfo(np.float64).max)
    labels = np.eye(2, 4, dtype=bool)
    with pytest.raises(ValueError, match="No finite float64 presence cutoff"):
        calibrate_zone_policy(scores, labels, ["a", "b"])
