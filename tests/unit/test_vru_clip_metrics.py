import json

import pytest

from hybrid_drive.experiments.vru_clip_metrics import summarize_vru_clips


def _detection(key, x, previous=None, *, kind="person"):
    return {"object_id": key, "kind": kind, "xy_m": [x, 0.], "previous_object_id": previous}


def _target(key, x, *, kind="person"):
    return {"track_id": key, "kind": kind, "xy_m": [x, 0.]}


def _clip(token="clip", recording="drive", x=6., *, history=True):
    identity = {"sample_token": token, "recording": recording}
    predicted = {**identity, "frames": {
        "current": [_detection("now", x, "past" if history else None)],
        "previous": [_detection("past", x - .5, "old")],
        "older": [_detection("old", x - 1.)],
    }}
    labels = {**identity, "primary_id": "focus", "frames": {
        "current": [_target("focus", x)],
        "previous": [_target("focus", x - .5)],
        "older": [_target("focus", x - 1.)],
    }}
    return predicted, labels


def _score(predicted, labels):
    return summarize_vru_clips([predicted], [labels], expected_count=1, required_pass=1)


def _row(predicted, labels):
    return _score(predicted, labels)["rows"][0]


def test_older_alignment_can_verify_an_unmatched_intermediate_link():
    predicted, labels = _clip()
    predicted["frames"]["previous"][0]["xy_m"] = [10., 0.]

    row = _row(predicted, labels)

    assert row["passed"]
    assert row["history"] == {
        "previous": {"status": "unverified", "object_id": "past", "track_id": None},
        "older": {"status": "same_track", "object_id": "old", "track_id": "focus"},
    }


@pytest.mark.parametrize("wrong_time", ["previous", "older"])
def test_known_wrong_historical_link_vetoes_a_pass_even_if_the_other_link_verifies(wrong_time):
    predicted, labels = _clip()
    detected = predicted["frames"][wrong_time][0]
    wrong_x = detected["xy_m"][0] + 5.
    detected["xy_m"][0] = wrong_x
    labels["frames"][wrong_time].append(_target("other", wrong_x))

    row = _row(predicted, labels)

    assert not row["passed"]
    assert row["reason"] == "historical_link_other_track"
    assert row["history"][wrong_time]["track_id"] == "other"
    assert {link["status"] for link in row["history"].values()} == {"same_track", "other_track"}


def test_an_unverified_chain_never_counts_as_a_historical_match():
    predicted, labels = _clip()
    for time in ("previous", "older"):
        predicted["frames"][time][0]["xy_m"][0] += 10.

    assert _row(predicted, labels)["reason"] == "historical_link_unverified"


def test_old_prediction_is_not_evidence_if_current_does_not_link_to_it():
    predicted, labels = _clip(history=False)

    row = _row(predicted, labels)

    assert row["reason"] == "no_historical_link"
    assert all(link["object_id"] is None for link in row["history"].values())


def test_current_center_at_the_radius_boundary_counts_but_just_outside_does_not():
    predicted, labels = _clip()
    point = predicted["frames"]["current"][0]
    point["xy_m"][0] = 8.

    row = _row(predicted, labels)
    assert row["passed"] and row["current_error_m"] == 2.

    point["xy_m"][0] = 8.001
    assert _row(predicted, labels)["reason"] == "current_not_matched"


def test_current_matching_abstains_on_close_competing_predictions_and_truth():
    predicted, labels = _clip()
    original = predicted["frames"]["current"]
    duplicate = _detection("second", 6.1)
    original.append(duplicate)
    assert _row(predicted, labels)["reason"] == "current_not_matched"

    original.pop()
    labels["frames"]["current"].append(_target("neighbor", 6.1))
    assert _row(predicted, labels)["reason"] == "current_not_matched"


def test_other_class_at_same_center_does_not_replace_primary_or_create_ambiguity():
    predicted, labels = _clip()
    person = predicted["frames"]["current"].pop()
    predicted["frames"]["current"].append(_detection("bike", 6., kind="bicycle"))
    labels["frames"]["current"].append(_target("labeled-bike", 6., kind="bicycle"))

    assert _row(predicted, labels)["reason"] == "current_not_matched"

    predicted["frames"]["current"].append(person)
    assert _row(predicted, labels)["passed"]


def test_locally_reused_ids_at_different_times_are_not_duplicates_or_cycles():
    predicted, labels = _clip()
    for frame in predicted["frames"].values():
        frame[0]["object_id"] = "local"
        if frame[0]["previous_object_id"] is not None:
            frame[0]["previous_object_id"] = "local"

    assert _row(predicted, labels)["passed"]


@pytest.mark.parametrize("problem", ["missing_primary", "changed_class", "missing_frame", "duplicate_detection", "duplicate_truth"])
def test_invalid_truth_or_frame_is_an_error_not_a_lower_denominator(problem):
    predicted, labels = _clip()
    if problem == "missing_primary":
        labels["frames"]["older"] = []
    elif problem == "changed_class":
        labels["frames"]["older"][0]["kind"] = "bicycle"
    elif problem == "missing_frame":
        del predicted["frames"]["older"]
    elif problem == "duplicate_detection":
        predicted["frames"]["older"].append(_detection("old", 6.2))
    else:
        labels["frames"]["older"].append(_target("focus", 6.2))

    with pytest.raises(ValueError):
        _row(predicted, labels)


@pytest.mark.parametrize("problem", ["skip_frame", "two_to_one", "before_clip", "cross_class"])
def test_links_have_to_bind_causally_to_the_exact_previous_frame(problem):
    predicted, labels = _clip(history=False)
    frames = predicted["frames"]
    if problem == "skip_frame":
        frames["current"][0]["previous_object_id"] = "old"
    elif problem == "two_to_one":
        frames["current"][0]["previous_object_id"] = "past"
        frames["current"].append(_detection("another", 12., "past"))
    elif problem == "before_clip":
        frames["older"][0]["previous_object_id"] = "past"
    else:
        frames["current"].append(_detection("bike", 12., "past", kind="bicycle"))

    with pytest.raises(ValueError):
        _row(predicted, labels)


def test_order_and_cardinality_are_fixed_including_failed_or_repeated_clips():
    first, first_label = _clip("first")
    second, second_label = _clip("second", history=False)
    options = {"expected_count": 2, "required_pass": 1}

    with pytest.raises(ValueError, match="complete fixed clip count"):
        summarize_vru_clips([first], [first_label, second_label], **options)
    with pytest.raises(ValueError, match="Ordered clip identities"):
        summarize_vru_clips([second, first], [first_label, second_label], **options)
    with pytest.raises(ValueError, match="Ordered clip identities"):
        summarize_vru_clips([first, first], [first_label, first_label], **options)
    second["recording"] = "wrong-recording"
    with pytest.raises(ValueError, match="Ordered clip identities"):
        summarize_vru_clips([first, second], [first_label, second_label], **options)


def test_full_benchmark_pass_gate_and_grouped_denominators():
    clips = [_clip(f"clip-{n}", f"drive-{n // 40}", x=(7.9, 8., 16.)[n // 40], history=n < 80) for n in range(120)]
    predicted, labels = zip(*clips)

    result = summarize_vru_clips(predicted, labels)

    assert result["gate"]["status"] == "passed"
    assert result["counts"] == {"clips": 120, "passed": 80, "failed": 40, "current_matched": 120}
    assert result["failures"] == {"no_historical_link": 40}
    assert result["by_recording"]["drive-0"]["counts"]["passed"] == 40
    assert result["by_distance"]["8_to_lt_16m"]["counts"]["passed"] == 40
    assert result["by_distance"]["gte_16m"]["counts"]["failed"] == 40
    assert json.loads(json.dumps(result)) == result

    predicted[79]["frames"]["current"][0]["previous_object_id"] = None
    result = summarize_vru_clips(predicted, labels)
    assert result["gate"]["status"] == "failed"
    assert result["gate"]["additional_passes_needed"] == 1
