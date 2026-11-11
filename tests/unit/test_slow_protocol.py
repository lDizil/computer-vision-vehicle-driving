from dataclasses import replace
import json

import numpy as np
import pytest

from hybrid_drive import Pose2D, Trajectory
from hybrid_drive.contracts.motion import EgoMotion
from hybrid_drive.contracts.planning import Candidate, CandidateSet
from hybrid_drive.contracts.slow import RiskStatus, SlowRequest
from hybrid_drive.slow.context import CAMERA_NAMES, CameraFrame, SlowContext
from hybrid_drive.slow.protocol import MAX_RESPONSE_BYTES, parse_response


@pytest.fixture
def context():
    plan = CandidateSet("plan", "snapshot", 10, Pose2D(0, 0, 0), (
        Candidate("left", Trajectory([[1, 0.1, 0]], 0.5)),
        Candidate("straight", Trajectory([[1, 0, 0]], 0.5)),
    ))
    return SlowContext(SlowRequest("request", plan, 100, 101, 10.5, "trusted-model"), EgoMotion(2, 0, 0, 0),
                       tuple(CameraFrame(name, np.zeros((1, 1, 3), np.uint8)) for name in CAMERA_NAMES))


def response(context, **changes):
    payload = {"context_hash": context.content_hash, "status": "UNKNOWN",
               "vetoed_candidate_ids": [], "explanation": "Insufficient visible evidence."}
    payload.update(changes)
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


@pytest.mark.parametrize("status,ids", [
    ("UNKNOWN", []), ("NO_VETO", []), ("VETO", ["straight"]), ("VETO", ["straight", "left"]),
])
def test_valid_response_stamps_trusted_envelope_and_keeps_explicit_status(context, status, ids):
    raw = response(context, status=status, vetoed_candidate_ids=ids, explanation="VETO is just a word here.")
    value = parse_response(raw, context)
    value.validate_for(context.request)
    assert value.status is RiskStatus(status)
    assert value.vetoed_candidate_ids == tuple(ids)
    assert value.explanation == "VETO is just a word here."
    assert (value.request_id, value.plan_id, value.snapshot_id, value.model_version) == (
        "request", "plan", "snapshot", "trusted-model",
    )


@pytest.mark.parametrize("key,value", [
    ("request_id", "forged"), ("model_version", "forged"), ("deadline_monotonic_s", 100000),
    ("snapshot_id", "other"), ("plan_id", "other"), ("candidates_hash", "0" * 64),
])
def test_model_cannot_supply_correspondence_or_timing_fields(context, key, value):
    with pytest.raises(ValueError):
        parse_response(response(context, **{key: value}), context)


@pytest.mark.parametrize("key", ["context_hash", "status", "vetoed_candidate_ids", "explanation"])
def test_missing_or_duplicate_keys_are_not_silently_defaulted(context, key):
    payload = json.loads(response(context))
    del payload[key]
    with pytest.raises(ValueError):
        parse_response(json.dumps(payload), context)
    original = response(context)
    duplicate = original[:-1] + "," + json.dumps(key) + ":" + json.dumps(json.loads(original)[key]) + "}"
    with pytest.raises(ValueError):
        parse_response(duplicate, context)


@pytest.mark.parametrize("wrap", [
    lambda raw: "```json\n" + raw + "\n```", lambda raw: "Result: " + raw,
    lambda raw: raw + " done", lambda raw: raw + raw, lambda raw: "[" + raw + "]",
    lambda raw: raw + "\x00", lambda raw: "\ufeff" + raw,
])
def test_prose_fences_arrays_and_trailing_data_are_invalid(context, wrap):
    with pytest.raises(ValueError):
        parse_response(wrap(response(context)), context)


@pytest.mark.parametrize("raw", [None, b"{}", "", "null", "true", "3", "{}", "[", "{" * 2000])
def test_invalid_structure_is_always_a_value_error(context, raw):
    with pytest.raises(ValueError):
        parse_response(raw, context)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_nonstandard_nonfinite_json_is_rejected_even_in_unused_fields(context, constant):
    raw = response(context)[:-1] + ',"unused":' + constant + "}"
    with pytest.raises(ValueError):
        parse_response(raw, context)


@pytest.mark.parametrize("status,ids", [
    ("VETO", []), ("UNKNOWN", ["left"]), ("NO_VETO", ["left"]),
    ("veto", ["left"]), ("SAFE", []), (None, []), (True, []),
    ("VETO", ["missing"]), ("VETO", ["left", "left"]),
    ("VETO", [1]), ("VETO", [True]), ("VETO", [None]), ("VETO", [""]),
    ("VETO", "left"), ("UNKNOWN", None), ("VETO", {"left": True}),
])
def test_status_and_candidate_ids_are_strict(context, status, ids):
    with pytest.raises(ValueError):
        parse_response(response(context, status=status, vetoed_candidate_ids=ids), context)


@pytest.mark.parametrize("digest", ["0" * 64, "", None, 1, []])
def test_response_must_match_the_entire_context_hash(context, digest):
    with pytest.raises(ValueError):
        parse_response(response(context, context_hash=digest), context)


def test_reusing_request_ids_with_other_pixels_still_rejects_old_response(context):
    original = response(context)
    frames = list(context.cameras)
    frames[-1] = CameraFrame("CAM_R0", np.ones((1, 1, 3), np.uint8))
    changed = replace(context, cameras=frames)
    with pytest.raises(ValueError):
        parse_response(original, changed)


@pytest.mark.parametrize("explanation", [None, 1, [], "x" * 513, "\ud800"])
def test_explanation_is_bounded_text_not_protocol_or_arbitrary_json(context, explanation):
    with pytest.raises(ValueError):
        parse_response(response(context, explanation=explanation), context)


def test_explanation_character_limit_and_response_byte_limit_are_independent(context):
    raw = response(context, explanation="界" * 512)
    assert parse_response(raw, context).explanation == "界" * 512
    padded = raw + " " * (MAX_RESPONSE_BYTES - len(raw.encode("utf-8")))
    assert len(padded.encode("utf-8")) == MAX_RESPONSE_BYTES
    assert parse_response(padded, context).status is RiskStatus.UNKNOWN
    with pytest.raises(ValueError):
        parse_response(padded + " ", context)
    multibyte_padding = raw + " " * (MAX_RESPONSE_BYTES - len(raw))
    assert len(multibyte_padding) == MAX_RESPONSE_BYTES
    with pytest.raises(ValueError):
        parse_response(multibyte_padding, context)


def test_json_whitespace_and_escaped_duplicate_key_are_handled_strictly(context):
    assert parse_response(" \n\t" + response(context) + "\r\n", context).status is RiskStatus.UNKNOWN
    escaped_duplicate = response(context)[:-1] + ',"\\u0073tatus":"NO_VETO"}'
    with pytest.raises(ValueError):
        parse_response(escaped_duplicate, context)


def test_escaped_unpaired_surrogate_is_invalid_even_when_raw_text_is_utf8(context):
    payload = json.loads(response(context))
    payload["explanation"] = "\ud800"
    raw = json.dumps(payload, ensure_ascii=True)
    raw.encode("utf-8")
    with pytest.raises(ValueError):
        parse_response(raw, context)
