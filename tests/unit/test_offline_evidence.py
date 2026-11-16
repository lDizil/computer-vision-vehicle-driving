import json
import shutil

import pytest

from test_real_replay import bundle, FixtureFactory
from hybrid_drive.experiments.offline_evidence import (
    load_frozen_evidence, write_frozen_evidence, _evidence_hash,
)
from hybrid_drive.experiments.real_replay import run_real_replay
from hybrid_drive.slow.context import CAMERA_NAMES


@pytest.fixture
def exported(bundle, tmp_path):
    replay, _ = run_real_replay(bundle, FixtureFactory("unused", {"model_version": "fixture"}),
                                tmp_path / "replay", revision="a" * 40)
    destination = tmp_path / "evidence"
    written = write_frozen_evidence(bundle, replay, destination)
    return destination, written, replay


def rewrite(path, change):
    file = path / "evidence.json"
    data = json.loads(file.read_text())
    change(data)
    images = [(path / (name + ".png")).read_bytes() for name in CAMERA_NAMES]
    data["evidence_hash"] = _evidence_hash(data, images)
    file.write_text(json.dumps(data))


def test_roundtrip_preserves_original_clocks_raw_and_pixels_without_rebasing(bundle, exported):
    path, written, replay = exported
    loaded = load_frozen_evidence(bundle, path)
    record = json.loads((replay / "records.json").read_text())[0]
    assert loaded.context.content_hash == record["context_hash"]
    assert loaded.context.request.submitted_at_monotonic_s == record["submitted_at_monotonic_s"]
    assert loaded.context.request.deadline_monotonic_s == record["deadline_monotonic_s"]
    assert loaded.received_at_monotonic_s == record["received_at_monotonic_s"]
    assert loaded.decision_checkpoint_monotonic_s == record["decision_checkpoint_monotonic_s"]
    assert loaded.raw_response == record["raw_response"]
    assert loaded.evidence_hash == written.evidence_hash
    assert loaded.result.status.value == "NO_VETO" and loaded.cleanup_confirmed
    assert loaded.context.request.submitted_at_monotonic_s <= loaded.received_at_monotonic_s <= loaded.decision_checkpoint_monotonic_s


@pytest.mark.parametrize("change", [
    lambda d: d.update(bundle_hash="f" * 64),
    lambda d: d["request"].update(request_id="another-request"),
    lambda d: d["request"].update(valid_until_sim_s=0),
    lambda d: d.update(context_hash="f" * 64),
    lambda d: d.update(protocol="projected-overlap-v1"),
    lambda d: d.update(model_version="other-model"),
    lambda d: d.update(raw_response="not-json"),
    lambda d: d.update(received_at_monotonic_s=d["request"]["submitted_at_monotonic_s"] - 1),
    lambda d: d.update(decision_checkpoint_monotonic_s=d["received_at_monotonic_s"] - 1),
    lambda d: d.update(cleanup_confirmed=False),
    lambda d: d["prompts"].__setitem__(0, "invented prompt"),
    lambda d: d["cameras"][0].update(path="../outside.png"),
])
def test_rehashed_tampering_cannot_change_context_protocol_or_temporal_binding(bundle, exported, change):
    path, _, _ = exported
    rewrite(path, change)
    with pytest.raises(ValueError): load_frozen_evidence(bundle, path)


def test_png_corruption_and_extra_files_are_not_ignored(bundle, exported):
    path, _, _ = exported
    image = path / "CAM_F0.png"
    image.write_bytes(b"corrupt")
    with pytest.raises((ValueError, OSError)): load_frozen_evidence(bundle, path)


def test_symlink_is_rejected_even_when_pointing_to_identical_png(bundle, exported, tmp_path):
    path, _, _ = exported
    image = path / "CAM_F0.png"
    copy = tmp_path / "original.png"; shutil.copyfile(image, copy)
    image.unlink(); image.symlink_to(copy)
    with pytest.raises(ValueError): load_frozen_evidence(bundle, path)


def test_late_decision_checkpoint_is_preserved_for_arbiter_not_rewritten_as_fresh(bundle, exported):
    path, _, _ = exported
    rewrite(path, lambda d: d.update(decision_checkpoint_monotonic_s=d["request"]["deadline_monotonic_s"] + 1))
    loaded = load_frozen_evidence(bundle, path)
    assert loaded.decision_checkpoint_monotonic_s > loaded.context.request.deadline_monotonic_s
    assert loaded.received_at_monotonic_s < loaded.context.request.deadline_monotonic_s


@pytest.mark.parametrize("kind", ["malformed", "timeout"])
def test_negative_execution_evidence_never_becomes_a_prediction(bundle, tmp_path, kind):
    replay, _ = run_real_replay(bundle, FixtureFactory("unused", {"model_version": "fixture"}, kind=kind),
                                tmp_path / "replay", revision="a" * 40,
                                deadline_s=.1 if kind == "timeout" else 10)
    path = tmp_path / "evidence"
    write_frozen_evidence(bundle, replay, path)
    loaded = load_frozen_evidence(bundle, path)
    assert loaded.result is None and loaded.outcome in ("invalid_output", "timeout")
    assert loaded.cleanup_confirmed
