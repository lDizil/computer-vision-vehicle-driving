from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import pytest

from hybrid_drive.assets.files import PreparationError
from hybrid_drive.experiments import vru_clip_data as data
from hybrid_drive.experiments import vru_clip_development_data as dev
from hybrid_drive.experiments.bounded_io import canonical_json


_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
_REPLAY = _SCRIPTS / "vru-fusion-holdout-sources.json"
_CHAIN = _SCRIPTS / "vru-chain-control-sources.json"
_CONTROL = _SCRIPTS / "vru-120-sources.json"


def test_development_pins_old_archives_and_all_ten_old_groups_are_excluded_from_control():
    source = dev._development_source(_REPLAY, _CHAIN)
    assert sha256(canonical_json(source)).hexdigest() == dev.DEVELOPMENT_SOURCE_SHA256
    control = data._load_source(_CONTROL)
    development_groups = {data._group(log["recording"]) for log in source["logs"]}
    assert len(development_groups) == 10
    assert development_groups <= {data._group(name) for name in control["previous_recordings"]}
    assert development_groups.isdisjoint(data._group(log["recording"]) for log in control["logs"])
    assert set(source["camera_archives"]) == {"3", "5", "11", "16", "22"}


def test_wrong_parent_manifest_role_is_rejected():
    with pytest.raises(PreparationError, match="in that order"):
        dev._development_source(_CHAIN, _REPLAY)


def _fixture_candidates(spec, _frames):
    group = spec["ordinal"]
    recording = f"2020.10.{group:02d}.00.00.00_veh-00"
    return [data._Clip("1", recording + "_00000_00100", recording, i * 3_000_000,
                       tuple(f"{group * 1000 + i * 10 + j:016x}" for j in range(3)),
                       (data._Primary((group - 1) // 2, f"{group:016x}", "still"),)) for i in range(20)]


def _mock_candidates(monkeypatch):
    monkeypatch.setattr(data, "_read_recordings", lambda *args: [({"ordinal": k}, []) for k in range(1, 7)])
    monkeypatch.setattr(data, "_candidates", _fixture_candidates)


def test_shared_defaults_leave_the_original_control_plan_bytes_unchanged(monkeypatch):
    _mock_candidates(monkeypatch)
    plan = data._plan(None, {"camera_archives": {"1": {"size_bytes": 100}}})
    payload = canonical_json(plan) + b"\n"
    assert sha256(payload).hexdigest() == "12130bb1185de40dfb417159d00417592c2f3407fe5cc75dbdefa40325de7728"


def test_separate_development_plan_has_a_distinct_schema_and_balanced_sixty_clip_selection(monkeypatch, tmp_path):
    _mock_candidates(monkeypatch)
    monkeypatch.setattr(dev, "_require_linux", lambda: None)
    source = {"camera_archives": {"1": {"size_bytes": 100}}, "schema": "development-test"}
    monkeypatch.setattr(dev, "_development_source", lambda *_: source)
    output = tmp_path / "development-plan.json"
    summary = dev.development_plan(None, None, None, output)
    payload = output.read_bytes()
    plan = json.loads(payload)
    assert summary["clips"] == len(plan["clips"]) == 60
    assert list(summary["bands"].values()) == [20, 20, 20]
    assert len(summary["groups"]) == 6
    assert max(summary["groups"].values()) <= 16
    assert plan["schema"] == dev.DEVELOPMENT_PLAN_SCHEMA
    assert plan["source_sha256"] == dev.DEVELOPMENT_SOURCE_SHA256
    assert summary["plan_sha256"] == sha256(payload).hexdigest()
    with pytest.raises(PreparationError, match="must not already exist"):
        dev.development_plan(None, None, None, output)


def _fixture_materialize(_metadata, _source, plan):
    matrix = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    transform = [row + [0] for row in matrix] + [[0, 0, 0, 1]]
    frames, labels = {}, []
    for j, clip in enumerate(plan["clips"]):
        recording = clip["recording"]
        x = (5, 10, 20)[(int(recording[8:10]) - 1) // 2]
        for offset, key in enumerate(("older_token", "previous_token", "sample_token")):
            token = clip[key]
            views = []
            for name in data.CAMERAS:
                path = f"rgb/{token}/{name}.jpg"
                views.append({"camera": name, "path": path, "sha256": sha256(path.encode()).hexdigest(),
                              "intrinsic": matrix, "camera_to_ego_rotation": matrix,
                              "camera_to_ego_translation": [0, 0, 0], "distortion": [0] * 5})
            frames[token] = {"recording": recording, "ref_time_us": 1_000_000 + j * 3_000_000 + offset * 500_000,
                             "ego_to_world": transform, "views": views}
        labels.append({"sample_token": clip["sample_token"], "recording": recording, "primary_id": clip["primary_id"],
                       "frames": {phase: [{"track_id": clip["primary_id"], "kind": "person", "xy_m": [x, 0]}]
                                  for phase in data._PHASES}})
    return frames, {}, labels


def test_development_materializer_separates_inputs_and_labels_and_rejects_cross_profile(
        monkeypatch, tmp_path):
    _mock_candidates(monkeypatch)
    monkeypatch.setattr(dev, "_require_linux", lambda: None)
    monkeypatch.setattr(dev, "_development_source", lambda *_: {"camera_archives": {"1": {"size_bytes": 100}}})
    recordings = frozenset(_fixture_candidates({"ordinal": k}, None)[0].recording for k in range(1, 7))
    monkeypatch.setattr(dev, "_PROFILE", replace(dev._PROFILE, recordings=recordings))
    monkeypatch.setattr(data, "_materialize", _fixture_materialize)
    monkeypatch.setattr(data, "_extract_cameras", lambda _root, _archives, _wanted, inputs: inputs.mkdir(parents=True))
    plan = tmp_path / "plan.json"
    output = tmp_path / "dev"
    dev.development_plan(None, None, None, plan)
    summary = dev.prepare_development_corpus(None, None, None, None, plan, output)

    inputs, labels = output / "inputs", output / "labels" / "labels.json"
    loaded = dev.load_development_inputs(inputs)
    assert loaded["schema"] == dev.DEVELOPMENT_INPUT_SCHEMA
    assert loaded["source_sha256"] == dev.DEVELOPMENT_SOURCE_SHA256
    assert len(loaded["anchors"]) == 60
    assert len(loaded["frames"]) == 180
    assert summary["camera_jpegs"] == 1080
    assert len(list(inputs.iterdir())) == 1 and "primary_id" not in (inputs / "index.json").read_text()
    truth = dev.load_development_labels(inputs, labels)
    assert truth["schema"] == dev.DEVELOPMENT_LABEL_SCHEMA
    assert len(truth["rows"]) == 60
    assert summary["labels_sha256"] == sha256(labels.read_bytes()).hexdigest()

    with pytest.raises(ValueError, match="index shape or provenance"):
        data.load_clip_inputs(inputs)
    index_file = inputs / "index.json"
    loaded["anchors"][0]["recording"] = "2021.05.25.14.16.10_veh-35"
    index_file.write_bytes(canonical_json(loaded) + b"\n")
    with pytest.raises(ValueError, match="outside this corpus profile"):
        dev.load_development_inputs(inputs)

    loaded["schema"] = data.INPUT_SCHEMA
    index_file.write_bytes(canonical_json(loaded) + b"\n")
    with pytest.raises(ValueError, match="index shape or provenance"):
        dev.load_development_inputs(inputs)


def test_development_rejects_changed_plan_before_materializing(monkeypatch, tmp_path):
    _mock_candidates(monkeypatch)
    monkeypatch.setattr(dev, "_require_linux", lambda: None)
    monkeypatch.setattr(dev, "_development_source", lambda *_: {"camera_archives": {"1": {"size_bytes": 100}}})
    pending = tmp_path / "dev"
    plan = tmp_path / "plan.json"
    dev.development_plan(None, None, None, plan)
    plan.write_bytes(plan.read_bytes().replace(b'"primary_id"', b'"edited_id"', 1))
    with pytest.raises(PreparationError, match="Frozen clip plan differs"):
        dev.prepare_development_corpus(None, None, None, None, plan, pending)
    assert not pending.exists()
