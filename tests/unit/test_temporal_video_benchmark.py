import hashlib
import json
from types import SimpleNamespace

import pytest

from hybrid_drive.experiments import temporal_video_benchmark as bench
from hybrid_drive.experiments.protocol import content_hash
from hybrid_drive.experiments.temporal_spatial_types import FrameObservation


def _pack(role, directory, *, delta_us=500_000):
    observations, targets = [], []
    logits = ((0.9, -1.2, 0.15, -0.8), (0.1, 0.75, -0.9, -0.3), (-1.0, 0.5, -1.2, 0.8))
    for recording in range(2):
        label = f"{role}-rec-{recording}"
        for clip in range(2):
            for i in range(8):
                index = 10 + i + clip * 12
                id_ = f"{label}-{clip}-{index}"
                x = (i % 3) * 0.005
                boxes = tuple((x + band * 0.25 + 0.03, 0.25, x + band * 0.25 + 0.19, 0.48) for band in range(4))
                present = (i % 4 != 0, i % 3 == 2, False, i >= 6)
                observations.append(FrameObservation(label, f"{label}-clip-{clip}", id_, index,
                    1_000_000 + index * delta_us, logits[i % len(logits)], boxes))
                targets.append({"sample_id": id_, "presence": present,
                                "boxes_2d": tuple(b if flag else None for b, flag in zip(boxes, present))})
    pack = SimpleNamespace(root=directory, manifest={"role": role, "tags": list(targets)},
                           frames=tuple(observations), manifest_hash=hashlib.sha256(role.encode()).hexdigest())
    return pack, SimpleNamespace(frames=tuple(observations))


def _setup(tmp_path, monkeypatch):
    roles = ("development", "holdout", "stress")
    packs, traces = {}, {}
    for role in roles:
        packs[role], traces[role] = _pack(role, tmp_path / role)
    root = tmp_path / "output"
    capture = SimpleNamespace(packs=packs, traces=traces, manifest_sha256="c" * 64, manifest={
        "model_receipt": {"calibrated_policy": {"absent_below": -0.5, "present_at_or_above": 0.5}},
        "model_hash": "d" * 64, "revision": "a" * 40,
    })
    opened = []

    def read_labels(_, data, split):
        role = data["role"]
        assert split == "holdout"
        opened.append(role)
        if role != "development":
            saved = list(root.glob("*/frozen-development.json"))
            assert len(saved) == 1
            gate = json.loads(saved[0].read_text())
            assert gate["capture_sha256"] == capture.manifest_sha256
            assert gate["selection"]["name"] in bench.RECIPES
        return data["tags"]

    monkeypatch.setattr(bench, "read_spatial_targets", read_labels)

    def gallery(path, pack, targets, outcomes, *, selected_name):
        assert pack is packs["holdout"] and len(targets) == len(pack.frames)
        assert set(outcomes) == set(bench.RECIPES) and selected_name in outcomes
        assert opened[-3:] == list(roles)
        assert all(len(arm) == len(pack.frames) for arm in outcomes.values())
        path.write_text("local demo")

    monkeypatch.setattr(bench, "write_temporal_spatial_gallery", gallery)
    return root, capture, opened


def _run(tmp, output, capture):
    return bench.run_temporal_video_benchmark(capture, tmp / "capture", output, revision="b" * 40)


def test_matrix_is_frozen_before_holdout_and_repeatable_without_gallery(tmp_path, monkeypatch):
    output, capture, opened = _setup(tmp_path, monkeypatch)
    run, result, digest = _run(tmp_path, output, capture)
    assert result["status"] == "passed", result.get("error")
    assert opened == ["development", "holdout", "stress"]
    matrix = result["matrix"]
    assert matrix["experiment_counts"] == {"clean": 27, "stress": 36, "total": 63}
    assert matrix["selection"]["name"] == matrix["holdout_diagnostic"]["evaluated_recipe"]
    assert result["matrix_hash"] == content_hash(matrix)
    assert all(all(checks.values()) for checks in matrix["fault_invariants"].values())
    assert (run / "video.html").exists() and 'href="video.html"' in (run / "report.html").read_text()

    def no_rgb(*args, **kwargs):
        pytest.fail("Replay verification must not render images")

    monkeypatch.setattr(bench, "write_temporal_spatial_gallery", no_rgb)
    checked, proof = bench.verify_temporal_video_replay(
        run, capture, tmp_path / "capture", tmp_path / "verify", expected_benchmark_result_sha256=digest,
    )
    assert opened == ["development", "holdout", "stress"] * 2
    assert proof["status"] == "identical" and proof["matrix_hash"] == result["matrix_hash"]
    assert proof["inference_repeated"] is False and proof["rgb_loaded"] is False
    assert (checked / "replay-proof.json").exists()
    assert not (checked / "video.html").exists()


def test_changed_gate_prevents_first_final_label_access(tmp_path, monkeypatch):
    output, capture, opened = _setup(tmp_path, monkeypatch)
    write = bench.write_json

    def faulty(path, data):
        if path.name == "frozen-development.json":
            data = {**data, "capture_sha256": "0" * 64}
        write(path, data)

    monkeypatch.setattr(bench, "write_json", faulty)
    _, result, _ = _run(tmp_path, output, capture)
    assert result["status"] == "failed" and "did not round-trip" in result["error"]
    assert opened == ["development"]


def test_replay_checks_original_result_and_frozen_gate_before_any_recalculation(tmp_path, monkeypatch):
    output, capture, opened = _setup(tmp_path, monkeypatch)
    run, result, digest = _run(tmp_path, output, capture)
    assert result["status"] == "passed"
    opened.clear()
    with pytest.raises(ValueError, match="benchmark byte hash"):
        bench.verify_temporal_video_replay(
            run, capture, tmp_path / "capture", tmp_path / "repeat", expected_benchmark_result_sha256="0" * 64,
        )
    path = run / "frozen-development.json"
    data = json.loads(path.read_text())
    data["selection"]["name"] = "tampered"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="dev gate was tampered"):
        bench.verify_temporal_video_replay(
            run, capture, tmp_path / "capture", tmp_path / "repeat", expected_benchmark_result_sha256=digest,
        )
    assert opened == []


def test_fast_stress_clip_rejected_before_labels_instead_of_emulating_wrong_event_order(tmp_path, monkeypatch):
    output, capture, opened = _setup(tmp_path, monkeypatch)
    pack, trace = _pack("stress", tmp_path / "fast", delta_us=300_000)
    capture.packs["stress"], capture.traces["stress"] = pack, trace
    _, result, _ = _run(tmp_path, output, capture)
    assert result["status"] == "failed" and "too fast" in result["error"]
    assert opened == []


def test_stress_metrics_match_only_surviving_inputs(tmp_path, monkeypatch):
    output, capture, _ = _setup(tmp_path, monkeypatch)
    _, result, _ = _run(tmp_path, output, capture)
    assert result["status"] == "passed"
    rows = result["matrix"]["faults"]["drop_every_fifth"]
    assert rows["framewise"]["frames"] == len(capture.traces["stress"].frames) - 4
    assert all(run["evaluation_identity"] == rows["framewise"]["evaluation_identity"] for run in rows.values())
    assert all(run["replay"]["resets"]["gap"] == 4 for run in rows.values())


def test_output_failure_is_not_a_passed_run(tmp_path, monkeypatch):
    out, capture, _ = _setup(tmp_path, monkeypatch)

    def no_space(*args, **kwargs):
        raise OSError("injected disk failure")

    monkeypatch.setattr(bench, "write_temporal_video_report", no_space)
    run, result, _ = _run(tmp_path, out, capture)
    assert result["status"] == "failed" and "injected disk failure" in result["report_error"]
    assert json.loads((run / "result.json").read_text())["status"] == "failed"
