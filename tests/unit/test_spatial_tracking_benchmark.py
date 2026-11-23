from dataclasses import replace
import hashlib
import json
from types import SimpleNamespace

import pytest

from hybrid_drive.experiments import spatial_tracking_benchmark as bench
from hybrid_drive.experiments.protocol import content_hash
from hybrid_drive.experiments.temporal_spatial_types import FrameObservation


def _prepare(role, path, model_hash):
    observations, labels, aliases = [], [], {}
    for i, (band, center) in enumerate(((0, .215), (0, .235), (1, .255), (1, .285), (1, .315), (1, .345))):
        sid = f"{role}-frame-{i}"
        corners = [(.04 + b * .25, .23, .18 + b * .25, .50) for b in range(4)]
        corners[band] = (center - .055, .25, center + .055, .52)
        scores = tuple(.8 if b == band else -.8 for b in range(4))
        observation = FrameObservation(role, f"{role}-clip", sid, i, i * 500_000 + 1_200_000, scores, corners)
        row = {"sample_id": sid, "boxes_2d": tuple(c if b == band else None for b, c in enumerate(corners))}
        labels.append(row)
        aliases[sid] = tuple(hashlib.sha256(role.encode()).hexdigest() if b == band else None for b in range(4))
        observations.append(observation)
    frames = tuple(observations)
    pack_hash = hashlib.sha256(path.name.encode()).hexdigest()
    pack = SimpleNamespace(root=path, frames=frames, manifest_hash=pack_hash,
                           manifest={"role": role, "labels": labels, "aliases": aliases})
    trace = SimpleNamespace(role=role, frames=frames, pack_hash=pack_hash, model_hash=model_hash)
    return pack, trace


def _setup(tmp_path, monkeypatch):
    output = tmp_path / "out"
    all_roles = (*bench.ROLES, bench.TRACKING_ROLE)
    model_hash = "b" * 64
    pairs = {r: _prepare(r, tmp_path / r, model_hash) for r in all_roles}
    previous = SimpleNamespace(
        packs={r: pairs[r][0] for r in bench.ROLES},
        traces={r: pairs[r][1] for r in bench.ROLES},
        manifest={"model_hash": model_hash,
                  "model_receipt": {"calibrated_policy": {"absent_below": -0.4, "present_at_or_above": .4}}},
        manifest_sha256="a" * 64,
    )
    truth = {r: bench.TrackingTruthFile(tmp_path / f"{r}.json", str(i) * 64)
             for i, r in enumerate(all_roles, 1)}
    opened = []

    def read_truth(pack, path, *, expected_sha256):
        role = pack.manifest["role"]
        opened.append(role)
        assert (path, expected_sha256) == (truth[role].path, truth[role].sha256)
        if role == bench.TRACKING_ROLE:
            files = list(output.glob("*/development-gate.json"))
            assert len(files) == 1
            gate = json.loads(files[0].read_text())
            assert gate["selection"]["name"] in bench.TRACK_RECIPES and gate["development_hash"]
        return pack.manifest["aliases"]

    def gallery(path, pack, targets, predictions, *, selected):
        assert pack is pairs[bench.TRACKING_ROLE][0] and len(targets) == len(pack.frames)
        assert set(predictions) == set(bench.TRACK_RECIPES)
        assert selected in predictions
        path.write_text("tracking comparison")

    monkeypatch.setattr(bench, "read_track_truth", read_truth)
    monkeypatch.setattr(bench, "read_spatial_targets", lambda root, manifest, role: manifest["labels"])
    monkeypatch.setattr(bench, "write_tracking_gallery", gallery)
    return previous, pairs[bench.TRACKING_ROLE], truth, output, opened


def _run(args):
    old, (pack, trace), truth, output, _ = args
    return bench.run_tracking_benchmark(
        old, pack, trace, truth, output, revision="c" * 40, tracking_capture_sha256="d" * 64,
    )


def test_gate_precedes_opening_fresh_truth_and_geometry_is_fixed(tmp_path, monkeypatch):
    args = _setup(tmp_path, monkeypatch)
    directory, result, digest = _run(args)
    assert result["status"] == "passed", result.get("error")
    assert args[-1] == [*bench.ROLES, bench.TRACKING_ROLE]
    matrix = result["matrix"]
    assert matrix["configurations"] == 12 and matrix["holdout_frames"] == 6
    assert matrix["holdout_diagnostic"]["evaluated_recipe"] == matrix["selection"]["name"]
    assert matrix["holdout"]["zone"]["cross_zone_links"] == 1
    assert matrix["holdout"]["zone"]["cross_zone_correct"] == 0
    assert matrix["holdout"]["iou"]["cross_zone_correct"] == 1
    assert len({e["evaluation_identity"] for e in matrix["holdout"].values()}) == 1
    assert len({e["detections_matched"] for e in matrix["holdout"].values()}) == 1
    assert content_hash(matrix) == result["matrix_hash"]
    assert hashlib.sha256((directory / "result.json").read_bytes()).hexdigest() == digest
    assert 'href="tracking.html"' in (directory / "report.html").read_text()


def test_invalid_model_bound_to_final_trace_never_opens_targets(tmp_path, monkeypatch):
    old, (pack, trace), truth, output, opened = _setup(tmp_path, monkeypatch)
    trace = SimpleNamespace(**{**vars(trace), "model_hash": "e" * 64})
    with pytest.raises(ValueError, match="same camera model"):
        bench.run_tracking_benchmark(old, pack, trace, truth, output,
                                     revision="c"*40, tracking_capture_sha256="d"*64)
    assert not opened


def test_changed_development_gate_never_opens_final_sidecar(tmp_path, monkeypatch):
    args = _setup(tmp_path, monkeypatch)
    write = bench.write_json

    def faulty(path, data):
        if path.name == "development-gate.json":
            data = {**data, "development_hash": "tampered"}
        write(path, data)

    monkeypatch.setattr(bench, "write_json", faulty)
    _, r, _ = _run(args)
    assert r["status"] == "failed" and "did not match the frozen gate" in r["error"]
    assert args[-1] == list(bench.ROLES)


def test_identity_arm_is_not_allowed_to_recompute_frame_choices(tmp_path, monkeypatch):
    args = _setup(tmp_path, monkeypatch)
    actual = bench.replay_vehicle_tracks

    def mutating(frames, recipe):
        rows = list(actual(frames, recipe))
        rows[0] = replace(rows[0], decision=replace(rows[0].decision))
        return tuple(rows)

    monkeypatch.setattr(bench, "replay_vehicle_tracks", mutating)
    _, r, _ = _run(args)
    assert r["status"] == "failed" and "must not replace" in r["error"]
    assert args[-1] == list(bench.ROLES)
