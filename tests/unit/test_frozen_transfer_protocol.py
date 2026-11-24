from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
import json
from types import SimpleNamespace as NS

import pytest
import torch

from hybrid_drive.experiments import frozen_transfer_predict as prediction
from hybrid_drive.experiments import frozen_transfer_score as scoring
from hybrid_drive.experiments.frozen_transfer_protocol import METHODS, SCORE_ARMS, donor_anchors
from hybrid_drive.experiments.frozen_transfer_score import bind_transfer_rows, score_transfer, summarize_transfer
from hybrid_drive.fast.forecast_proposals import ForecastObject


def _bound_rows():
    rows = [{"sample_token": key, "recording": group, "ref_time_us": time}
            for key, group, time in (("a", "east", 5), ("b", "east", 6), ("c", "west", 1))]
    anchors = [{"sample_token": r["sample_token"], "recording": r["recording"]} for r in rows]
    observed = {"anchors": anchors, "frames": {r["sample_token"]: r for r in rows}}
    selected = [{**r, "frame_index": i+2, "population": "uniform"} for i, r in enumerate(rows)]
    digests = {row["sample_token"]: f"{i+1}"*64 for i, row in enumerate(rows)}
    donors = donor_anchors(anchors)
    predicted = []
    for row in rows:
        donor = donors[row["recording"]]["sample_token"]
        predicted.append({**row, "snapshot_sha256": digests[row["sample_token"]], "status": "ok", "candidates": [],
                          "donor_snapshot_id": donor, "donor_snapshot_sha256": digests[donor]})
    return observed, selected, predicted


def test_three_stage_binding_disallows_unrelated_or_reordered_rows():
    observed, selected, predicted = _bound_rows()
    assert list(bind_transfer_rows(observed, selected, predicted)) == ["east", "west"]
    with pytest.raises(ValueError, match="identity"):
        bind_transfer_rows(observed, selected[::-1], predicted)
    predicted[2]["donor_snapshot_id"], predicted[2]["donor_snapshot_sha256"] = "b", "2"*64
    with pytest.raises(ValueError, match="Camera donor"):
        bind_transfer_rows(observed, selected, predicted)


def test_unsupported_window_is_retained_and_cannot_invent_candidates():
    observed, selected, predicted = _bound_rows()
    predicted[0].update(status="out_of_envelope")
    assert bind_transfer_rows(observed, selected, predicted)["east"][1][0]["status"] == "out_of_envelope"
    predicted[0]["candidates"] = [{}]
    with pytest.raises(ValueError, match="invented"):
        bind_transfer_rows(observed, selected, predicted)


def test_bad_seal_fails_before_labels_or_input_files_are_read(tmp_path):
    raw = tmp_path / "predictions.json"
    raw.write_bytes(b"{}")
    missing = tmp_path / "does-not-exist"
    with pytest.raises(ValueError, match="seal"):
        score_transfer(raw, "0" * 64, missing, missing, missing, missing, tmp_path / "out")


def test_score_replays_selection_and_rejects_posthoc_stratum_switch(monkeypatch):
    frames = object()
    choices = [{"frame_index": 4, "population": "uniform"}, {"frame_index": 9, "population": "vulnerable"}]

    def predeclared(actual):
        assert actual is frames
        return [(4, "uniform"), (9, "vulnerable")], {}

    monkeypatch.setattr(scoring, "_select_windows", predeclared)
    scoring._verify_selection(frames, choices)
    choices[0]["population"] = "vulnerable"
    with pytest.raises(ValueError, match="fixed observation-only"):
        scoring._verify_selection(frames, choices)


def _window(recording, *, wide=False, bad=(), vulnerable=False, n=2, population="uniform"):
    pairs = []
    for i in range(n):
        left, right = f"a{i}", f"b{i}"
        # Truth: only the left plan contacts an object.
        prefer = {method: left if method in bad else right for method in METHODS}
        if wide:
            prefer["fast_gap"] = None
        pairs.append({"candidate_ids": [left, right], "vehicle_outcome": "neither_observed" if vulnerable else "left_contact",
                      "vulnerable_outcome": "left_contact" if vulnerable else "neither_observed",
                      "fast_binary_tied": True, "vulnerable_only": vulnerable, "preferred": prefer})
    return {"recording": recording, "population": population, "window_status": "complete", "pairs": pairs}


def test_window_weighting_not_pair_count_and_ties_are_not_correct():
    rows = [_window("one", n=9, wide=True), _window("two", n=1, bad=("trained",))]
    result = summarize_transfer(rows)
    profile = result["slices"]["vehicle_uniform"]
    assert (profile["pairs"], profile["windows"], profile["recordings"]) == (10, 2, 2)
    assert profile["methods"]["trained"]["correct"] == 9
    assert profile["methods"]["trained"]["window_macro_accuracy"] == .5
    assert profile["methods"]["fast_gap"]["tied"] == 9
    assert result["vehicle_transfer"]["status"] == "insufficient_support"


def test_visual_gate_needs_both_camera_controls_and_neutral_vehicle_evaluation():
    rows = []
    for recording in "abc":
        for _ in range(2):
            rows.extend((_window(recording, bad=("fast", "fast_gap", "untrained", "no_camera")),
                         _window(recording, vulnerable=True, population="vulnerable", bad=("no_camera", "shuffled_camera"))))
    result = summarize_transfer(rows)
    assert Counter(result["populations"]) == {"uniform": 6, "vulnerable": 6}
    assert result["vehicle_transfer"]["status"] == "passed_transfer_probe"
    assert result["visual_contribution_vehicle"]["status"] == "direct_camera_contribution_not_established"
    assert result["visual_contribution_vulnerable"]["status"] == "visual_signal_detected"


@dataclass(frozen=True)
class _Features:
    snapshot: object
    encoder_id: str
    feature_maps: torch.Tensor


@dataclass(frozen=True)
class _Scene:
    content_sha256: str
    plan: object
    boxes: tuple


def test_camera_and_motion_controls_change_only_the_named_input_after_fast():
    a = ForecastObject("actor", (9., 0., 1.), (2., 4., 2.), 0., (2., 5.), .8)
    candidate = NS(candidate_id="keep", trajectory=NS(interval_s=.5, poses=torch.tensor([[1., 0., 0.]])))
    planned = NS(candidates=(candidate,))
    scene = _Scene("scene", planned, (a,))
    fast = NS(scene=scene, paired_road=NS(ordered=("keep",)), base=NS(diagnostics=(NS(
        candidate_id="keep", first_overlap_s=None, min_axis_gap_m=4.5),)))
    origin = _Features(NS(snapshot_id="A"), "same", torch.ones(2, 3))
    other = _Features(NS(snapshot_id="B", content_sha256="donor"), "same", torch.full((2, 3), 2.))
    captured = []

    def readout(requests, language):
        captured.extend(requests)
        return tuple(torch.tensor([i + 1.]) for i in range(len(requests)))

    answer = prediction._score_requests(readout, readout, object(), origin, fast, other)
    original, blank, still, swapped, initial = captured
    assert original.features is initial.features is origin
    assert blank.scene is swapped.scene is original.scene is scene
    assert still.features is origin and still.scene.plan is planned and still.scene.boxes[0].velocity_xy == (0., 0.)
    assert scene.boxes[0].velocity_xy == (2., 5.)
    assert blank.features.feature_maps.count_nonzero() == 0 and swapped.features.feature_maps is other.feature_maps
    assert original.features.snapshot is blank.features.snapshot is swapped.features.snapshot
    assert answer["donor_snapshot_id"] == "B" and set(answer["candidates"][0]["margins"]) == set(SCORE_ARMS)
    assert answer["candidates"][0]["fast_axis_gap_m"] == 4.5


def test_checkpoint_identifies_exact_fit_and_exact_bytes_before_loading_model(tmp_path, monkeypatch):
    weights = b"exact immutable example"
    weight_sha = sha256(weights).hexdigest()
    fit = {"schema": prediction.FIT_SCHEMA, "revision": prediction.FIT_REVISION, "future_annotations_used": False,
           "checkpoint_sha256": weight_sha}
    manifest = json.dumps(fit).encode()
    (tmp_path / "fit.json").write_bytes(manifest)
    (tmp_path / "bridge-projector.safetensors").write_bytes(weights)
    monkeypatch.setattr(prediction, "FIT_SHA256", sha256(manifest).hexdigest())
    monkeypatch.setattr(prediction, "CHECKPOINT_SHA256", weight_sha)
    assert prediction._verified_original_fit(tmp_path) == (fit, weights)
    (tmp_path / "bridge-projector.safetensors").write_bytes(b"altered")
    with pytest.raises(ValueError, match="checkpoint"):
        prediction._verified_original_fit(tmp_path)
