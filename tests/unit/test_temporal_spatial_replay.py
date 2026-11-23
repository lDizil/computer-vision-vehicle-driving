from dataclasses import replace

import pytest

from hybrid_drive.experiments.temporal_spatial_filter import RECIPES
from hybrid_drive.experiments.temporal_spatial_replay import (
    BAD_INPUTS, FRAME_DROP, LATE, STRESS_SCENARIOS, TIMELY, replay_temporal, verify_stress_receipts,
)
from hybrid_drive.experiments.temporal_spatial_types import FrameObservation
from hybrid_drive.experiments.spatial_decision_policy import ZoneDecisionPolicy


POLICY = ZoneDecisionPolicy(-0.2, 0.5)


def _frames(clip, *, n=12, start=20, epoch=1_000_000, recording="recording"):
    logits = (0.9, 0.85, 0.25, 0.65, -0.3, 0.7)
    for i in range(n):
        offset = (i % 4) * 0.004
        boxes = tuple((0.025 + band * 0.25 + offset, 0.25, 0.18 + band * 0.25 + offset, 0.65) for band in range(4))
        yield FrameObservation(recording, clip, f"{recording}-{clip}-{i}", start+i, epoch + 500_000*i,
                               tuple(logits[(i+band) % len(logits)] for band in range(4)), boxes)


@pytest.mark.parametrize("name", list(RECIPES))
def test_all_methods_have_accountable_and_atomic_failures(name):
    frames = tuple(_frames("clip-1")) + tuple(_frames("clip-2", start=45, epoch=30_000_000))
    recipe = RECIPES[name]
    clean = replay_temporal(frames, POLICY, recipe)
    stresses = {scenario: replay_temporal(frames, POLICY, recipe, scenario=scenario) for scenario in STRESS_SCENARIOS}
    assert all(verify_stress_receipts(clean, stresses).values())
    assert clean.resets == {"initial": 2}
    assert stresses[LATE].rejected == {"stale": 4}
    assert stresses[FRAME_DROP].resets == {"gap": 4, "initial": 2}
    assert stresses[BAD_INPUTS].rejected == {
        "chronology": 4, "early": 2, "invalid_availability": 2, "invalid_observation": 2, "wrong_clip": 2,
    }
    for mode in (LATE, FRAME_DROP):
        assert all((row.observation.source_index - 20) not in (4, 9)
                   for row in stresses[mode].decisions if row.observation.clip_id == "clip-1")
        for row in stresses[mode].decisions:
            if row.reset_reason:
                assert row.boxes == row.observation.boxes and row.scores == row.observation.scores
    assert stresses[TIMELY].decisions == stresses[BAD_INPUTS].decisions == clean.decisions


def test_recipe_history_does_not_cross_a_clip_cut_or_recording():
    a = tuple(_frames("clip", n=5, recording="a"))
    b = tuple(_frames("clip", n=5, recording="b"))
    args = (POLICY, RECIPES["confirm_box"])
    joined = replay_temporal(a+b, *args)
    assert joined.decisions == replay_temporal(a, *args).decisions + replay_temporal(b, *args).decisions


def test_rejects_duplicate_identity_and_noncontiguous_clip_blocks():
    rows = tuple(_frames("c"))
    extra = tuple(_frames("e"))
    args = (POLICY, RECIPES["box_ema"])
    with pytest.raises(ValueError, match="unique"):
        replay_temporal(rows + (rows[0],), *args)
    with pytest.raises(ValueError, match="contiguous block"):
        replay_temporal(rows[:3] + extra + rows[3:], *args)
    with pytest.raises(ValueError, match="advance"):
        replay_temporal((rows[1], rows[0], *rows[2:]), *args)
    with pytest.raises(ValueError, match="Unknown"):
        replay_temporal(rows, *args, scenario="unknown")


def test_fault_verifier_checks_observable_state_not_only_counters():
    rows = tuple(_frames("c"))
    args = (POLICY, RECIPES["score_box"])
    clean = replay_temporal(rows, *args)
    stresses = {s: replay_temporal(rows, *args, scenario=s) for s in STRESS_SCENARIOS}
    bad = stresses[BAD_INPUTS]
    first = bad.decisions[0]
    wrong = replace(first, states=(0, 0, 0, 0))
    stresses[BAD_INPUTS] = replace(bad, decisions=(wrong, *bad.decisions[1:]))
    with pytest.raises(ValueError, match="rejected mutation changed"):
        verify_stress_receipts(clean, stresses)
