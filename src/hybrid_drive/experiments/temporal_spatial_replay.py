"""Deterministic source-clock replay with explicit faults; never a motion decision."""

from collections import Counter
from dataclasses import dataclass, replace
from itertools import groupby

from .temporal_spatial_filter import (
    DEFAULT_MAX_AGE_US,
    DEFAULT_MAX_GAP_US,
    TemporalSpatialFilter,
)
from .temporal_spatial_types import FrameDecision, FrameObservation, TemporalInputError


CLEAN = "clean"
FRAME_DROP = "drop_every_fifth"
LATE = "stale_every_fifth"
TIMELY = "timely_delivery_jitter"
BAD_INPUTS = "invalid_insertions"
STRESS_SCENARIOS = (FRAME_DROP, LATE, TIMELY, BAD_INPUTS)
_TIMELY_OFFSETS = (0, 50_000, 180_000, 100_000)


@dataclass(frozen=True)
class ReplayResult:
    scenario: str
    decisions: tuple[FrameDecision, ...]
    source_frames: int
    dropped: int
    withheld_stale: int
    rejected: dict
    resets: dict

    def receipt(self):
        return {
            "scenario": self.scenario, "source_frames": self.source_frames, "accepted": len(self.decisions),
            "dropped": self.dropped, "withheld_stale": self.withheld_stale,
            "rejected": self.rejected.copy(), "resets": self.resets.copy(),
        }


def replay_temporal(
    frames, policy, recipe, *, scenario=CLEAN, max_gap_us=DEFAULT_MAX_GAP_US, max_age_us=DEFAULT_MAX_AGE_US,
):
    if scenario not in (CLEAN, *STRESS_SCENARIOS):
        raise ValueError("Unknown offline temporal scenario")
    frames = tuple(frames)
    if not frames or any(not isinstance(frame, FrameObservation) for frame in frames):
        raise ValueError("Replay needs source-ordered observations")
    if len({f.sample_id for f in frames}) != len(frames):
        raise ValueError("Source sample identities must be unique")

    decisions, clips = [], set()
    dropped, stale = 0, 0
    rejected, resets = Counter(), Counter()
    for identity, chunk in groupby(frames, key=lambda f: (f.recording_id, f.clip_id)):
        if identity in clips:
            raise ValueError("Each replay clip must occupy one contiguous block")
        clips.add(identity)
        values = tuple(chunk)
        if any(
            right.source_index <= left.source_index or right.timestamp_us <= left.timestamp_us
            for left, right in zip(values, values[1:])
        ):
            raise ValueError("A replay clip must advance indices and timestamps before fault injection")
        actor = TemporalSpatialFilter(policy, recipe, max_gap_us=max_gap_us, max_age_us=max_age_us)
        for i, frame in enumerate(values):
            if (i + 1) % 5 == 0 and scenario == FRAME_DROP:
                dropped += 1
                continue
            if (i + 1) % 5 == 0 and scenario == LATE:
                _must_reject(actor, frame, frame.timestamp_us + max_age_us + 1, "stale", rejected)
                stale += 1
                continue
            if scenario == BAD_INPUTS and i == 2:
                _guard_faults(actor, values[:2], frame, rejected)
            # Never go past the max admission age even for a small custom budget.
            offset = min(_TIMELY_OFFSETS[i % len(_TIMELY_OFFSETS)], max_age_us) if scenario == TIMELY else 0
            decision = actor.push(frame, available_at_us=frame.timestamp_us + offset)
            if decision.reset_reason:
                if decision.boxes != frame.boxes or decision.scores != frame.scores:
                    raise ValueError("A cold observation inherited old scores or geometry")
                resets[decision.reset_reason] += 1
            decisions.append(decision)

    if len(decisions) + dropped + stale != len(frames):
        raise ValueError("An omitted observation was not accounted for")
    return ReplayResult(
        scenario, tuple(decisions), len(frames), dropped, stale, dict(sorted(rejected.items())), dict(sorted(resets.items())),
    )


def _guard_faults(actor, history, current, rejects):
    now = current.timestamp_us
    # Each mutation is followed by the original frame. The suite verifies
    # that the eventual accepted stream is identical to a clean replay.
    _must_reject(actor, history[-1], now, "chronology", rejects)
    _must_reject(actor, history[0], now, "chronology", rejects)
    other = "unrelated-clip" if current.clip_id != "unrelated-clip" else "another-clip"
    _must_reject(actor, replace(current, clip_id=other), now, "wrong_clip", rejects)
    _must_reject(actor, current, now - 1, "early", rejects)
    _must_reject(actor, current, 0.5, "invalid_availability", rejects)
    _must_reject(actor, object(), now, "invalid_observation", rejects)


def _must_reject(actor, frame, now, reason, rejects):
    try:
        actor.push(frame, available_at_us=now)
    except TemporalInputError as exc:
        if exc.reason != reason:
            raise ValueError(f"Expected {reason} rejection, got {exc.reason}") from exc
        rejects[reason] += 1
        return
    raise ValueError(f"Temporal filter accepted {reason} input")


def verify_stress_receipts(clean, stresses):
    """Check causal equivalences using outputs, not just injected counters."""
    if set(stresses) != set(STRESS_SCENARIOS):
        raise ValueError("Expected all declared stress scenarios")
    if clean.scenario != CLEAN or clean.dropped or clean.withheld_stale or clean.rejected:
        raise ValueError("Stress baseline was not clean")
    drop, late = stresses[FRAME_DROP], stresses[LATE]
    jitter, bad = stresses[TIMELY], stresses[BAD_INPUTS]
    if (
        not drop.dropped or drop.dropped != late.withheld_stale
        or any(run.source_frames != clean.source_frames for run in stresses.values())
        or late.rejected != {"stale": late.withheld_stale}
        or drop.decisions != late.decisions
        or drop.resets != late.resets
    ):
        raise ValueError("Dropping vs rejecting the same stale observations produced different outputs")
    if jitter.decisions != clean.decisions or jitter.rejected:
        raise ValueError("A timely source-clock delivery change modified camera evidence")
    if bad.decisions != clean.decisions or not bad.rejected:
        raise ValueError("A rejected mutation changed the subsequent stream")
    return {
        "drop_and_late_equal": True,
        "timely_jitter_equal": True,
        "invalid_insertions_atomic": True,
        "all_dropped_or_stale_accounted_for": True,
    }
