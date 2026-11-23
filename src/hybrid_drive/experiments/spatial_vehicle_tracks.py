"""Associate observed boxes in source time; no interpolation or new detections."""

from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
import math
from types import MappingProxyType
from typing import Literal

from .spatial_decision_policy import PRESENT, UNCERTAIN
from .temporal_spatial_filter import DEFAULT_MAX_GAP_US
from .temporal_spatial_types import Box, FrameDecision, Reasons, TemporalInputError


_MIN_IOU = 0.10
_MAX_SIZE_RATIO = 3.0
_AMBIGUITY_MARGIN = 0.04
_DORMANT_COST = 0.05
_MAX_CENTER_SPEED = 0.35  # normalized image units per second, not vehicle velocity
_MAX_PREDICT_S = 1.0

TrackIds = tuple[int | None, int | None, int | None, int | None]


@dataclass(frozen=True, slots=True)
class TrackingRecipe:
    method: Literal["zone", "iou", "motion"]
    bridge_uncertainty: bool = False
    reject_ambiguous: bool = False

    def __post_init__(self):
        if self.method not in ("zone", "iou", "motion"):
            raise ValueError("Unknown association method")
        if type(self.bridge_uncertainty) is not bool or type(self.reject_ambiguous) is not bool:
            raise ValueError("Association options must be boolean")
        if self.method == "zone" and (self.bridge_uncertainty or self.reject_ambiguous):
            raise ValueError("Zone baseline does not use a geometry gate or bridge")


TRACK_RECIPES = MappingProxyType({
    "zone": TrackingRecipe("zone"),
    "iou": TrackingRecipe("iou"),
    "motion": TrackingRecipe("motion"),
    "iou_gap": TrackingRecipe("iou", bridge_uncertainty=True),
    "motion_gap": TrackingRecipe("motion", bridge_uncertainty=True),
    "motion_strict": TrackingRecipe("motion", reject_ambiguous=True),
})


@dataclass(frozen=True, slots=True)
class TrackedFrame:
    decision: FrameDecision
    track_ids: TrackIds
    reasons: Reasons

    def __post_init__(self):
        if not isinstance(self.decision, FrameDecision):
            raise ValueError("Expected the unchanged per-frame decision")
        if not isinstance(self.track_ids, (tuple, list)) or len(self.track_ids) != 4:
            raise ValueError("Expected four track slots")
        if not isinstance(self.reasons, (tuple, list)) or len(self.reasons) != 4:
            raise ValueError("Expected four association reasons")
        for state, track_id, reason in zip(self.decision.states, self.track_ids, self.reasons):
            if state == PRESENT:
                if type(track_id) is not int or track_id < 1:
                    raise ValueError("A PRESENT slot must have a positive integer track ID")
            elif track_id is not None:
                raise ValueError("Cannot emit a track for an unconfirmed slot")
            if not isinstance(reason, str) or not reason:
                raise ValueError("Association reasons must be nonempty strings")
        visible = tuple(track_id for track_id in self.track_ids if track_id is not None)
        if len(visible) != len(set(visible)):
            raise ValueError("One track cannot explain two slots at the same time")
        object.__setattr__(self, "track_ids", tuple(self.track_ids))
        object.__setattr__(self, "reasons", tuple(self.reasons))


@dataclass(frozen=True, slots=True)
class _Track:
    track_id: int
    band: int
    box: Box
    timestamp_us: int
    velocity: tuple[float, float] | None = None
    dormant: bool = False


class VehicleTracker:
    """Causal bounded tracker for one clip; IDs are local to that clip."""

    def __init__(self, recipe: TrackingRecipe):
        if not isinstance(recipe, TrackingRecipe):
            raise ValueError("Expected a TrackingRecipe")
        self._recipe = recipe
        self._tracks: tuple[_Track, ...] = ()
        self._previous: FrameDecision | None = None
        self._next_id = 1

    def push(self, decision: FrameDecision) -> TrackedFrame:
        if not isinstance(decision, FrameDecision):
            raise TemporalInputError("invalid_decision", "Expected a FrameDecision")
        continuous = self._check_next(decision)
        frame = decision.observation
        present = tuple(band for band, state in enumerate(decision.states) if state == PRESENT)
        previous = self._tracks if continuous else ()

        edges = _association_edges(previous, present, decision, self._recipe)
        ambiguous = set()
        if self._recipe.reject_ambiguous:
            original = edges
            edges = _unambiguous_edges(edges, _AMBIGUITY_MARGIN)
            accepted_bands = {band for _, band in edges}
            ambiguous = {band for _, band in original if band not in accepted_bands}
        assignments = _assign(edges, present)

        next_id = self._next_id
        by_id = {track.track_id: track for track in previous}
        used_ids = set()
        active = []
        ids: list[int | None] = [None] * 4
        reasons = ["uncertain" if state == UNCERTAIN else "absent" for state in decision.states]
        for band in present:
            track = by_id.get(assignments.get(band))
            box = decision.boxes[band]
            if track is None:
                tracked = _Track(next_id, band, box, frame.timestamp_us)
                next_id += 1
                reason = "ambiguous_new_track" if band in ambiguous else "new_track"
            else:
                used_ids.add(track.track_id)
                tracked = _Track(
                    track.track_id, band, box, frame.timestamp_us,
                    _observed_velocity(track, box, frame.timestamp_us) if self._recipe.method == "motion" else None,
                )
                reason = _link_reason(track, band)
            active.append(tracked)
            ids[band] = tracked.track_id
            reasons[band] = reason

        # A dormant track owns the last observed geometry privately. It survives
        # only the first observed UNCERTAIN at that slot, and is never emitted.
        if self._recipe.bridge_uncertainty:
            for track in previous:
                if (
                    track.track_id not in used_ids
                    and not track.dormant
                    and decision.states[track.band] == UNCERTAIN
                ):
                    active.append(_Track(
                        track.track_id, track.band, track.box, track.timestamp_us,
                        track.velocity, dormant=True,
                    ))

        output = TrackedFrame(decision, tuple(ids), tuple(reasons))
        self._tracks = tuple(active)
        self._previous = decision
        self._next_id = next_id
        return output

    def _check_next(self, decision):
        if self._previous is None:
            return False
        last = self._previous.observation
        obs = decision.observation
        if (last.recording_id, last.clip_id) != (obs.recording_id, obs.clip_id):
            raise TemporalInputError("wrong_clip", "Tracker already belongs to another recording/clip")
        if (
            obs.source_index <= last.source_index
            or obs.timestamp_us <= last.timestamp_us
            or obs.sample_id == last.sample_id
        ):
            raise TemporalInputError("chronology", "Tracker requires advancing source time and indices")
        return (
            obs.source_index == last.source_index + 1
            and obs.timestamp_us - last.timestamp_us <= DEFAULT_MAX_GAP_US
            and decision.reset_reason is None
        )


def replay_vehicle_tracks(decisions: Sequence[FrameDecision], recipe: TrackingRecipe) -> tuple[TrackedFrame, ...]:
    """Keep original decisions, resetting the identity namespace at each clip cut."""
    if not isinstance(recipe, TrackingRecipe):
        raise ValueError("Expected a TrackingRecipe")
    tracker = None
    previous = None
    recording_ids, clip_keys, sample_ids = set(), set(), set()
    outputs = []
    for decision in decisions:
        if not isinstance(decision, FrameDecision):
            raise TemporalInputError("invalid_decision", "Expected a FrameDecision")
        obs = decision.observation
        if obs.sample_id in sample_ids:
            raise TemporalInputError("chronology", "Duplicate sample ID in tracking replay")
        sample_ids.add(obs.sample_id)
        key = (obs.recording_id, obs.clip_id)
        if previous is None or key != (previous.recording_id, previous.clip_id):
            if key in clip_keys:
                raise TemporalInputError("chronology", "A clip must occupy one contiguous replay block")
            clip_keys.add(key)
            if previous is None or obs.recording_id != previous.recording_id:
                if obs.recording_id in recording_ids:
                    raise TemporalInputError("chronology", "A recording must occupy one contiguous replay block")
                recording_ids.add(obs.recording_id)
            elif obs.source_index <= previous.source_index or obs.timestamp_us <= previous.timestamp_us:
                raise TemporalInputError("chronology", "Source time must advance between clips in a recording")
            tracker = VehicleTracker(recipe)
        outputs.append(tracker.push(decision))
        previous = obs
    return tuple(outputs)


def _association_edges(previous, present, decision, recipe):
    edges = {}
    for track in previous:
        for band in present:
            if recipe.method == "zone":
                if band == track.band:
                    edges[track.track_id, band] = 0.0
                continue
            box = decision.boxes[band]
            if abs(band - track.band) > 1 or not _comparable_size(track.box, box):
                continue
            measured = _iou(track.box, box)
            gated = affinity = measured
            if recipe.method == "motion" and track.velocity is not None:
                projected = _iou(_translate(track, decision.observation.timestamp_us), box)
                # Preserve the measured fallback gate, but prefer projected
                # overlap at equal measured and projected quality.
                gated = max(measured, projected)
                affinity = max(0.9 * measured, projected)
            if gated >= _MIN_IOU:
                edges[track.track_id, band] = 1.0 - affinity + _DORMANT_COST * track.dormant
    return edges


def _unambiguous_edges(edges, margin):
    # Require separation in both directions, against the entire original
    # candidate set. Removing one competitor cannot make another clear.
    return {
        (track, band): cost
        for (track, band), cost in edges.items()
        if all(
            alternative - cost >= margin
            for (other_track, other_band), alternative in edges.items()
            if (other_track, other_band) != (track, band) and (other_track == track or other_band == band)
        )
    }


def _assign(edges, bands):
    """Largest feasible one-to-one match, then minimum cost, then lowest IDs."""
    if not edges:
        return {}
    track_ids = sorted({track_id for track_id, _ in edges})
    bits = {track_id: 1 << i for i, track_id in enumerate(track_ids)}
    missing = track_ids[-1] + 1
    candidates = tuple(
        tuple((track_id, edges[track_id, band]) for track_id in track_ids if (track_id, band) in edges)
        for band in bands
    )

    @lru_cache(maxsize=None)
    def solve(slot, used):
        if slot == len(bands):
            return 0, 0.0, ()
        count, cost, rest = solve(slot + 1, used)
        best = count, cost, (missing, *rest)
        for track_id, distance in candidates[slot]:
            bit = bits[track_id]
            if used & bit:
                continue
            count, cost, rest = solve(slot + 1, used | bit)
            best = min(best, (count - 1, cost + distance, (track_id, *rest)))
        return best

    _, _, choice = solve(0, 0)
    return {band: track_id for band, track_id in zip(bands, choice) if track_id != missing}


def _link_reason(track, band):
    if track.dormant:
        return "uncertainty_bridge_cross_zone" if track.band != band else "uncertainty_bridge"
    return "cross_zone" if track.band != band else "linked"


def _observed_velocity(track, box, timestamp_us):
    seconds = (timestamp_us - track.timestamp_us) / 1_000_000
    x0, y0 = _center(track.box)
    x1, y1 = _center(box)
    vx, vy = (x1 - x0) / seconds, (y1 - y0) / seconds
    speed = math.hypot(vx, vy)
    scale = _MAX_CENTER_SPEED / speed if speed > _MAX_CENTER_SPEED else 1.0
    return vx * scale, vy * scale


def _translate(track, timestamp_us):
    seconds = min(_MAX_PREDICT_S, (timestamp_us - track.timestamp_us) / 1_000_000)
    vx, vy = track.velocity
    shift_x, shift_y = vx * seconds, vy * seconds
    x1, y1, x2, y2 = track.box
    return x1 + shift_x, y1 + shift_y, x2 + shift_x, y2 + shift_y


def _center(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2) / 2


def _comparable_size(a, b):
    for first, last in ((0, 2), (1, 3)):
        wa, wb = a[last] - a[first], b[last] - b[first]
        if max(wa, wb) > _MAX_SIZE_RATIO * min(wa, wb):
            return False
    return True


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0.0
