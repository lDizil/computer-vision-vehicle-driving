"""Identity diagnostics on frozen, causal four-zone camera detections."""

from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from hashlib import sha256
from itertools import groupby
import json
from math import fsum, isfinite
from numbers import Integral, Real

from .spatial_decision_policy import PRESENT
from .temporal_spatial_filter import DEFAULT_MAX_GAP_US
from .temporal_spatial_metrics import TemporalTarget
from .temporal_spatial_types import Box, FrameDecision


_BANDS = 4
_MIN_IDF1_GAIN = 0.005
_IDENTITY_SCHEMA = "hybrid-drive-fixed-frame-track-metrics-v1"
_FIXED_COUNTS = (
    "frames", "labeled_boxes", "output_boxes", "detections_matched",
    "adjacent_links", "cross_zone_links",
)


@dataclass(frozen=True, slots=True)
class TrackingTarget:
    track_ids: tuple[str | None, str | None, str | None, str | None]
    boxes: tuple[Box | None, Box | None, Box | None, Box | None]

    def __post_init__(self):
        ids = _four(self.track_ids, "Target track IDs")
        known = [alias for alias in ids if alias is not None]
        if any(
            not isinstance(alias, str) or not 0 < len(alias) <= 256
            or not alias.strip() or any(c in alias for c in ("\0", "\n", "\r"))
            for alias in known
        ):
            raise ValueError("Expected nonempty bounded target track aliases")
        if len(known) != len(set(known)):
            raise ValueError("A target track ID cannot occupy two zones of one frame")
        projected = TemporalTarget(tuple(alias is not None for alias in ids), self.boxes)
        object.__setattr__(self, "track_ids", ids)
        object.__setattr__(self, "boxes", projected.boxes)


def match_tracking_frame(
    decision: FrameDecision, target: TrackingTarget, *, min_iou=0.5,
) -> tuple[tuple[int, int], ...]:
    """Return deterministic (prediction band, target band) pairs before using any IDs."""
    threshold = _checked_min_iou(min_iou)
    if not isinstance(decision, FrameDecision) or not isinstance(target, TrackingTarget):
        raise ValueError("Expected FrameDecision and TrackingTarget for fixed frame matching")
    active = tuple(band for band, state in enumerate(decision.states) if state == PRESENT)
    labeled = tuple(band for band, alias in enumerate(target.track_ids) if alias is not None)
    return _match_frame(decision.boxes, active, target.boxes, labeled, threshold)


def _checked_min_iou(value):
    if not isinstance(value, Real) or isinstance(value, bool) or not 0 < value <= 1:
        raise ValueError("Expected a minimum IoU in (0, 1]")
    return float(value)


def evaluate_tracking(frames, targets: Mapping[str, TrackingTarget], *, min_iou=0.5) -> dict:
    """Score identity labels without changing which detections match spatially.

    Fixed frame matches maximize admissible pair count, then total IoU, with a
    stable band-order tie-break. Predicted IDs do not participate. Within each
    uninterrupted part of a clip, choose the one-to-one trajectory mapping that
    retains the most of these fixed matches. The resulting fixed-match IDF1 is
    a diagnostic, not the official MOTChallenge evaluation.

    Adjacent links count matched GT identities visible at both ends of an
    adjacent edge. A wrong link is one emitted ID with different matched GT
    aliases on consecutive emissions, at most one observed frame apart.
    Unmatched emissions do not validate an identity link. Cuts, skipped source
    indices and long time gaps break both types of links and the ID assignment.
    """
    threshold = _checked_min_iou(min_iou)
    rows = _checked_rows(frames)
    if not isinstance(targets, Mapping) or set(targets) != {row.source.sample_id for row in rows}:
        raise ValueError("Expected exactly one tracking target per evaluated sample ID")
    if any(not isinstance(target, TrackingTarget) for target in targets.values()):
        raise ValueError("Expected TrackingTarget values by sample ID")

    total, per_recording = _Counts(), {}
    for recording, segments in groupby(_segments(rows), key=lambda segment: segment[0].source.recording_id):
        observed = _Counts()
        for segment in segments:
            observed.add(_measure_segment(segment, targets, threshold))
        total.add(observed)
        per_recording[recording] = observed.summary()

    summary = total.summary()
    summary["per_recording"] = per_recording
    # Include the whole frozen decision rather than only the number of
    # detections. Swapped boxes or labels can preserve all aggregate counts.
    signature = [[asdict(row.decision), asdict(targets[row.source.sample_id])] for row in rows]
    payload = [_IDENTITY_SCHEMA, DEFAULT_MAX_GAP_US, float(min_iou), signature]
    summary["evaluation_identity"] = sha256(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    return summary


def select_tracking_recipe(dev_results, *, baseline_name="zone") -> dict:
    """Prefer a better development identity assignment without more wrong links."""
    if not isinstance(dev_results, Mapping) or baseline_name not in dev_results:
        raise ValueError("Development metrics must include the zone baseline")
    base = dev_results[baseline_name]
    assessments, accepted = {}, []
    for name, trial in sorted(dev_results.items()):
        if name == baseline_name:
            continue
        if not base.get("evaluation_identity") or trial.get("evaluation_identity") != base["evaluation_identity"]:
            assessments[name] = {"eligible": False, "reasons": ["different_frozen_geometry_or_targets"]}
            continue
        if any(key not in base or trial.get(key) != base[key] for key in _FIXED_COUNTS):
            assessments[name] = {"eligible": False, "reasons": ["different_fixed_counts"]}
            continue
        first, second = base["fixed_match_idf1"], trial["fixed_match_idf1"]
        gain = second - first if first is not None and second is not None else None
        wrong = trial["wrong_links"] - base["wrong_links"]
        switches = trial["adjacent_switches"] - base["adjacent_switches"]
        reasons = []
        if gain is None or not isfinite(gain) or gain < _MIN_IDF1_GAIN - 1e-12:
            reasons.append("insufficient_idf1_improvement")
        if wrong > 0:
            reasons.append("more_wrong_links")
        if switches > 0:
            reasons.append("more_adjacent_switches")
        assessments[name] = {
            "eligible": not reasons, "reasons": reasons, "idf1_change": gain,
            "extra_wrong_links": wrong, "extra_adjacent_switches": switches,
        }
        if not reasons:
            accepted.append((-second, trial["wrong_links"], trial["adjacent_switches"], name))
    return {
        "name": min(accepted)[-1] if accepted else baseline_name,
        "baseline_name": baseline_name,
        "summary": {
            "scope": "development_only",
            "minimum_idf1_gain": _MIN_IDF1_GAIN,
            "maximum_extra_wrong_links": 0,
            "maximum_extra_adjacent_switches": 0,
            "tie_break": ["higher_fixed_match_idf1", "fewer_wrong_links", "fewer_adjacent_switches", "lexical_recipe_name"],
        },
        "candidates": assessments,
    }


@dataclass(frozen=True, slots=True)
class _Row:
    decision: FrameDecision
    track_ids: tuple[int | None, ...]

    @property
    def source(self):
        return self.decision.observation


def _checked_rows(frames):
    try:
        frames = iter(frames)
    except TypeError as exc:
        raise ValueError("Expected a sequence of tracked frames") from exc
    rows = []
    sample_ids, seen_recordings, seen_clips = set(), set(), set()
    previous = None
    for tracked in frames:
        decision = getattr(tracked, "decision", None)
        if not isinstance(decision, FrameDecision):
            raise ValueError("Expected tracking output with its original FrameDecision")
        ids = _four(getattr(tracked, "track_ids", None), "Predicted track IDs")
        for state, number in zip(decision.states, ids):
            active = number is not None
            if active != (state == PRESENT):
                raise ValueError("Only PRESENT detections must have track IDs")
            if active and (
                not isinstance(number, Integral) or isinstance(number, bool) or not 0 < number < 2**63
            ):
                raise ValueError("Predicted track IDs must be positive bounded integers")
        active_ids = [number for number in ids if number is not None]
        if len(active_ids) != len(set(active_ids)):
            raise ValueError("A predicted track ID cannot occupy two zones of one frame")
        row = _Row(decision, tuple(int(number) if number is not None else None for number in ids))
        source = row.source
        if source.sample_id in sample_ids:
            raise ValueError("Duplicate evaluated sample ID")
        sample_ids.add(source.sample_id)
        if previous is None or source.recording_id != previous.recording_id:
            if source.recording_id in seen_recordings:
                raise ValueError("Each recording must occupy one source block")
            seen_recordings.add(source.recording_id)
        elif source.source_index <= previous.source_index or source.timestamp_us <= previous.timestamp_us:
            raise ValueError("Source indices and timestamps must increase within a recording")

        clip_key = source.recording_id, source.clip_id
        if previous is None or clip_key != (previous.recording_id, previous.clip_id):
            if clip_key in seen_clips:
                raise ValueError("Each clip must occupy one source block within its recording")
            seen_clips.add(clip_key)
        previous = source
        rows.append(row)
    if not rows:
        raise ValueError("Expected at least one tracked frame")
    return rows


def _four(values, label):
    if isinstance(values, (str, bytes, Mapping)):
        raise ValueError(f"{label} must have exactly four entries")
    try:
        result = tuple(values)
    except TypeError as exc:
        raise ValueError(f"{label} must have exactly four entries") from exc
    if len(result) != _BANDS:
        raise ValueError(f"{label} must have exactly four entries")
    return result


def _segments(rows):
    chunk, previous = [], None
    for row in rows:
        source = row.source
        if previous is not None and (
            (source.recording_id, source.clip_id) != (previous.recording_id, previous.clip_id)
            or source.source_index != previous.source_index + 1
            or source.timestamp_us - previous.timestamp_us > DEFAULT_MAX_GAP_US
        ):
            yield chunk
            chunk = []
        chunk.append(row)
        previous = source
    yield chunk


@dataclass
class _Counts:
    frames: int = 0
    labeled_boxes: int = 0
    output_boxes: int = 0
    detections_matched: int = 0
    idtp: int = 0
    adjacent_links: int = 0
    adjacent_switches: int = 0
    cross_zone_links: int = 0
    cross_zone_correct: int = 0
    wrong_links: int = 0
    inspected_links: int = 0

    def add(self, other):
        for field, value in vars(other).items():
            setattr(self, field, getattr(self, field) + value)

    def summary(self):
        denominator = self.labeled_boxes + self.output_boxes
        return {
            **vars(self),
            "fixed_match_idf1": 2 * self.idtp / denominator if denominator else None,
            "idfp": self.output_boxes - self.idtp,
            "idfn": self.labeled_boxes - self.idtp,
        }


def _measure_segment(rows, targets, min_iou):
    counts = _Counts()
    occurrences = Counter()
    previous_match = {}
    last_emission = {}
    for row in rows:
        target = targets[row.source.sample_id]
        active = tuple(band for band, state in enumerate(row.decision.states) if state == PRESENT)
        labeled = tuple(band for band, alias in enumerate(target.track_ids) if alias is not None)
        pairs = _match_frame(row.decision.boxes, active, target.boxes, labeled, min_iou)
        counts.frames += 1
        counts.labeled_boxes += len(labeled)
        counts.output_boxes += len(active)
        counts.detections_matched += len(pairs)

        matched = {}
        emitted = dict.fromkeys(row.track_ids[band] for band in active)
        for band, truth_band in pairs:
            number, alias = row.track_ids[band], target.track_ids[truth_band]
            matched[alias] = (number, truth_band)
            emitted[number] = alias
            occurrences[number, alias] += 1
            if alias in previous_match:
                prev_number, prev_band = previous_match[alias]
                preserved = prev_number == number
                counts.adjacent_links += 1
                counts.adjacent_switches += not preserved
                if prev_band != truth_band:
                    counts.cross_zone_links += 1
                    counts.cross_zone_correct += preserved

        index = row.source.source_index
        for number, alias in emitted.items():
            prior = last_emission.get(number)
            if alias is not None and prior is not None:
                prev_index, prev_alias = prior
                if prev_alias is not None and index - prev_index <= 2:
                    counts.inspected_links += 1
                    counts.wrong_links += prev_alias != alias
            last_emission[number] = index, alias
        previous_match = matched

    counts.idtp = _maximum_cooccurrence(occurrences)
    return counts


def _match_frame(output, active, truth, labeled, min_iou):
    # Four slots allow exhaustive matching with a transparent deterministic
    # tie-break. No IDs or scores appear in the edge or ranking expressions.
    edges = {
        band: tuple((j, iou) for j in labeled if (iou := _iou(output[band], truth[j])) >= min_iou)
        for band in active
    }
    best = (0, 0.0, ())

    def visit(position, used, pairs, overlaps):
        nonlocal best
        if position == len(active):
            best = min(best, (-len(pairs), -fsum(overlaps), pairs))
            return
        band = active[position]
        visit(position + 1, used, pairs, overlaps)
        for j, overlap in edges[band]:
            if not (used & (1 << j)):
                visit(position + 1, used | (1 << j), (*pairs, (band, j)), (*overlaps, overlap))

    visit(0, 0, (), ())
    return best[2]


def _iou(a, b):
    intersection = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union


def _maximum_cooccurrence(occurrences):
    if not occurrences:
        return 0
    output = sorted({number for number, _ in occurrences})
    truth = sorted({alias for _, alias in occurrences})
    if len(output) <= len(truth):
        weights = [[occurrences.get((number, alias), 0) for alias in truth] for number in output]
    else:
        weights = [[occurrences.get((number, alias), 0) for number in output] for alias in truth]
    return _max_rectangular_assignment(weights)


def _max_rectangular_assignment(weights):
    # Shortest augmenting path variant of Hungarian for integer weights,
    # with rows <= columns. Dummy-equivalent zero edges permit trajectories
    # having no shared detections and cannot raise the identity score.
    n, m = len(weights), len(weights[0])
    row_dual, col_dual = [0] * (n + 1), [0] * (m + 1)
    match, via = [0] * (m + 1), [0] * (m + 1)
    for row in range(1, n + 1):
        match[0] = row
        slack, used = [None] * (m + 1), [False] * (m + 1)
        col = 0
        while True:
            used[col] = True
            occupied = match[col]
            best_cost, next_col = None, 0
            for j in range(1, m + 1):
                if used[j]:
                    continue
                cost = -weights[occupied - 1][j - 1] - row_dual[occupied] - col_dual[j]
                if slack[j] is None or cost < slack[j]:
                    slack[j], via[j] = cost, col
                if best_cost is None or slack[j] < best_cost:
                    best_cost, next_col = slack[j], j
            for j in range(m + 1):
                if used[j]:
                    row_dual[match[j]] += best_cost
                    col_dual[j] -= best_cost
                elif j:
                    slack[j] -= best_cost
            col = next_col
            if not match[col]:
                break
        while col:
            prev = via[col]
            match[col] = match[prev]
            col = prev
    return sum(weights[row - 1][j - 1] for j, row in enumerate(match) if j and row)
