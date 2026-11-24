"""Outcome-only, offline paired contact check on a fixed OpenScene recording."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
from itertools import combinations
from math import atan2, comb, cos, hypot, sin
from pathlib import Path
import pickle
import sys

import numpy as np

from hybrid_drive.assets.files import check_components
from hybrid_drive.contracts._validation import finite_real, identifier
from hybrid_drive.contracts.sensors import timestamp_us
from hybrid_drive.experiments.bounded_io import canonical_json, decode_json, read_bounded, require_hex
from hybrid_drive.experiments.box_motion_data import (
    SOURCES_SHA256,
    _METADATA_LIMIT,
    _chronological,
    _pose,
    _read_selected,
    _vehicle_boxes,
)
from hybrid_drive.experiments.io import write_json
from hybrid_drive.geometry.rigid import invert_pose, matrix


_PREDICTION_SCHEMA = "hybrid-drive-paired-slow-natural-predictions-v1"
_EVALUATION_SCHEMA = "hybrid-drive-paired-slow-natural-score-v1"
_STEPS = 5
_STEP_US = 500_000
_MAX_LAG_US = 100_000
_GAP_RANGE_US = (250_000, 750_000)
_SCORE_TIE = 1e-6
_EGO_CENTER_OFFSET_M = 1.25
_EGO_HALF_EXTENT_M = (2.25, 0.95)
_EGO_PAD_M = 0.10
_METHODS = ("fast_first", "slow", "untrained")


@dataclass(frozen=True)
class _Candidate:
    candidate_id: str
    poses: np.ndarray
    fast_overlap_s: float | None
    slow_margin: float
    untrained_margin: float


@dataclass(frozen=True)
class _Forecast:
    sample_token: str
    snapshot_sha256: str
    ref_time_us: int
    candidates: tuple[_Candidate, ...]


def score_natural(
    predictions: Path,
    sealed_sha256: str,
    metadata_archive: Path,
    sources: Path,
    input_index: Path,
    output: Path,
) -> dict:
    """Score saved predictions without opening any labels until the seal matches."""
    output = Path(output)
    check_components(output)
    if output.exists():
        raise ValueError("Natural score output must be new")
    require_hex(sealed_sha256, 64, "natural prediction seal")
    raw = read_bounded(Path(predictions), 2 * 1024**2)
    if sha256(raw).hexdigest() != sealed_sha256:
        raise ValueError("Natural predictions differ from the seal")
    predicted = decode_json(raw)
    if (not isinstance(predicted, dict) or predicted.get("schema") != _PREDICTION_SCHEMA
            or predicted.get("heldout_targets_opened") is not False):
        raise ValueError("Expected sealed natural predictions without labels")
    require_hex(predicted.get("revision"), 40, "natural prediction revision")
    source = decode_json(read_bounded(Path(sources), 16 * 1024))
    if sha256(canonical_json(source)).hexdigest() != SOURCES_SHA256:
        raise ValueError("Natural outcome source differs from the checked-in manifest")

    heldout = source["logs"][1]
    index_data = read_bounded(Path(input_index), 4 * 1024**2)
    forecasts = _bind_predictions(predicted["rows"], decode_json(index_data), heldout)
    if sys.platform != "linux":
        raise ValueError("Natural OpenScene score requires the isolated Linux image")
    recording = _load_verified_recording(Path(metadata_archive), source["metadata_archive"], heldout)
    rows = _score_recording(recording, heldout, forecasts)
    result = {
        "schema": _EVALUATION_SCHEMA,
        "revision": predicted["revision"],
        "prediction_seal": sealed_sha256,
        "input_index_sha256": sha256(index_data).hexdigest(),
        "metadata_archive_sha256": source["metadata_archive"]["sha256"],
        "metadata_member_sha256": heldout["sha256"],
        "recording": heldout["recording"],
        "protocol": {
            "scope": "single_heldout_recording_observed_annotated_vehicle_contact_at_samples_only",
            "future_sample_times_s": [i * _STEP_US / 1e6 for i in range(1, _STEPS + 1)],
            "max_abs_time_lag_s": _MAX_LAG_US / 1e6,
            "consecutive_frame_gaps_s": [gap / 1e6 for gap in _GAP_RANGE_US],
            "ego_unpadded_length_width_m": [2 * extent for extent in _EGO_HALF_EXTENT_M],
            "ego_center_forward_m_from_rear_axle": _EGO_CENTER_OFFSET_M,
            "ego_border_padding_m": _EGO_PAD_M,
            "margin_tie_tolerance": _SCORE_TIE,
            "preference": "lower_blocked_margin_or_earlier_frozen_fast_candidate",
        },
        "summary": _summarize(rows),
        "rows": rows,
    }
    output.mkdir(parents=True)
    write_json(output / "natural-score.json", result)
    return result


def _bind_predictions(rows, index, heldout) -> tuple[_Forecast, ...]:
    if (index.get("schema") != "hybrid-drive-box-motion-inputs-v1"
            or index.get("source", {}).get("source_sha256") != SOURCES_SHA256):
        raise ValueError("Expected fixed observation index from the motion cohort")
    recording = heldout["recording"]
    tokens = [a["sample_token"] for a in index["anchors"] if a["recording"] == recording]
    if (len(tokens) != len(heldout["anchor_indices"]) or len(set(tokens)) != len(tokens)
            or not isinstance(rows, list) or len(rows) != len(tokens)):
        raise ValueError("Natural score must include every anchor from the fixed held-out recording")
    lookup = {}
    for row in rows:
        token = row["sample_token"]
        require_hex(token, 16, "natural prediction sample token")
        digest = row["snapshot_sha256"]
        require_hex(digest, 64, "natural prediction snapshot digest")
        stamp = timestamp_us(row["ref_time_us"], "natural prediction anchor time")
        if (token not in tokens or token in lookup or row["recording"] != recording
                or index["frames"][token]["recording"] != recording
                or index["frames"][token]["ref_time_us"] != stamp):
            raise ValueError("Natural prediction does not match a unique held-out observation")
        values = row["candidates"]
        if not isinstance(values, list) or not 1 <= len(values) <= 8:
            raise ValueError("Expected between one and eight natural candidates")
        candidates = tuple(_parse_candidate(item) for item in values)
        if len({c.candidate_id for c in candidates}) != len(candidates):
            raise ValueError("Duplicate natural candidate ID")
        lookup[token] = _Forecast(token, digest, stamp, candidates)
    return tuple(lookup[token] for token in tokens)


def _parse_candidate(item) -> _Candidate:
    if finite_real(item["dt_s"], "dt_s") != _STEP_US / 1e6:
        raise ValueError("Natural candidate and outcome grids must both be sampled at 0.5 s")
    poses = matrix(item["poses"], (_STEPS, 3), "natural candidate poses")
    if np.abs(poses[:, :2]).max() > 1_000:
        raise ValueError("Natural candidate poses exceed the local scoring extent")
    overlap = item["fast_overlap_s"]
    if overlap is not None:
        overlap = finite_real(overlap, "fast_overlap_s")
        if not 0 <= overlap <= _STEPS * _STEP_US / 1e6:
            raise ValueError("Fast contact does not lie in the frozen candidate horizon")
    return _Candidate(
        identifier(item["candidate_id"], "candidate_id"),
        poses,
        overlap,
        finite_real(item["slow_margin"], "slow_margin"),
        finite_real(item["untrained_margin"], "untrained_margin"),
    )


def _load_verified_recording(archive, spec, heldout):
    [(_, data)] = _read_selected(archive, spec, {heldout["member"]}, _METADATA_LIMIT)
    if len(data) != heldout["size_bytes"] or sha256(data).hexdigest() != heldout["sha256"]:
        raise ValueError("Pinned natural outcome member differs from its size or SHA256")
    frames = pickle.loads(data)
    if (not isinstance(frames, list) or len(frames) != heldout["frame_count"]
            or any(frame.get("log_name") != heldout["log_name"] for frame in frames)):
        raise ValueError("Natural outcome member does not match the pinned recording")
    return _chronological(frames)


def _score_recording(frames, heldout, forecasts):
    rows = []
    for anchor_index, entry in zip(heldout["anchor_indices"], forecasts, strict=True):
        anchor = frames[anchor_index]
        if anchor["token"] != entry.sample_token or anchor["timestamp"] != entry.ref_time_us:
            raise ValueError("Pinned natural outcome anchor and prediction identity differ")
        window = frames[anchor_index:anchor_index + _STEPS + 1]
        rows.append(_score_window(window, entry))
    return rows


def _score_window(window, entry):
    row = {
        "sample_token": entry.sample_token,
        "snapshot_sha256": entry.snapshot_sha256,
        "candidate_count": len(entry.candidates),
        "candidate_pairs": comb(len(entry.candidates), 2),
        "pairs": [],
    }
    failure, lags = _timing(window)
    row["future_lag_us"] = lags
    if failure:
        return {**row, "window_status": failure}

    if any(not isinstance(f.get("anns"), dict)
           or not {"gt_names", "track_tokens", "gt_boxes"} <= f["anns"].keys() for f in window):
        return {**row, "window_status": "missing_annotations"}
    world_to_anchor = invert_pose(_pose(window[0]))
    boxes = [_boxes_in_anchor(frame, world_to_anchor) for frame in window]
    row["annotated_vehicle_count_by_sample"] = [len(sample) for sample in boxes]
    if _overlapping_ids((0, 0, 0), boxes[0]):
        return {**row, "window_status": "shared_observed_initial_contact"}

    row["window_status"] = "complete"
    candidates = []
    for c in entry.candidates:
        contacts = [_overlapping_ids(pose, sample) for pose, sample in zip(c.poses, boxes[1:], strict=True)]
        candidates.append({
            "candidate_id": c.candidate_id,
            "fast_overlap_s": c.fast_overlap_s,
            "slow_margin": c.slow_margin,
            "untrained_margin": c.untrained_margin,
            "observed_contact_sample_times_s": [
                step * _STEP_US / 1e6 for step, ids in enumerate(contacts, start=1) if ids
            ],
        })
    row["candidates"] = candidates
    row["pairs"] = [_compare(a, b) for a, b in combinations(candidates, 2)]
    return row


def _timing(window):
    if len(window) != _STEPS + 1:
        return "missing_future_samples", []
    anchor = window[0]
    if any(f["log_token"] != anchor["log_token"] or f["log_name"] != anchor["log_name"] for f in window[1:]):
        return "recording_boundary", []
    stamps = [timestamp_us(f["timestamp"], "annotation time") for f in window]
    low, high = _GAP_RANGE_US
    if any(not low <= b - a <= high for a, b in zip(stamps, stamps[1:])):
        return "noncontinuous_future", []
    lags = [stamp - stamps[0] - step * _STEP_US for step, stamp in enumerate(stamps[1:], start=1)]
    if any(abs(lag) > _MAX_LAG_US for lag in lags):
        return "future_time_mismatch", lags
    return None, lags


def _boxes_in_anchor(frame, world_to_anchor):
    transform = world_to_anchor @ _pose(frame)
    rotation, offset = transform[:3, :3], transform[:3, 3]
    boxes = {}
    for token, box in _vehicle_boxes(frame).items():
        center = rotation @ box[:3] + offset
        heading = rotation @ np.array([cos(box[6]), sin(box[6]), 0.0])
        if not np.isfinite(center).all() or hypot(*heading[:2]) <= 1e-6:
            raise ValueError("Annotated vehicle is not representable in the current ego plane")
        boxes[token] = (center[:2], atan2(heading[1], heading[0]), (box[3] / 2, box[4] / 2))
    return boxes


def _overlapping_ids(pose, boxes):
    x, y, yaw = pose
    ego_axes = _planar_axes(yaw)
    ego_center = np.array((x, y)) + _EGO_CENTER_OFFSET_M * ego_axes[0]
    ego_half = np.array(_EGO_HALF_EXTENT_M) + _EGO_PAD_M
    hits = []
    for token, (center, heading, half) in boxes.items():
        vehicle_axes = _planar_axes(heading)
        axes = np.vstack((ego_axes, vehicle_axes))
        support = np.abs(axes @ ego_axes.T) @ ego_half + np.abs(axes @ vehicle_axes.T) @ half
        if np.all(np.abs(axes @ (center - ego_center)) <= support + 1e-9):
            hits.append(token)
    return hits


def _planar_axes(yaw):
    return np.array(((cos(yaw), sin(yaw)), (-sin(yaw), cos(yaw))))


def _compare(a, b):
    left, right = a["candidate_id"], b["candidate_id"]
    a_hit, b_hit = bool(a["observed_contact_sample_times_s"]), bool(b["observed_contact_sample_times_s"])
    pair = {"candidate_ids": [left, right]}
    if a_hit == b_hit:
        pair["status"] = "both_contact" if a_hit else "neither_contact_observed"
        return pair
    return {
        **pair,
        "status": "comparable",
        "candidate_without_observed_contact": right if a_hit else left,
        "preferred": {
            "fast_first": left,
            "slow": _lower(left, right, a["slow_margin"], b["slow_margin"]),
            "untrained": _lower(left, right, a["untrained_margin"], b["untrained_margin"]),
        },
    }


def _lower(left, right, a, b):
    if abs(a - b) <= _SCORE_TIE:
        return None
    return left if a < b else right


def _summarize(rows):
    windows = Counter(row["window_status"] for row in rows)
    pairs = Counter(pair["status"] for row in rows for pair in row["pairs"])
    total_pairs = sum(row["candidate_pairs"] for row in rows)
    support = pairs["comparable"]
    methods = {}
    for name in _METHODS:
        counts = Counter(correct=0, incorrect=0, tied=0)
        for row in rows:
            for pair in row["pairs"]:
                if pair["status"] != "comparable":
                    continue
                choice = pair["preferred"][name]
                if choice is None:
                    counts["tied"] += 1
                else:
                    correct = choice == pair["candidate_without_observed_contact"]
                    counts["correct" if correct else "incorrect"] += 1
        methods[name] = {**counts, "accuracy_including_ties": counts["correct"] / support if support else None}
    return {
        "anchors": len(rows),
        "window_statuses": dict(sorted(windows.items())),
        "windows_with_comparable_pairs": sum(any(p["status"] == "comparable" for p in row["pairs"]) for row in rows),
        "candidate_pairs": total_pairs,
        "complete_window_pairs": sum(pairs.values()),
        "unscorable_window_pairs": total_pairs - sum(pairs.values()),
        "comparable_pairs": support,
        "both_contact_pairs": pairs["both_contact"],
        "neither_contact_observed_pairs": pairs["neither_contact_observed"],
        "method_counts": methods,
    }
