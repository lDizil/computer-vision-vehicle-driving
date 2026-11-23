"""Self-contained view of the exact ordered camera frames compared by temporal replay."""

from __future__ import annotations

from base64 import b64encode
from collections.abc import Mapping, Sequence
from io import BytesIO
from itertools import groupby
import json
from math import isfinite
from numbers import Integral, Real
import os
from pathlib import Path

import numpy as np
from PIL import Image

from hybrid_drive.assets.alignment_manifest import IMAGE_SIZE
from hybrid_drive.assets.files import AtomicFile, check_components
from hybrid_drive.assets.spatial_manifest import BAND_COUNT, read_spatial_rgb

from .spatial_decision_policy import ABSENT, PRESENT, UNCERTAIN
from .temporal_spatial_gallery_player import DOCUMENT_END, DOCUMENT_START


_BASELINE = "framewise"
_MAX_CLIPS = 6
_MAX_FRAMES_PER_CLIP = 24
_MAX_MODES = 9
_MAX_HTML_BYTES = 16 * 1024**2
_UNSIGNED_INT64 = 2**63 - 1
_MODE_LABELS = {
    "framewise": "Покадровый",
    "confirm2": "Подтверждение ×2",
    "confirm3": "Подтверждение ×3",
    "hold1": "Удержание на одной неопределённости",
    "score_ema": "Сглаживание оценки",
    "box_ema": "Сглаживание рамки",
    "score_box": "Сглаживание оценки и рамки",
    "confirm_box": "Подтверждение ×2 + сглаживание рамки",
    "hold_box": "Удержание + сглаживание рамки",
}
_JSON_ESCAPES = str.maketrans({
    "<": r"\u003c", ">": r"\u003e", "&": r"\u0026", "\u2028": r"\u2028", "\u2029": r"\u2029",
})


def write_temporal_spatial_gallery(path, pack, targets, outcomes, *, selected_name):
    """Publish a complete comparison of untouched final frames, not model output footage.

    The caller supplies verified per-frame reference targets and decisions. Each
    mode must match the full ordered identity and timestamp of every frame.
    JPEG compression is display-only; boxes remain in normalized image coordinates.
    """
    root, path = (Path(os.path.abspath(value)) for value in (pack.root, path))
    check_components(root)
    check_components(path)
    if path.is_relative_to(root):
        raise ValueError("Temporal gallery must be written outside the image pack")
    if path.exists():
        raise FileExistsError(f"Temporal gallery already exists: {path}")

    frames, layout = _layout(pack.frames)
    modes, decisions = _checked_modes(outcomes, selected_name, len(frames))
    if not isinstance(targets, Mapping) or set(targets) != {frame.sample_id for frame in frames}:
        raise ValueError("Temporal gallery needs one reference target per exact final frame")

    # Fail before opening any images if even an undisplayed mode has stale evidence.
    clips = []
    for (recording, clip), rows in layout:
        first = rows[0][1].timestamp_us
        items = []
        for index, frame in rows:
            items.append({
                "sample": frame.sample_id,
                "index": frame.source_index,
                "offset_us": frame.timestamp_us - first,
                "truth": _target_overlay(targets[frame.sample_id]),
                "predictions": [_prediction_overlay(mode[index], frame) for mode in decisions],
            })
        clips.append({"recording": recording, "clip": clip, "frames": items})

    payload = {
        "schema": "mv-temporal-gallery-v1",
        "baseline": _BASELINE,
        "selected": selected_name,
        "modes": [{"id": key, "label": _MODE_LABELS.get(key, key)} for key in modes],
        "clips": clips,
    }
    out_frames = [frame for clip in clips for frame in clip["frames"]]
    for frame, view in zip(frames, out_frames):
        view["image"] = _jpeg_uri(read_spatial_rgb(root, frame.rgb))

    data = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")).translate(_JSON_ESCAPES)
    document = (DOCUMENT_START + data + DOCUMENT_END).encode("utf-8")
    if len(document) > _MAX_HTML_BYTES:
        raise ValueError("Temporal gallery exceeds the one-file size budget")
    with AtomicFile(path) as file:
        file.out.write(document)
        file.commit()


def _layout(values):
    if not isinstance(values, Sequence) or not 1 <= len(values) <= _MAX_CLIPS * _MAX_FRAMES_PER_CLIP:
        raise ValueError("Temporal gallery needs a bounded, ordered sequence of frames")
    frames = tuple(values)
    seen_samples, seen_recordings, seen_clips = set(), set(), set()
    previous = None
    for frame in frames:
        recording, clip, sample, index, timestamp = _identity(frame)
        if any(not _valid_text(text) for text in (recording, clip, sample)):
            raise ValueError("Temporal gallery frame identity must use bounded nonempty text")
        if any(type(number) is not int or not 0 <= number <= _UNSIGNED_INT64 for number in (index, timestamp)):
            raise ValueError("Temporal gallery frame indices and timestamps must be integer source values")
        if sample in seen_samples:
            raise ValueError("Duplicate sample in temporal gallery")
        seen_samples.add(sample)
        if previous is None or previous.recording_id != recording:
            if recording in seen_recordings:
                raise ValueError("A recording cannot reenter the temporal gallery sequence")
            seen_recordings.add(recording)
        elif index <= previous.source_index or timestamp <= previous.timestamp_us:
            raise ValueError("Temporal gallery source frames must remain chronological")
        if previous is not None and previous.recording_id == recording and previous.clip_id == clip:
            if index != previous.source_index + 1:
                raise ValueError("A temporal gallery clip is missing a source frame")
            if not 250_000 <= timestamp - previous.timestamp_us <= 800_000:
                raise ValueError("A temporal gallery clip has a broken source cadence")
        else:
            if clip in seen_clips:
                raise ValueError("A clip cannot reenter the temporal gallery sequence")
            seen_clips.add(clip)
        previous = frame

    blocks = [
        (identity, tuple(group))
        for identity, group in groupby(enumerate(frames), key=lambda pair: (pair[1].recording_id, pair[1].clip_id))
    ]
    if len(blocks) > _MAX_CLIPS or any(len(rows) > _MAX_FRAMES_PER_CLIP for _, rows in blocks):
        raise ValueError("Temporal gallery clip dimensions exceed the prepared pack")
    return frames, blocks


def _checked_modes(outcomes, selected, frame_count):
    if not isinstance(outcomes, Mapping) or not 1 <= len(outcomes) <= _MAX_MODES:
        raise ValueError("Temporal gallery mode count must fit the input scope")
    if not isinstance(selected, str) or _BASELINE not in outcomes or selected not in outcomes:
        raise ValueError("Temporal gallery requires both the framewise and selected modes")
    if any(not _valid_text(key, limit=64) for key in outcomes):
        raise ValueError("Temporal gallery modes must have nonempty bounded names")
    names = [_BASELINE, *(name for name in outcomes if name != _BASELINE)]
    modes = [outcomes[name] for name in names]
    if any(not isinstance(rows, Sequence) or len(rows) != frame_count for rows in modes):
        raise ValueError("Every temporal gallery mode must cover the full ordered pack")
    return names, modes


def _valid_text(value, *, limit=256):
    return (
        isinstance(value, str) and 0 < len(value) <= limit and bool(value.strip())
        and not any(char in value for char in ("\x00", "\n", "\r"))
    )


def _identity(frame):
    try:
        return frame.recording_id, frame.clip_id, frame.sample_id, frame.source_index, frame.timestamp_us
    except AttributeError as exc:
        raise ValueError("Missing identity in a temporal gallery frame") from exc


def _target_overlay(target):
    try:
        labels, raw_boxes = target.present, target.boxes
    except AttributeError as exc:
        raise ValueError("Missing target for a temporal gallery frame") from exc
    labels, boxes = _four(labels, "reference labels"), _four(raw_boxes, "reference boxes")
    if any(type(value) is not bool for value in labels):
        raise ValueError("Temporal gallery reference labels must be boolean")
    if any(label != (box is not None) for label, box in zip(labels, boxes)):
        raise ValueError("Only present reference zones may carry a box")
    return [None if box is None else _box(box) for box in boxes]


def _prediction_overlay(decision, ref):
    try:
        observation = decision.observation
        states, raw_boxes, reasons, reset = decision.states, decision.boxes, decision.reasons, decision.reset_reason
    except AttributeError as exc:
        raise ValueError("Missing decision fields in a temporal gallery frame") from exc
    if _identity(observation) != _identity(ref):
        raise ValueError("A temporal gallery decision does not match its frame identity or order")
    states = _four(states, "decision states")
    if any(
        not isinstance(value, Integral) or isinstance(value, bool) or value not in (ABSENT, UNCERTAIN, PRESENT)
        for value in states
    ):
        raise ValueError("Temporal gallery decision states must be present, absent or uncertain")
    boxes = tuple(_box(box) for box in _four(raw_boxes, "decision boxes"))
    reasons = _four(reasons, "decision reasons")
    if any(not _valid_text(value, limit=128) for value in reasons) or reset not in (None, "initial", "gap"):
        raise ValueError("Temporal gallery decision reasons are invalid")
    return {
        "states": list(map(int, states)),
        "boxes": [box if state == PRESENT else None for state, box in zip(states, boxes)],
        "reasons": reasons,
        "reset": reset,
    }


def _four(values, name):
    if not isinstance(values, (list, tuple)) or len(values) != BAND_COUNT:
        raise ValueError(f"Temporal gallery {name} must have four entries")
    return tuple(values)


def _box(values):
    coords = _four(values, "normalized xyxy box")
    if any(not isinstance(value, Real) or isinstance(value, bool) or not 0 <= value <= 1 for value in coords):
        raise ValueError("Temporal gallery boxes must be finite and normalized")
    x1, y1, x2, y2 = box = tuple(float(value) for value in coords)
    if not all(map(isfinite, box)) or not (x1 < x2 and y1 < y2):
        raise ValueError("Temporal gallery boxes must have positive area")
    return box


def _jpeg_uri(pixels):
    if (
        not isinstance(pixels, np.ndarray) or pixels.shape != (IMAGE_SIZE, IMAGE_SIZE, 3)
        or pixels.dtype != np.uint8
    ):
        raise ValueError("Temporal gallery needs prepared 224×224 RGB pixels")
    output = BytesIO()
    Image.fromarray(pixels).save(output, format="JPEG", quality=85, optimize=True)
    return "data:image/jpeg;base64," + b64encode(output.getvalue()).decode("ascii")
