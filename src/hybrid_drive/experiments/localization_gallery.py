"""Local camera gallery with projected reference boxes and dev-gated overlays."""

from __future__ import annotations

from base64 import b64encode
from collections.abc import Mapping
from dataclasses import dataclass
from html import escape
from io import BytesIO
import math
from numbers import Real
import os
from pathlib import Path

import numpy as np

from hybrid_drive.assets.alignment_manifest import IMAGE_SIZE, midpoint_indices
from hybrid_drive.assets.files import AtomicFile, check_components
from hybrid_drive.assets.spatial_manifest import BAND_COUNT, read_spatial_rgb, read_spatial_targets


@dataclass(frozen=True)
class GalleryPredictions:
    scores: np.ndarray
    boxes: np.ndarray
    threshold: float


def write_localization_gallery(
    path: Path,
    data_root: Path,
    manifest: dict,
    variants: Mapping[str, GalleryPredictions],
    *,
    examples_per_recording: int = 6,
) -> None:
    """Render aligned holdout predictions only after the caller's holdout gate.

    Prediction rows follow the verified manifest order, not gallery order.
    Publication is atomic and refuses to overwrite an existing gallery.
    """
    path = Path(os.path.abspath(path))
    data_root = Path(os.path.abspath(data_root))
    if path.is_relative_to(data_root):
        raise ValueError("Localization gallery must be outside the input dataset")
    check_components(path)
    if path.exists():
        raise FileExistsError(f"Localization gallery already exists: {path}")
    if type(examples_per_recording) is not int or examples_per_recording <= 0:
        raise ValueError("Expected a positive integer number of examples per recording")

    samples, selection = _select_examples(manifest, examples_per_recording)
    checked = _validate_variants(variants, len(samples))
    targets = read_spatial_targets(data_root, manifest, "holdout")
    if [target["sample_id"] for target in targets] != [sample["sample_id"] for sample in samples]:
        raise ValueError("Gallery targets differ from the holdout frame order")

    with AtomicFile(path) as output:
        output.out.write(_page_start(selection, checked).encode())
        for recording_number, (recording, indices) in enumerate(selection, start=1):
            output.out.write(
                (
                    f'<section class="recording"><h2>Запись {recording_number}</h2>'
                    f'<p class="source">{escape(recording)}</p><div class="frames">'
                ).encode()
            )
            for index in indices:
                sample = samples[index]
                rgb = read_spatial_rgb(data_root, sample["rgb"])
                output.out.write(_frame(index, sample, targets[index], rgb, checked).encode())
            output.out.write(b"</div></section>")
        output.out.write(b"</main></body></html>\n")
        output.commit()


def _select_examples(manifest, limit):
    samples, selection = [], []
    recording_ids, sample_ids = set(), set()
    for recording in manifest["splits"]["holdout"]["recordings"]:
        identity = recording["recording_id"]
        rows = recording["samples"]
        if not rows or identity in recording_ids:
            raise ValueError("Expected distinct nonempty holdout recordings")
        source_indices = [row["source_list_index"] for row in rows]
        if any(a >= b for a, b in zip(source_indices, source_indices[1:])):
            raise ValueError("Gallery samples must be in source-list order")
        for row in rows:
            if row["sample_id"] in sample_ids:
                raise ValueError("Duplicate holdout frame identity in gallery")
            sample_ids.add(row["sample_id"])
        offset = len(samples)
        indices = midpoint_indices(len(rows), min(limit, len(rows)))
        selection.append((identity, tuple(offset + index for index in indices)))
        samples.extend(rows)
        recording_ids.add(identity)
    if not selection:
        raise ValueError("Expected at least one holdout recording")
    return samples, selection


def _validate_variants(variants, frame_count):
    if not isinstance(variants, Mapping) or not 1 <= len(variants) <= 3:
        raise ValueError("Gallery requires one to three ordered prediction variants")
    checked = []
    shape = (frame_count, BAND_COUNT)
    for label, prediction in variants.items():
        if not isinstance(label, str) or not label.strip():
            raise ValueError("Expected a nonempty gallery variant label")
        if not isinstance(prediction, GalleryPredictions):
            raise ValueError("Gallery variants must contain GalleryPredictions")
        scores, boxes, threshold = prediction.scores, prediction.boxes, prediction.threshold
        if not isinstance(threshold, Real) or isinstance(threshold, bool) or not math.isfinite(threshold):
            raise ValueError("Expected a finite dev threshold for the gallery")
        if (
            not isinstance(scores, np.ndarray)
            or scores.shape != shape
            or scores.dtype.kind != "f"
            or not np.isfinite(scores).all()
        ):
            raise ValueError("Gallery scores must be finite floats in holdout [N,4] order")
        if (
            not isinstance(boxes, np.ndarray)
            or boxes.shape != (*shape, 4)
            or boxes.dtype.kind != "f"
            or not np.isfinite(boxes).all()
            or np.any(boxes[..., 2:] <= boxes[..., :2])
        ):
            raise ValueError("Gallery boxes must be finite positive-area xyxy floats in holdout [N,4,4] order")
        # Match spatial_metrics at the cutoff, including when the saved threshold
        # is just above a float32 score and would round down in its original dtype.
        shown = scores.astype(np.float64) >= float(threshold)
        checked.append((label, boxes, shown, float(threshold)))
    return checked


def _page_start(selection, variants):
    frame_count = sum(len(indices) for _, indices in selection)
    rules = "".join(
        f".view-{i}:checked ~ .image .mode-{i},.view-{i}:checked ~ .counts .mode-{i}"
        "{display:block}"
        for i in range(len(variants))
    )
    thresholds = "".join(
        f'<li>{escape(label)} <code>{threshold!r}</code></li>'
        for label, _, _, threshold in variants
    )
    return f"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Галерея 2D-локализации · OpenScene</title>
<style>{_STYLE}{rules}</style></head>
<body><main>
<header class="hero">
<p class="eyebrow">OPENSCENE · CAM_F0 · ОФЛАЙН</p>
<h1>Где модель видит машины</h1>
<p>Один и тот же кадр для всех вариантов. Переключатель меняет только предсказанные рамки.</p>
<div class="stats">
<div><b>{len(selection)}</b>записей</div>
<div><b>{frame_count}</b>кадров в галерее</div>
<div><b>{len(variants)}</b>вариантов модели</div>
</div>
</header>
<aside class="key">
<p><span class="swatch truth-key"></span>Эталон: проекция исходного 3D-бокса<br>
<span class="swatch prediction-key"></span>Предсказание: оценка не ниже порога</p>
<details><summary>Сохранённые пороги из dev</summary><ul>{thresholds}</ul></details>
<p class="note">В каждой из четырёх вертикальных зон эталон отмечает ближайшую машину
с подходящей аннотацией; пустая зона не доказывает отсутствие машины.
Кадры выбраны равномерно по каждой записи, до учёта меток и оценок.
Это проекции, не ручные пиксельные рамки и не проверка вождения.
Для боксов, выходящих за кадр, показана только видимая часть; метрики не изменены.</p>
</aside>
"""


def _frame(index, sample, target, rgb, variants):
    image = _png_data_uri(rgb)
    sample_index = sample["source_list_index"]
    truth = [box for present, box in zip(target["presence"], target["boxes_2d"]) if present]
    buttons, overlays, counts = [], [], []
    for variant, (label, boxes, shown, _) in enumerate(variants):
        radio_id = f"frame-{index}-view-{variant}"
        selected = " checked" if variant == 0 else ""
        buttons.append(
            f'<input type="radio" class="view view-{variant}" name="frame-{index}" id="{radio_id}"{selected}>'
            f'<label for="{radio_id}">{escape(label)}</label>'
        )
        predicted = boxes[index][shown[index]]
        visible = [part for box in predicted if (part := _visible_box(box)) is not None]
        overlays.append(_overlay(f"prediction mode-{variant}", visible, "prediction-box"))
        outside = len(predicted) - len(visible)
        note = f" · Вне кадра: {outside}" if outside else ""
        counts.append(
            f'<p class="switchable mode-{variant}">Эталон: {len(truth)} / 4 · '
            f'Предсказано: {len(predicted)} / 4{note}</p>'
        )

    bounds = "".join(f"M{IMAGE_SIZE * i / BAND_COUNT:g} 0V{IMAGE_SIZE}" for i in range(1, BAND_COUNT))
    quarters = f'<path class="quarters" d="{bounds}"/>'
    return (
        f'<fieldset class="frame" data-frame-index="{index}"><legend>Кадр {sample_index}</legend>'
        f'{"".join(buttons)}<div class="image">'
        f'<img alt="Фронтальная камера, кадр {sample_index}" width="{IMAGE_SIZE}" height="{IMAGE_SIZE}" '
        f'loading="lazy" src="{image}">{"".join(overlays)}'
        f'{_overlay("reference", truth, "reference-box", quarters)}</div>'
        f'<div class="counts">{"".join(counts)}</div></fieldset>'
    )


def _png_data_uri(rgb):
    from PIL import Image

    buffer = BytesIO()
    Image.fromarray(rgb).save(buffer, format="PNG")
    return "data:image/png;base64," + b64encode(buffer.getvalue()).decode("ascii")


def _overlay(css, boxes, box_style, extra=""):
    return (
        f'<svg aria-hidden="true" class="overlay {css}" viewBox="0 0 {IMAGE_SIZE} {IMAGE_SIZE}">'
        f'{extra}{"".join(_rect(box_style, box) for box in boxes)}</svg>'
    )


def _visible_box(box):
    x1, y1, x2, y2 = (min(max(float(coordinate), 0.0), 1.0) for coordinate in box)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _rect(css, box):
    x1, y1, x2, y2 = map(float, box)
    x, y, width, height = (v * IMAGE_SIZE for v in (x1, y1, x2 - x1, y2 - y1))
    return (
        f'<rect class="{css}" x="{x:.4f}" y="{y:.4f}" width="{width:.4f}" height="{height:.4f}" '
        'vector-effect="non-scaling-stroke"/>'
    )


_STYLE = """
*{box-sizing:border-box}
:root{color-scheme:light;--ink:#10283b;--muted:#456377;--line:#d6e4ec;--paper:#fff;
      --mint:#65ffd5;--amber:#ffcc63}
body{margin:0;background:#edf4f8;color:var(--ink);font:16px/1.5 system-ui,sans-serif}
main{max-width:1200px;margin:auto;padding:24px clamp(12px,4vw,32px) 44px}
.hero{background:linear-gradient(118deg,#122b3e,#154761);color:#f5fbff;
      border-radius:24px;padding:clamp(18px,5vw,38px)}
.hero h1{font-size:clamp(27px,5vw,48px);line-height:1.1;margin:.25em 0 .42em}
.hero p{margin:.35em 0}.eyebrow{font-size:13px;letter-spacing:.09em;font-weight:700;color:#a4dfef}
.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-top:21px}
.stats div{padding:9px 12px;border-radius:12px;background:#f0fbff1c;font-size:.88rem}
.stats b{display:block;font-size:1.38rem}
.key{background:var(--paper);border:1px solid var(--line);border-radius:14px;padding:14px 18px;margin:16px 0 24px}
.key p{margin:0 0 .5em}.key .note{color:var(--muted);margin:.65em 0 0}
.swatch{display:inline-block;vertical-align:middle;width:25px;margin-right:8px;border-radius:2px}
.truth-key{border-top:3px dashed #aa6300}.prediction-key{border-top:3px solid #007459}
.key summary{color:#104e70;cursor:pointer}.key li{overflow-wrap:anywhere}
.recording{margin:20px 0 30px}.recording h2{margin:0}.source{margin:.1em 0 .8em;color:var(--muted)}
.frames{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,305px),1fr));gap:14px}
.frame{min-inline-size:0;min-width:0;margin:0;padding:7px 9px 0;background:var(--paper);
       border:1px solid var(--line);border-radius:13px}
.frame legend{padding:0 8px;color:#164a67;font-weight:700}
.view{position:absolute;width:1px;height:1px;clip-path:inset(50%);white-space:nowrap}
.view + label{display:inline-block;margin:0 3px 6px 0;padding:4px 8px;cursor:pointer;
              border:1px solid var(--line);border-radius:8px;font-size:.85rem;overflow-wrap:anywhere}
.view:checked + label{background:#175479;color:#fff;border-color:#175479;font-weight:700}
.view:focus-visible + label{outline:2px solid #175479;outline-offset:2px}
.image{position:relative;overflow:hidden;border-radius:9px;background:#142331}
.image img{display:block;width:100%;height:auto}
.overlay{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}
.prediction,.switchable{display:none}
.reference-box,.prediction-box{fill:none;stroke-width:2;filter:drop-shadow(0 0 1px #092337)}
.reference-box{stroke:var(--amber);stroke-dasharray:5 3}
.prediction-box{stroke:var(--mint)}
.quarters{stroke:#fff;stroke-opacity:.32;stroke-width:.55;stroke-dasharray:2 4}
.counts p{margin:7px 0;color:var(--muted);font-size:.86rem}
@media(max-width:460px){.stats div{padding:6px;font-size:.75rem}}
"""
