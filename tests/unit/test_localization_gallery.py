from hashlib import sha256
from html import escape
from io import BytesIO
import json
import re

import numpy as np
from PIL import Image
import pytest

from hybrid_drive.assets.files import PreparationError
from hybrid_drive.experiments.localization_gallery import GalleryPredictions, write_localization_gallery


def _record(root, path, data, **fields):
    file = root / path
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_bytes(data)
    return {"path": path, "size_bytes": len(data), "sha256": sha256(data).hexdigest(), **fields}


def _pack(root):
    recordings, targets = [], []
    for number, n in enumerate((6, 5)):
        rows = []
        for index in range(n):
            identity = f"example-{number}-{index}"
            rgb = np.full((224, 224, 3), len(targets), dtype=np.uint8)
            buffer = BytesIO()
            Image.fromarray(rgb).save(buffer, format="PNG")
            image = _record(
                root, f"rgb/{identity}.png", buffer.getvalue(), pixels_sha256=sha256(rgb.tobytes()).hexdigest()
            )
            rows.append(dict(sample_id=identity, source_list_index=10 * index + 2, rgb=image))
            targets.append(
                dict(
                    sample_id=identity,
                    presence=[True, False, False, False],
                    boxes_2d=[[0.0, 0.125, 0.125, 0.375], None, None, None],
                )
            )
        recordings.append({"recording_id": f"recording-{number}<&", "samples": rows})
    target_series = {"schema": "hybrid-drive-spatial-loc-targets-v1", "split": "holdout", "rows": targets}
    partition = {
        "recordings": recordings,
        "targets": _record(root, "targets/holdout.json", json.dumps(target_series).encode()),
    }
    boxes = np.broadcast_to(np.array([0.25, 0.25, 0.75, 0.75]), (len(targets), 4, 4)).copy()
    scores = np.broadcast_to(np.array([0.5, 0.25, -1.0, -2.0], dtype=np.float32), boxes.shape[:-1]).copy()
    return {"splits": {"holdout": partition}}, scores, boxes


def _render(tmp_path, manifest, variants):
    path = tmp_path / "report" / "gallery.html"
    write_localization_gallery(path, tmp_path / "assets", manifest, variants, examples_per_recording=2)
    return path.read_text(encoding="utf-8")


def test_same_uniform_frames_and_escaped_variant_switches(tmp_path):
    data = tmp_path / "assets"
    manifest, scores, boxes = _pack(data)
    label = '<script type="text">alert(1)</script>'
    variants = {"Базовый": GalleryPredictions(scores, boxes, 0.0), label: GalleryPredictions(scores, boxes, 0.0)}
    first = _render(tmp_path, manifest, variants)
    # New scores and boxes cannot change frame selection. Select two fixed midpoint rows per recording.
    changed = {"Другой": GalleryPredictions(-scores, boxes[::-1], 1.0)}
    second_path = tmp_path / "other" / "gallery.html"
    write_localization_gallery(second_path, data, manifest, changed, examples_per_recording=2)
    second = second_path.read_text(encoding="utf-8")
    assert re.findall(r'data-frame-index="(\d+)"', first) == ["1", "4", "7", "9"]
    assert re.findall(r'data-frame-index="(\d+)"', second) == ["1", "4", "7", "9"]
    assert re.findall(r'src="(data:image/png;base64,[^"]+)"', first) == re.findall(
        r'src="(data:image/png;base64,[^"]+)"', second
    )
    assert first.count("data:image/png;base64,") == 4
    assert first.count("class=\"reference-box\"") == 4
    assert label not in first and escape(label) in first and "recording-0&lt;&amp;" in first
    assert first.count("checked>") == 4  # each frame starts on its own first variant


def test_cutoff_is_inclusive_and_float64_like_metrics(tmp_path):
    manifest, scores, boxes = _pack(tmp_path / "assets")
    variants = {
        "Exact": GalleryPredictions(scores, boxes, 0.5),
        "Just above": GalleryPredictions(scores, boxes, np.nextafter(0.5, np.inf)),
        "Second box also": GalleryPredictions(scores, boxes, 0.25),
    }
    html = _render(tmp_path, manifest, variants)
    for index, expected in enumerate((1, 0, 2)):
        groups = re.findall(rf'class="overlay prediction mode-{index}"[^>]*>(.*?)</svg>', html)
        assert len(groups) == 4
        assert all(group.count("prediction-box") == expected for group in groups)
    assert html.count("Эталон: 1 / 4 · Предсказано: 1 / 4") == 4
    assert html.count("Эталон: 1 / 4 · Предсказано: 0 / 4") == 4
    assert html.count("Эталон: 1 / 4 · Предсказано: 2 / 4") == 4


def test_clips_rendered_boxes_without_mutating_raw_model_output(tmp_path):
    manifest, scores, boxes = _pack(tmp_path / "assets")
    boxes[:, 0, :] = [-0.125, 0.25, 0.875, 1.125]
    boxes[:, 1, :] = [1.125, 0.25, 1.375, 0.75]
    saved = boxes.copy()
    html = _render(tmp_path, manifest, {"baseline": GalleryPredictions(scores, boxes, 0.0)})
    assert html.count('class="prediction-box" x="0.0000" y="56.0000" width="196.0000" height="168.0000"') == 4
    assert html.count('class="prediction-box"') == 4
    assert html.count("Предсказано: 2 / 4 · Вне кадра: 1") == 4
    assert np.array_equal(boxes, saved)


@pytest.mark.parametrize("problem", ["cutoff", "score", "extent", "rows"])
def test_rejects_misaligned_or_nonfinite_overlays(tmp_path, problem):
    manifest, scores, boxes = _pack(tmp_path / "assets")
    threshold = 0.0
    if problem == "cutoff":
        threshold = float("nan")
    elif problem == "score":
        scores[0, 0] = np.inf
    elif problem == "extent":
        boxes[-1, -1, 2] = 0.01  # reject even when this box would not have been drawn
    elif problem == "rows":
        scores = scores[:-1]
    with pytest.raises(ValueError, match="Expected a finite dev threshold|Gallery scores|Gallery boxes"):
        _render(tmp_path, manifest, {"model": GalleryPredictions(scores, boxes, threshold)})
    assert not (tmp_path / "report" / "gallery.html").exists()


def test_bad_image_leaves_no_partial_and_stale_target_is_preserved(tmp_path):
    root = tmp_path / "assets"
    manifest, scores, boxes = _pack(root)
    variant = {"baseline": GalleryPredictions(scores, boxes, 0.0)}
    recording = manifest["splits"]["holdout"]["recordings"][0]
    image = root / recording["samples"][4]["rgb"]["path"]
    image.write_bytes(b"changed after manifest")
    path = tmp_path / "report" / "gallery.html"
    with pytest.raises(PreparationError, match="frozen size or SHA256"):
        write_localization_gallery(path, root, manifest, variant, examples_per_recording=2)
    assert list(path.parent.iterdir()) == []

    path.write_text("Do not overwrite", encoding="utf-8")
    with pytest.raises(FileExistsError, match="gallery already exists"):
        write_localization_gallery(path, root, manifest, variant, examples_per_recording=2)
    assert path.read_text(encoding="utf-8") == "Do not overwrite"
    assert list(path.parent.iterdir()) == [path]


def test_refuses_to_write_gallery_inside_the_readonly_pack(tmp_path):
    root = tmp_path / "assets"
    manifest, scores, boxes = _pack(root)
    variant = {"baseline": GalleryPredictions(scores, boxes, 0.0)}
    target = root / ".." / "assets" / "gallery.html"
    with pytest.raises(ValueError, match="outside the input dataset"):
        write_localization_gallery(target, root, manifest, variant)
    assert not target.exists()


def test_rejects_target_order_before_creating_output(tmp_path):
    root = tmp_path / "assets"
    manifest, scores, boxes = _pack(root)
    partition = manifest["splits"]["holdout"]
    series = json.loads((root / partition["targets"]["path"]).read_text())
    series["rows"][0], series["rows"][1] = series["rows"][1], series["rows"][0]
    partition["targets"] = _record(root, "targets/holdout.json", json.dumps(series).encode())
    with pytest.raises(PreparationError, match="target identity"):
        _render(tmp_path, manifest, {"baseline": GalleryPredictions(scores, boxes, 0.0)})
    assert not (tmp_path / "report").exists()
