from base64 import b64decode
from dataclasses import replace
from io import BytesIO
import json
import re
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from hybrid_drive.assets.files import PreparationError
from hybrid_drive.assets.spatial_temporal import TemporalFrameRef, TemporalSpatialPack
from hybrid_drive.experiments import temporal_spatial_gallery as gallery
from hybrid_drive.experiments.temporal_spatial_metrics import TemporalTarget
from hybrid_drive.experiments.temporal_spatial_types import FrameDecision, FrameObservation


_BOXES = tuple((.07 + .25 * zone, .2, .17 + .25 * zone, .64) for zone in range(4))
_MODES = (
    "framewise", "confirm2", "confirm3", "hold1", "score_ema", "box_ema", "score_box", "confirm_box", "hold_box",
)
_DATA_PATTERN = re.compile(r'<script type="application/json" id="video-data">(.*?)</script>', re.S)


def _clip_pack(tmp_path, *, clips=2, frames_per_clip=3, modes=("framewise", "score_box")):
    root = tmp_path / "images"
    root.mkdir()
    refs, targets = [], {}
    outcomes = {key: [] for key in modes}
    for clip in range(clips):
        recording = f"recording-{clip // 3}"
        clip_id = f"{recording}-clip-{clip % 3:02}"
        for i in range(frames_per_clip):
            sample = f"sample-{clip}-{i}"
            index = 10 + 40 * (clip % 3) + i
            ts = 1_600_000_000_000_000 + 100_000_000 * clip + i * 500_000 + (i % 2) * 211
            refs.append(TemporalFrameRef(recording, clip_id, sample, index, ts, {"intensity": len(refs) or 21}))
            targets[sample] = TemporalTarget((True, False, True, False), (_BOXES[0], None, _BOXES[2], None))
            observation = FrameObservation(recording, clip_id, sample, index, ts, (1, -0.2, -1, 0.7), _BOXES)
            for mode in modes:
                baseline = mode == "framewise"
                outcomes[mode].append(FrameDecision(
                    observation, (1, 0, -1, 1) if baseline else (0, 1, -1, 1), observation.scores, _BOXES,
                    ("raw",) * 4 if baseline else ("pending_confirmation", "score_ema+box_ema", "raw", "score_ema"),
                    "initial" if i == 0 else None,
                ))
    return TemporalSpatialPack(root, {}, tuple(refs)), targets, outcomes


@pytest.fixture
def rgb_reader(monkeypatch):
    read = []

    def pixels(root, ref):
        read.append((root, ref))
        return np.full((224, 224, 3), ref["intensity"], np.uint8)

    monkeypatch.setattr(gallery, "read_spatial_rgb", pixels)
    return read


def _render(tmp_path, pack, targets, outcomes, selected="score_box"):
    path = tmp_path / "gallery.html"
    gallery.write_temporal_spatial_gallery(path, pack, targets, outcomes, selected_name=selected)
    page = path.read_text(encoding="utf-8")
    data = json.loads(_DATA_PATTERN.search(page)[1])
    return page, data


def test_gallery_reuses_one_image_and_uses_exact_source_time_and_layout(tmp_path, rgb_reader):
    pack, targets, outcomes = _clip_pack(tmp_path)
    page, data = _render(tmp_path, pack, targets, outcomes)
    actual = [frame for clip in data["clips"] for frame in clip["frames"]]
    assert data["selected"] == "score_box"
    assert [mode["id"] for mode in data["modes"]] == ["framewise", "score_box"]
    assert [row["sample"] for row in actual] == [row.sample_id for row in pack.frames]
    assert [row["index"] for row in actual] == [10, 11, 12, 50, 51, 52]
    assert [row["offset_us"] for row in actual] == [0, 500211, 1000000] * 2
    assert rgb_reader == [(pack.root, ref.rgb) for ref in pack.frames]
    assert page.count("data:image/jpeg;base64,") == len(pack.frames)
    assert '<canvas id="left-image"' in page and '<canvas id="right-image"' in page
    for expected, rendered in zip(pack.frames, actual):
        jpeg = rendered["image"].split(",", 1)[1]
        with Image.open(BytesIO(b64decode(jpeg, validate=True))) as decoded:
            assert decoded.size == (224, 224)
            assert decoded.getpixel((0, 0)) == (expected.rgb["intensity"],) * 3


def test_rendered_boxes_follow_states_but_do_not_suppress_reference(tmp_path, rgb_reader):
    pack, targets, outcomes = _clip_pack(tmp_path)
    _, data = _render(tmp_path, pack, targets, outcomes)
    frame = data["clips"][0]["frames"][0]
    left, right = frame["predictions"]
    assert frame["truth"] == [list(_BOXES[0]), None, list(_BOXES[2]), None]
    assert left["states"] == [1, 0, -1, 1]
    assert left["boxes"] == [list(_BOXES[0]), None, None, list(_BOXES[3])]
    assert right["states"] == [0, 1, -1, 1]
    assert right["boxes"] == [None, list(_BOXES[1]), None, list(_BOXES[3])]
    assert right["reset"] == "initial"
    assert right["reasons"][1] == "score_ema+box_ema"


def test_full_nine_arm_gallery_covers_all_six_clips_within_budget(tmp_path, rgb_reader):
    pack, targets, outcomes = _clip_pack(tmp_path, clips=6, frames_per_clip=24, modes=_MODES)
    page, data = _render(tmp_path, pack, targets, outcomes)
    assert len(data["clips"]) == 6
    assert all(len(clip["frames"]) == 24 for clip in data["clips"])
    assert all(len(row["predictions"]) == 9 for clip in data["clips"] for row in clip["frames"])
    assert len(rgb_reader) == 144 == page.count("data:image/jpeg;base64,")
    assert len(page.encode()) < 16 * 1024**2


def test_script_payload_cannot_close_data_tag_or_add_html(tmp_path, rgb_reader):
    injected = '</script><img src=x onerror="alert(7)">'
    pack, targets, outcomes = _clip_pack(tmp_path, modes=("framewise", injected))
    previous = pack.frames[0]
    changed = replace(previous, sample_id=injected + "\u2028\u2029")
    pack = replace(pack, frames=(changed, *pack.frames[1:]))
    targets[changed.sample_id] = targets.pop(previous.sample_id)
    for rows in outcomes.values():
        rows[0] = replace(rows[0], observation=replace(rows[0].observation, sample_id=changed.sample_id))
    page, data = _render(tmp_path, pack, targets, outcomes, selected=injected)
    assert data["selected"] == injected
    assert data["clips"][0]["frames"][0]["sample"] == changed.sample_id
    assert injected not in page and "\u2028" not in page and "\u2029" not in page
    assert page.count("</script>") == 2
    assert "default-src 'none'" in page
    assert "innerHTML" not in page


@pytest.mark.parametrize("mismatch", ["identity", "timestamp", "clip", "recording", "source", "count", "target"])
def test_rejects_misaligned_modes_as_a_whole_before_reading_pixels(tmp_path, rgb_reader, mismatch):
    pack, targets, outcomes = _clip_pack(tmp_path)
    mode = outcomes["score_box"]
    if mismatch == "count":
        mode.pop()
    elif mismatch == "target":
        targets.pop(pack.frames[-1].sample_id)
    else:
        field = {
            "identity": "sample_id", "timestamp": "timestamp_us", "clip": "clip_id",
            "recording": "recording_id", "source": "source_index",
        }[mismatch]
        value = getattr(mode[-1].observation, field)
        mode[-1] = replace(mode[-1], observation=replace(
            mode[-1].observation, **{field: value + 1 if isinstance(value, int) else value + "-wrong"}
        ))
    with pytest.raises(ValueError, match="must cover|target per exact|frame identity or order"):
        _render(tmp_path, pack, targets, outcomes)
    assert not rgb_reader
    assert not (tmp_path / "gallery.html").exists()


@pytest.mark.parametrize("fault", ["missing", "backward", "long_gap", "duplicate", "seventh_clip"])
def test_refuses_broken_source_timeline_without_silently_skipping_frames(tmp_path, rgb_reader, fault):
    pack, targets, outcomes = _clip_pack(tmp_path, clips=7 if fault == "seventh_clip" else 2)
    if fault == "missing":
        pack = replace(pack, frames=(pack.frames[0], *pack.frames[2:]))
    elif fault == "backward":
        pack = replace(pack, frames=(pack.frames[1], pack.frames[0], *pack.frames[2:]))
    elif fault == "long_gap":
        pack = replace(pack, frames=(
            pack.frames[0], replace(pack.frames[1], timestamp_us=pack.frames[0].timestamp_us + 810_000),
            *pack.frames[2:],
        ))
    elif fault == "duplicate":
        pack = replace(pack, frames=(pack.frames[0], pack.frames[0], *pack.frames[2:]))
    with pytest.raises(ValueError, match="clip dimensions|missing a source frame|chronological|broken source|Duplicate"):
        _render(tmp_path, pack, targets, outcomes)
    assert not rgb_reader


def test_bad_hidden_boxes_or_nonboolean_reference_rejected_even_when_not_drawn(tmp_path, rgb_reader):
    pack, targets, outcomes = _clip_pack(tmp_path)
    original = outcomes["framewise"][-1]
    outcomes["framewise"][-1] = SimpleNamespace(
        observation=original.observation, states=original.states, reasons=original.reasons,
        reset_reason=None, boxes=(_BOXES[0], (float("nan"), .3, .6, .7), *_BOXES[2:]),
    )
    with pytest.raises(ValueError, match="boxes must be finite"):
        _render(tmp_path, pack, targets, outcomes)
    outcomes["framewise"][-1] = original
    targets[pack.frames[-1].sample_id] = SimpleNamespace(present=(1, False, False, False), boxes=(None,) * 4)
    with pytest.raises(ValueError, match="labels must be boolean"):
        _render(tmp_path, pack, targets, outcomes)
    assert not rgb_reader


def test_missing_baseline_or_selected_mode_cannot_render(tmp_path, rgb_reader):
    pack, targets, outcomes = _clip_pack(tmp_path)
    with pytest.raises(ValueError, match="both the framewise and selected"):
        _render(tmp_path, pack, targets, outcomes, selected="not-run")
    del outcomes["framewise"]
    with pytest.raises(ValueError, match="both the framewise and selected"):
        _render(tmp_path, pack, targets, outcomes)
    assert not rgb_reader


def test_bad_pixels_leave_no_published_or_partial_gallery(tmp_path, monkeypatch):
    pack, targets, outcomes = _clip_pack(tmp_path)

    def defective(_, ref):
        if ref is pack.frames[-1].rgb:
            return np.zeros((223, 224, 3), np.uint8)
        return np.zeros((224, 224, 3), np.uint8)

    monkeypatch.setattr(gallery, "read_spatial_rgb", defective)
    path = tmp_path / "output" / "gallery.html"
    with pytest.raises(ValueError, match="224×224"):
        gallery.write_temporal_spatial_gallery(path, pack, targets, outcomes, selected_name="score_box")
    assert not path.parent.exists()


def test_atomic_output_refuses_existing_file_pack_path_and_symlink(tmp_path, rgb_reader):
    pack, targets, outcomes = _clip_pack(tmp_path)
    path = tmp_path / "gallery.html"
    path.write_text("keep me")
    with pytest.raises(FileExistsError):
        gallery.write_temporal_spatial_gallery(path, pack, targets, outcomes, selected_name="score_box")
    assert path.read_text() == "keep me"
    with pytest.raises(ValueError, match="outside"):
        gallery.write_temporal_spatial_gallery(pack.root / "new.html", pack, targets, outcomes, selected_name="score_box")
    alias = tmp_path / "alias"
    alias.symlink_to(pack.root, target_is_directory=True)
    with pytest.raises(PreparationError, match="Symlink"):
        gallery.write_temporal_spatial_gallery(alias / "new.html", pack, targets, outcomes, selected_name="score_box")
    assert not rgb_reader and not (pack.root / "new.html").exists()
