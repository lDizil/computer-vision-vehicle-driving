import base64
from dataclasses import FrozenInstanceError, replace
from hashlib import sha256
import json
from math import sqrt
import struct

import numpy as np
import pytest

from hybrid_drive.experiments import spatial_appearance_features as features
from hybrid_drive.experiments.spatial_decision_policy import ABSENT, PRESENT, UNCERTAIN
from hybrid_drive.experiments.temporal_spatial_types import FrameDecision, FrameObservation


BOXES = (
    (0.0, 0.0, 0.25, 0.5),
    (0.25, 0.0, 0.5, 0.5),
    (0.5, 0.0, 0.75, 0.5),
    (0.75, 0.0, 1.0, 0.5),
)
MODEL_HASH = "a" * 64
CAPTURE_HASH = "b" * 64
NONE = (None,) * 4


def decision(sample="sample-0", *, states=(PRESENT, UNCERTAIN, ABSENT, ABSENT), boxes=BOXES):
    observation = FrameObservation("recording", "clip", sample, 2, 2_000_000, (0.8, 0.3, 0.1, 0.2), boxes)
    return FrameDecision(observation, states, observation.scores, observation.boxes, ("same",) * 4, None)


def pack_row(sample="sample-0"):
    return features.FrameAppearance(sample, NONE, ((-0.2, 0.125), None, None, None), NONE)


def write(path, decisions, values):
    return features.write_appearance_trace(
        path, decisions, values, model_hash=MODEL_HASH, capture_hash=CAPTURE_HASH,
    )


def read(path, decisions, expected, **override):
    options = dict(model_hash=MODEL_HASH, capture_hash=CAPTURE_HASH)
    options.update(override)
    return features.read_appearance_trace(path, decisions, expected_sha256=expected, **options)


def tamper(path, edit):
    value = json.loads(path.read_bytes())
    edit(value)
    raw = json.dumps(value, separators=(",", ":"), allow_nan=False).encode()
    path.write_bytes(raw)
    return sha256(raw).hexdigest()


def test_only_visible_box_palettes_affect_color_profile():
    rgb = np.zeros((3, 24, 24), dtype=np.float32)
    rgb[0, :12, :6] = 1.0  # red
    rgb[2, :12, 6:12] = 1.0  # blue
    grid = np.broadcast_to(np.array([3.0, -4.0], dtype=np.float32)[:, None, None], (2, 4, 4))
    d = decision(states=(PRESENT, PRESENT, UNCERTAIN, ABSENT))

    first = features._extract_frame(d, rgb, grid)
    red = np.array(first.rgb[0]).reshape(3, 8)
    blue = np.array(first.rgb[1]).reshape(3, 8)
    assert np.argmax(red, axis=1).tolist() == [7, 0, 0]
    assert np.argmax(blue, axis=1).tolist() == [0, 0, 7]
    assert np.linalg.norm(red) == pytest.approx(1.0)
    assert first.dino[0] == pytest.approx((0.6, -0.8))
    assert first.contrast == NONE  # no contrast in a uniform patch grid

    rgb[:, :12, 12:] = 1.0  # change two boxes that cannot emit appearance
    assert features._extract_frame(d, rgb, grid) == first
    assert first.rgb[2:] == first.dino[2:] == (None, None)


def test_fractional_cell_overlap_and_nearest_context_ring():
    # The ROI covers 3/4 of the first patch column and 1/4 of the
    # second by area. The opposite half of the image is not a local background.
    rgb = np.zeros((3, 16, 32), dtype=np.float32)
    grid = np.zeros((2, 4, 4), dtype=np.float32)
    grid[0, :, 0] = 4.0
    grid[1, :, 1] = 8.0
    grid[:, :, 3] = 100.0
    boxes = ((0.0625, 0.25, 0.3125, 0.75), *BOXES[1:])

    row = features._extract_frame(decision(boxes=boxes), rgb, grid)
    assert row.dino[0] == pytest.approx((3 / sqrt(13), 2 / sqrt(13)))
    # Hand-integrated expanded area is .4375 × .75. Context contributions
    # exclude the entire inner box's weighted area, not just its center pixel.
    outer = np.array([4 * .25, 8 * .1875]) * .75
    inner = np.array([3., 2.]) * .25 * .5
    outside = (outer - inner) / (.4375 * .75 - .25 * .5)
    expected = np.array([3., 2.]) - outside
    expected /= np.linalg.norm(expected)
    assert row.contrast[0] == pytest.approx(expected)
    assert row.dino[1:] == row.contrast[1:] == (None, None, None)


def test_half_open_crop_does_not_read_border_pixel_at_upper_center():
    rgb = np.zeros((3, 8, 16), dtype=np.float32)
    # xhi at centre of pixel 4, excluded. Pixel 0 also excluded.
    rgb[:, :, 0] = rgb[:, :, 4] = 1.0
    box = (0.5 / 16 + 1e-8, 0.0, 4.5 / 16, .5)
    frame = decision(boxes=(box, *BOXES[1:]))
    result = features._extract_frame(frame, rgb, np.ones((2, 2, 2), dtype=np.float32))
    assert result.rgb[0] is not None
    palette = np.array(result.rgb[0]).reshape(3, 8)
    assert np.count_nonzero(palette) == 3
    assert [np.argmax(ch) for ch in palette] == [0, 0, 0]


def test_tiny_visible_crop_and_zero_patch_vector_are_unavailable():
    rgb = np.zeros((3, 16, 16), dtype=np.float32)
    grid = np.zeros((2, 4, 4), dtype=np.float32)
    state = (PRESENT, PRESENT, ABSENT, ABSENT)
    d = decision(states=state, boxes=((0.02, 0.02, .03, .03), *BOXES[1:]))
    row = features._extract_frame(d, rgb, grid)

    assert row.rgb[0] is None
    assert row.rgb[1] is not None
    assert row.dino == row.contrast == NONE


def test_frame_vectors_are_deeply_immutable_even_if_given_mutable_sources():
    source = [[1, -0.375], None, None, None]
    row = features.FrameAppearance("sample-0", [None] * 4, source, [None] * 4)
    source[0][0] = 4.0
    assert row.dino == ((1., -0.375), None, None, None)
    with pytest.raises(FrozenInstanceError):
        row.dino = NONE


def test_bound_compact_roundtrip_and_no_overwrite(tmp_path):
    path = tmp_path / "appearance.json"
    d = (decision(), decision("sample-1"))
    row = (pack_row(), pack_row("sample-1"))
    digest = write(path, d, row)

    raw = path.read_bytes()
    assert digest == sha256(raw).hexdigest()
    doc = json.loads(raw)
    assert doc["dimensions"] == {"rgb": None, "dino": 2, "contrast": None}
    first = doc["frames"][0]
    assert struct.unpack("<2f", base64.b64decode(first["dino"][0])) == row[0].dino[0]
    assert set(first) == {"sample_id", *features.APPEARANCE_KINDS}
    assert read(path, d, digest) == row

    with pytest.raises(FileExistsError):
        write(path, d, row)
    assert path.read_bytes() == raw
    assert list(tmp_path.iterdir()) == [path]


def test_absent_vectors_have_null_dimension_and_roundtrip(tmp_path):
    path = tmp_path / "empty-support.json"
    row = features.FrameAppearance("sample-0", NONE, NONE, NONE)
    # A positive decision still has no appearance when support is unavailable.
    digest = write(path, [decision()], [row])
    assert json.loads(path.read_bytes())["dimensions"] == dict.fromkeys(features.APPEARANCE_KINDS)
    assert read(path, [decision()], digest) == (row,)


def test_reopened_trace_binds_full_decisions_model_and_capture(tmp_path):
    path = tmp_path / "appearance.json"
    d = decision()
    digest = write(path, (d,), (pack_row(),))

    for override in ({"model_hash": "c" * 64}, {"capture_hash": "c" * 64}):
        with pytest.raises(ValueError, match="bound inputs"):
            read(path, (d,), digest, **override)

    changed_box = ((0.01, .02, .25, .5), *BOXES[1:])
    changed = (
        replace(d, scores=(0.9, *d.scores[1:])),
        replace(d, boxes=changed_box),
        replace(d, observation=replace(d.observation, timestamp_us=2_000_001)),
        replace(d, reasons=("revised", *d.reasons[1:])),
    )
    for alternative in changed:
        with pytest.raises(ValueError, match="bound inputs"):
            read(path, (alternative,), digest)


def test_recompute_byte_hash_does_not_suppress_alignment_or_decision_mask(tmp_path):
    path = tmp_path / "appearance.json"
    d = (decision(), decision("sample-1"))

    write(path, d, (pack_row(), pack_row("sample-1")))
    digest = tamper(path, lambda doc: doc["frames"].reverse())
    with pytest.raises(ValueError, match="order changed"):
        read(path, d, digest)

    path.unlink()
    write(path, d, (pack_row(), pack_row("sample-1")))
    digest = tamper(path, lambda doc: doc["frames"][0]["dino"].__setitem__(1, doc["frames"][0]["dino"][0]))
    with pytest.raises(ValueError, match="non-present"):
        read(path, d, digest)


@pytest.mark.parametrize("invalid", ("nan", "garbage", "shape", "protocol", "extra"))
def test_trace_reader_rejects_rehashed_malformed_content(tmp_path, invalid):
    path = tmp_path / "appearance.json"
    d = (decision(),)
    write(path, d, (pack_row(),))

    def edit(doc):
        if invalid == "nan":
            doc["frames"][0]["dino"][0] = base64.b64encode(struct.pack("<2f", float("nan"), 1)).decode()
        elif invalid == "garbage":
            doc["frames"][0]["dino"][0] = "!" * 12
        elif invalid == "shape":
            doc["dimensions"]["dino"] = 384
        elif invalid == "protocol":
            doc["protocol"]["min_rgb_pixels_per_axis"] = False
        else:
            doc["frames"][0]["labels"] = [0, 0, 0, 0]

    digest = tamper(path, edit)
    with pytest.raises(ValueError, match="finite float32|base64|byte shape|protocol|unknown keys"):
        read(path, d, digest)


def test_accepts_only_exact_bound_bytes_even_for_equivalent_json(tmp_path):
    path = tmp_path / "appearance.json"
    d = (decision(),)
    digest = write(path, d, (pack_row(),))
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="byte hash differs"):
        read(path, d, digest)


def test_writer_checks_full_order_duplicate_sample_and_consistent_width(tmp_path):
    path = tmp_path / "appearance.json"
    a, b = decision(), decision("sample-1")
    wrong_width = features.FrameAppearance("sample-1", NONE, ((0.5,), None, None, None), NONE)

    with pytest.raises(ValueError, match="order changed"):
        write(path, (a, b), (pack_row("sample-1"), pack_row()))
    with pytest.raises(ValueError, match="duplicate sample"):
        write(path, (a, a), (pack_row(), pack_row()))
    with pytest.raises(ValueError, match="Inconsistent dino"):
        write(path, (a, b), (pack_row(), wrong_width))
    with pytest.raises(ValueError, match="non-present"):
        write(path, (decision(states=(ABSENT,) * 4),), (pack_row(),))
    assert not path.exists()


@pytest.mark.parametrize("invalid", ((0, 0), (1.0, True), (1.0, 4e45), (1.0, float("inf"))))
def test_frame_rejects_non_embedding(invalid):
    with pytest.raises(ValueError, match="vector|embedding"):
        features.FrameAppearance("sample-0", NONE, (invalid, None, None, None), NONE)


def test_recipe_copy_never_changes_frozen_protocol():
    writable = features.descriptor_protocol()
    writable["version"] = 101
    assert features.descriptor_protocol()["version"] == 1
    with pytest.raises(TypeError):
        features.DESCRIPTOR_PROTOCOL["version"] = 101


def test_cpu_torch_entry_point_preserves_input_and_batch_order():
    torch = pytest.importorskip("torch")
    a, b = decision(), decision("sample-1")
    image = torch.zeros(2, 1, 3, 24, 24)
    image[0, 0, 0, :, :] = 1.0
    image[1, 0, 2, :, :] = 1.0
    grid = torch.ones(2, 1, 4, 4, 4)
    before = image.clone()

    rows = features.extract_spatial_appearance((a, b), image, grid)
    assert torch.equal(before, image)
    assert [t.sample_id for t in rows] == ["sample-0", "sample-1"]
    assert rows[0].rgb[0] != rows[1].rgb[0]
    assert rows[0].dino[0] == (0.5,) * 4
    assert rows[0].contrast[0] is None

    with pytest.raises(ValueError, match="CPU FP32"):
        features.extract_spatial_appearance((a, b), image.double(), grid)
    with pytest.raises(ValueError, match="three RGB channels"):
        features.extract_spatial_appearance((a, b), image + 0.1, grid)
    with pytest.raises(ValueError, match="Nonfinite DINO"):
        features.extract_spatial_appearance((a, b), image, grid * float("nan"))
