import sys
from textwrap import dedent

import numpy as np
import pytest

from hybrid_drive.experiments.vru_depthpro2 import DepthPro2, DepthPro2Error, _to_optical_depth
from hybrid_drive.geometry.camera import CameraCalibration


_READ_FRAME = """
from struct import pack, unpack
from sys import stdin, stdout

states = {}
while header := stdin.buffer.read(8):
    magic, slot, reset, reserved = unpack("<4sBBH", header)
    assert (magic, reserved) == (b"DPQ2", 0)
    assert 0 <= slot < 6 and reset in (0, 1)
    rgb = stdin.buffer.read(640 * 360 * 3)
    assert len(rgb) == 640 * 360 * 3
    assert list(rgb[:3]) == list(rgb[-3:]) == [10, 20, 30]
"""


def _start(tmp_path, body):
    worker = tmp_path / "worker"
    worker.write_text("#!" + sys.executable + "\n" + dedent(_READ_FRAME).lstrip() + body)
    worker.chmod(0o700)
    model = tmp_path / "model.mlmodelc"
    model.mkdir()
    return DepthPro2(worker, model, timeout_s=8.0)


def _calibration(name):
    return CameraCalibration(name, 768, 432, [[576., 2., 383.5], [0., 580., 215.5], [0., 0., 1.]], np.eye(4))


def _rgb():
    rgb = np.empty((432, 768, 3), dtype=np.uint8)
    rgb[:] = (10, 20, 30)
    return rgb


def test_camera_stream_history_isolated_and_clip_reset(tmp_path):
    body = """
    states[slot] = 1 if reset else states[slot] + 1
    payload = pack("<f", 0.125 / states[slot]) * (640 * 360)
    stdout.buffer.write(pack("<4sI", b"DPR2", len(payload)) + payload)
    stdout.buffer.flush()
"""
    rgb, a, b = _rgb(), _calibration("front"), _calibration("right")
    with _start(tmp_path, body) as depth:
        first = depth.predict(rgb, a, stream_key="a")
        assert first.shape == (432, 768) and first.dtype == np.float32
        np.testing.assert_array_equal(first, 6.0)
        assert depth.predict(rgb, b, stream_key="b")[0, 0] == 6.0
        assert depth.predict(rgb, a, stream_key="a")[0, 0] == 12.0
        with pytest.raises(ValueError, match="before changing its camera"):
            depth.predict(rgb, b, stream_key="a")
        assert depth.predict(rgb, a, stream_key="a", reset=True)[0, 0] == 6.0
        assert depth.predict(rgb, b, stream_key="b")[0, 0] == 12.0
        depth.reset_streams()
        assert depth.predict(rgb, b, stream_key="a")[0, 0] == 6.0
    with pytest.raises(DepthPro2Error, match="closed"):
        depth.predict(rgb, a, stream_key="a")


def test_truncated_output_cannot_advance_shared_camera_state(tmp_path):
    body = """
    stdout.buffer.write(pack("<4sI", b"DPR2", 640 * 360 * 4) + b"x")
    stdout.buffer.flush()
    break
"""
    rgb, a = _rgb(), _calibration("front")
    with _start(tmp_path, body) as depth:
        with pytest.raises(DepthPro2Error, match="closed its reply early"):
            depth.predict(rgb, a, stream_key="a")
        with pytest.raises(DepthPro2Error, match="worker is closed"):
            depth.predict(rgb, a, stream_key="a")


def test_undo_resize_keeps_invalid_and_outline_depth_discontinuities():
    inverse = np.full((360, 640), 0.125, dtype="<f4")
    inverse[0, :4] = (0.0, np.nan, -1.0, np.inf)
    inverse[:, 639] = 0.5
    depth = _to_optical_depth(inverse.tobytes(), 480.0)
    assert np.isnan(depth[0, :5]).all()
    assert depth[-1, 0] == 6.0
    assert depth[0, -1] == 1.5
    assert depth[-1, -1] == 1.5
    assert depth[-1, -2] == 6.0
