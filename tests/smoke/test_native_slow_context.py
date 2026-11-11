from dataclasses import replace
import json

import numpy as np
from PIL import Image
import pytest
from navsim.common.dataclasses import AgentInput, Camera, Cameras

from hybrid_drive import Pose2D, Trajectory
from hybrid_drive.adapters.slow_context import context_from_navsim
from hybrid_drive.contracts.motion import EgoMotion
from hybrid_drive.contracts.planning import Candidate, CandidateSet
from hybrid_drive.contracts.slow import SlowRequest
from hybrid_drive.slow.protocol import make_prompt, parse_response


def inputs():
    images = [np.arange(h * w * 3, dtype=np.uint8).reshape(h, w, 3) + offset
              for h, w, offset in ((7, 9, 0), (5, 6, 30), (8, 4, 60))]
    current = Cameras(cam_l0=Camera(image=images[0]), cam_f0=Camera(image=images[1]),
                      cam_r0=Camera(image=images[2]), cam_l1=Camera(), cam_l2=Camera(),
                      cam_r1=Camera(), cam_r2=Camera(), cam_b0=Camera())
    plan = CandidateSet("native-plan", "native-input-association", 10, Pose2D(0, 0, 0),
                        (Candidate("fast", Trajectory([[1, 0, 0], [2, 0, 0]], 0.5)),))
    request = SlowRequest("native-request", plan, 100, 101, 10.5, "fixture-v1")
    return request, EgoMotion(2, 0, 0, 0), AgentInput([], [current], []), images


def test_native_current_views_resize_independently_without_history_or_privileged_inputs():
    request, motion, native, images = inputs()

    class ForbiddenHistory:
        def __getattr__(self, name):
            raise AssertionError("Historical or privileged inputs must not be read")

    native.cameras.insert(0, ForbiddenHistory())
    native.ego_statuses = ForbiddenHistory()
    native.lidars = ForbiddenHistory()
    value = context_from_navsim(request, motion, native, width=4, height=3)
    assert [frame.name for frame in value.cameras] == ["CAM_L0", "CAM_F0", "CAM_R0"]
    for frame, image in zip(value.cameras, images):
        expected = np.asarray(Image.fromarray(image).resize((4, 3), resample=Image.Resampling.BILINEAR))
        np.testing.assert_array_equal(frame.rgb, expected)
        assert frame.rgb.shape == (3, 4, 3)
        with pytest.raises(ValueError):
            frame.rgb.setflags(write=True)
    original_hash = value.content_hash
    for image in images:
        image[:] = 255
    assert value.content_hash == original_hash
    text = make_prompt(value)
    payload = json.loads(text.split("Context JSON:\n", 1)[1])
    assert payload["request"]["plan"]["snapshot_id"] == "native-input-association"
    raw = json.dumps({"context_hash": value.content_hash, "status": "VETO",
                      "vetoed_candidate_ids": ["fast"], "explanation": "Visible front obstruction."})
    parse_response(raw, value).validate_for(request)


@pytest.mark.parametrize("rgb", [
    None, np.zeros((2, 2, 4), np.uint8), np.zeros((2, 2, 3), np.float32),
    np.zeros((0, 2, 3), np.uint8), [[[1, 2, 3]]],
])
def test_native_adapter_does_not_silently_convert_invalid_rgb(rgb):
    request, motion, native, _ = inputs()
    native.cameras[-1].cam_f0.image = rgb
    with pytest.raises(ValueError):
        context_from_navsim(request, motion, native)


@pytest.mark.parametrize("width,height", [
    (0, 1), (1, 0), (641, 1), (1, 481), (True, 1), (1, False), (4.0, 3), (4, 3.0),
])
def test_native_adapter_requires_explicit_bounded_integer_output_dimensions(width, height):
    request, motion, native, _ = inputs()
    with pytest.raises(ValueError):
        context_from_navsim(request, motion, native, width=width, height=height)


def test_native_adapter_requires_all_current_views_not_a_stale_camera_history():
    request, motion, native, _ = inputs()
    native.cameras.append(replace(native.cameras[-1], cam_r0=Camera()))
    with pytest.raises(ValueError):
        context_from_navsim(request, motion, native)
    native.cameras.clear()
    with pytest.raises(ValueError):
        context_from_navsim(request, motion, native)
