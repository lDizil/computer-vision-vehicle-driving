from dataclasses import FrozenInstanceError, replace
import json
import pickle

import numpy as np
import pytest

from hybrid_drive import Pose2D, Trajectory
from hybrid_drive.contracts.motion import EgoMotion
from hybrid_drive.contracts.planning import Candidate, CandidateSet
from hybrid_drive.contracts.slow import SlowRequest
from hybrid_drive.slow.context import CAMERA_NAMES, MAX_METADATA_BYTES, CameraFrame, SlowContext
from hybrid_drive.slow.protocol import MAX_PROMPT_BYTES, make_prompt


def context(*, candidates=2, points=2):
    plan = CandidateSet("plan", "snapshot", 10, Pose2D(3, 4, 0.5), tuple(
        Candidate(f"candidate-{index}", Trajectory([[index + t, 0.2, 0.1] for t in range(points)], 0.25))
        for index in range(candidates)
    ))
    request = SlowRequest("request", plan, 100, 101, 10.5, "fixture-v1")
    cameras = tuple(CameraFrame(name, np.full((3, 5, 3), i, np.uint8)) for i, name in enumerate(CAMERA_NAMES))
    return SlowContext(request, EgoMotion(1, 0.1, 0.2, -0.1), cameras)


def test_pixels_are_owned_immutable_even_after_pickle_and_metadata_changes():
    source = np.arange(3 * 5 * 3, dtype=np.uint8).reshape(3, 5, 3)
    expected = source.copy()
    frame = CameraFrame("CAM_L0", source[:, ::-1])
    expected = expected[:, ::-1]
    source[:] = 255
    restored = pickle.loads(pickle.dumps(frame))
    for value in (frame, restored):
        np.testing.assert_array_equal(value.rgb, expected)
        with pytest.raises(ValueError):
            value.rgb.setflags(write=True)
        with pytest.raises(ValueError):
            value.rgb[0, 0, 0] = 12
        exposed = value.rgb
        exposed.shape = (5, 3, 3)
        exposed.dtype = np.int8
        assert value.rgb.shape == (3, 5, 3)
        assert value.rgb.dtype == np.uint8
        with pytest.raises(FrozenInstanceError):
            value.width = 4


@pytest.mark.parametrize("rgb", [
    None, [[[1, 2, 3]]], np.zeros((2, 2, 3)), np.zeros((2, 2, 3), np.int8),
    np.zeros((0, 2, 3), np.uint8), np.zeros((2, 0, 3), np.uint8),
    np.zeros((2, 2), np.uint8), np.zeros((2, 2, 4), np.uint8),
    np.zeros((481, 2, 3), np.uint8), np.zeros((2, 641, 3), np.uint8),
])
def test_frame_rejects_invalid_rgb_without_conversion_or_resize(rgb):
    with pytest.raises(ValueError):
        CameraFrame("CAM_F0", rgb)


@pytest.mark.parametrize("name", [None, "", "cam_f0", "CAM_B0", 1])
def test_frame_rejects_unknown_camera_names(name):
    with pytest.raises(ValueError):
        CameraFrame(name, np.zeros((1, 1, 3), np.uint8))


def test_context_preserves_maximum_pixels_and_bounded_plan_without_truncation():
    value = context(candidates=8, points=32)
    full_frame = CameraFrame("CAM_L0", np.zeros((480, 640, 3), np.uint8))
    value = replace(value, cameras=(full_frame,) + value.cameras[1:])
    restored = pickle.loads(pickle.dumps(value))
    assert restored.content_hash == value.content_hash
    assert restored.cameras[0].rgb.shape == (480, 640, 3)
    with pytest.raises(ValueError):
        restored.cameras[0].rgb.setflags(write=True)
    with pytest.raises(ValueError):
        restored.request.plan.candidates[0].trajectory.poses.setflags(write=True)
    payload = json.loads(make_prompt(restored).split("Context JSON:\n", 1)[1])
    candidates = payload["request"]["plan"]["candidates"]
    assert len(candidates) == 8
    assert all(len(candidate["poses"]) == len(candidate["times_s"]) == 32 for candidate in candidates)
    assert len(make_prompt(restored).encode("utf-8")) <= MAX_PROMPT_BYTES


@pytest.mark.parametrize("candidates,points", [(9, 2), (1, 33)])
def test_context_rejects_unbounded_candidate_work(candidates, points):
    with pytest.raises(ValueError):
        context(candidates=candidates, points=points)


@pytest.mark.parametrize("change", [
    {"request": None}, {"motion": (1, 0, 0, 0)}, {"cameras": None}, {"cameras": "CAM_F0"},
    {"cameras": ()}, {"cameras": (None, None, None)},
])
def test_context_rejects_missing_or_untyped_inputs(change):
    with pytest.raises(ValueError):
        replace(context(), **change)


def test_context_requires_exact_view_order_and_snapshots_camera_collection():
    value = context()
    with pytest.raises(ValueError):
        replace(value, cameras=value.cameras[::-1])
    with pytest.raises(ValueError):
        replace(value, cameras=(value.cameras[0],) * 3)
    frames = list(value.cameras)
    snapshot = replace(value, cameras=iter(frames))
    frames.clear()
    assert len(snapshot.cameras) == 3
    assert snapshot.content_hash == value.content_hash


@pytest.mark.parametrize("field,value", [
    ("request_id", "other"), ("model_version", "other"),
    ("submitted_at_monotonic_s", 99), ("deadline_monotonic_s", 102), ("valid_until_sim_s", 11),
])
def test_context_hash_binds_request_identity_and_both_clocks(field, value):
    original = context()
    changed = replace(original, request=replace(original.request, **{field: value}))
    assert changed.content_hash != original.content_hash


@pytest.mark.parametrize("field,value", [
    ("plan_id", "other"), ("snapshot_id", "other"), ("t_ref_s", 9),
    ("reference_pose", Pose2D(3.1, 4, 0.5)),
])
def test_context_hash_binds_plan_reference(field, value):
    original = context()
    plan = replace(original.request.plan, **{field: value})
    assert replace(original, request=replace(original.request, plan=plan)).content_hash != original.content_hash


def test_context_hash_binds_all_pixels_dimensions_motion_and_actual_trajectory():
    original = context()
    for index in range(3):
        pixels = original.cameras[index].rgb.copy()
        pixels[-1, -1, -1] += 1
        frames = list(original.cameras)
        frames[index] = CameraFrame(frames[index].name, pixels)
        assert replace(original, cameras=frames).content_hash != original.content_hash
    changed = CameraFrame("CAM_L0", np.zeros((5, 3, 3), np.uint8))
    assert replace(original, cameras=(changed,) + original.cameras[1:]).content_hash != original.content_hash
    for field in ("vx_mps", "vy_mps", "ax_mps2", "ay_mps2"):
        motion = replace(original.motion, **{field: getattr(original.motion, field) + 0.01})
        assert replace(original, motion=motion).content_hash != original.content_hash
    first, second = original.request.plan.candidates
    poses = first.trajectory.poses.copy()
    poses[-1, 1] += 0.1
    plan = replace(original.request.plan, candidates=(replace(first, trajectory=Trajectory(poses, 0.25)), second))
    changed = replace(original, request=replace(original.request, plan=plan))
    assert changed.content_hash != original.content_hash


def test_prompt_contains_actual_poses_grid_motion_views_and_expected_binding():
    value = context()
    prompt = make_prompt(value)
    payload = json.loads(prompt.split("Context JSON:\n", 1)[1])
    assert payload["context_hash"] == value.content_hash
    assert payload["motion"] == {"vx_mps": 1.0, "vy_mps": 0.1, "ax_mps2": 0.2, "ay_mps2": -0.1}
    assert [view["name"] for view in payload["cameras"]] == list(CAMERA_NAMES)
    first = payload["request"]["plan"]["candidates"][0]
    assert first == {"candidate_id": "candidate-0", "interval_s": 0.25,
                     "times_s": [0.25, 0.5], "poses": [[0.0, 0.2, 0.1], [1.0, 0.2, 0.1]]}
    assert "forward x metres, left y metres, heading radians" in prompt
    assert "not a measured camera timestamp" in prompt
    assert "do not invent geometric alignment" in prompt
    assert "not hidden reasoning or chain of thought" in prompt


@pytest.mark.parametrize("identifier", ["x" * MAX_METADATA_BYTES, "界" * (MAX_METADATA_BYTES // 6)])
def test_context_rejects_unbounded_metadata_before_worker_serialization(identifier):
    value = context()
    request = replace(value.request, request_id=identifier)
    with pytest.raises(ValueError, match="metadata exceeds 32 KiB"):
        replace(value, request=request)


def test_prompt_refuses_oversized_text_instead_of_omitting_candidates():
    value = context()
    payload = json.loads(make_prompt(value).split("Context JSON:\n", 1)[1])
    del payload["context_hash"]
    metadata_bytes = len(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    identifier = "x" * (MAX_METADATA_BYTES - metadata_bytes + len(value.request.request_id))
    request = replace(value.request, request_id=identifier)
    bounded_context = replace(value, request=request)
    with pytest.raises(ValueError, match="prompt exceeds 32 KiB"):
        make_prompt(bounded_context)
