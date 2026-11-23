import numpy as np
import pytest

from hybrid_drive.contracts.motion import EgoMotion
from hybrid_drive.contracts.sensors import CameraSnapshot, SURROUND_ORDER, SurroundView, TimingBasis
from hybrid_drive.geometry.camera import CameraCalibration
from hybrid_drive.geometry.rigid import invert_pose, pose_from_quaternion


def _view(direction, *, capture=None, name=None, value=32):
    calibration = CameraCalibration(
        name or direction.value, 7, 5, [[3, 0, 2], [0, 4, 1], [0, 0, 1]], np.eye(4)
    )
    return SurroundView(direction, calibration, np.full((5, 7, 3), value, dtype=np.uint8), capture_us=capture)


def _snapshot(views, *, max_skew_us=None, pose=None):
    return CameraSnapshot(
        "snap-a", 10_000, np.eye(4) if pose is None else pose, EgoMotion(4, -0.25, 1, 0.1),
        views, rig_id="fixture-rig", max_skew_us=max_skew_us,
    )


def test_snapshot_orders_views_and_binds_pixels_calibration_and_time():
    views = [_view(direction, capture=9_990 + i, value=20+i) for i, direction in enumerate(SURROUND_ORDER)]
    shuffled = _snapshot(reversed(views), max_skew_us=10)
    assert tuple(v.direction for v in shuffled.views) == SURROUND_ORDER
    assert shuffled.measured_skew_us == 10  # Includes the reference, outside the camera-only span.
    assert shuffled.timing is TimingBasis.CAMERA_CAPTURE
    assert shuffled.content_sha256 == _snapshot(views, max_skew_us=12).content_sha256
    changed = list(views)
    changed[2] = _view(SURROUND_ORDER[2], capture=9_992, value=99)
    assert shuffled.content_sha256 != _snapshot(changed, max_skew_us=10).content_sha256
    changed[2] = SurroundView(changed[2].direction, changed[2].calibration.resized(3, 2),
                              np.full((2, 3, 3), 22, np.uint8), capture_us=9_992)
    assert shuffled.content_sha256 != _snapshot(changed, max_skew_us=10).content_sha256
    changed[2] = _view(SURROUND_ORDER[2], capture=9_993, value=22)
    assert shuffled.content_sha256 != _snapshot(changed, max_skew_us=10).content_sha256


@pytest.mark.parametrize("violation, message", [
    ("missing", "exactly six"), ("duplicate_direction", "unique"), ("duplicate_name", "unique"),
    ("mixed_clocks", "Cannot mix"), ("outside", "skew"), ("no_limit", "explicit"),
    ("unverifiable_limit", "association-only"),
])
def test_incomplete_or_falsely_synchronized_snapshot_rejected(violation, message):
    views = [_view(direction, capture=10_000) for direction in SURROUND_ORDER]
    limit = 30
    if violation == "missing":
        views.pop()
    elif violation == "duplicate_direction":
        views[1] = views[0]
    elif violation == "duplicate_name":
        views[1] = _view(SURROUND_ORDER[1], name=views[0].calibration.name, capture=10_000)
    elif violation == "mixed_clocks":
        views[1] = _view(SURROUND_ORDER[1])
    elif violation == "outside":
        views[4] = _view(SURROUND_ORDER[4], capture=10_016)
        views[5] = _view(SURROUND_ORDER[5], capture=9_985)
    elif violation == "no_limit":
        limit = None
    else:
        views = [_view(direction) for direction in SURROUND_ORDER]
    with pytest.raises(ValueError, match=message):
        _snapshot(views, max_skew_us=limit)


def test_association_snapshot_keeps_unknown_clock_and_owns_rasters_and_pose():
    raw = np.full((5, 7, 3), 71, np.uint8)
    base = [_view(direction) for direction in SURROUND_ORDER]
    base[0] = SurroundView(base[0].direction, base[0].calibration, raw)
    pose = np.eye(4)
    pose[:3, 3] = [2, -4, 1]
    snap = _snapshot(base, pose=pose)
    raw.fill(9)
    pose[:3, 3] = 0
    base.reverse()
    assert snap.timing is TimingBasis.SOURCE_ASSOCIATION
    assert snap.measured_skew_us is None and all(v.capture_us is None for v in snap.views)
    assert np.all(snap.views[0].rgb == 71)
    np.testing.assert_array_equal(snap.ego_to_world[:3, 3], [2, -4, 1])
    with pytest.raises(ValueError, match="read-only"):
        snap.views[0].rgb[0, 0, 0] = 9
    with pytest.raises(ValueError, match="read-only"):
        snap.ego_to_world[0, 3] = 0


def test_snapshot_rejects_incompatible_raster_and_reflection():
    view = _view(SURROUND_ORDER[0])
    with pytest.raises(ValueError, match="matching calibration"):
        SurroundView(view.direction, view.calibration, np.zeros((7, 5, 3), np.uint8))
    with pytest.raises(ValueError, match="proper orthonormal"):
        _snapshot([_view(d) for d in SURROUND_ORDER], pose=np.diag([-1, 1, 1, 1]))


def test_quaternion_pose_and_inverse_use_column_vector_wxyz_convention():
    q = np.sqrt(0.5)
    pose = pose_from_quaternion((3, 5, 2), (q, 0, 0, q))
    np.testing.assert_allclose(pose @ [2, 1, -1, 1], [2, 7, 1, 1], atol=1e-12)
    np.testing.assert_allclose(invert_pose(pose) @ [2, 7, 1, 1], [2, 1, -1, 1], atol=1e-12)
    with pytest.raises(ValueError, match="unit quaternion"):
        pose_from_quaternion((1, 2, 3), (2, 0, 0, 0))
