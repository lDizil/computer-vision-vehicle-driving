from math import sqrt
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from hybrid_drive.adapters.nuscenes_surround import load_nuscenes_surround
from hybrid_drive.contracts.motion import EgoMotion
from hybrid_drive.contracts.sensors import ViewDirection


_CAMERAS = (
    "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT", "CAM_BACK_RIGHT", "CAM_BACK",
)
_DIRECTION_ORDER = (
    ViewDirection.FRONT, ViewDirection.FRONT_LEFT, ViewDirection.FRONT_RIGHT,
    ViewDirection.REAR_LEFT, ViewDirection.REAR_RIGHT, ViewDirection.REAR,
)
_SAMPLE = "record-at-t0"
_REFERENCE_US = 1_532_402_927_647_951
_OFFSETS_US = (-35_000, -20_000, 10_000, -10_000, 20_000, 5_000)
_SPAN_US = 55_000


def _write_table(root: Path, name: str, rows):
    (root / "v1.0-mini" / f"{name}.json").write_text(json.dumps(rows), encoding="utf-8")


def _selected(table, token):
    return next(row for row in table if row["token"] == token)


@pytest.fixture
def mini(tmp_path):
    root = tmp_path / "nuScenes"
    db = root / "v1.0-mini"
    db.mkdir(parents=True)
    tables = {
        "sample": [
            # Deliberately no devkit-injected "data" member.
            {"token": _SAMPLE, "timestamp": _REFERENCE_US},
            {"token": "future-sample", "timestamp": _REFERENCE_US + 500_000},
        ],
        "sample_data": [
            {
                "token": "lidar-row", "sample_token": _SAMPLE, "is_key_frame": True,
                "timestamp": _REFERENCE_US, "calibrated_sensor_token": "lidar-cal",
                "ego_pose_token": "lidar-pose", "filename": "not-downloaded.bin",
            },
            {
                "token": "future-frame", "sample_token": "future-sample", "is_key_frame": True,
                "filename": "not-downloaded-future.jpg",
            },
            {
                "token": "front-sweep", "sample_token": _SAMPLE, "is_key_frame": False,
                "calibrated_sensor_token": "CAM_FRONT-cal", "filename": "unavailable-sweep.jpg",
            },
        ],
        "sensor": [{"token": "lidar-sensor", "channel": "LIDAR_TOP", "modality": "lidar"}],
        "calibrated_sensor": [{
            "token": "lidar-cal", "sensor_token": "lidar-sensor",
            # If these are used as the reference the projection assertion fails.
            "translation": [99.0, 7.0, 4.0], "rotation": [0.0, 1.0, 0.0, 0.0],
        }],
        "ego_pose": [{
            "token": "lidar-pose", "timestamp": _REFERENCE_US,
            "translation": [10.0, 20.0, 0.0], "rotation": [sqrt(0.5), 0.0, 0.0, sqrt(0.5)],
        }],
    }
    for index, (name, offset) in enumerate(zip(_CAMERAS, _OFFSETS_US)):
        capture = _REFERENCE_US + offset
        path = Path("samples", name, "capture.jpg")
        jpeg = root / path
        jpeg.parent.mkdir(parents=True)
        Image.new("RGB", (24, 18), color=(25 * index,) * 3).save(jpeg)
        tables["sensor"].append({"token": f"{name}-sensor", "channel": name, "modality": "camera"})
        tables["sample_data"].append({
            "token": f"{name}-row", "sample_token": _SAMPLE, "is_key_frame": True, "timestamp": capture,
            "ego_pose_token": f"{name}-pose", "calibrated_sensor_token": f"{name}-cal",
            "filename": path.as_posix(), "fileformat": "jpg", "width": 24, "height": 18,
        })
        tables["ego_pose"].append({
            "token": f"{name}-pose", "timestamp": capture,
            "translation": [8.0, 21.0, 0.0], "rotation": [0.0, 0.0, 0.0, 1.0],
        })
        tables["calibrated_sensor"].append({
            "token": f"{name}-cal", "sensor_token": f"{name}-sensor",
            "translation": [1.0, -0.5, 1.5], "rotation": [0.5, -0.5, 0.5, -0.5],
            "camera_intrinsic": [[14.0, 0.0, 12.0], [0.0, 14.0, 9.0], [0.0, 0.0, 1.0]],
        })
    for name, table in tables.items():
        table.reverse()
        _write_table(root, name, table)
    (db / "sample_annotation.json").write_text("invalid and never read", encoding="utf-8")
    return root, tables


def test_surround_uses_sensor_clock_and_motion_compensated_anchor(mini):
    root, _ = mini
    motion = EgoMotion(3.25, -0.1, 0.15, 0.0)
    snapshot = load_nuscenes_surround(root, _SAMPLE, max_skew_us=_SPAN_US, motion=motion)

    assert snapshot.snapshot_id == _SAMPLE
    assert snapshot.rig_id == "nuscenes-v1.0-mini"
    assert snapshot.ref_time_us == _REFERENCE_US
    assert snapshot.measured_skew_us == _SPAN_US
    assert snapshot.motion == motion
    assert tuple(v.direction for v in snapshot.views) == _DIRECTION_ORDER
    assert tuple(v.calibration.name for v in snapshot.views) == _CAMERAS
    assert tuple(v.capture_us for v in snapshot.views) == tuple(_REFERENCE_US + v for v in _OFFSETS_US)
    assert [int(v.rgb[0, 0, 0]) for v in snapshot.views] == [25 * i for i in range(6)]
    np.testing.assert_allclose(snapshot.ego_to_world, [
        [0, -1, 0, 10], [1, 0, 0, 20], [0, 0, 1, 0], [0, 0, 0, 1],
    ], atol=1e-12)

    # Probe (1, 10, 0) at reference -> (0, 21, 0) in the world ->
    # (8, 0, 0) in the pi-rotated exposure ego frame. Camera right/down/forward
    # of the offset point (7, .5, -1.5) is (-.5, 1.5, 7).
    front = snapshot.views[0].calibration
    xyz = front.ego_to_camera @ [1.0, 10.0, 0.0, 1.0]
    np.testing.assert_allclose(xyz, [-0.5, 1.5, 7.0, 1.0], atol=1e-12)
    pixel = front.intrinsic @ xyz[:3]
    np.testing.assert_allclose(pixel[:2] / pixel[2], [11.0, 12.0], atol=1e-12)


def test_unknown_ego_motion_is_not_fabricated(mini):
    snapshot = load_nuscenes_surround(mini[0], _SAMPLE, max_skew_us=_SPAN_US)
    assert snapshot.motion is None


def test_rejects_camera_skew_of_one_microsecond_over_bound(mini):
    with pytest.raises(ValueError, match="skew"):
        load_nuscenes_surround(mini[0], _SAMPLE, max_skew_us=_SPAN_US - 1)


@pytest.mark.parametrize("name", ["CAM_BACK-row", "lidar-row"])
def test_refuses_to_borrow_keyframe_from_other_sample(mini, name):
    root, tables = mini
    _selected(tables["sample_data"], name)["sample_token"] = "future-sample"
    _write_table(root, "sample_data", tables["sample_data"])
    with pytest.raises(ValueError, match="Missing keyframe channels"):
        load_nuscenes_surround(root, _SAMPLE, max_skew_us=_SPAN_US)


def test_rejects_duplicate_keyframe_without_guessing(mini):
    root, tables = mini
    duplicate = dict(_selected(tables["sample_data"], "CAM_FRONT-row"), token="other-camera-row")
    tables["sample_data"].append(duplicate)
    _write_table(root, "sample_data", tables["sample_data"])
    with pytest.raises(ValueError, match="Duplicate keyframe channel CAM_FRONT"):
        load_nuscenes_surround(root, _SAMPLE, max_skew_us=_SPAN_US)


def test_rejects_ego_pose_for_different_sensor_capture(mini):
    root, tables = mini
    row = _selected(tables["sample_data"], "CAM_FRONT-row")
    row["ego_pose_token"] = "CAM_BACK-pose"
    _write_table(root, "sample_data", tables["sample_data"])
    with pytest.raises(ValueError, match="ego_pose timestamp differs"):
        load_nuscenes_surround(root, _SAMPLE, max_skew_us=_SPAN_US)


@pytest.mark.parametrize("filename", [
    "/tmp/capture.jpg",
    "samples/CAM_FRONT_LEFT/capture.jpg",
    "samples/CAM_FRONT/../../outside.jpg",
    "samples/CAM_FRONT//capture.jpg",
])
def test_rejects_paths_not_in_camera_allowlist(mini, filename):
    root, tables = mini
    _selected(tables["sample_data"], "CAM_FRONT-row")["filename"] = filename
    _write_table(root, "sample_data", tables["sample_data"])
    with pytest.raises(ValueError, match="requires its own samples"):
        load_nuscenes_surround(root, _SAMPLE, max_skew_us=_SPAN_US)


def test_jpeg_symlink_cannot_escape_the_dataset(mini):
    root, _ = mini
    jpeg = root / "samples" / "CAM_FRONT" / "capture.jpg"
    outside = root.parent / "outside.jpg"
    jpeg.rename(outside)
    jpeg.symlink_to(outside)
    with pytest.raises(ValueError, match="inside the extracted dataset"):
        load_nuscenes_surround(root, _SAMPLE, max_skew_us=_SPAN_US)


@pytest.mark.parametrize("wrong_jpeg", [False, True])
def test_rejects_image_format_or_shape_mismatch(mini, wrong_jpeg):
    root, tables = mini
    if wrong_jpeg:
        Image.new("RGB", (24, 18)).save(root / "samples" / "CAM_FRONT" / "capture.jpg", format="PNG")
    else:
        _selected(tables["sample_data"], "CAM_FRONT-row")["width"] = 22
        _write_table(root, "sample_data", tables["sample_data"])
    with pytest.raises(ValueError, match="JPEG"):
        load_nuscenes_surround(root, _SAMPLE, max_skew_us=_SPAN_US)
