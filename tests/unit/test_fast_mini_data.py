from io import BytesIO
import hashlib
import json
from math import pi, sqrt
from pathlib import Path
import tarfile

import numpy as np
from PIL import Image
import pytest

from hybrid_drive.adapters.nuscenes_surround import load_nuscenes_surround
from hybrid_drive.experiments import fast_mini_data as mini


def _fixture():
    tables = {name: [] for name in mini._TABLES}
    images = {}
    for channel in mini._CHANNELS:
        is_camera = channel != "LIDAR_TOP"
        tables["sensor"].append({
            "token": channel, "channel": channel, "modality": "camera" if is_camera else "lidar",
        })
        calibration = {
            "token": f"cal:{channel}", "sensor_token": channel,
            "translation": [0., 0., 0.], "rotation": [1., 0., 0., 0.],
        }
        if is_camera:
            calibration["camera_intrinsic"] = [[8., 0., 5.], [0., 8., 4.], [0., 0., 1.]]
        tables["calibrated_sensor"].append(calibration)

    names = ("scene-0001", "scene-0796", "scene-0103", "scene-0916")
    for scene_no, name in enumerate(names):
        tokens = [f"{name}:{i}" for i in range(4)]
        tables["scene"].append({"token": name, "name": name, "log_token": f"log:{scene_no}"})
        for i, token in enumerate(tokens):
            t_ref = 1_000_000 + i * 500_000 + scene_no * 5_000_000
            tables["sample"].append({
                "token": token, "timestamp": t_ref + 20_000, "scene_token": name,
                "prev": tokens[i-1] if i else "", "next": tokens[i+1] if i < 3 else "",
            })
            for view, channel in enumerate(mini._CHANNELS):
                camera = channel != "LIDAR_TOP"
                pose = {
                    "token": f"pose:{token}:{channel}",
                    "timestamp": t_ref + 7_000 * view if camera else t_ref,
                    "translation": [float(i), 0., 0.], "rotation": [1., 0., 0., 0.],
                }
                tables["ego_pose"].append(pose)
                row = {
                    "token": f"data:{token}:{channel}", "sample_token": token, "is_key_frame": True,
                    "timestamp": pose["timestamp"], "ego_pose_token": pose["token"],
                    "calibrated_sensor_token": f"cal:{channel}",
                }
                if camera:
                    path = f"samples/{channel}/{token}.jpg"
                    png = BytesIO()
                    Image.new("RGB", (12, 10), color=(30 + 50 * scene_no, 20 + i * 50, 20 * view)).save(png, "JPEG")
                    images[path] = png.getvalue()
                    row.update(filename=path, fileformat="jpg", width=12, height=10)
                else:
                    row["filename"] = "not-extracted-lidar.bin"
                tables["sample_data"].append(row)
    return tables, images


def _archive(path, tables, images, monkeypatch):
    members = {f"v1.0-mini/{name}.json": json.dumps(rows).encode() for name, rows in tables.items()}
    with tarfile.open(path, mode="w:gz") as dest:
        for name, body in (members | images | {"not-extracted-lidar.bin": b"data"}).items():
            entry = tarfile.TarInfo(name)
            entry.size = len(body)
            dest.addfile(entry, BytesIO(body))
    data = path.read_bytes()
    monkeypatch.setattr(mini, "_ARCHIVE_SIZE", len(data))
    monkeypatch.setattr(mini, "_ARCHIVE_SHA256", hashlib.sha256(data).hexdigest())
    return path


def test_source_to_surround_adapter_has_only_current_and_previous(tmp_path, monkeypatch):
    tables, images = _fixture()
    archive = _archive(tmp_path / "synthetic.tgz", tables, images, monkeypatch)
    output = tmp_path / "output"
    receipt = mini.prepare_fast_mini(archive, output)
    assert receipt["anchors"] == {"train": 2, "dev": 2, "test": 4}
    assert receipt["observed_keyframes"] == 12
    assert receipt["jpegs"] == 72
    assert receipt["largest_camera_skew_us"] == 35_000
    inputs = output / "inputs"
    index = json.loads((inputs / "index.json").read_text())
    assert index["schema"] == "hybrid-drive-fast-mini-inputs-v1"
    for name, role in {
        "scene-0001": "train", "scene-0796": "dev",
        "scene-0103": "test", "scene-0916": "test",
    }.items():
        anchors = [row for row in index["splits"][role] if row["scene"] == name]
        assert [(row["sample_token"], row["previous_token"]) for row in anchors] == [
            (f"{name}:1", f"{name}:0"), (f"{name}:2", f"{name}:1"),
        ]
        assert not (inputs / "samples" / "CAM_FRONT" / f"{name}:3.jpg").exists()

    assert {p.stem for p in (inputs / "v1.0-mini").glob("*.json")} == {
        "sample", "sample_data", "ego_pose", "calibrated_sensor", "sensor",
    }
    assert not any(inputs.rglob("*.bin"))
    assert not (inputs / "targets").exists()
    targets = json.loads((output / "targets" / "test.json").read_text())
    assert targets["scene-0103:1"] == {"next_dt_s": 0.5, "boxes": []}

    snapshot = load_nuscenes_surround(inputs, "scene-0103:1", max_skew_us=50_000)
    assert snapshot.ref_time_us == 11_500_000
    assert snapshot.measured_skew_us == 35_000
    np.testing.assert_allclose(snapshot.ego_to_world[:3, 3], [1, 0, 0])


def test_binary_provenance_and_cross_role_duplicate_rejected(tmp_path, monkeypatch):
    tables, images = _fixture()
    images["samples/CAM_FRONT/scene-0796:1.jpg"] = images["samples/CAM_FRONT/scene-0001:1.jpg"]
    archive = _archive(tmp_path / "synthetic.tgz", tables, images, monkeypatch)
    output = tmp_path / "result"
    with pytest.raises(ValueError, match="JPEG payload crosses"):
        mini.prepare_fast_mini(archive, output)
    assert not output.exists()
    assert not list(tmp_path.glob(".result-*"))
    monkeypatch.setattr(mini, "_ARCHIVE_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="SHA256"):
        mini.prepare_fast_mini(archive, output)


def test_same_recording_never_crosses_role_even_with_different_scene_name():
    info, _ = _fixture()
    scenes = {row["token"]: row for row in info["scene"]}
    scenes["scene-0998"] = {"name": "scene-0998", "log_token": scenes["scene-0796"]["log_token"]}
    assert mini._scene_roles(scenes)["scene-0998"] == "dev"
    scenes["scene-0998"]["log_token"] = scenes["scene-0916"]["log_token"]
    with pytest.raises(ValueError, match="recording overlaps"):
        mini._scene_roles(scenes)


def test_gate_checks_both_raw_lidar_intervals_and_preceding_camera_skew():
    samples = {
        "prev": {"token": "prev", "scene_token": "scene", "prev": "", "next": "now"},
        "now": {"token": "now", "scene_token": "scene", "prev": "prev", "next": "next"},
        "next": {"token": "next", "scene_token": "scene", "prev": "now", "next": ""},
    }
    refs = {"prev": 0, "now": 450_000, "next": 1_020_000}
    assert mini._eligible(samples["now"], samples, refs, {"now": 50_000, "prev": 50_000})
    assert not mini._eligible(samples["now"], samples, refs, {"now": 49_999, "prev": 50_001})
    assert not mini._eligible(samples["now"], samples, refs | {"next": 1_100_001}, {"now": 12_000, "prev": 9_000})
    assert not mini._eligible(samples["now"], samples, refs | {"prev": 100_001}, {"now": 12_000, "prev": 9_000})
    samples["next"]["scene_token"] = "other"
    with pytest.raises(ValueError, match="scene boundary"):
        mini._eligible(samples["now"], samples, refs, {"now": 50_000, "prev": 50_000})


def test_current_and_future_world_boxes_use_same_current_reference_and_immediate_instance():
    poses = {
        "current-pose": {"translation": [10., 20., 1.], "rotation": [sqrt(0.5), 0., 0., sqrt(0.5)]},
        "next-pose": {"translation": [100., -50., 99.], "rotation": [0., 1., 0., 0.]},
    }
    frame_data = {
        "now": {"LIDAR_TOP": {"ego_pose_token": "current-pose"}},
        "next": {"LIDAR_TOP": {"ego_pose_token": "next-pose"}},
    }

    def annotation(token, sample, instance, visibility, xyz):
        return dict(token=token, sample_token=sample, instance_token=instance, visibility_token=visibility,
                    translation=xyz, size=[2., 4., 1.5], rotation=[0., 0., 0., 1.])

    rows = [
        annotation("a", "now", "motor", "2", [8., 21., 2.5]),
        annotation("b", "next", "motor", "1", [7., 22., 3.]),
        annotation("c", "now", "gone", "4", [6., 21., 2.]),
        annotation("d", "later", "gone", "3", [9., 24., 1.75]),
        annotation("e", "now", "occluded", "1", [5., 22., 2.5]),
        annotation("f", "now", "cycle", "4", [5., 24., 2.5]),
    ]
    tables = {
        "category": {"car": {"name": "vehicle.car"}, "bicycle": {"name": "vehicle.bicycle"}},
        "instance": {key: {"category_token": "bicycle" if key == "cycle" else "car"}
                     for key in ("motor", "gone", "occluded", "cycle")},
        "sample_annotation": {row["token"]: row for row in rows},
    }
    selected = mini._target("now", "next", mini._vehicle_annotations(tables), frame_data, poses,
                            {"now": 1_000_000, "next": 1_550_000})
    assert selected["next_dt_s"] == 0.55
    assert [box["instance_id"] for box in selected["boxes"]] == ["gone", "motor"]
    gone, motor = selected["boxes"]
    assert gone["future_center"] is None
    np.testing.assert_allclose(motor["center"], [1, 2, 1.5], atol=1e-12)
    np.testing.assert_allclose(motor["future_center"], [2, 3, 2], atol=1e-12)
    assert motor["yaw"] == pytest.approx(pi / 2)
    assert motor["size_wlh"] == [2, 4, 1.5]
