import gzip
import hashlib
import io
import json
from math import sqrt
from pathlib import Path
import pickle
import tarfile
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from hybrid_drive.adapters import openscene_surround as adapter
from hybrid_drive.assets.files import PreparationError
from hybrid_drive.contracts.sensors import SURROUND_ORDER, TimingBasis


SOURCES = Path(__file__).resolve().parents[2] / "scripts" / "surround-openscene-sources.json"


def test_source_pin_rejects_a_caller_changed_metadata_digest(tmp_path, monkeypatch):
    monkeypatch.setattr(adapter, "sys", SimpleNamespace(platform="linux"))
    changed = json.loads(SOURCES.read_text())
    changed["log"]["sha256"] = "0" * 64
    override = tmp_path / "override.json"
    override.write_text(json.dumps(changed))
    with pytest.raises(PreparationError, match="pinned manifest"):
        adapter.load_openscene_surround(tmp_path / "no-meta.tgz", tmp_path / "no-cameras.tgz", override)


def test_loader_reuses_only_the_pinned_archives_and_uses_all_six_matched_views(monkeypatch):
    source = adapter._load_source(SOURCES)
    official = json.loads((SOURCES.parent / "alignment-data-sources.json").read_text())
    assert source["metadata_archive"] == official["metadata_archive"]
    assert source["camera_archive"] == official["camera_archive"]

    frame, images = _scene(source)
    # Resize actual small RGB JPEGs while keeping all production associations.
    source["native_size"] = [8, 4]
    source["rectified_size"] = [4, 2]
    monkeypatch.setattr(adapter, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(adapter, "_load_source", lambda _: source)
    calls = []

    def read_members(path, spec, entries):
        calls.append((path, spec, [entry["member"] for entry in entries]))
        if path == "metadata":
            return {source["log"]["member"]: pickle.dumps([frame] * source["log"]["frame_count"])}
        return images

    monkeypatch.setattr(adapter, "_read_verified_members", read_members)
    snapshot = adapter.load_openscene_surround("metadata", "cameras", SOURCES)

    assert calls == [
        ("metadata", official["metadata_archive"], [source["log"]["member"]]),
        ("cameras", official["camera_archive"], [entry["member"] for entry in source["cameras"]]),
    ]
    assert snapshot.snapshot_id == "openscene-v1.1-mini:68b1d9f6bfbe5650"
    assert snapshot.rig_id == "openscene-v1.1-mini-six-selected-v1"
    assert snapshot.ref_time_us == 1_633_542_871_299_594
    assert snapshot.timing is TimingBasis.SOURCE_ASSOCIATION
    assert snapshot.measured_skew_us is None and snapshot.max_skew_us is None
    assert all(view.capture_us is None for view in snapshot.views)
    assert tuple(view.direction for view in snapshot.views) == SURROUND_ORDER
    assert [view.calibration.name for view in snapshot.views] == [
        "CAM_F0", "CAM_L0", "CAM_R0", "CAM_L2", "CAM_R2", "CAM_B0",
    ]
    assert snapshot.motion.vx_mps == 4
    assert snapshot.motion.ay_mps2 == -0.25
    np.testing.assert_allclose(snapshot.ego_to_world @ [1, 0, 0, 1], [9, 6, 2, 1], atol=1e-12)
    for index, view in enumerate(snapshot.views):
        assert view.rgb.shape == (2, 4, 3)
        assert view.rgb[1, 2, 0] == pytest.approx(30 * (index + 1), abs=1)
        np.testing.assert_allclose(view.calibration.intrinsic, [[3, 0, 1.25], [0, 3, 0.25], [0, 0, 1]])
        np.testing.assert_array_equal(
            view.calibration.ego_to_camera,
            [[0, -1, 0, index + 2], [0, 0, -1, 3], [1, 0, 0, -1], [0, 0, 0, 1]],
        )


def test_frame_with_a_different_identity_cannot_reach_snapshot_construction(monkeypatch):
    source = adapter._load_source(SOURCES)
    frame, _ = _scene(source)
    frame["scene_token"] = "unrelated-recording"
    verified = {source["log"]["member"]: pickle.dumps([frame] * source["log"]["frame_count"])}
    monkeypatch.setattr(adapter, "_read_verified_members", lambda *_: verified)
    with pytest.raises(PreparationError, match="pinned identity"):
        adapter._load_frame("metadata", source)


def test_camera_cannot_be_substituted_with_a_member_from_another_frame():
    source = adapter._load_source(SOURCES)
    frame, images = _scene(source)
    frame["cams"]["CAM_F0"]["data_path"] = frame["cams"]["CAM_L0"]["data_path"]
    with pytest.raises(PreparationError, match="pinned frame association"):
        adapter._snapshot(source, frame, images)


def test_member_reader_keeps_only_pinned_payloads_in_memory(tmp_path):
    archive, spec = _archive(tmp_path, [("unselected.jpg", b"skip"), ("camera.jpg", b"pixels")])
    assert adapter._read_verified_members(archive, spec, [_entry("camera.jpg", b"pixels")]) == {
        "camera.jpg": b"pixels",
    }


@pytest.mark.parametrize(("rows", "expected"), [
    ([("camera.jpg", b"alter!")], "pinned SHA256"),
    ([("camera.jpg", b"pixels-more")], "pinned size"),
    ([("unrelated.jpg", b"pixels")], "missing a pinned member"),
    ([("camera.jpg", b"pixels"), ("camera.jpg", b"pixels")], "Duplicate OpenScene tar member"),
])
def test_member_reader_rejects_replacements_even_with_a_matching_archive_digest(tmp_path, rows, expected):
    archive, spec = _archive(tmp_path, rows)
    with pytest.raises(PreparationError, match=expected):
        adapter._read_verified_members(archive, spec, [_entry("camera.jpg", b"pixels")])


def test_archive_reader_rejects_inconsistent_full_archive_digest(tmp_path):
    archive, spec = _archive(tmp_path, [("camera.jpg", b"pixels")])
    with pytest.raises(PreparationError, match="archive failed pinned SHA256"):
        adapter._read_verified_members(archive, {**spec, "sha256": "0" * 64}, [_entry("camera.jpg", b"pixels")])


def _scene(source):
    frame = {
        "log_name": source["log"]["log_name"],
        "frame_idx": source["frame"]["index"],
        "token": source["frame"]["token"],
        "scene_token": source["frame"]["scene_token"],
        "timestamp": source["frame"]["timestamp_us"],
        "ego2global_translation": [9, 5, 2],
        "ego2global_rotation": [sqrt(0.5), 0, 0, sqrt(0.5)],
        "ego_dynamic_state": [4, 1, -0.5, -0.25],
        "cams": {},
    }
    images = {}
    archive_prefix = source["camera_archive"]["archive_prefix"]
    for index, entry in enumerate(source["cameras"]):
        frame["cams"][entry["camera"]] = {
            "data_path": entry["member"].removeprefix(archive_prefix + "/"),
            "cam_intrinsic": [[6, 0, 3], [0, 6, 1], [0, 0, 1]],
            "sensor2lidar_rotation": [[0, 0, 1], [-1, 0, 0], [0, -1, 0]],
            "sensor2lidar_translation": [1, index + 2, 3],
            "distortion": [0, 0, 0, 0, 0],
        }
        buf = io.BytesIO()
        Image.new("RGB", (8, 4), (30 * (index + 1),) * 3).save(buf, format="JPEG")
        images[entry["member"]] = buf.getvalue()
    return frame, images


def _archive(tmp_path, rows):
    path = tmp_path / "test.tgz"
    with gzip.open(path, "wb") as zipped:
        with tarfile.open(fileobj=zipped, mode="w", format=tarfile.USTAR_FORMAT) as archive:
            for name, payload in rows:
                member = tarfile.TarInfo(name)
                member.size = len(payload)
                archive.addfile(member, io.BytesIO(payload))
    spec = _entry(path.name, path.read_bytes())
    return path, spec


def _entry(name, payload):
    return {"member": name, "size_bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
