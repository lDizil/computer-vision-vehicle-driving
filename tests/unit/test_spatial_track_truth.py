import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import pickle
import tarfile
from types import SimpleNamespace

import numpy as np
import pytest

from hybrid_drive.adapters.real_camera import OPENSCENE_NATIVE_PROFILE, openscene_camera_calibrations
from hybrid_drive.assets import spatial_manifest as spatial
from hybrid_drive.assets import spatial_track_truth as truth
from hybrid_drive.assets.alignment_manifest import ARCHIVE_VERIFICATION, CAMERA, IMAGE_SIZE, digest_json, rgb_path
from hybrid_drive.assets.alignment_sample import _frame_calibration_receipt
from hybrid_drive.assets.files import PreparationError
from hybrid_drive.assets.spatial_sample import _spatial_target
from hybrid_drive.assets.spatial_temporal import read_temporal_spatial_pack, verify_independent_temporal_final


SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
ROLE = "development"
NATIVE_WIDTH, NATIVE_HEIGHT = 640, 360
RAW_TRACKS = ("fe7f9102307baabc", "54daaf13022baabf")


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _sources(role):
    return json.loads((SCRIPTS / f"spatial-temporal-{role.replace('_', '-')}-sources.json").read_text())


def _archive(path, entries):
    with path.open("wb") as fd, gzip.GzipFile(fileobj=fd, mode="wb", mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode="w", format=tarfile.USTAR_FORMAT) as tar:
            for name, raw in entries:
                member = tarfile.TarInfo(name)
                member.size = len(raw)
                tar.addfile(member, io.BytesIO(raw))
    data = path.read_bytes()
    return {"size_bytes": len(data), "sha256": _sha(data)}


def _store(root, relative, payload):
    data = json.dumps(payload).encode()
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {"path": relative, "size_bytes": len(data), "sha256": _sha(data)}


def _fixture(tmp, monkeypatch, *, fault=None, role=ROLE):
    monkeypatch.setattr(truth, "sys", SimpleNamespace(platform="linux"))
    sources = _sources(role)
    root = tmp / "pack"
    root.mkdir()

    camera = {
        "cam_intrinsic": [[250, 0, 320], [0, 250, 180], [0, 0, 1]],
        "sensor2lidar_rotation": [[0, 0, 1], [-1, 0, 0], [0, -1, 0]],
        "sensor2lidar_translation": [0, 0, 1.5],
        "distortion": [0, 0, 0, 0, 0],
    }
    _, calibration, _ = openscene_camera_calibrations(
        CAMERA, NATIVE_WIDTH, NATIVE_HEIGHT, camera["cam_intrinsic"], camera["sensor2lidar_rotation"],
        camera["sensor2lidar_translation"], camera["distortion"], width=IMAGE_SIZE, height=IMAGE_SIZE,
        profile=OPENSCENE_NATIVE_PROFILE,
    )
    receipt = _frame_calibration_receipt(camera, NATIVE_WIDTH, NATIVE_HEIGHT, calibration)
    calibration_sha = digest_json(receipt)
    rows, recordings, logs = [], [], []
    for number, spec in enumerate(sources["logs"]["holdout"], 1):
        spec["frame_count"] = 72
        log_name, recording_id = spec["log_name"], spec["recording_id"]
        prefix = sources["camera_archives"][str(spec["camera_shard"])]["archive_prefix"]
        log_token = f"{number:016x}"
        metadata, samples = [], []
        for index in range(spec["frame_count"]):
            token = f"{number * 1000 + index:016x}"
            relative = f"{log_name}/CAM_F0/{token}.jpg"
            # Same physical track crosses the centerline between two boxes.
            offset = 4 if index % 2 else -4
            tracks = list(RAW_TRACKS)
            first = number == 1 and index == 0
            if first and fault == "duplicate":
                tracks[1] = tracks[0]
            elif first and fault == "shape":
                tracks.pop()
            anns = {
                "gt_names": ["vehicle", "vehicle"],
                "gt_boxes": np.array([[20, offset, 0.8, 4, 1.7, 1.5, 0], [20, -offset, 0.8, 4, 1.7, 1.5, 0]]),
                "instance_tokens": [f"{number * 10000 + index * 2 + i:016x}" for i in range(2)],
                "track_tokens": tracks,
            }
            frame = {"token": token, "timestamp": 1_000_000 + 500_000 * index, "frame_idx": index,
                     "scene_token": f"{number * 10:016x}", "log_token": log_token, "log_name": log_name,
                     "cams": {CAMERA: {**camera, "data_path": relative}}, "anns": anns}
            metadata.append(frame)
            sample = spatial.sample_id("holdout", recording_id, index)
            rgb = rgb_path(sample)
            pixels = f"{recording_id}:{index}".encode()
            path = root / rgb
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(pixels)  # The sidecar must never try to decode these bytes.
            samples.append({
                "sample_id": sample, "source_list_index": index, "frame_token": token,
                "scene_token": frame["scene_token"], "frame_idx": index,
                "timestamp_us": frame["timestamp"] + (1 if first and fault == "timestamp" else 0),
                "calibration_sha256": "1" * 64 if first and fault == "calibration" else calibration_sha,
                "jpeg": {"member": f"{prefix}/{relative}", "width": NATIVE_WIDTH, "height": NATIVE_HEIGHT,
                         "size_bytes": len(pixels), "sha256": _sha(b"jpeg" + pixels)},
                "rgb": {"path": rgb, "size_bytes": len(pixels), "sha256": _sha(pixels), "pixels_sha256": _sha(pixels)},
            })
            target = _spatial_target(anns, calibration, sample)
            assert target["presence"] == [False, True, True, False]
            if first and fault == "labels":
                target["presence"][1], target["boxes_2d"][1] = False, None
            rows.append(target)
        payload = pickle.dumps(metadata)
        spec.update(size_bytes=len(payload), sha256=_sha(payload))
        logs.append((spec["member"], payload))
        recordings.append({"recording_id": recording_id, "log_token": log_token, "samples": samples})

    archive = tmp / "metadata.tgz"
    sources["metadata_archive"].update(_archive(archive, logs))
    monkeypatch.setitem(spatial._TEMPORAL_SOURCE_HASHES, role, digest_json(sources))
    targets = _store(root, "targets/holdout.json", {"schema": spatial.TARGET_SCHEMA, "split": "holdout", "rows": rows})
    pack = {"schema": spatial.MANIFEST_SCHEMA, "sources": sources, "archive_verification": ARCHIVE_VERIFICATION,
            "splits": {"holdout": {"recordings": recordings, "targets": targets}}}
    pack["manifest_hash"] = digest_json(pack)
    _store(root, spatial.MANIFEST_NAME, pack)
    return read_temporal_spatial_pack(root, role=role), archive, tmp / "truth.json"


@pytest.mark.parametrize("role", [ROLE, spatial.ZOOM_GATE_HOLDOUT_ROLE])
def test_tracks_follow_raw_track_tokens_across_bands_and_have_recording_scope(tmp_path, monkeypatch, role):
    pack, archive, path = _fixture(tmp_path, monkeypatch, role=role)
    original = os.open

    def no_rgb(file, *args, **kwargs):
        if not isinstance(file, int) and Path(file).parent == pack.root / "rgb":
            pytest.fail("Tracking truth opened RGB bytes")
        return original(file, *args, **kwargs)

    monkeypatch.setattr(os, "open", no_rgb)
    digest = truth.prepare_track_truth(pack, archive, path)
    assert _sha(path.read_bytes()) == digest
    assert all(token not in path.read_text() for token in RAW_TRACKS)
    assert json.loads(path.read_bytes())["temporal_role"] == role
    labels = truth.read_track_truth(pack, path, expected_sha256=digest)
    assert list(labels) == [frame.sample_id for frame in pack.frames]
    a, b, other = [labels[pack.frames[index].sample_id] for index in (0, 1, 72)]
    assert a[0] is a[3] is None and a[1] == b[2] and a[2] == b[1] and a[1] != a[2]
    assert set(a[1:3]).isdisjoint(other[1:3])
    first = pack.frames[0]
    assert a[1] == _sha(f"{first.recording_id}\0{RAW_TRACKS[1]}".encode())
    with pytest.raises(FileExistsError, match="already exists"):
        truth.prepare_track_truth(pack, archive, path)
    assert digest == _sha(path.read_bytes())


@pytest.mark.parametrize(("fault", "error"), [
    ("duplicate", "Duplicate source vehicle"),
    ("shape", "tracking IDs do not align"),
    ("calibration", "metadata calibration differs"),
    ("labels", "prepared target boxes or presence"),
    ("timestamp", "metadata frame identity differs"),
])
def test_preparation_rejects_misalignment_before_publishing(tmp_path, monkeypatch, fault, error):
    pack, archive, path = _fixture(tmp_path, monkeypatch, fault=fault)
    with pytest.raises(ValueError, match=error):
        truth.prepare_track_truth(pack, archive, path)
    assert not path.exists()


def test_sidecar_targets_are_role_bound_ordered_and_shape_checked(tmp_path, monkeypatch):
    pack, archive, path = _fixture(tmp_path, monkeypatch)
    digest = truth.prepare_track_truth(pack, archive, path)
    original = json.loads(path.read_bytes())
    with pytest.raises(PreparationError, match="byte hash differs"):
        truth.read_track_truth(pack, path, expected_sha256="0" * 64)

    original["temporal_role"] = "holdout"
    path.write_text(json.dumps(original))
    with pytest.raises(PreparationError, match="does not bind"):
        truth.read_track_truth(pack, path, expected_sha256=_sha(path.read_bytes()))

    original["temporal_role"] = ROLE
    rows = original["rows"]
    rows[0], rows[1] = rows[1], rows[0]
    path.write_text(json.dumps(original))
    with pytest.raises(PreparationError, match="prepared target order"):
        truth.read_track_truth(pack, path, expected_sha256=_sha(path.read_bytes()))

    rows[0], rows[1] = rows[1], rows[0]
    rows[0]["tracks"][1] = None
    path.write_text(json.dumps(original))
    with pytest.raises(PreparationError, match="track presence differs"):
        truth.read_track_truth(pack, path, expected_sha256=_sha(path.read_bytes()))


def test_path_inside_immutable_pack_or_changed_metadata_is_rejected(tmp_path, monkeypatch):
    pack, archive, path = _fixture(tmp_path, monkeypatch)
    with pytest.raises(PreparationError, match="outside the prepared pack"):
        truth.prepare_track_truth(pack, archive, pack.root / "new-truth.json")
    with archive.open("ab") as fd:
        fd.write(b"changed after pinning")
    with pytest.raises(PreparationError, match="size mismatch|size or SHA|archive.*size|Archive"):
        truth.prepare_track_truth(pack, archive, path)
    assert not path.exists()


def test_tracking_holdout_role_and_pins_preserve_legacy_roles(tmp_path, monkeypatch):
    new_role = spatial.TRACKING_HOLDOUT_ROLE
    profile = _sources(new_role)
    assert spatial.validate_spatial_sources(profile) == ("holdout",)
    assert profile["temporal_role"] == "tracking_holdout"
    assert all(log["sample_count"] == 72 for log in profile["logs"]["holdout"])
    for old_role in ("development", "holdout", "stress"):
        old = _sources(old_role)
        assert spatial.validate_spatial_sources(old) == ("holdout",)
        assert profile["selection"] == old["selection"] and profile["loc"] == old["loc"]
        assert not {spec["recording_id"] for spec in old["logs"]["holdout"]}.intersection(
            spec["recording_id"] for spec in profile["logs"]["holdout"]
        )
    profile["temporal_role"] = "holdout"
    with pytest.raises(PreparationError, match="pinned recordings"):
        spatial.validate_spatial_sources(profile)

    # The dedicated evaluator must opt in to the new role; old final readers stay strict.
    from hybrid_drive.assets import spatial_temporal

    pack = SimpleNamespace(manifest={"sources": {"schema": spatial.TEMPORAL_SOURCE_SCHEMA, "temporal_role": new_role}})
    with pytest.raises(PreparationError, match="pinned temporal holdout"):
        verify_independent_temporal_final(pack, [tmp_path])
    checked = []
    monkeypatch.setattr(spatial_temporal, "read_spatial_manifest", lambda root: root)
    monkeypatch.setattr(spatial_temporal, "verify_disjoint_spatial_manifests",
                        lambda fresh, previous, **_: checked.append((fresh, previous)))
    verify_independent_temporal_final(pack, [tmp_path], expected_role=new_role)
    assert checked == [(pack.manifest, tmp_path)]
