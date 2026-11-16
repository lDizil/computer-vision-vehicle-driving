import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from hybrid_drive.adapters.real_camera import CalibratedView, OPENSCENE_NATIVE_PROFILE
from hybrid_drive.contracts.motion import EgoMotion
from hybrid_drive.contracts.trajectory import Trajectory
from hybrid_drive.experiments.protocol import trajectory_hash
from hybrid_drive.experiments.real_bundle import (
    MAX_MANIFEST_BYTES, MAX_PNG_BYTES, RealSceneBundle, load_real_bundle, write_real_bundle,
)
from hybrid_drive.geometry.camera import CameraCalibration
from hybrid_drive.slow.context import CAMERA_NAMES, CameraFrame
from scripts import export_real_bundle as exporter


@pytest.fixture
def bundle():
    trajectory = Trajectory([[1, 0, 0], [2, 0.2, 0.1]], 0.5)
    views = tuple(CalibratedView(
        CameraFrame(name, np.full((6, 8, 3), index * 70, np.uint8)),
        CameraCalibration(name, 8, 6, [[5, 0, 4], [0, 5, 3], [0, 0, 1]], np.eye(4)),
    ) for index, name in enumerate(CAMERA_NAMES))
    provenance = {
        "source_revision": "a" * 40, "navsim_revision": "b" * 40, "nuplan_revision": "c" * 40,
        "metadata_sha256": "d" * 64, "source_assets_manifest_hash": "e" * 64,
        "source_experiment_manifest_hash": "f" * 64, "raw_candidate_hash": trajectory_hash(trajectory),
        "checkpoint_sha256": "1" * 64, "source_image_hashes": {name: "2" * 64 for name in CAMERA_NAMES},
        "profile": OPENSCENE_NATIVE_PROFILE,
    }
    return RealSceneBundle("0123456789abcdef", 1234567, trajectory, EgoMotion(2, 0, 0, 0), views, provenance)


def test_roundtrip_preserves_pixels_calibration_candidate_and_hash_without_rectification(tmp_path, bundle, monkeypatch):
    path = tmp_path / "bundle"
    write_real_bundle(bundle, path)
    monkeypatch.setattr(exporter, "prepare_openscene_camera", lambda *a, **k: pytest.fail("Repeated rectification"))
    restored = load_real_bundle(path)
    assert restored.bundle_hash == bundle.bundle_hash
    assert restored.provenance == bundle.provenance
    assert restored.frame_timestamp_us == bundle.frame_timestamp_us
    np.testing.assert_array_equal(restored.trajectory.poses, bundle.trajectory.poses)
    for before, after in zip(bundle.views, restored.views):
        np.testing.assert_array_equal(before.frame.rgb, after.frame.rgb)
        np.testing.assert_array_equal(before.calibration.intrinsic, after.calibration.intrinsic)
        np.testing.assert_array_equal(before.calibration.ego_to_camera, after.calibration.ego_to_camera)
    manifest = json.loads((path / "bundle.json").read_text())
    assert all(camera["capture_timestamp_us"] is None for camera in manifest["cameras"])
    assert all(camera["state"] == "rectified" and camera["effective_distortion"] == [0] * 5
               for camera in manifest["cameras"])


def test_hash_is_stable_and_provenance_owned(tmp_path, bundle):
    provenance = bundle.provenance
    clone = RealSceneBundle(bundle.scene_token, bundle.frame_timestamp_us, bundle.trajectory,
                            bundle.motion, bundle.views, provenance)
    provenance["source_image_hashes"]["CAM_F0"] = "9" * 64
    clone.provenance["source_revision"] = "0" * 40
    assert clone.bundle_hash == bundle.bundle_hash
    write_real_bundle(bundle, tmp_path / "one")
    write_real_bundle(clone, tmp_path / "two")
    assert (tmp_path / "one/bundle.json").read_bytes() == (tmp_path / "two/bundle.json").read_bytes()


@pytest.mark.parametrize("change", [
    lambda m: m.update(annotations=[]),
    lambda m: m["provenance"].update(deadline_monotonic_s=1),
    lambda m: m["trajectory"].update(reference_poses=[]),
    lambda m: m["cameras"][0].update(path="../CAM_L0.png"),
    lambda m: m["cameras"][0].update(width=9),
    lambda m: m["cameras"][0].update(width=True),
    lambda m: m["cameras"][0].update(state="original"),
    lambda m: m["cameras"][0].update(effective_distortion=[0.1, 0, 0, 0, 0]),
    lambda m: m["cameras"][0].update(capture_timestamp_us=123),
    lambda m: m["motion"].update(vx_mps=float("nan")),
    lambda m: m["motion"].update(vx_mps=10**400),
    lambda m: m["trajectory"].update(poses=[[1, 0, 0]] * 33),
])
def test_manifest_tamper_or_unknown_contract_rejected(tmp_path, bundle, change):
    path = tmp_path / "bundle"
    write_real_bundle(bundle, path)
    manifest = json.loads((path / "bundle.json").read_text())
    change(manifest)
    (path / "bundle.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        load_real_bundle(path)


@pytest.mark.parametrize("kind", ["duplicate", "manifest_size", "png_size", "png_tamper", "symlink", "extra"])
def test_bounded_files_and_no_symlinks(tmp_path, bundle, kind):
    path = tmp_path / "bundle"
    write_real_bundle(bundle, path)
    image = path / "CAM_L0.png"
    if kind == "duplicate":
        manifest = path / "bundle.json"
        manifest.write_text(manifest.read_text().replace('"schema":', '"schema":"duplicate","schema":', 1))
    elif kind == "manifest_size":
        (path / "bundle.json").write_bytes(b" " * (MAX_MANIFEST_BYTES + 1))
    elif kind == "png_size":
        with image.open("wb") as output:
            output.truncate(MAX_PNG_BYTES + 1)
    elif kind == "png_tamper":
        image.write_bytes(image.read_bytes() + b"tamper")
    elif kind == "symlink":
        target = tmp_path / "other.png"
        image.rename(target)
        image.symlink_to(target)
    else:
        (path / "labels.json").write_text("{}")
    with pytest.raises((ValueError, OSError)):
        load_real_bundle(path)


def test_reject_unknown_provenance_or_candidate_mismatch(bundle):
    for provenance in ({**bundle.provenance, "metrics": {}}, {**bundle.provenance, "raw_candidate_hash": "0" * 64}):
        with pytest.raises(ValueError):
            RealSceneBundle(bundle.scene_token, bundle.frame_timestamp_us, bundle.trajectory,
                            bundle.motion, bundle.views, provenance)


def test_writer_never_overwrites_existing_bundle(tmp_path, bundle):
    path = tmp_path / "bundle"
    write_real_bundle(bundle, path)
    with pytest.raises(FileExistsError):
        write_real_bundle(bundle, path)
    assert load_real_bundle(path).bundle_hash == bundle.bundle_hash


def test_metadata_hash_is_checked_before_pickle(tmp_path, monkeypatch):
    (tmp_path / "log.pkl").write_bytes(b"untrusted")
    sources = {"log": {"path": "log.pkl", "size_bytes": 9, "sha256": exporter.VERIFIED_METADATA_SHA256}}
    monkeypatch.setattr(exporter.pickle, "loads", lambda data: pytest.fail("Unverified pickle was deserialized"))
    with pytest.raises(ValueError, match="before pickle"):
        exporter._verified_current_frame(tmp_path, sources, {"frame_index": 3})


def test_current_frame_selection_does_not_touch_future_or_privileged_fields(tmp_path, monkeypatch):
    data = b"verified-fixture"
    digest = hashlib.sha256(data).hexdigest()
    (tmp_path / "log.pkl").write_bytes(data)
    monkeypatch.setattr(exporter, "VERIFIED_METADATA_SHA256", digest)

    class Current(dict):
        def __getitem__(self, name):
            assert name in {"token", "frame_idx"}, "Privileged metadata was read"
            return super().__getitem__(name)

    class Forbidden:
        def __getattribute__(self, name):
            raise AssertionError("Noncurrent frame was read")

    current = Current(token="0123456789abcdef", frame_idx=3)
    monkeypatch.setattr(exporter.pickle, "loads", lambda data: [Forbidden()] * 3 + [current, Forbidden()])
    selected = exporter._verified_current_frame(
        tmp_path, {"log": {"path": "log.pkl", "size_bytes": len(data), "sha256": digest}},
        {"frame_index": 3, "scene_token": "0123456789abcdef"},
    )
    assert selected is current


def test_export_reads_only_current_camera_motion_and_historical_fast(tmp_path, bundle, monkeypatch):
    asset_root = tmp_path / "assets"
    asset_root.mkdir()
    entries, records, cameras = [], [], {}
    for view in bundle.views:
        name = view.frame.name
        relative = "sensor_blobs/" + name + ".jpg"
        path = asset_root / relative
        path.parent.mkdir(exist_ok=True)
        Image.fromarray(view.frame.rgb).save(path, format="JPEG")
        entries.append({"camera": name, "path": relative})
        records.append({"camera": name, "path": relative, "size_bytes": path.stat().st_size,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "width": 8, "height": 6})
        cameras[name] = {"data_path": name + ".jpg", "cam_intrinsic": view.calibration.intrinsic,
                         "sensor2lidar_rotation": np.eye(3), "sensor2lidar_translation": np.zeros(3),
                         "distortion": np.zeros(5)}
    fast_sources = {"scene_token": bundle.scene_token, "cameras": {"files": entries}}
    sources = {"log": {"sha256": bundle.provenance["metadata_sha256"]}}
    assets = {"cameras": {"files": records}}
    p = bundle.provenance
    manifest = {"source_revision": p["source_revision"], "navsim_revision": p["navsim_revision"],
                "nuplan_revision": p["nuplan_revision"], "assets_manifest_hash": p["source_assets_manifest_hash"],
                "manifest_hash": p["source_experiment_manifest_hash"], "raw_candidate_hash": p["raw_candidate_hash"],
                "model": {"checkpoint_sha256": p["checkpoint_sha256"]}}

    class Current(dict):
        def __getitem__(self, name):
            assert name in {"cams", "ego_dynamic_state", "timestamp"}, "Future/privileged data was read"
            return super().__getitem__(name)

    current = Current(cams=cameras, ego_dynamic_state=[2, 0, 0, 0], timestamp=1234567)
    monkeypatch.setattr(exporter, "_json_file", lambda path: fast_sources if path.name.startswith("ltf-") else sources)
    monkeypatch.setattr(exporter, "verify_prepared", lambda *args: assets)
    monkeypatch.setattr(exporter, "_verified_baseline", lambda *args: (bundle.trajectory, manifest))
    monkeypatch.setattr(exporter, "_verified_current_frame", lambda *args: current)
    prepared = []

    def prepare(name, rgb, *args, **kwargs):
        prepared.append(name)
        assert kwargs["profile"] == OPENSCENE_NATIVE_PROFILE
        return next(view for view in bundle.views if view.frame.name == name)

    monkeypatch.setattr(exporter, "prepare_openscene_camera", prepare)
    result = exporter.export_real_bundle(tmp_path, asset_root, tmp_path, tmp_path / "output")
    assert prepared == list(CAMERA_NAMES)
    assert load_real_bundle(tmp_path / "output").bundle_hash == result.bundle_hash


def baseline_files(bundle, *, legacy=False):
    from hybrid_drive.experiments.protocol import make_manifest

    sources = {"navsim_revision": "b" * 40, "nuplan_revision": "c" * 40, "log": {"path": "logs/log.pkl"}}
    fast = {"scene_token": bundle.scene_token, "checkpoint": {"sha256": "d" * 64}}
    assets = {"sources": {}, "models": [], "cameras": {"files": []}}
    model = {"checkpoint_sha256": "d" * 64, "navsim_revision": "b" * 40}
    manifest = make_manifest(revision="a" * 40, sources=sources, assets=assets, model=model,
                             scene_token=bundle.scene_token, log_name="log", seed=0, policy={}, candidate=bundle.trajectory)
    native = bundle.trajectory.poses.tolist()
    if not legacy:
        native = {"poses": native, "interval_s": bundle.trajectory.interval_s}
    result = {"status": "success", "source_revision": "a" * 40, "scene_token": bundle.scene_token,
              "model": model.copy(), "records": [{"mode": "baseline", "failed": False,
              "scene_token": bundle.scene_token, "native_trajectory": native}]}
    return sources, fast, assets, result, manifest


def save_baseline(tmp_path, result, manifest):
    (tmp_path / "result.json").write_text(json.dumps(result))
    (tmp_path / "experiment-manifest.json").write_text(json.dumps(manifest))


@pytest.mark.parametrize("legacy", [False, True])
def test_historical_baseline_requires_full_association_before_reuse(tmp_path, bundle, legacy):
    sources, fast, assets, result, manifest = baseline_files(bundle, legacy=legacy)
    save_baseline(tmp_path, result, manifest)
    trajectory, verified = exporter._verified_baseline(tmp_path, sources, fast, assets)
    np.testing.assert_array_equal(trajectory.poses, bundle.trajectory.poses)
    assert trajectory.interval_s == bundle.trajectory.interval_s
    assert verified == manifest


@pytest.mark.parametrize("corrupt", [
    lambda r, m: r.update(status="failed"),
    lambda r, m: r.update(source_revision="f" * 40),
    lambda r, m: r.update(scene_token="f" * 16),
    lambda r, m: r["records"][0].update(scene_token="f" * 16),
    lambda r, m: r["records"][0].update(failed=True),
    lambda r, m: r["records"].append(r["records"][0].copy()),
    lambda r, m: r["records"][0]["native_trajectory"]["poses"][0].__setitem__(0, 99),
    lambda r, m: r["records"][0]["native_trajectory"].update(interval_s=1),
    lambda r, m: m.update(raw_candidate_hash="0" * 64),
    lambda r, m: m.update(assets_manifest_hash="0" * 64),
    lambda r, m: m.update(dataset_source_hash="0" * 64),
    lambda r, m: m.update(sampling={"num_poses": 99, "interval_s": .5}),
    lambda r, m: r["model"].update(checkpoint_sha256="0" * 64),
])
def test_changed_historical_result_or_validly_rehashed_manifest_is_rejected(tmp_path, bundle, corrupt):
    from hybrid_drive.experiments.protocol import content_hash

    sources, fast, assets, result, manifest = baseline_files(bundle)
    corrupt(result, manifest)
    manifest.pop("manifest_hash")
    manifest["manifest_hash"] = content_hash(manifest)
    save_baseline(tmp_path, result, manifest)
    with pytest.raises(ValueError):
        exporter._verified_baseline(tmp_path, sources, fast, assets)
