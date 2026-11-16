from dataclasses import FrozenInstanceError
import hashlib
import json
import math
import pickle

import numpy as np
import pytest

from hybrid_drive.adapters.real_camera import CalibratedView, OPENSCENE_NATIVE_PROFILE
from hybrid_drive.contracts.motion import EgoMotion
from hybrid_drive.contracts.motion_observation import MotionObservation, MotionSample
from hybrid_drive.contracts.trajectory import Trajectory
from hybrid_drive.experiments import motion_observation as observer
from hybrid_drive.experiments.protocol import trajectory_hash
from hybrid_drive.experiments.real_bundle import RealSceneBundle, load_real_bundle, write_real_bundle
from hybrid_drive.geometry.camera import CameraCalibration
from hybrid_drive.slow.context import CAMERA_NAMES, CameraFrame


def sample(index, **changes):
    angle = index / 10
    values = dict(
        frame_token=f"{index:016x}", metadata_scene_token="a" * 16,
        source_list_index=index, metadata_frame_idx=index, frame_timestamp_us=1_000_000 + index * 500_000,
        global_translation_m=(index, 2, 3), quaternion_wxyz=(math.cos(angle / 2), 0, 0, math.sin(angle / 2)),
        body_velocity_mps=(2, 0, .1), body_acceleration_mps2=(0, 0, .2), body_angular_rate_radps=(0, 0, .2),
    )
    return MotionSample(**(values | changes))


@pytest.fixture
def bundle():
    trajectory = Trajectory([[1, 0, 0], [2, .2, .1]], .5)
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
    return RealSceneBundle(sample(3).frame_token, sample(3).frame_timestamp_us,
                           trajectory, EgoMotion(2, 0, 0, 0), views, provenance)


def observation(bundle, samples=None):
    return MotionObservation(bundle.bundle_hash, "1" * 40, "b" * 40, "c" * 40, "d" * 64, "e" * 64,
                             tuple(sample(i) for i in range(4)) if samples is None else samples)


def test_owned_immutable_raw_inputs_and_unknown_capture_uncertainty(bundle):
    quaternion = np.array([1., 0, 0, 0])
    velocities = [2, 0, .1]
    current = sample(3, quaternion_wxyz=quaternion, body_velocity_mps=velocities)
    values = [sample(0), sample(1), sample(2), current]
    value = observation(bundle, values)
    quaternion[:] = 9
    velocities[:] = [9] * 3
    values.clear()
    assert value.current.quaternion_wxyz == (1, 0, 0, 0)
    assert value.current.body_velocity_mps == (2, 0, .1)
    assert value.current.pose_capture_timestamp_us is None
    assert value.current.motion_capture_timestamp_us is None
    assert value.current.heading_rate_uncertainty_radps is None
    with pytest.raises(FrozenInstanceError):
        value.current.frame_timestamp_us = 1
    assert pickle.loads(pickle.dumps(value)) == value


def test_interval_mean_is_not_current_associated_rate_and_unwraps(bundle):
    values = tuple(sample(i, quaternion_wxyz=(math.cos((3.1 + .1 * i) / 2), 0, 0,
                                             math.sin((3.1 + .1 * i) / 2)),
                          body_angular_rate_radps=(0, 0, .7)) for i in range(4))
    value = observation(bundle, values)
    assert value.past_interval_mean_heading_rates_radps == pytest.approx((.2, .2, .2))
    assert value.current.associated_planar_heading_rate_radps == pytest.approx(.7)
    assert value.current.heading_rad < 0


def _multiply(a, b):
    w, x, y, z = a
    v, i, j, k = b
    return (w*v-x*i-y*j-z*k, w*i+x*v+y*k-z*j,
            w*j-x*k+y*v+z*i, w*k+x*j-y*i+z*v)


def _orientation_at(q, omega, time):
    speed = math.hypot(*omega)
    rotation = (math.cos(speed*time/2), *(value/speed*math.sin(speed*time/2) for value in omega))
    return _multiply(q, rotation)


def test_heading_derivative_matches_body_rotation_not_gyro_z():
    q = (.6, .2, -.3, .7)
    omega = (.7, -.4, .2)
    current = sample(3, quaternion_wxyz=q, body_angular_rate_radps=omega)
    assert current.heading_rad == pytest.approx(1.7561442767905913)
    dt = 1e-5
    before = sample(3, quaternion_wxyz=_orientation_at(q, omega, -dt)).heading_rad
    after = sample(3, quaternion_wxyz=_orientation_at(q, omega, dt)).heading_rad
    assert current.associated_planar_heading_rate_radps == pytest.approx((after-before)/(2*dt), rel=1e-8)
    assert abs(current.associated_planar_heading_rate_radps-omega[2]) > .01
    negated = sample(3, quaternion_wxyz=tuple(-value for value in q), body_angular_rate_radps=omega)
    scaled = sample(3, quaternion_wxyz=tuple(10*value for value in q), body_angular_rate_radps=omega)
    assert negated.heading_rad == current.heading_rad
    assert scaled.associated_planar_heading_rate_radps == pytest.approx(current.associated_planar_heading_rate_radps)
    assert current.quaternion_wxyz == q


def test_singular_heading_and_ambiguous_interval_remain_unknown(bundle):
    singular = sample(3, quaternion_wxyz=(math.sqrt(.5), 0, math.sqrt(.5), 0))
    assert singular.heading_rad is None
    assert singular.associated_planar_heading_rate_radps is None
    assert singular.heading_rate_failure == "heading_singularity"
    value = observation(bundle, [sample(0, quaternion_wxyz=(1, 0, 0, 0)),
                                 sample(1, quaternion_wxyz=(0, 0, 0, 1)), sample(2), singular])
    assert value.past_interval_mean_heading_rates_radps[0] is None
    assert value.past_interval_mean_heading_rates_radps[-1] is None


def test_subnormal_projected_rate_is_preserved_or_explicitly_unrepresentable():
    smallest = math.ulp(0.0)
    current = sample(3, quaternion_wxyz=(1, 0, 0, 0), body_angular_rate_radps=(0, 0, smallest))
    assert current.associated_planar_heading_rate_radps == smallest
    assert current.heading_rate_failure is None
    transverse = sample(3, quaternion_wxyz=(1, 0, 0, 0), body_angular_rate_radps=(smallest, 0, 0))
    assert transverse.associated_planar_heading_rate_radps == 0
    assert transverse.heading_rate_failure is None
    lost = sample(3, quaternion_wxyz=(1, 0, smallest, 0), body_angular_rate_radps=(.1, 0, 0))
    assert lost.associated_planar_heading_rate_radps is None
    assert lost.heading_rate_failure == "numeric_range"


@pytest.mark.parametrize("changes", [
    {"frame_timestamp_us": True}, {"frame_timestamp_us": 1.5}, {"frame_timestamp_us": -1},
    {"quaternion_wxyz": (0, 0, 0, 0)}, {"body_velocity_mps": (True, 0, 0)},
    {"body_angular_rate_radps": (0, 0, math.nan)}, {"global_translation_m": (1, 2)},
    {"pose_capture_timestamp_us": 2_500_000}, {"motion_capture_timestamp_us": 2_500_000},
])
def test_invalid_sample_is_rejected(changes):
    with pytest.raises(ValueError):
        sample(3, **changes)


@pytest.mark.parametrize("changes", [
    {"frame_timestamp_us": 1_500_000}, {"source_list_index": 9},
    {"metadata_frame_idx": 9}, {"metadata_scene_token": "b" * 16}, {"frame_token": f"{2:016x}"},
])
def test_noncontiguous_observation_is_rejected(bundle, changes):
    with pytest.raises(ValueError):
        observation(bundle, [sample(0), sample(1), sample(2), sample(3, **changes)])


def test_roundtrip_hash_and_legacy_bundle_remain_unchanged(tmp_path, bundle):
    legacy = tmp_path / "legacy"
    write_real_bundle(bundle, legacy)
    before = {p.name: p.read_bytes() for p in legacy.iterdir()}
    value = observation(bundle)
    path = tmp_path / "observations" / "motion-observation.json"
    observer.write_motion_observation(bundle, value, path)
    assert observer.load_motion_observation(bundle, path) == value
    manifest = json.loads(path.read_text())
    assert manifest["observation_hash"] == observer.motion_observation_hash(value)
    assert {p.name: p.read_bytes() for p in legacy.iterdir()} == before
    assert load_real_bundle(legacy).bundle_hash == bundle.bundle_hash
    with pytest.raises(FileExistsError):
        observer.write_motion_observation(bundle, value, path)
    with pytest.raises(ValueError, match="outside"):
        observer.write_motion_observation(bundle, value, legacy / "motion-observation.json")


@pytest.mark.parametrize("corruption", ["hash", "derived", "base", "timestamp", "extra", "capture", "metadata", "velocity"])
def test_sidecar_corruption_rejected_even_with_rehashed_payload(tmp_path, bundle, corruption):
    path = tmp_path / "motion-observation.json"
    observer.write_motion_observation(bundle, observation(bundle), path)
    manifest = json.loads(path.read_text())
    if corruption == "derived": manifest["samples"][-1]["associated_planar_heading_rate_radps"] = 999
    elif corruption == "base": manifest["base_bundle_hash"] = "0" * 64
    elif corruption == "timestamp": manifest["samples"][-1]["frame_timestamp_us"] = 1
    elif corruption == "extra": manifest["unexpected"] = 1
    elif corruption == "capture": manifest["samples"][-1]["motion_capture_timestamp_us"] = 2_500_000
    elif corruption == "metadata": manifest["provenance"]["metadata_sha256"] = "0" * 64
    elif corruption == "velocity": manifest["samples"][-1]["body_velocity_mps"][0] = 3
    if corruption != "hash":
        manifest["observation_hash"] = observer._hash({k: v for k, v in manifest.items() if k != "observation_hash"})
    else: manifest["observation_hash"] = "0" * 64
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        observer.load_motion_observation(bundle, path)


def test_sidecar_rejects_symlink_duplicate_json_and_unbounded_file(tmp_path, bundle):
    path = tmp_path / "motion-observation.json"
    observer.write_motion_observation(bundle, observation(bundle), path)
    link = tmp_path / "link.json"
    link.symlink_to(path)
    with pytest.raises(ValueError): observer.load_motion_observation(bundle, link)
    path.write_text('{"schema":1,"schema":2}')
    with pytest.raises(ValueError, match="Duplicate"): observer.load_motion_observation(bundle, path)
    path.write_bytes(b" " * (observer.MAX_OBSERVATION_BYTES+1))
    with pytest.raises(ValueError, match="budget"): observer.load_motion_observation(bundle, path)


def _frame(index):
    value = sample(index)
    return dict(token=value.frame_token, scene_token=value.metadata_scene_token, frame_idx=index,
                timestamp=value.frame_timestamp_us, ego2global_translation=value.global_translation_m,
                ego2global_rotation=value.quaternion_wxyz, ego_dynamic_state=[2, 0, 0, 0],
                can_bus=[*value.global_translation_m, *value.quaternion_wxyz,
                         *value.body_acceleration_mps2, *value.body_velocity_mps, *value.body_angular_rate_radps, 0, 0])


def _sources(count=4):
    return dict(navsim_revision="b"*40, nuplan_revision="c"*40, log={"sha256":"d"*64},
                selection={"num_source_frames":count,"timestamp_step_us":[499000,501000]})


def _spec(index=3):
    value = sample(index)
    return dict(source_list_index=index, metadata_frame_idx=index, metadata_scene_token=value.metadata_scene_token,
                scene_token=value.frame_token, frame_timestamp_us=value.frame_timestamp_us)


def test_exporter_only_touches_selected_past_current_fields(bundle):
    allowed = set(_frame(0))
    class Current(dict):
        def __getitem__(self, name):
            assert name in allowed, "Privileged/non-motion field was read"
            return super().__getitem__(name)
    class Future:
        def __getattribute__(self, name): raise AssertionError("Future frame was read")
    frames = [Current(_frame(i), anns="forbidden", cams="forbidden") for i in range(4)] + [Future()]
    value = observer._from_frames(bundle, _spec(), frames, _sources(5), revision="1"*40)
    assert value.current.frame_token == bundle.scene_token
    assert value.current.body_velocity_mps == (2, 0, .1)
    frames[2]["timestamp"] += 100_000
    with pytest.raises(ValueError, match="discontinuous"):
        observer._from_frames(bundle, _spec(), frames, _sources(5), revision="1"*40)


def test_metadata_sha_is_verified_before_pickle(tmp_path, monkeypatch):
    data = pickle.dumps([_frame(i) for i in range(4)])
    path = tmp_path / "metadata.pkl"
    path.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    sources = _sources() | {"log":{"path":path.name,"sha256":digest,"size_bytes":len(data)}}
    monkeypatch.setattr(observer, "VERIFIED_METADATA_SHA256", digest)
    assert len(observer._verified_frames(tmp_path, sources)) == 4
    path.write_bytes(b"x" * len(data))
    monkeypatch.setattr(observer.pickle, "loads", lambda data: pytest.fail("Unverified pickle was read"))
    with pytest.raises(ValueError, match="before pickle"):
        observer._verified_frames(tmp_path, sources)


@pytest.mark.parametrize("corruption", ["gyro_shape", "motion_bool", "motion_mismatch", "quaternion_mismatch"])
def test_explicit_fields_must_match_verified_can_bus_layout(corruption):
    frame = _frame(0)
    if corruption == "gyro_shape": frame["can_bus"].pop()
    elif corruption == "motion_bool": frame["ego_dynamic_state"][1] = False
    elif corruption == "motion_mismatch": frame["ego_dynamic_state"][0] = 3
    elif corruption == "quaternion_mismatch": frame["ego2global_rotation"] = (0, 0, 0, 1)
    with pytest.raises(ValueError): observer._sample(frame, 0)


def _export_fixture(tmp_path, bundle, monkeypatch, source_error=False):
    from hybrid_drive.assets import cohort as assets
    from hybrid_drive.experiments import cohort_fast

    fast_root = tmp_path / "fast"
    sources = _sources(24)
    sources["cases"] = [{"case_id":f"r{i:02d}", **_spec(4*i+3)} for i in range(6)]
    rows = []
    for spec in sources["cases"]:
        current_bundle = RealSceneBundle(spec["scene_token"], spec["frame_timestamp_us"], bundle.trajectory,
                                         bundle.motion, bundle.views, bundle.provenance)
        relative = spec["case_id"] + "/bundle"
        write_real_bundle(current_bundle, fast_root / relative)
        rows.append({"case_id":spec["case_id"],"scene_token":spec["scene_token"],
                     "source_list_index":spec["source_list_index"],"bundle_path":relative,
                     "bundle_hash":current_bundle.bundle_hash,"status":"success"})
    cohort = {"sources_hash":observer._hash(sources),"cohort_hash":"9"*64,"cases":rows}
    monkeypatch.setattr(assets,"load_cohort_sources",lambda path:sources)
    monkeypatch.setattr(cohort_fast,"load_fast_cohort",lambda path:cohort)
    def verified(*args):
        if source_error: raise ValueError("source unavailable")
        return [_frame(i) for i in range(24)]
    monkeypatch.setattr(observer,"_verified_frames",verified)
    before = {str(p.relative_to(fast_root)):p.read_bytes() for p in fast_root.rglob('*') if p.is_file()}
    output,index = observer.export_motion_observations(tmp_path, fast_root, tmp_path/"observations", revision="1"*40)
    return output, index, fast_root, before, sources, cohort


@pytest.mark.parametrize("source_error", [False, True])
def test_six_case_export_index_retains_all_attempts_without_modifying_fast(tmp_path, bundle, monkeypatch, source_error):
    output, index, fast_root, before, sources, cohort = _export_fixture(tmp_path, bundle, monkeypatch, source_error)
    assert len(index["cases"]) == index["planned_cases"] == 6
    assert index["index_hash"] == observer._hash({k:v for k,v in index.items() if k != "index_hash"})
    assert observer.load_motion_observation_index(output,fast_cohort_hash=cohort["cohort_hash"],sources_hash=cohort["sources_hash"]) == index
    for row in index["cases"]:
        assert (row["status"] == "failed") is source_error
        if source_error:
            assert row["observation_hash"] is row["observation_path"] is None
            assert "source unavailable" in row["error"]
        else:
            value = observer.load_motion_observation(load_real_bundle(fast_root/row["case_id"]/"bundle"), output/row["observation_path"])
            assert observer.motion_observation_hash(value) == row["observation_hash"]
    assert {str(p.relative_to(fast_root)):p.read_bytes() for p in fast_root.rglob('*') if p.is_file()} == before
    with pytest.raises(ValueError,match="outside"):
        observer.export_motion_observations(tmp_path,fast_root,fast_root/"observations",revision="1"*40)


@pytest.mark.parametrize("corruption", [
    "hash", "cohort", "source", "producer", "count", "order", "token", "source_index", "bool_index",
    "path", "absolute_path", "observation_hash", "base_hash", "status", "extra", "success_error",
])
def test_index_rejects_rehashed_corruption(tmp_path, bundle, monkeypatch, corruption):
    output,index,_,_,_,cohort = _export_fixture(tmp_path,bundle,monkeypatch)
    row = index["cases"][0]
    if corruption == "cohort": index["fast_cohort_hash"] = "0"*64
    elif corruption == "source": index["sources_hash"] = "0"*64
    elif corruption == "producer": index["source_revision"] = "0"*40
    elif corruption == "count": index["cases"].pop()
    elif corruption == "order": index["cases"][:2] = reversed(index["cases"][:2])
    elif corruption == "token": row["scene_token"] = "0"*16
    elif corruption == "source_index": row["source_list_index"] += 1
    elif corruption == "bool_index": row["source_list_index"] = True
    elif corruption == "path": row["observation_path"] = "r00/../r01/motion-observation.json"
    elif corruption == "absolute_path": row["observation_path"] = str(output/"r00/motion-observation.json")
    elif corruption == "observation_hash": row["observation_hash"] = "0"*64
    elif corruption == "base_hash": row["base_bundle_hash"] = "0"*64
    elif corruption == "status": row["status"] = "skipped"
    elif corruption == "extra": index["extra"] = 1
    elif corruption == "success_error": row["error"] = "not allowed on success"
    index["index_hash"] = "0"*64 if corruption == "hash" else observer._hash({k:v for k,v in index.items() if k != "index_hash"})
    (output/"index.json").write_text(json.dumps(index))
    with pytest.raises(ValueError):
        observer.load_motion_observation_index(output,fast_cohort_hash=cohort["cohort_hash"],sources_hash=cohort["sources_hash"])


@pytest.mark.parametrize("corruption", ["producer", "source_manifest", "timestamp", "sidecar_symlink", "index_symlink", "oversize"])
def test_index_checks_sidecar_source_binding_and_bounded_regular_paths(tmp_path,bundle,monkeypatch,corruption):
    output,index,_,_,_,cohort = _export_fixture(tmp_path,bundle,monkeypatch)
    row = index["cases"][0]
    path = output/row["observation_path"]
    if corruption in ("producer","source_manifest","timestamp"):
        manifest = json.loads(path.read_text())
        if corruption == "producer": manifest["provenance"]["source_revision"] = "0"*40
        elif corruption == "source_manifest": manifest["provenance"]["source_manifest_hash"] = "0"*64
        else:
            for sample_value in manifest["samples"]: sample_value["frame_timestamp_us"] += 500_000
        manifest["observation_hash"] = observer._hash({k:v for k,v in manifest.items() if k != "observation_hash"})
        row["observation_hash"] = manifest["observation_hash"]
        path.write_text(json.dumps(manifest))
        index["index_hash"] = observer._hash({k:v for k,v in index.items() if k != "index_hash"})
        (output/"index.json").write_text(json.dumps(index))
    elif corruption in ("sidecar_symlink","index_symlink"):
        target = path if corruption == "sidecar_symlink" else output/"index.json"
        saved = tmp_path/"saved.json"
        target.rename(saved)
        target.symlink_to(saved)
    else: (output/"index.json").write_bytes(b" "*(observer.MAX_INDEX_BYTES+1))
    with pytest.raises(ValueError):
        observer.load_motion_observation_index(output,fast_cohort_hash=cohort["cohort_hash"],sources_hash=cohort["sources_hash"])


@pytest.mark.parametrize("corruption", ["missing_error", "blank_error", "fake_path", "fake_hash"])
def test_failed_index_rows_cannot_drop_cause_or_fabricate_observation(tmp_path,bundle,monkeypatch,corruption):
    output,index,_,_,_,cohort = _export_fixture(tmp_path,bundle,monkeypatch,source_error=True)
    row = index["cases"][0]
    if corruption == "missing_error": row.pop("error")
    elif corruption == "blank_error": row["error"] = " "
    elif corruption == "fake_path": row["observation_path"] = "r00/motion-observation.json"
    else: row["observation_hash"] = "0"*64
    index["index_hash"] = observer._hash({k:v for k,v in index.items() if k != "index_hash"})
    (output/"index.json").write_text(json.dumps(index))
    with pytest.raises(ValueError):
        observer.load_motion_observation_index(output,fast_cohort_hash=cohort["cohort_hash"],sources_hash=cohort["sources_hash"])


def test_wire_identity_excludes_platform_dependent_derived_numbers(tmp_path, bundle):
    from dataclasses import replace

    path = tmp_path / "motion-observation.json"
    value = observation(bundle)
    observer.write_motion_observation(bundle, value, path)
    manifest = json.loads(path.read_text())
    assert "past_interval_mean_heading_rates_radps" not in manifest
    assert set(manifest["samples"][0]) == observer.RAW_SAMPLE_KEYS
    loaded = observer.load_motion_observation(bundle, path)
    assert loaded.current.associated_planar_heading_rate_radps == pytest.approx(.2)
    # A different runtime's last-bit derived result cannot change source identity.
    different = replace(value)
    object.__setattr__(different.current, "associated_planar_heading_rate_radps", math.nextafter(.2, math.inf))
    assert observer.motion_observation_hash(different) == manifest["observation_hash"]
