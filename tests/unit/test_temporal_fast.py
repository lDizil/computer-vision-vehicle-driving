import copy
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from hybrid_drive.adapters.real_camera import CalibratedView, OPENSCENE_NATIVE_PROFILE
from hybrid_drive.contracts.motion import EgoMotion
from hybrid_drive.experiments import temporal_fast
from hybrid_drive.experiments.protocol import content_hash
from hybrid_drive.experiments.real_bundle import RealSceneBundle
from hybrid_drive.geometry.camera import CameraCalibration
from hybrid_drive.slow.context import CAMERA_NAMES, CameraFrame


def test_metadata_hash_precedes_pickle_and_selection_reads_no_annotations_or_future_motion(tmp_path,monkeypatch):
    sources = copy.deepcopy(temporal_fast.load_temporal_sources())
    data = b"verified-fixture"
    path = tmp_path/"metadata.pkl"
    path.write_bytes(data)
    sources["log"] = {"path":path.name,"size_bytes":len(data),"sha256":hashlib.sha256(data).hexdigest()}
    class Forbidden:
        def __getattribute__(self,name): raise AssertionError("Unselected frame was inspected")
    class Frame(dict):
        def __getitem__(self,name):
            assert name in {"token","scene_token","frame_idx","timestamp"} | ({"cams"} if 27 <= self.index <= 46 else set())
            return super().__getitem__(name)
    frames = [Forbidden() for _ in range(738)]
    cases = {case["source_list_index"]:case for case in sources["cases"]}
    for identity in sources["selection"]["identity_window"]:
        idx = identity["source_list_index"]
        value = Frame({k:v for k,v in identity.items() if k != "source_list_index"});value.index = idx
        if idx in cases:value["cams"] = {c["camera"]:{"data_path":c["path"].removeprefix("sensor_blobs/")} for c in cases[idx]["cameras"]}
        frames[idx] = value
    monkeypatch.setattr(temporal_fast.pickle,"loads",lambda data:frames)
    assert len(temporal_fast._current_frames(tmp_path,sources)) == 20
    frames[27]["cams"]["CAM_L0"]["data_path"] = "wrong.jpg"
    with pytest.raises(ValueError,match="camera"):temporal_fast._current_frames(tmp_path,sources)
    path.write_bytes(b"unverified")
    monkeypatch.setattr(temporal_fast.pickle,"loads",lambda data:pytest.fail("Unverified pickle"))
    with pytest.raises(ValueError,match="before pickle"):temporal_fast._current_frames(tmp_path,sources)


def freeze_fixture(tmp_path,monkeypatch,*,source_error=False,preflight_error=False):
    sources = temporal_fast.load_temporal_sources()
    verified = []
    monkeypatch.setattr(temporal_fast,"configure_environment",lambda *a:None)
    monkeypatch.setattr(temporal_fast,"verify_temporal_sample",lambda *a:{"files":[{"case_id":s["case_id"]} for s in sources["cases"]]})
    def current(*args):
        if source_error:raise ValueError("source identity mismatch")
        verified.append("selection")
        return {case["case_id"]:{} for case in sources["cases"]}
    monkeypatch.setattr(temporal_fast,"_current_frames",current)
    monkeypatch.setattr(temporal_fast,"_verified_models",lambda *a:[])
    class Fast:
        def provenance(self):return {"checkpoint_sha256":"1"*64}
        def get_sensor_config(self):return None
        def compute_trajectory(self,value):
            return SimpleNamespace(poses=np.array([[1,0,0],[2,0,0],[3,0,0]],float),
                                   trajectory_sampling=SimpleNamespace(interval_length=.5,num_poses=3))
    def initialize(*args):
        assert verified == ["selection","preflight"]
        return Fast()
    monkeypatch.setattr(temporal_fast,"_initialize_fast",initialize)
    def loader(*args,**kwargs):
        verified.append("camera_inputs" if "sensor_root" in kwargs else "preflight")
        if preflight_error and "sensor_root" not in kwargs:raise ValueError("preflight token mismatch")
        return SimpleNamespace(get_agent_input_from_token=lambda token:token),[]
    monkeypatch.setattr(temporal_fast,"load_cohort_inputs",loader)
    def bundle(spec,current,raw,root,records,manifest,*,metadata_sha256):
        views = tuple(CalibratedView(CameraFrame(name,np.zeros((6,8,3),np.uint8)),
                        CameraCalibration(name,8,6,[[5,0,4],[0,5,3],[0,0,1]],np.eye(4))) for name in CAMERA_NAMES)
        provenance = {"source_revision":manifest["source_revision"],"navsim_revision":sources["navsim_revision"],
            "nuplan_revision":sources["nuplan_revision"],"metadata_sha256":metadata_sha256,
            "source_assets_manifest_hash":manifest["assets_manifest_hash"],"source_experiment_manifest_hash":manifest["manifest_hash"],
            "raw_candidate_hash":manifest["raw_candidate_hash"],"checkpoint_sha256":"1"*64,
            "source_image_hashes":{name:"2"*64 for name in CAMERA_NAMES},"profile":OPENSCENE_NATIVE_PROFILE}
        return RealSceneBundle(spec["scene_token"],spec["frame_timestamp_us"],raw,EgoMotion(2,0,0,0),views,provenance)
    monkeypatch.setattr(temporal_fast,"_make_bundle",bundle)
    # make_manifest needs complete camera identities, not an image/model runtime.
    records = [{"case_id":s["case_id"],**camera,"size_bytes":1,"sha256":"2"*64,"width":8,"height":6,"format":"JPEG","mode":"RGB"}
               for s in sources["cases"] for camera in s["cameras"]]
    monkeypatch.setattr(temporal_fast,"verify_temporal_sample",lambda *a:{"files":records})
    return temporal_fast.run_temporal_fast(tmp_path/"data",tmp_path/"models",tmp_path/"inputs",tmp_path/"fast",revision="a"*40)


def test_twenty_outputs_and_separate_cpu_wall_latencies_are_frozen_without_scoring(tmp_path,monkeypatch):
    root,manifest = freeze_fixture(tmp_path,monkeypatch)
    assert temporal_fast.load_temporal_fast(root) == manifest
    assert len(temporal_fast.load_temporal_bundles(root,manifest)) == 20
    assert all(case["fast_cpu_s"] >= 0 and case["fast_wall_s"] >= 0 for case in manifest["cases"])
    assert not list(root.glob("**/*score*"))


def test_source_failure_prevents_model_initialization_and_preserves_all_attempts(tmp_path,monkeypatch):
    root,manifest = freeze_fixture(tmp_path,monkeypatch,source_error=True)
    assert manifest["model"] is None
    assert len(manifest["cases"]) == 20
    assert all(case["status"] == "failed" and "source identity mismatch" in case["error"] for case in manifest["cases"])
    assert temporal_fast.load_temporal_fast(root,require_success=False) == manifest
    with pytest.raises(ValueError,match="twenty successful"):temporal_fast.load_temporal_fast(root)


def test_navsim_token_preflight_failure_prevents_any_model_initialization(tmp_path,monkeypatch):
    _,manifest = freeze_fixture(tmp_path,monkeypatch,preflight_error=True)
    assert manifest["model"] is None
    assert all(case["status"] == "failed" and "preflight token mismatch" in case["error"] for case in manifest["cases"])


@pytest.mark.parametrize("corruption", ["count","path","source_index","latency_bool","latency_none","hash","execution"])
def test_temporal_fast_reader_rejects_rehashed_corruption(tmp_path,monkeypatch,corruption):
    root,manifest = freeze_fixture(tmp_path,monkeypatch)
    row = manifest["cases"][0]
    if corruption == "count":manifest["cases"].pop()
    elif corruption == "path":row["bundle_path"] = "../bundle"
    elif corruption == "source_index":row["source_list_index"] = True
    elif corruption == "latency_bool":row["fast_cpu_s"] = False
    elif corruption == "latency_none":row["fast_cpu_s"] = None
    elif corruption == "execution":manifest["execution"] = "scored_before_capture"
    else:manifest["temporal_fast_hash"] = "0"*64
    if corruption != "hash":manifest["temporal_fast_hash"] = content_hash({k:v for k,v in manifest.items() if k != "temporal_fast_hash"})
    (root/"temporal-fast.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):temporal_fast.load_temporal_fast(root)
