from copy import deepcopy
from hashlib import sha256
from io import BytesIO
import json
from math import sqrt
from pathlib import Path
import pickle
import tarfile
from types import SimpleNamespace

import pytest

from hybrid_drive.assets.files import PreparationError
from hybrid_drive.experiments import box_motion_data as data
from hybrid_drive.experiments.bounded_io import canonical_json


_LOG = "2021.07.24.23.50.16_veh-17_01696_02071"
_PREFIX = "openscene-v1.1/sensor_blobs/mini"
_CAMERA = {"cam_intrinsic": [[2, 0, .5], [0, 2, .5], [0, 0, 1]],
           "sensor2lidar_rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
           "sensor2lidar_translation": [0, 0, 0], "distortion": [0]*5}


def _f(i, *, x=None):
    x = i*.6 if x is None else x
    return {
        "token": f"{i+1:016x}", "log_name": _LOG, "log_token": "1"*16, "frame_idx": i % 4,
        "timestamp": i*500_000, "ego2global_translation": [0, 0, 0], "ego2global_rotation": [1, 0, 0, 0],
        "anns": {"gt_names": ["vehicle", "vehicle"], "track_tokens": ["2"*16, "3"*16],
                 "gt_boxes": [[x, 1, .8, 4, 2, 1.5, 0], [x, -1, .8, 4, 2, 1.5, 0]]},
        "cams": {
            c: {**_CAMERA, "data_path": f"{_LOG}/{c}/{1000 + 6*i+j:016x}.jpg"}
            for j, c in enumerate(data.CAMERAS)
        },
    }


def test_target_world_compensation_and_width_length_order():
    current, next_frame = _f(9, x=2), _f(10, x=1)
    orientation = [sqrt(.5), 0, 0, sqrt(.5)]
    current.update(ego2global_translation=[10,20,0], ego2global_rotation=orientation)
    next_frame.update(ego2global_translation=[10,21,0], ego2global_rotation=orientation)
    # The first world's position stays (9,22); the second goes from (11,22) to (11,24).
    next_frame["anns"]["gt_boxes"][1][0] = 3
    label, moving = data._target(current, next_frame)
    assert moving == 1 and label["next_dt_s"] == .5
    one, two = label["boxes"]
    assert one["center"] == [2,1,.8] and one["future_center"] == pytest.approx([2,1,.8])
    assert two["center"] == [2,-1,.8] and two["future_center"] == pytest.approx([4,-1,.8])
    assert one["size_wlh"] == [2,4,1.5]


def test_missing_track_is_null_and_never_uses_wrong_category():
    current, after = _f(3), _f(4)
    after["anns"]["gt_names"][0] = "pedestrian"
    label, moving = data._target(current, after)
    assert label["boxes"][0]["future_center"] is None
    assert label["boxes"][1]["future_center"] == pytest.approx([2.4,-1,.8])
    assert moving == 1


def test_temporal_windows_allow_boundary_reset_but_reject_discontinuities():
    frames = [_f(i) for i in range(5)]
    assert [f["frame_idx"] for f in frames] == [0,1,2,3,0]
    assert data._valid_window(frames)
    dropped = deepcopy(frames)
    dropped[4]["timestamp"] += 750_000
    assert not data._valid_window(dropped)
    other = deepcopy(frames)
    other[2]["log_token"] = "8"*16
    assert not data._valid_window(other)
    dropped = deepcopy(frames)
    dropped[2]["cams"].pop("CAM_L2")
    assert not data._valid_window(dropped)
    with pytest.raises(PreparationError, match="ambiguous"):
        data._chronological([frames[0], deepcopy(frames[0])])


def test_curation_is_fixed_and_nonoverlapping():
    frames = data._chronological(list(reversed([_f(i) for i in range(60)])))
    chosen, stats = data._select_anchors(frames, 3)
    assert [i for i,_ in chosen] == [13, 33, 53]
    assert stats == {"eligible_windows": 56, "nonoverlapping_eligible_windows": 12,
                     "selected_anchors": 3, "selected_moving_targets": 6}
    with pytest.raises(PreparationError, match="non-overlapping"):
        data._select_anchors(frames, 14)


def test_source_manifest_pins_fresh_day_and_vehicle_groups():
    sources = json.loads(Path("scripts/box-motion-sources.json").read_bytes())
    assert sha256(canonical_json(sources)).hexdigest() == data.SOURCES_SHA256
    data._check_recording_groups(sources)
    sources["previous_recordings"].append("2021.07.24.14.03.00_veh-17")  # not the exact same prefix
    with pytest.raises(PreparationError, match="day/vehicle"):
        data._check_recording_groups(sources)


def _tar(path, members):
    with tarfile.open(path, "w:gz") as archive:
        for name, payload in members:
            item = tarfile.TarInfo(name)
            item.size = len(payload)
            archive.addfile(item, BytesIO(payload))
    payload = path.read_bytes()
    return {"size_bytes":len(payload), "sha256":sha256(payload).hexdigest()}


def test_synthetic_preparation_never_exports_next_frame_or_annotations(tmp_path, monkeypatch):
    frames = [_f(i) for i in range(5)]
    member = f"openscene-v1.1/meta_datas/mini/{_LOG}.pkl"
    payload = pickle.dumps(frames)
    p, q = tmp_path/"meta.tgz", tmp_path/"camera.tgz"
    metadata = _tar(p, [(member, payload)])
    jpeg = b"\xff\xd8\xff"  # data prep only checks the signature; renderer verifies decoded pixels later.
    camera = _tar(q, [(f'{_PREFIX}/{view["data_path"]}', jpeg+view["data_path"].encode())
                      for f in frames for view in f["cams"].values()])
    camera["archive_prefix"] = _PREFIX
    config = {
        "logs":[{"member":member,"frame_count":len(frames),"recording":_LOG.rsplit("_",2)[0],
                 "log_name":_LOG,"size_bytes":len(payload),"sha256":sha256(payload).hexdigest(),"anchor_indices":[3]}],
        "metadata_archive":metadata, "camera_archive":camera, "camera_shard":0,
        "publisher_mapping_sha256":"0"*64, "previous_recordings":[], "selection":"fixture",
    }
    encoded = canonical_json(config)
    src = tmp_path/"source.json"
    src.write_bytes(encoded)
    monkeypatch.setattr(data, "SOURCES_SHA256", sha256(encoded).hexdigest())
    monkeypatch.setattr(data, "_ANCHORS_PER_LOG", 1)
    monkeypatch.setattr(data, "sys", SimpleNamespace(platform="linux"))
    out = tmp_path/"out"
    result = data.prepare_motion_holdout(p, q, src, out)
    assert (result["anchors"],result["observed_frames"],result["camera_jpegs"]) == (1,4,24)
    blob = (out/"inputs/index.json").read_bytes()
    index = json.loads(blob)
    assert {f["token"] for f in frames[:4]} == set(index["frames"])
    assert b"future_center" not in blob and b"gt_boxes" not in blob
    stored = list((out/"inputs").rglob("*.jpg"))
    assert len(stored) == 24 and not (out/"inputs/rgb"/frames[-1]["token"]).exists()
    for frame in index["frames"].values():
        for view in frame["views"]:
            assert sha256((out/"inputs"/view["path"]).read_bytes()).hexdigest() == view["sha256"]
    truth = json.loads((out/"targets/targets.json").read_text())
    assert set(truth) == {frames[3]["token"]} and truth[frames[3]["token"]]["next_dt_s"] == .5
