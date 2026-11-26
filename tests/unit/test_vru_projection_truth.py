from hashlib import sha256
from io import BytesIO
import json
from math import pi, sqrt
import pickle
import tarfile
from types import SimpleNamespace

import numpy as np
import pytest

from hybrid_drive.assets.files import PreparationError
from hybrid_drive.experiments import vru_fusion_holdout_data as data, vru_projection_truth as truth
from hybrid_drive.experiments.bounded_io import canonical_json
from hybrid_drive.experiments.box_motion_data import CAMERAS, _observation
from hybrid_drive.experiments.temporal_vru_data import _label_window
from hybrid_drive.geometry.camera import CameraCalibration


_RECORDING = "2021.05.12.23.36.44_veh-35"
_LOG = _RECORDING + "_00000_00004"
_PREFIX = "openscene-v1.1/sensor_blobs/mini"
_PERSON, _BICYCLE = "f1" * 8, "f2" * 8
_RIG = {"camera": "CAM_L2", "camera_to_ego_rotation": [[0, 0, 1], [-1, 0, 0], [0, -1, 0]],
        "camera_to_ego_translation": [0, 0, 0], "distortion": [0] * 5,
        "intrinsic": [[1000, 0, 960], [0, 900, 540], [0, 0, 1]]}


def test_projected_envelope_uses_half_pixel_resizing_and_yaw():
    camera = truth._calibration(_RIG)
    unrotated = np.array([5., 0., 0., 2., 1., 2., 0.])
    spun = unrotated.copy()
    spun[6] = pi / 2
    a = truth._project_target("person", unrotated, (camera,))["views"][0]
    b = truth._project_target("person", spun, (camera,))["views"][0]
    assert a["camera_name"] == "CAM_L2"
    assert a["xyxy"] == pytest.approx([383.7 - 50, 215.7 - 90, 383.7 + 50, 215.7 + 90])
    assert b["xyxy"] == pytest.approx([383.7 - 400/4.5, 215.7 - 360/4.5, 383.7 + 400/4.5, 215.7 + 360/4.5])


def test_near_intersections_clip_to_viewport_without_projecting_back_faces():
    camera = truth._calibration(_RIG)
    box = np.array([.15, 0., 0., .2, .6, .6, 0.])
    partial = truth._project_target("bicycle", box, (camera,))["views"]
    assert partial == [{"camera_name": "CAM_L2", "xyxy": [0., 0., 767., 431.]}]
    box[0] = -.5
    assert truth._project_target("bicycle", box, (camera,))["views"] == []
    box[:3] = [5, -10, 0]
    assert truth._project_target("bicycle", box, (camera,))["views"] == []


def test_silhouette_outside_the_viewport_does_not_inherit_an_empty_aabb_corner():
    rotation = np.eye(4)
    rotation[:2, :2] = np.array([[1, -1], [1, 1]]) / sqrt(2)
    camera = CameraCalibration("rolled", 768, 432, [[100, 0, 0], [0, 100, 0], [0, 0, 1]], rotation)
    # Both positive pixel axes are reached separately, but no part of the
    # diamond touches the first quadrant where the viewport starts.
    offscreen = np.array([-sqrt(8), 0., 10., 4., 4., 2., 0.])
    assert truth._project_target("person", offscreen, (camera,))["views"] == []


def _frame(i):
    token = f"{i:016x}"
    cameras = {name: {"data_path": f"{_LOG}/{name}/{token}.jpg",
                      "sensor2lidar_rotation": _RIG["camera_to_ego_rotation"],
                      "sensor2lidar_translation": _RIG["camera_to_ego_translation"],
                      "distortion": _RIG["distortion"], "cam_intrinsic": _RIG["intrinsic"]} for name in CAMERAS}
    return {"token": token, "log_name": _LOG, "log_token": "projection-test-recording", "timestamp": i * 500_000,
            "ego2global_rotation": [1., 0., 0., 0.], "ego2global_translation": [float(i), 0., 0.], "cams": cameras,
            "anns": {"gt_names": ["bicycle", "pedestrian"], "track_tokens": [_BICYCLE, _PERSON],
                     "gt_boxes": [[12., 0., .5, 1., 1., 1., .1], [6. + .5 * i, 1., .9, 1., 1., 1.8, .2]]}}


@pytest.fixture
def cohort(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    frames = [_frame(i) for i in range(4)]
    previous, now = frames[:2]
    _, tracks, elapsed = _label_window(frames, [frame["timestamp"] for frame in frames])
    archive, inputs = root / "metadata.tgz", root / "inputs"
    payload = pickle.dumps(frames)
    item = tarfile.TarInfo(f"openscene-v1.1/meta_datas/mini/{_LOG}.pkl")
    item.size = len(payload)
    with tarfile.open(archive, "w:gz") as target:
        target.addfile(item, BytesIO(payload))
    source = {"metadata_archive": {"size_bytes": archive.stat().st_size, "sha256": sha256(archive.read_bytes()).hexdigest()},
              "camera_archives": {"3": {"archive_prefix": _PREFIX}},
              "logs": [{"recording": _RECORDING, "camera_shard": 3, "log_name": _LOG, "member": item.name,
                        "size_bytes": len(payload), "sha256": sha256(payload).hexdigest()}]}
    sources_path = root / "sources.json"
    sources_path.write_bytes(canonical_json(source))
    source_sha = sha256(sources_path.read_bytes()).hexdigest()
    observed = {}
    for frame in (previous, now):
        obs = _observation(frame, _RECORDING, _LOG, _PREFIX, {})
        for view in obs["views"]:
            view["sha256"] = sha256(view["path"].encode()).hexdigest()
        observed[frame["token"]] = obs
    anchor = {"sample_token": now["token"], "previous_token": previous["token"], "older_token": None, "recording": _RECORDING}
    index_bytes = canonical_json({"schema": data.INPUT_SCHEMA, "source_sha256": source_sha,
                                  "anchors": [anchor], "frames": observed})
    inputs.mkdir()
    (inputs / "index.json").write_bytes(index_bytes)
    labels = root / "labels.json"
    labels.write_bytes(canonical_json({"schema": data.TARGET_SCHEMA, "input_sha256": sha256(index_bytes).hexdigest(),
                                        "rows": [{"sample_token": now["token"], "tracks": tracks, "target_dt_s": elapsed}]}))
    monkeypatch.setattr(truth, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(truth, "_load_source", lambda _: source)
    monkeypatch.setattr(data, "SOURCE_SHA256", source_sha)
    monkeypatch.setattr(data, "_RECORDINGS", frozenset((_RECORDING,)))
    return archive, sources_path, inputs, labels, root / "projection-truth.json"


def test_recovers_original_order_and_boxes_without_jpegs_or_track_tokens(cohort):
    summary = truth.prepare_projection_truth(*cohort)
    _, _, inputs, labels, output = cohort
    person, bicycle = truth.load_projection_truth(inputs, labels, output)["rows"][0]["targets"]
    assert person["kind"] == "person" and person["box_ego"] == [6.5, 1., .9, 1., 1., 1.8, .2]
    assert bicycle["kind"] == "bicycle" and bicycle["box_ego"] == [12., 0., .5, 1., 1., 1., .1]
    assert [view["camera_name"] for view in person["views"]] == list(CAMERAS)
    assert summary["camera_envelopes"] == 12 and summary["truth_sha256"] == sha256(output.read_bytes()).hexdigest()
    assert _PERSON.encode() not in output.read_bytes() and _BICYCLE.encode() not in output.read_bytes()


def test_refuses_future_label_and_current_calibration_drift(cohort):
    _, _, inputs, labels, output = cohort
    original_labels = json.loads(labels.read_bytes())
    wrong_future = json.loads(labels.read_bytes())
    wrong_future["rows"][0]["tracks"][0]["centers_xy_m"][1][0] += .25
    labels.write_bytes(canonical_json(wrong_future))
    with pytest.raises(PreparationError, match="current and future targets"):
        truth.prepare_projection_truth(*cohort)
    assert not output.exists()
    index_path = inputs / "index.json"
    index = json.loads(index_path.read_bytes())
    token = index["anchors"][0]["sample_token"]
    index["frames"][token]["views"][0]["intrinsic"][0][0] += 1
    changed = canonical_json(index)
    index_path.write_bytes(changed)
    original_labels["input_sha256"] = sha256(changed).hexdigest()
    labels.write_bytes(canonical_json(original_labels))
    with pytest.raises(PreparationError, match="observation metadata differs"):
        truth.prepare_projection_truth(*cohort)
    assert not output.exists()


def test_loader_checks_exact_bindings_and_ordered_center(cohort):
    _, _, inputs, labels, output = cohort
    truth.prepare_projection_truth(*cohort)
    original = labels.read_bytes()
    labels.write_bytes(original + b"\n")
    with pytest.raises(ValueError, match="exact observation index and labels"):
        truth.load_projection_truth(inputs, labels, output)
    labels.write_bytes(original)
    emitted = json.loads(output.read_bytes())
    emitted["rows"][0]["targets"].reverse()
    output.write_bytes(canonical_json(emitted))
    with pytest.raises(ValueError, match="ordered center"):
        truth.load_projection_truth(inputs, labels, output)
