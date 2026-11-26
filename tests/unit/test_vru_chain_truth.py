from contextlib import contextmanager
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import pickle
import tarfile
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from hybrid_drive.assets.files import PreparationError
from hybrid_drive.experiments import vru_chain_truth as truth, vru_fusion_holdout_data as data
from hybrid_drive.experiments.bounded_io import canonical_json
from hybrid_drive.experiments.box_motion_data import CAMERAS, _observation
from hybrid_drive.experiments.temporal_vru_data import _label_window


_RECORDING = "2021.05.12.23.36.44_veh-35"
_LOG = _RECORDING + "_00000_00004"
_PREFIX = "openscene-v1.1/sensor_blobs/mini"
_TRACK_A = "f1" * 8
_TRACK_B = "f2" * 8


def _frame(i):
    token = f"{i:016x}"
    cameras = {name: {"data_path": f"{_LOG}/{name}/{token}.jpg",
                      "sensor2lidar_rotation": [[0, 0, 1], [-1, 0, 0], [0, -1, 0]],
                      "sensor2lidar_translation": [0, 0, 0], "distortion": [0] * 5,
                      "cam_intrinsic": [[700, 0, 960], [0, 700, 540], [0, 0, 1]]} for name in CAMERAS}
    return {"token": token, "log_name": _LOG, "log_token": "history-test-recording", "timestamp": i * 500_000,
            "ego2global_rotation": [1., 0., 0., 0.], "ego2global_translation": [float(i), 0., 0.], "cams": cameras,
            "anns": {"gt_names": ["bicycle", "pedestrian"], "track_tokens": [_TRACK_B, _TRACK_A],
                     "gt_boxes": [[12., 0., 0., 1., 1., 1., 0.], [6. + .5 * i, 1., 0., 1., 1., 1., 0.]]}}


@contextmanager
def _fixture(root, *, anchor_i=2, previous_annotations=True, older_annotations=True, pinned_member=True):
    frames = [_frame(i) for i in range(5)]
    if anchor_i == 2:
        frames[0]["ego2global_rotation"] = [0., 0., 0., 1.]
        frames[0]["anns"] = {key: values[1:] for key, values in frames[0]["anns"].items()}
        if not older_annotations:
            frames[0].pop("anns")
    if not previous_annotations:
        frames[anchor_i - 1].pop("anns")
    older = frames[anchor_i - 2] if anchor_i == 2 else None
    window = frames[anchor_i - 1:anchor_i + 3]
    previous, now = window[:2]
    _, tracks, elapsed = _label_window(window, [frame["timestamp"] for frame in window])

    archive = root / "metadata.tgz"
    payload = pickle.dumps(frames)
    member = tarfile.TarInfo(f"openscene-v1.1/meta_datas/mini/{_LOG}.pkl")
    member.size = len(payload)
    with tarfile.open(archive, "w:gz") as target:
        target.addfile(member, BytesIO(payload))
    source = {
        "metadata_archive": {"size_bytes": archive.stat().st_size, "sha256": sha256(archive.read_bytes()).hexdigest()},
        "camera_archives": {"3": {"archive_prefix": _PREFIX}},
        "logs": [{"recording": _RECORDING, "camera_shard": 3, "log_name": _LOG, "member": member.name,
                  "size_bytes": len(payload), "sha256": sha256(payload).hexdigest() if pinned_member else "0" * 64}],
    }
    sources_path = root / "sources.json"
    sources_path.write_bytes(canonical_json(source))
    source_sha = sha256(sources_path.read_bytes()).hexdigest()
    observed = {}
    for frame in (previous, now) if older is None else (older, previous, now):
        obs = _observation(frame, _RECORDING, _LOG, _PREFIX, {})
        for view in obs["views"]:
            view["sha256"] = sha256(view["path"].encode()).hexdigest()
        observed[frame["token"]] = obs
    anchor = {"sample_token": now["token"], "previous_token": previous["token"],
              "older_token": older["token"] if older is not None else None, "recording": _RECORDING}
    index = {"schema": data.INPUT_SCHEMA, "source_sha256": source_sha, "anchors": [anchor], "frames": observed}
    inputs, labels_path = root / "inputs", root / "labels.json"
    inputs.mkdir()
    index_bytes = canonical_json(index)
    (inputs / "index.json").write_bytes(index_bytes)
    labels_path.write_bytes(canonical_json({"schema": data.TARGET_SCHEMA, "input_sha256": sha256(index_bytes).hexdigest(),
                                           "rows": [{"sample_token": now["token"], "tracks": tracks,
                                                     "target_dt_s": elapsed}]}))
    with (patch.object(truth, "sys", SimpleNamespace(platform="linux")),
          patch.object(truth, "_load_source", return_value=source),
          patch.object(data, "SOURCE_SHA256", source_sha),
          patch.object(data, "_RECORDINGS", frozenset((_RECORDING,)))):
        yield archive, sources_path, inputs, labels_path, root / "history-truth.json"


class VruChainTruthTest(TestCase):
    def test_recovers_track_order_and_rotates_past_positions_without_requiring_images(self):
        with TemporaryDirectory() as directory, _fixture(Path(directory).resolve()) as paths:
            _, _, inputs, labels, output = paths
            report = truth.prepare_vru_chain_truth(*paths)
            recovered = truth.load_vru_chain_truth(inputs, labels, output)
            row, = recovered["rows"]
            self.assertEqual((row["previous_status"], row["older_status"]), ("available", "available"))
            self.assertEqual(row["tracks"], [{"previous_xy_m": [5.5, 1.], "older_xy_m": [-8., -1.]},
                                             {"previous_xy_m": [11., 0.], "older_xy_m": None}])
            self.assertEqual((report["windows"], report["tracks"], report["previous_centers"], report["older_centers"]),
                             (1, 2, 2, 1))
            self.assertEqual(report["truth_sha256"], sha256(output.read_bytes()).hexdigest())
            self.assertNotIn(_TRACK_A.encode(), output.read_bytes())
            self.assertNotIn(_TRACK_B.encode(), output.read_bytes())

    def test_distinguishes_missing_frame_and_missing_annotation_bundle(self):
        for anchor_i, expected_older in ((1, "missing_frame"), (2, "missing_annotations")):
            with (self.subTest(anchor=anchor_i), TemporaryDirectory() as directory,
                  _fixture(Path(directory).resolve(), anchor_i=anchor_i, previous_annotations=False,
                           older_annotations=False) as paths):
                _, _, inputs, labels, output = paths
                truth.prepare_vru_chain_truth(*paths)
                row, = truth.load_vru_chain_truth(inputs, labels, output)["rows"]
                self.assertEqual((row["previous_status"], row["older_status"]), ("missing_annotations", expected_older))
                self.assertEqual(row["tracks"], [{"previous_xy_m": None, "older_xy_m": None}] * 2)

    def test_rejects_modified_current_future_labels_without_writing_truth(self):
        with TemporaryDirectory() as directory, _fixture(Path(directory).resolve()) as paths:
            labels, output = paths[-2:]
            corrupted = json.loads(labels.read_bytes())
            corrupted["rows"][0]["tracks"][0]["centers_xy_m"][1][0] += .125
            labels.write_bytes(canonical_json(corrupted))
            with self.assertRaisesRegex(PreparationError, "frozen current and future targets"):
                truth.prepare_vru_chain_truth(*paths)
            self.assertFalse(output.exists())

    def test_rejects_archive_member_that_does_not_match_the_pin_before_deserialization(self):
        with TemporaryDirectory() as directory, _fixture(Path(directory).resolve(), pinned_member=False) as paths:
            output = paths[-1]
            with patch("hybrid_drive.experiments.visual_vru_data.pickle.loads") as decode:
                with self.assertRaisesRegex(PreparationError, "pinned SHA256"):
                    truth.prepare_vru_chain_truth(*paths)
                decode.assert_not_called()
            self.assertFalse(output.exists())

    def test_rejects_stale_truth_and_impossible_missing_annotation_center(self):
        with (TemporaryDirectory() as directory,
              _fixture(Path(directory).resolve(), previous_annotations=False) as paths):
            _, _, inputs, labels, output = paths
            truth.prepare_vru_chain_truth(*paths)
            original = labels.read_bytes()
            labels.write_bytes(original + b"\n")
            with self.assertRaisesRegex(ValueError, "exact observation index and labels"):
                truth.load_vru_chain_truth(inputs, labels, output)
            labels.write_bytes(original)
            row = json.loads(output.read_bytes())
            row["rows"][0]["tracks"][0]["previous_xy_m"] = [3., 1.]
            output.write_bytes(canonical_json(row))
            with self.assertRaisesRegex(ValueError, "cannot have a center without annotations"):
                truth.load_vru_chain_truth(inputs, labels, output)
