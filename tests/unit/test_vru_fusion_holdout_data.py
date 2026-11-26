from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import tarfile
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from hybrid_drive.assets.files import PreparationError
from hybrid_drive.experiments import vru_fusion_holdout_data as data
from hybrid_drive.experiments.bounded_io import canonical_json


SOURCE = Path("scripts/vru-fusion-holdout-sources.json")
PREFIX = "openscene-v1.1/sensor_blobs/mini"
POSE_ROTATION = [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]
INTRINSIC = [[700, 0, 960], [0, 700, 540], [0, 0, 1]]


def _frame(i, log):
    cams = {c: {"data_path": f"{log}/{c}/{i:016x}.jpg", "sensor2lidar_rotation": POSE_ROTATION,
                "sensor2lidar_translation": [0, 0, 0], "cam_intrinsic": INTRINSIC, "distortion": [0] * 5}
            for c in data.CAMERAS}
    return {"timestamp": i * 500_000, "token": f"{i:016x}", "log_name": log, "log_token": "one-recording",
            "ego2global_translation": [0., 0., 0.], "ego2global_rotation": [1., 0., 0., 0.], "cams": cams}


def _fixture(root, *, duplicate=False):
    spec = data._load_source(SOURCE)["logs"][0]
    frames = [_frame(i, spec["log_name"]) for i in range(18)]
    frames[6]["cams"].pop(data.CAMERAS[0])
    path = root / "synthetic.tgz"
    with tarfile.open(path, "w:gz") as tar:
        for frame in frames:
            for view in frame["cams"].values():
                name = view["data_path"]
                payload = b"\xff\xd8\xff" + (b"identical" if duplicate else name.encode())
                info = tarfile.TarInfo(f"{PREFIX}/{name}")
                info.size = len(payload)
                tar.addfile(info, BytesIO(payload))
    archive = {"filename": path.name, "archive_prefix": PREFIX, "size_bytes": path.stat().st_size,
               "sha256": sha256(path.read_bytes()).hexdigest()}
    return spec, frames, {"camera_archives": {str(spec["camera_shard"]): archive}}


def _prepare(root, *, duplicate=False):
    spec, frames, source = _fixture(root, duplicate=duplicate)
    tracks = [{"centers_xy_m": [[8., 1.], [8.5, 1.], [9., 1.]]}]
    selected = [(1, tracks, [.5, 1.]), (8, [], [.5, 1.]), (14, [], [.5, 1.])]
    output = root / "dataset"
    with (patch.object(data, "_RECORDINGS", {spec["recording"]}),
          patch.object(data, "_load_source", return_value=source),
          patch.object(data, "_read_recordings", return_value=[(spec, frames)]),
          patch.object(data, "_select_samples", return_value=(selected, {})),
          patch.object(data, "sys", SimpleNamespace(platform="linux"))):
        return output, data.prepare_fusion_holdout(root / "unused.tgz", root, SOURCE, output), spec["recording"]


class FusionHoldoutDataTest(TestCase):
    def test_manifest_excludes_every_prior_day_vehicle_and_pins_each_shard(self):
        source = data._load_source(SOURCE)
        self.assertEqual(set(source["camera_archives"]), {"3", "5", "22"})
        self.assertEqual(len(source["previous_recordings"]), 12)
        self.assertEqual(len(source["logs"]), 6)
        group = lambda recording: (recording[:10], recording.rsplit("_", 1)[-1])
        self.assertFalse({group(x["recording"]) for x in source["logs"]} & {group(x) for x in source["previous_recordings"]})

    def test_source_guard_rejects_same_day_vehicle_even_when_stamps_and_pin_differ(self):
        source = data._load_source(SOURCE)
        earlier = source["previous_recordings"][0]
        later = f"{earlier[:10]}.01.02.03_{earlier.rsplit('_', 1)[-1]}"
        log = f"{later}_00001_00010"
        source["logs"][0].update(recording=later, log_name=log, member=f"openscene-v1.1/meta_datas/mini/{log}.pkl")
        with TemporaryDirectory() as directory:
            fixed = Path(directory).resolve() / "unsafe.json"
            fixed.write_bytes(canonical_json(source))
            with patch.object(data, "SOURCE_SHA256", sha256(fixed.read_bytes()).hexdigest()):
                with self.assertRaisesRegex(PreparationError, "intersects old or selected"):
                    data._load_source(fixed)

    def test_deterministic_curated_caps_and_spacing_precede_any_prediction(self):
        frames = [{"timestamp": i * 500_000} for i in range(600)]
        def choose_label(_frames, stamps):
            return ("moving", "quiet", "other", "empty")[stamps[1] // 500_000 % 4], [], [.5, 1.]
        with patch.object(data, "_label_window", side_effect=choose_label):
            selected, summary = data._select_samples(frames)
            self.assertEqual(data._select_samples(frames), (selected, summary))
        self.assertEqual(summary["selected"], {"moving": 8, "quiet": 4, "empty": 2})
        self.assertTrue(all(b[0] - a[0] >= 6 for a, b in zip(selected, selected[1:])))

    def test_prepare_retains_windows_with_missing_history_and_seals_all_labels(self):
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            output, meta, recording = _prepare(root)
            index_only = root / "index-only"
            index_only.mkdir()
            raw = (output / "inputs/index.json").read_bytes()
            (index_only / "index.json").write_bytes(raw)
            with patch.object(data, "_RECORDINGS", {recording}):
                index = data.load_fusion_inputs(index_only)
                labels = data.load_fusion_labels(index_only, output / "labels/labels.json")
            self.assertEqual([a["older_token"] for a in index["anchors"]], [None, None, f"{12:016x}"])
            self.assertEqual(meta["recordings"][recording]["older_statuses"], {
                "missing_older": 1, "missing_older_cameras": 1, "ready": 1,
            })
            self.assertEqual((meta["anchors"], meta["observed_frames"], meta["camera_jpegs"]), (3, 7, 42))
            self.assertEqual(labels["input_sha256"], sha256(raw).hexdigest())
            self.assertEqual(meta["input_sha256"], labels["input_sha256"])
            self.assertNotIn(b"centers_xy_m", raw)
            self.assertNotIn(b"target_dt_s", raw)
            for frame in index["frames"].values():
                for view in frame["views"]:
                    self.assertEqual(view["sha256"], sha256((output / "inputs" / view["path"]).read_bytes()).hexdigest())

    def test_load_rejects_wrong_cadence_and_stale_or_out_of_grid_labels(self):
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            output, _, recording = _prepare(root)
            inputs, labels_path = output / "inputs", output / "labels/labels.json"
            index_path = inputs / "index.json"
            original = index_path.read_bytes()
            labels = json.loads(labels_path.read_bytes())
            with patch.object(data, "_RECORDINGS", {recording}):
                index = json.loads(original)
                old = index["anchors"][-1]["older_token"]
                index["frames"][old]["ref_time_us"] -= 100_001
                index_path.write_bytes(canonical_json(index))
                with self.assertRaisesRegex(ValueError, "cadence"):
                    data.load_fusion_inputs(inputs)
                index_path.write_bytes(original + b"\n")
                with self.assertRaisesRegex(ValueError, "exact observation index"):
                    data.load_fusion_labels(inputs, labels_path)
                index_path.write_bytes(original)
                labels["rows"][0]["tracks"][0]["centers_xy_m"][2] = [1000., 1.]
                labels_path.write_bytes(canonical_json(labels))
                with self.assertRaisesRegex(ValueError, "outside the labelled grid"):
                    data.load_fusion_labels(inputs, labels_path)

    def test_duplicate_jpeg_cannot_publish_partial_observations(self):
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            with self.assertRaisesRegex(PreparationError, "unique JPEG"):
                _prepare(root, duplicate=True)
            self.assertFalse((root / "dataset").exists())

    def test_future_token_reuse_prevents_index_publication(self):
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            spec, frames, source = _fixture(root)
            frames[2]["token"] = frames[12]["token"]
            archive_key = str(spec["camera_shard"])
            with (patch.object(data, "_RECORDINGS", {spec["recording"]}),
                  patch.object(data, "_read_recordings", return_value=[(spec, frames)]),
                  patch.object(data, "_select_samples", return_value=([(1, [], [.5, 1.]), (14, [], [.5, 1.])], {}))):
                with self.assertRaisesRegex(PreparationError, "future observation"):
                    data._collect_observations(root / "unused.tgz", source, {archive_key: {}})
