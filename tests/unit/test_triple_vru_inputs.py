from copy import deepcopy
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import tarfile
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from hybrid_drive.assets.files import PreparationError
from hybrid_drive.experiments import temporal_vru_data as base_data
from hybrid_drive.experiments import triple_vru_inputs as data
from hybrid_drive.experiments.box_motion_data import CAMERAS, _observation
from hybrid_drive.experiments.io import write_json


_PREFIX = "openscene-v1.1/sensor_blobs/mini"


def _frame(i, log, group):
    number = 1000 * group + i
    return {
        "token": f"{number:016x}", "log_name": log, "log_token": f"{group:016x}", "timestamp": 500_000 * i,
        "ego2global_translation": [float(i), 0., 0.], "ego2global_rotation": [1., 0., 0., 0.],
        "cams": {cam: {"data_path": f"{log}/{cam}/{number*6+j:016x}.jpg",
                       "sensor2lidar_rotation": [[0, 0, 1], [-1, 0, 0], [0, -1, 0]],
                       "sensor2lidar_translation": [0, 0, 0], "distortion": [0] * 5,
                       "cam_intrinsic": [[700, 0, 960], [0, 700, 540], [0, 0, 1]]}
                 for j, cam in enumerate(CAMERAS)},
    }


def _jpeg(name):
    return b"\xff\xd8\xff" + name.encode()


def _case(root, *, repeat_from=None, missing_heldout_camera=True):
    pinned = base_data._load_source(Path("scripts/temporal-vru-sources.json"))
    logs, archives, groups = [], {}, {}
    previous_split_older = None
    base = root / "base"
    for group, split in enumerate(data.SPLITS, start=1):
        src = dict(next(row for row in pinned["logs"] if row["split"] == split), camera_shard=group)
        name, recording = src["log_name"], src["recording"]
        timeline = [_frame(i, name, group) for i in range(20)]
        timeline[13]["timestamp"] -= 200_000
        if split == "heldout" and missing_heldout_camera:
            del timeline[6]["cams"]["CAM_B0"]
        groups[split] = {recording}
        logs.append((src, timeline))
        archive = root / f"camera-{group}.tgz"
        older_camera = timeline[6]["cams"]["CAM_F0"]["data_path"]
        peer, replaced = None, None
        if repeat_from == "base" and split == "development":
            peer = timeline[7]["cams"]["CAM_F0"]["data_path"]
            replaced = older_camera
        elif repeat_from == "other_split" and split == "heldout":
            peer, replaced = previous_split_older, older_camera
        previous_split_older = older_camera
        with tarfile.open(archive, "w:gz") as tar:
            for frame in timeline:
                for camera in frame["cams"].values():
                    relative = camera["data_path"]
                    raw = _jpeg(peer if relative == replaced else relative)
                    member = tarfile.TarInfo(f"{_PREFIX}/{relative}")
                    member.size = len(raw)
                    tar.addfile(member, BytesIO(raw))
        content = archive.read_bytes()
        archives[str(group)] = {"filename": archive.name, "size_bytes": len(content),
                                "sha256": sha256(content).hexdigest(), "archive_prefix": _PREFIX}
        observed, anchors = {}, []
        for i in (1, 8, 15):
            previous, current = timeline[i - 1:i + 1]
            anchors.append({"sample_token": current["token"], "previous_token": previous["token"],
                            "recording": recording})
            for frame in (previous, current):
                raw = _observation(frame, recording, name, _PREFIX, {})
                for view in raw["views"]:
                    view["sha256"] = sha256(_jpeg(frame["cams"][view["camera"]]["data_path"])).hexdigest()
                observed[frame["token"]] = raw
        mounted = base / split / "inputs"
        mounted.mkdir(parents=True)
        write_json(mounted / "index.json", {"schema": base_data.INPUT_SCHEMA, "source_sha256": base_data.SOURCE_SHA256,
                                            "split": split, "anchors": anchors, "frames": observed})
    return base, groups, {"logs": [spec for spec, _ in logs], "camera_archives": archives}, logs


def _prepare(root, base, source, logs):
    # No archive unpickling; scan only tiny independently hashed synthetic camera files.
    with (patch.object(data, "_load_source", return_value=source), patch.object(data, "_read_recordings", return_value=logs),
          patch.object(data, "sys", SimpleNamespace(platform="linux"))):
        return data.prepare_third_observations(root / "not-mounted.tgz", root, root / "source.json", base, root / "third")


class ThirdVruInputTests(TestCase):
    def test_extension_preserves_original_windows_and_only_extracts_older_images(self):
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            base, groups, source, logs = _case(root)
            previous_bytes = {split: (base / split / "inputs/index.json").read_bytes() for split in data.SPLITS}
            with patch.object(base_data, "_RECORDINGS", groups):
                result = _prepare(root, base, source, logs)
                self.assertEqual((result["windows"], result["ready"], result["camera_jpegs"]), (6, 1, 6))
                for split, (_, timeline) in zip(data.SPLITS, logs):
                    baseline = base / split / "inputs"
                    split_root = root / "third" / split
                    raw = (split_root / "index.json").read_bytes()
                    ext = data.load_third_observations(baseline, split_root)
                    self.assertEqual(ext["base_input_sha256"], sha256(previous_bytes[split]).hexdigest())
                    self.assertEqual((baseline / "index.json").read_bytes(), previous_bytes[split])
                    self.assertEqual(ext["anchors"][1]["older_token"],
                                     timeline[6]["token"] if split == "development" else None)
                    self.assertEqual([a["status"] for a in ext["anchors"]], ["missing_older",
                                     "ready" if split == "development" else "missing_older_cameras", "older_cadence"])
                    self.assertNotIn(b"tracks", raw)
                    self.assertNotIn(b"target_dt_s", raw)
                    self.assertNotIn(b"anns", raw)
                    self.assertFalse((split_root / "labels").exists())
                    payloads = []
                    for frame in ext["frames"].values():
                        for view in frame["views"]:
                            jpg = split_root / view["path"]
                            payloads.append(jpg)
                            self.assertEqual(view["sha256"], sha256(jpg.read_bytes()).hexdigest())
                    self.assertCountEqual(payloads, list(split_root.glob("rgb/*/*.jpg")))
                    index_only = root / f"{split}-only"
                    index_only.mkdir()
                    (index_only / "index.json").write_bytes(raw)
                    self.assertEqual(data.load_third_observations(baseline, index_only), ext)

    def test_immediate_older_clock_distinguishes_cadence_and_missing_source(self):
        old, last = (_frame(i, "test_log", 1) for i in (2, 3))
        for valid in (400_000, 500_000, 600_000):
            self.assertEqual(data._third_status(dict(old, timestamp=last["timestamp"] - valid), last), "ready")
        for bad in (399_999, 600_001, -100_000):
            self.assertEqual(data._third_status(dict(old, timestamp=last["timestamp"] - bad), last), "older_cadence")
        self.assertEqual(data._third_status(dict(old, log_token="another"), last), "missing_older")
        self.assertEqual(data._third_status(None, last), "missing_older")
        self.assertEqual(data._third_status(dict(old, cams={}), last), "missing_older_cameras")

    def test_wrong_base_sha_or_row_order_is_rejected_even_for_shape_valid_indexes(self):
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            base, groups, source, logs = _case(root)
            with patch.object(base_data, "_RECORDINGS", groups):
                _prepare(root, base, source, logs)
                a, b = base / "development" / "inputs", root / "third" / "development"
                path = a / "index.json"
                original = path.read_bytes()
                path.write_bytes(original + b"\n")
                with self.assertRaisesRegex(ValueError, "exact base observation split"):
                    data.load_third_observations(a, b)
                path.write_bytes(original)
                ext = data.load_third_observations(a, b)
                ext["anchors"][0], ext["anchors"][1] = ext["anchors"][1], ext["anchors"][0]
                write_json(b / "index.json", ext)
                with self.assertRaisesRegex(ValueError, "order or identity"):
                    data.load_third_observations(a, b)

    def test_forged_older_time_token_or_digest_is_rejected(self):
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            base, groups, source, logs = _case(root)
            with patch.object(base_data, "_RECORDINGS", groups):
                _prepare(root, base, source, logs)
                baseline, ext_root = base / "development" / "inputs", root / "third" / "development"
                ext = data.load_third_observations(baseline, ext_root)
                old = ext["anchors"][1]["older_token"]
                previous = ext["anchors"][1]["previous_token"]
                changed = deepcopy(ext)
                changed["frames"][old]["ref_time_us"] += 101_000
                write_json(ext_root / "index.json", changed)
                with self.assertRaisesRegex(ValueError, "must precede"):
                    data.load_third_observations(baseline, ext_root)
                changed = deepcopy(ext)
                changed["anchors"][1]["older_token"] = previous
                changed["frames"][previous] = changed["frames"].pop(old)
                write_json(ext_root / "index.json", changed)
                with self.assertRaisesRegex(ValueError, "duplicated older"):
                    data.load_third_observations(baseline, ext_root)
                changed = deepcopy(ext)
                original_jpg = base_data.load_temporal_vru_inputs(baseline)["frames"][previous]["views"][0]["sha256"]
                changed["frames"][old]["views"][0]["sha256"] = original_jpg
                write_json(ext_root / "index.json", changed)
                with self.assertRaisesRegex(ValueError, "share JPEG bytes"):
                    data.load_third_observations(baseline, ext_root)

    def test_duplicate_source_jpeg_aborts_without_publishing_the_extension(self):
        for source_of_duplicate in ("base", "other_split"):
            with self.subTest(source_of_duplicate=source_of_duplicate), TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                base, groups, source, logs = _case(root, repeat_from=source_of_duplicate, missing_heldout_camera=False)
                with patch.object(base_data, "_RECORDINGS", groups):
                    with self.assertRaisesRegex(PreparationError, "share JPEG bytes"):
                        _prepare(root, base, source, logs)
                self.assertFalse((root / "third").exists())

    def test_preparation_refuses_a_base_pair_not_bound_to_the_raw_timeline(self):
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            base, groups, source, logs = _case(root)
            # Do not remove an anchor silently when the raw association would change.
            logs[0][1][7]["timestamp"] -= 50_000
            with patch.object(base_data, "_RECORDINGS", groups):
                with self.assertRaisesRegex(PreparationError, "differs from the pinned recording"):
                    _prepare(root, base, source, logs)
            self.assertFalse((root / "third").exists())
