from copy import deepcopy
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import pickle
import tarfile
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from hybrid_drive.assets.files import PreparationError
from hybrid_drive.experiments import frozen_transfer_data as data
from hybrid_drive.experiments.bounded_io import canonical_json


_PREFIX = "openscene-v1.1/sensor_blobs/mini"
_LOGS = ("2021.06.03.12.02.06_veh-35_00233_00609", "2021.06.28.15.02.02_veh-38_02398_02848")
_CALIBRATION = {"cam_intrinsic": [[2, 0, .5], [0, 2, .5], [0, 0, 1]],
                "sensor2lidar_rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "sensor2lidar_translation": [0, 0, 0], "distortion": [0] * 5}


def _frame(i, group=0, *, pedestrian=False):
    start = 10_000 * group
    log = _LOGS[group]
    return {"token": f"{start + i + 1:016x}", "log_name": log, "log_token": f"{group+1:016x}",
            "timestamp": i * 500_000, "ego2global_translation": [i / 10, 0, 0],
            "ego2global_rotation": [1, 0, 0, 0],
            "anns": {"gt_names": ["pedestrian" if pedestrian else "vehicle"],
                     "track_tokens": ["b" * 16], "gt_boxes": [[8, 2, .8, 1, .5, 1.5, 0]]},
            "cams": {name: {**_CALIBRATION, "data_path": f"{log}/{name}/{start + 1000 + 6*i+j:016x}.jpg"}
                     for j, name in enumerate(data.CAMERAS)}}


def _tar(path, members):
    with tarfile.open(path, "w:gz") as archive:
        for name, payload in members:
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            archive.addfile(member, BytesIO(payload))
    payload = path.read_bytes()
    return {"size_bytes": len(payload), "sha256": sha256(payload).hexdigest()}


def _fixture(root):
    source = {"schema": "hybrid-drive-frozen-transfer-sources-v1", "previous_recordings": [],
              "logs": [], "camera_archives": {}}
    members = []
    for group, log in enumerate(_LOGS):
        frames = [_frame(i, group, pedestrian=(i == 20)) for i in range(45)]
        member, payload = f"openscene-v1.1/meta_datas/mini/{log}.pkl", pickle.dumps(list(reversed(frames)))
        members.append((member, payload))
        source["logs"].append({"log_name": log, "recording": log.rsplit("_", 2)[0], "member": member,
                               "camera_shard": group, "size_bytes": len(payload), "sha256": sha256(payload).hexdigest()})
        filename = f"camera-{group}.tgz"
        cameras = _tar(root / filename, [(f'{_PREFIX}/{view["data_path"]}', b"\xff\xd8\xff" + view["data_path"].encode())
                                        for frame in frames for view in frame["cams"].values()])
        source["camera_archives"][str(group)] = {**cameras, "filename": filename, "archive_prefix": _PREFIX}
    source["metadata_archive"] = _tar(root / "metadata.tgz", members)
    return source


class FrozenTransferTests(unittest.TestCase):
    def test_checked_in_sources_pin_independent_recordings_and_members(self):
        original = Path("scripts/frozen-slow-transfer-sources.json")
        source = data.load_transfer_source(original)
        self.assertEqual(len(source["logs"]), 4)
        self.assertEqual(set(source["camera_archives"]), {"24", "26"})
        self.assertEqual(len({r["sha256"] for r in source["logs"]}), 4)
        self.assertTrue({"2021.06.09.14.03.17_veh-12", "2021.07.24.23.50.16_veh-17"} <= set(source["previous_recordings"]))
        source["previous_recordings"].append("2021.06.03.22.59.00_veh-35")
        with TemporaryDirectory() as folder:
            new = Path(folder).resolve() / "new.json"
            new.write_bytes(canonical_json(source))
            with self.assertRaisesRegex(PreparationError, "checked-in digest"):
                data.load_transfer_source(new)
            with patch.object(data, "SOURCES_SHA256", sha256(new.read_bytes()).hexdigest()):
                with self.assertRaisesRegex(PreparationError, "day/vehicle"):
                    data.load_transfer_source(new)

    def test_uniform_stratum_uses_time_and_never_current_annotations(self):
        frames = [_frame(i, pedestrian=i in (6, 41, 42)) for i in range(200)]
        with patch.object(data, "_UNIFORM_COUNT", 3), patch.object(data, "_VULNERABLE_CAP", 2):
            selected, counts = data._select_windows(frames)
            self.assertEqual(selected, [(6, "vulnerable"), (33, "uniform"), (41, "vulnerable"),
                                        (97, "uniform"), (161, "uniform")])
            self.assertEqual(counts["eligible_vulnerable_windows_after_uniform"], 3)
            # Changing labels never affects the time-selected anchors.
            for frame in frames:
                frame["anns"] = {}
            selected, _ = data._select_windows(frames)
            self.assertEqual(selected, [(33, "uniform"), (97, "uniform"), (161, "uniform")])

    def test_eligibility_requires_history_images_and_contiguous_log_stamps(self):
        previous, current = _frame(0), _frame(1)
        self.assertTrue(data._valid_history(previous, current, 500_000))
        self.assertFalse(data._valid_history(previous, current, 750_001))
        self.assertFalse(data._valid_history(previous, _frame(1, 1), 500_000))
        previous["cams"].pop("CAM_F0")
        self.assertFalse(data._valid_history(previous, current, 500_000))

    def test_challenge_uses_only_pedestrian_and_bicycle_in_current_corridor(self):
        f = _frame(30)
        self.assertFalse(data._current_vulnerable_in_corridor(f))
        f["anns"]["gt_names"] = ["bicycle"]
        self.assertTrue(data._current_vulnerable_in_corridor(f))
        f["anns"]["gt_boxes"][0][1] = 6.01
        self.assertFalse(data._current_vulnerable_in_corridor(f))
        f["anns"]["gt_names"] = []
        with self.assertRaisesRegex(PreparationError, "not aligned"):
            data._current_vulnerable_in_corridor(f)

    def test_preparation_separates_ledger_and_camera_inputs_and_supports_index_only_mount(self):
        with TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            source = _fixture(root)
            config = root / "source.json"
            config.write_bytes(canonical_json(source))
            with (patch.object(data, "sys", SimpleNamespace(platform="linux")),
                  patch.object(data, "SOURCES_SHA256", sha256(config.read_bytes()).hexdigest()),
                  patch.object(data, "_LOG_COUNT", 2), patch.object(data, "_UNIFORM_COUNT", 2),
                  patch.object(data, "_VULNERABLE_CAP", 1)):
                output = root / "staged"
                result = data.prepare_transfer(root / "metadata.tgz", root, config, output)
                self.assertEqual((result["anchors"], result["observed_frames"], result["camera_jpegs"]), (6, 12, 72))
                index_raw = (output / "inputs/index.json").read_bytes()
                ledger = json.loads((output / "evaluation/selection.json").read_bytes())
                self.assertEqual(ledger["input_sha256"], sha256(index_raw).hexdigest())
                self.assertNotIn(b"gt_boxes", index_raw)
                self.assertNotIn(b"population", index_raw)
                self.assertNotIn(b"frame_index", index_raw)
                self.assertNotIn(b"track_tokens", index_raw)
                isolated_index = root / "index-only"
                isolated_index.mkdir()
                (isolated_index / "index.json").write_bytes(index_raw)
                index = data.load_transfer_inputs(isolated_index)
                self.assertEqual([a["sample_token"] for a in index["anchors"]],
                                 [r["sample_token"] for r in ledger["rows"]])
                self.assertEqual([r["frame_index"] for r in ledger["rows"] if r["population"] == "vulnerable"], [20, 20])
                for frame in index["frames"].values():
                    for view in frame["views"]:
                        jpeg = output / "inputs" / view["path"]
                        self.assertEqual(sha256(jpeg.read_bytes()).hexdigest(), view["sha256"])
                dirty = deepcopy(index)
                dirty["anchors"][0]["population"] = "vulnerable"
                (isolated_index / "index.json").write_bytes(canonical_json(dirty))
                with self.assertRaisesRegex(ValueError, "unknown keys"):
                    data.load_transfer_inputs(isolated_index)
                dirty = deepcopy(index)
                dirty["anchors"][1]["history_tokens"] = dirty["anchors"][0]["history_tokens"]
                (isolated_index / "index.json").write_bytes(canonical_json(dirty))
                with self.assertRaisesRegex(ValueError, "shared frames"):
                    data.load_transfer_inputs(isolated_index)
                with self.assertRaisesRegex(PreparationError, "already exist"):
                    data.prepare_transfer(root / "metadata.tgz", root, config, output)

    def test_exact_member_is_verified_before_unpickling(self):
        with TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            source = _fixture(root)
            first = source["logs"][0]
            first["sha256"] = "0" * 64
            source_sha = sha256(canonical_json(source)).hexdigest()
            with (patch.object(data, "sys", SimpleNamespace(platform="linux")),
                  patch.object(data, "SOURCES_SHA256", source_sha),
                  patch.object(data.pickle, "loads", side_effect=AssertionError("Unsafe metadata was opened"))):
                with self.assertRaisesRegex(PreparationError, "member differs"):
                    next(data.read_transfer_recordings(root / "metadata.tgz", source))


if __name__ == "__main__":
    unittest.main()
