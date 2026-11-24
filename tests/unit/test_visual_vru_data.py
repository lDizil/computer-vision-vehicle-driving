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
from hybrid_drive.experiments import visual_vru_data as data
from hybrid_drive.experiments.bounded_io import canonical_json


_PREFIX = "openscene-v1.1/sensor_blobs/mini"
_LOGS = ("2021.06.03.12.02.06_veh-35_00233_00609", "2021.06.28.15.02.02_veh-38_02398_02848",
         "2021.06.07.18.53.26_veh-26_00005_00427", "2021.06.23.17.31.36_veh-16_00016_00377",
         "2021.06.07.12.54.00_veh-35_01843_02314", "2021.08.17.16.57.11_veh-08_01200_01636")
_SPLITS = ("train", "development", "heldout")
_CALIBRATION = {"cam_intrinsic": [[2, 0, .5], [0, 2, .5], [0, 0, 1]],
                "sensor2lidar_rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "sensor2lidar_translation": [0, 0, 0], "distortion": [0] * 5}
_LEFT = [[2.0 * i, i / 4, .05] for i in range(1, 6)]
_POSES = [_LEFT, [[x, -y, -yaw] for x, y, yaw in _LEFT]]


def _frame(i, group=0):
    start, log = 10_000 * group, _LOGS[group]
    return {"token": f"{start+i+1:016x}", "log_name": log, "log_token": f"{group+1:016x}",
            "timestamp": 500_000 * i, "ego2global_translation": [i / 10, 0, 0],
            "ego2global_rotation": [1, 0, 0, 0],
            "anns": {"gt_names": [], "gt_boxes": [], "track_tokens": []},
            "cams": {name: {**_CALIBRATION, "data_path": f"{log}/{name}/{start+1000+6*i+j:016x}.jpg"}
                     for j, name in enumerate(data.CAMERAS)}}


def _tar(path, members):
    with tarfile.open(path, "w:gz") as archive:
        for name, payload in members:
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            archive.addfile(member, BytesIO(payload))
    raw = path.read_bytes()
    return {"sha256": sha256(raw).hexdigest(), "size_bytes": len(raw)}


def _fixture(root, *, duplicate_jpegs=False):
    source = {"schema": "hybrid-drive-visual-vru-sources-v1", "camera_archives": {}, "logs": []}
    metadata, images = [], {i: [] for i in range(len(_SPLITS))}
    for group, log in enumerate(_LOGS):
        shard = group // 2
        split = _SPLITS[shard]
        frames = [_frame(i, group) for i in range(40)]
        member = f"openscene-v1.1/meta_datas/mini/{log}.pkl"
        payload = pickle.dumps(list(reversed(frames)))
        metadata.append((member, payload))
        source["logs"].append({"split": split, "recording": log.rsplit("_", 2)[0], "log_name": log, "member": member,
                               "camera_shard": shard, "sha256": sha256(payload).hexdigest(),
                               "size_bytes": len(payload), "frame_count": len(frames)})
        images[shard].extend(
            (f'{_PREFIX}/{view["data_path"]}',
             b"\xff\xd8\xff" + (b"duplicate" if duplicate_jpegs else view["data_path"].encode()))
            for frame in frames for view in frame["cams"].values()
        )
    for shard, members in images.items():
        name = f"camera-{shard}.tgz"
        source["camera_archives"][str(shard)] = {"filename": name, "archive_prefix": _PREFIX, **_tar(root / name, members)}
    source["metadata_archive"] = _tar(root / "metadata.tgz", metadata)
    return source


@contextmanager
def _fixture_config(root, source):
    config = root / "sources.json"
    config.write_bytes(canonical_json(source))
    groups = {split: frozenset(log.rsplit("_", 2)[0] for log in _LOGS[2*i:2*i+2]) for i, split in enumerate(_SPLITS)}
    with (patch.object(data, "SOURCE_SHA256", sha256(config.read_bytes()).hexdigest()),
          patch.object(data, "_RECORDINGS", groups), patch.object(data, "_SHARDS", dict(enumerate(_SPLITS))),
          patch.object(data, "sys", SimpleNamespace(platform="linux")),
          patch.object(data, "_PAIR_CAP", 2), patch.object(data, "_POSITIVE_CAP", 1), patch.object(data, "_EMPTY_CAP", 1),
          patch.object(data, "visible_vru_cells", side_effect=lambda f: ((4, 8),) if f["timestamp"] in (0, 3_000_000, 7_000_000) else ()),
          patch.object(data, "probe_pair", side_effect=lambda _, i: (_POSES, int(i == 14), int(i == 14))
                       if i in (6, 14) else None)):
        yield config


class VisualVruDataTests(TestCase):
    def test_repository_manifest_pins_the_exact_log_set_and_archive_splits(self):
        source = data._load_source(Path("scripts/visual-vru-sources.json"))
        self.assertEqual(set(source["camera_archives"]), {"4", "19", "23", "24", "26"})
        self.assertEqual(len({row["sha256"] for row in source["logs"]}), 10)
        for split in _SPLITS:
            self.assertEqual({row["recording"] for row in source["logs"] if row["split"] == split}, data._RECORDINGS[split])

    def test_selector_balances_sides_and_prefers_observable_future_when_possible(self):
        stamps = [i * 500_000 for i in range(60)]
        pairs = {5: (_POSES, 0, None), 6: (_POSES, 0, 0), 13: (_POSES, 1, None), 22: (_POSES, 1, 1), 30: (_POSES, 0, 1)}
        with patch.object(data, "_PAIR_CAP", 3):
            self.assertEqual(data._select_pairs(pairs, stamps), [6, 22, 30])
        frames = [_frame(i) for i in range(50)]
        frames[17].pop("anns")
        frames[20]["cams"].pop("CAM_F0")
        with (patch.object(data, "visible_vru_cells", return_value=()) as labels,
              patch.object(data, "probe_pair", side_effect=AssertionError("Empty corridors must not enter probe search"))):
            selected, stats = data._select_samples(frames)
        self.assertEqual(labels.call_count, 48)
        self.assertEqual(stats["available_empty"], 48)
        self.assertTrue(all(i not in (17, 20) for i, _, _ in selected))
        self.assertTrue(all(stamps[b[0]] - stamps[a[0]] >= 2_500_000 for a, b in zip(selected, selected[1:])))

    def test_end_to_end_pack_separates_labels_and_supports_index_only_mount(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            source = _fixture(root)
            with _fixture_config(root, source) as config:
                output = root / "prepared"
                prepared = data.prepare_vru_data(root / "metadata.tgz", root, config, output)
                self.assertEqual((prepared["anchors"], prepared["camera_jpegs"]), (24, 144))
                all_tokens, all_jpegs = set(), set()
                for split in _SPLITS:
                    inputs, labels = output / split / "inputs", output / split / "labels/labels.json"
                    raw = (inputs / "index.json").read_bytes()
                    self.assertNotIn(b"visible_cells", raw)
                    self.assertNotIn(b"blocked_index", raw)
                    self.assertNotIn(b"gt_boxes", raw)
                    isolated = root / (split + "-index-only")
                    isolated.mkdir()
                    (isolated / "index.json").write_bytes(raw)
                    index = data.load_vru_inputs(isolated)
                    targets = data.load_vru_labels(isolated, labels)
                    self.assertEqual(targets["input_sha256"], sha256(raw).hexdigest())
                    self.assertEqual(prepared["splits"][split]["probes"], 4)
                    self.assertEqual([row["blocked_index"] for row in targets["rows"] if row["blocked_index"] is not None], [0, 1, 0, 1])
                    self.assertFalse(all_tokens & index["frames"].keys())
                    all_tokens.update(index["frames"])
                    for frame in index["frames"].values():
                        for view in frame["views"]:
                            self.assertEqual(view["sha256"], sha256((inputs / view["path"]).read_bytes()).hexdigest())
                            self.assertNotIn(view["sha256"], all_jpegs)
                            all_jpegs.add(view["sha256"])
                    wrong = "heldout" if split != "heldout" else "train"
                    with self.assertRaisesRegex(ValueError, "exact observation index"):
                        data.load_vru_labels(isolated, output / wrong / "labels/labels.json")
                with self.assertRaisesRegex(PreparationError, "already exist"):
                    data.prepare_vru_data(root / "metadata.tgz", root, config, output)

    def test_a_failed_split_check_never_publishes_partial_input_or_labels(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            source = _fixture(root, duplicate_jpegs=True)
            with _fixture_config(root, source) as config:
                output = root / "prepared"
                with self.assertRaisesRegex(PreparationError, "share camera JPEG bytes"):
                    data.prepare_vru_data(root / "metadata.tgz", root, config, output)
                self.assertFalse(output.exists())

    def test_metadata_member_is_checked_before_any_pickle_is_opened(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            source = _fixture(root)
            source["logs"][0]["sha256"] = "a" * 64
            with (_fixture_config(root, source) as config,
                  patch.object(data.pickle, "loads", side_effect=AssertionError("Unsafe metadata reached pickle"))):
                with self.assertRaisesRegex(PreparationError, "metadata member differs"):
                    data.prepare_vru_data(root / "metadata.tgz", root, config, root / "out")

    def test_manifest_index_and_target_contamination_are_rejected(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            source = _fixture(root)
            with _fixture_config(root, source) as config:
                output = root / "prepared"
                data.prepare_vru_data(root / "metadata.tgz", root, config, output)
                inputs = output / "train/inputs"
                labels = output / "train/labels/labels.json"
                dirty = json.loads(labels.read_bytes())
                dirty["rows"][0]["visible_cells"] = [[True, 0]]
                labels.write_bytes(canonical_json(dirty))
                with self.assertRaisesRegex(ValueError, "visible cell coordinates"):
                    data.load_vru_labels(inputs, labels)
                index = data.load_vru_inputs(inputs)
                pair = next(anchor for anchor in index["anchors"] if anchor["probe_poses"] is not None)
                pair["probe_poses"][1][0][0] += 1
                (inputs / "index.json").write_bytes(canonical_json(index))
                with self.assertRaisesRegex(ValueError, "share forward progress"):
                    data.load_vru_inputs(inputs)
