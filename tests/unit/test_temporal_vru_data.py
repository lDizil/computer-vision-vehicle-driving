from copy import deepcopy
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import tarfile
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import numpy as np

from hybrid_drive.assets.files import PreparationError
from hybrid_drive.experiments import temporal_vru_data as data


PED = "a" * 16
VEHICLE = "f" * 16
PREFIX = "openscene-v1.1/sensor_blobs/mini"


def _frame(index, log="2021.10.05.07.10.04_veh-52_01442_01802", group=0, turned=False):
    key = (group + 1) * 100_000 + index
    translation = (100., 200. + index * 2, 0.) if turned else (100. + index * 2, 200., 0.)
    rotation = (2**-.5, 0., 0., 2**-.5) if turned else (1., 0., 0., 0.)
    return {
        "log_token": f"{group:016x}", "log_name": log, "token": f"{key:016x}", "timestamp": index * 500_000,
        "ego2global_translation": translation, "ego2global_rotation": rotation,
        "anns": {"gt_names": [], "track_tokens": [], "gt_boxes": []},
        "cams": {cam: {"data_path": f"{log}/{cam}/{key*6+j:016x}.jpg",
                       "sensor2lidar_rotation": [[0, 0, 1], [-1, 0, 0], [0, -1, 0]],
                       "sensor2lidar_translation": [0, 0, 0],
                       "cam_intrinsic": [[700, 0, 960], [0, 700, 540], [0, 0, 1]], "distortion": [0] * 5}
                 for j, cam in enumerate(data.CAMERAS)},
    }


def _add(frame, x, y, *, token=PED, name="pedestrian"):
    frame["anns"]["gt_names"].append(name)
    frame["anns"]["track_tokens"].append(token)
    frame["anns"]["gt_boxes"].append([x, y, .8, .6, .6, 1.5, 0.])


def _window():
    frames = [_frame(i, turned=True) for i in range(4)]
    _add(frames[1], 8., 1.)
    _add(frames[2], 7., 1.5)
    _add(frames[3], 6., 2.)
    return frames


def _label(window):
    return data._label_window(window, [f["timestamp"] for f in window])


def _fixture(root, *, duplicate_jpegs=False):
    pinned = data._load_source(Path("scripts/temporal-vru-sources.json"))
    picked = [next(row for row in pinned["logs"] if row["split"] == split) for split in data._RECORDINGS]
    by_split = {row["split"]: {row["recording"]} for row in picked}
    archives, logs = {}, []
    for group, row in enumerate(picked, start=1):
        frames = [_frame(i, row["log_name"], group) for i in range(40)]
        # A tracked moving object yields one label family; later frames are clear.
        for i in range(15):
            _add(frames[i], 8. + (i % 3) / 8, 1.)
        filename = f"camera-{group}.tgz"
        path = root / filename
        with tarfile.open(path, "w:gz") as tar:
            for frame in frames:
                for view in frame["cams"].values():
                    image = b"\xff\xd8\xff" + (b"same" if duplicate_jpegs else view["data_path"].encode())
                    info = tarfile.TarInfo(f'{PREFIX}/{view["data_path"]}')
                    info.size = len(image)
                    tar.addfile(info, BytesIO(image))
        raw = path.read_bytes()
        archives[str(group)] = {"filename": filename, "archive_prefix": PREFIX,
                                "size_bytes": len(raw), "sha256": sha256(raw).hexdigest()}
        logs.append(({**row, "camera_shard": group}, frames))
    return by_split, {"camera_archives": archives}, logs


class TemporalVruDataTests(TestCase):
    def test_checked_in_manifest_is_exact_and_heldout_adds_two_different_logs(self):
        source = data._load_source(Path("scripts/temporal-vru-sources.json"))
        self.assertEqual(set(source["camera_archives"]), {"1", "4", "19", "23", "24", "26"})
        self.assertEqual(len(source["logs"]), 12)
        self.assertEqual(sum(row["split"] == "heldout" for row in source["logs"]), 6)
        self.assertEqual([row["split"] for row in source["logs"] if row["camera_shard"] == 1], ["heldout"] * 2)

    def test_future_actor_centers_are_transformed_to_anchor_ego_not_future_ego(self):
        category, tracks, deltas = _label(_window())
        self.assertEqual(category, "moving")
        self.assertEqual(deltas, [.5, 1.])
        self.assertEqual(len(tracks), 1)
        np.testing.assert_allclose(tracks[0]["centers_xy_m"], [[8., 1.], [9., 1.5], [10., 2.]])

    def test_missing_type_changed_or_grid_exit_are_unknown_not_negative(self):
        missing, changed, departed = (_window() for _ in range(3))
        missing[2]["anns"] = {"gt_names": [], "track_tokens": [], "gt_boxes": []}
        changed[3]["anns"]["gt_names"][0] = "bicycle"
        departed[3]["anns"]["gt_boxes"][0][0] = 26.
        for window in (missing, changed, departed):
            with self.subTest(window=window):
                self.assertIsNone(_label(window))

    def test_negative_requires_empty_roi_for_all_three_label_times(self):
        window = [_frame(i) for i in range(4)]
        _add(window[1], 9., 1., token=VEHICLE, name="vehicle")
        self.assertEqual(_label(window), ("empty", [], [.5, 1.]))
        entrant = deepcopy(window)
        _add(entrant[2], 5., 1.)
        self.assertIsNone(_label(entrant))
        missing = deepcopy(window)
        missing[3].pop("anns")
        self.assertIsNone(_label(missing))
        behind_cameras = deepcopy(window)
        _add(behind_cameras[1], 9., 1.)
        for cam in behind_cameras[1]["cams"].values():
            cam["sensor2lidar_translation"] = [30., 0., 0.]
        self.assertIsNone(_label(behind_cameras))

    def test_observation_and_target_cadences_are_independently_checked(self):
        too_old, first_late, second_late, different_log, camera_missing = (_window() for _ in range(5))
        too_old[1]["timestamp"] += 101_000
        first_late[2]["timestamp"] += 101_000
        second_late[3]["timestamp"] += 101_000
        different_log[2]["log_token"] = "other"
        camera_missing[0]["cams"].pop("CAM_B0")
        for window in (too_old, first_late, second_late, different_log, camera_missing):
            with self.subTest(window=window):
                self.assertIsNone(_label(window))

    def test_curated_caps_keep_windows_spaced_across_categories(self):
        timeline = [_frame(i) for i in range(600)]
        def fake_label(_frames, stamps):
            i = stamps[1] // 500_000
            mode = ["moving", "quiet", "other", "empty"][i % 4]
            distance = {"moving": 1., "quiet": .0, "other": .5, "empty": None}[mode]
            labels = [] if distance is None else [{"centers_xy_m": [[8., 1.], [8., 1.], [8.+distance, 1.]]}]
            return mode, labels, [.5, 1.]
        with patch.object(data, "_label_window", side_effect=fake_label):
            result, summary = data._select_samples(timeline)
            self.assertEqual(data._select_samples(timeline)[0], result)
        self.assertEqual(summary["selected"]["moving"], 12)
        self.assertEqual(summary["selected"]["quiet"], 8)
        self.assertEqual(summary["selected"]["empty"], 4)
        self.assertTrue(all(b[0] - a[0] >= 6 for a, b in zip(result, result[1:])))

    def test_prepare_publishes_versioned_observations_and_targets_to_disjoint_mounts(self):
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            groups, source, logs = _fixture(root)
            with (patch.object(data, "_RECORDINGS", groups), patch.object(data, "_load_source", return_value=source),
                  patch.object(data, "_read_recordings", return_value=logs),
                  patch.object(data, "sys", SimpleNamespace(platform="linux"))):
                output = root / "cohort"
                meta = data.prepare_temporal_vru(root / "not-mounted.tgz", root, root / "source.json", output)
                self.assertGreater(meta["anchors"], 12)
                self.assertEqual(meta["camera_jpegs"], 12 * meta["anchors"])
                self.assertEqual(meta["observed_frames"], 2 * meta["anchors"])
                all_ids, all_digests = set(), set()
                for split in groups:
                    inputs, labels_path = output/split/"inputs", output/split/"labels"/"labels.json"
                    raw = (inputs/"index.json").read_bytes()
                    self.assertTrue(all(key not in raw for key in (b"tracks", b"gt_boxes", b"target_dt_s")))
                    index_only = root/f"{split}-input-only"
                    index_only.mkdir()
                    (index_only/"index.json").write_bytes(raw)
                    index = data.load_temporal_vru_inputs(index_only)
                    labels = data.load_temporal_vru_labels(index_only, labels_path)
                    self.assertEqual(labels["input_sha256"], sha256(raw).hexdigest())
                    self.assertEqual(meta["splits"][split]["input_sha256"], labels["input_sha256"])
                    self.assertTrue(any(r["tracks"] for r in labels["rows"]))
                    self.assertFalse(all_ids & index["frames"].keys())
                    all_ids.update(index["frames"])
                    for frame in index["frames"].values():
                        for view in frame["views"]:
                            self.assertNotIn(view["sha256"], all_digests)
                            all_digests.add(view["sha256"])
                            self.assertEqual(view["sha256"], sha256((inputs/view["path"]).read_bytes()).hexdigest())
                    contaminated = root/"changed.json"
                    labels["rows"][0]["target_dt_s"] = [1., 2.]
                    contaminated.write_text(json.dumps(labels))
                    with self.assertRaisesRegex(ValueError, "target cadence"):
                        data.load_temporal_vru_labels(index_only, contaminated)
                    foreign = next(name for name in groups if name != split)
                    with self.assertRaisesRegex(ValueError, "exact observation index"):
                        data.load_temporal_vru_labels(index_only, output/foreign/"labels"/"labels.json")
                    corrupt_input = json.loads(raw)
                    corrupt_input["anchors"][0]["previous_token"] = corrupt_input["anchors"][0]["sample_token"]
                    (index_only/"index.json").write_text(json.dumps(corrupt_input))
                    with self.assertRaisesRegex(ValueError, "share or miss"):
                        data.load_temporal_vru_inputs(index_only)

    def test_shared_camera_across_splits_cannot_publish_partial_cohort(self):
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            groups, source, logs = _fixture(root, duplicate_jpegs=True)
            with (patch.object(data, "_RECORDINGS", groups), patch.object(data, "_load_source", return_value=source),
                  patch.object(data, "_read_recordings", return_value=logs),
                  patch.object(data, "sys", SimpleNamespace(platform="linux"))):
                output = root/"not-published"
                with self.assertRaisesRegex(PreparationError, "share camera JPEG bytes"):
                    data.prepare_temporal_vru(root/"not-mounted.tgz", root, root/"source.json", output)
                self.assertFalse(output.exists())
