from copy import deepcopy
from unittest import TestCase

from hybrid_drive.experiments.box_motion_data import CAMERAS
from hybrid_drive.experiments.frozen_transfer_contacts import measure_recording


TOKEN = "3" * 16
STAMP = 12_000_000
PEDESTRIAN = "a" * 16
BICYCLE = "b" * 16
VEHICLE = "c" * 16


def _frames():
    front_camera = {
        "cam_intrinsic": ((700., 0., 960.), (0., 700., 540.), (0., 0., 1.)),
        "sensor2lidar_rotation": ((0., 0., 1.), (-1., 0., 0.), (0., -1., 0.)),
        "sensor2lidar_translation": (0., 0., 0.),
        "distortion": (0., 0., 0., 0., 0.),
    }
    result = [{
        "token": TOKEN if step == 0 else f"{step:016x}",
        "timestamp": STAMP + step * 500_000,
        "log_token": "physical-log",
        "log_name": "heldout-physical-log",
        "ego2global_translation": (100., 200., 0.),
        "ego2global_rotation": (1., 0., 0., 0.),
        "anns": {"gt_names": [], "track_tokens": [], "gt_boxes": []},
    } for step in range(6)]
    result[0]["cams"] = {name: deepcopy(front_camera) for name in CAMERAS}
    return result


def _track(frame, token, category, center, size=(.6, .6, 1.6), yaw=0.):
    annotation = frame["anns"]
    annotation["gt_names"].append(category)
    annotation["track_tokens"].append(token)
    annotation["gt_boxes"].append((*center, *size, yaw))


def _identity():
    selection = {"sample_token": TOKEN, "recording": "independent", "ref_time_us": STAMP}
    prediction = {**selection, "status": "ok", "snapshot_sha256": "e" * 64, "candidates": [
        _candidate("left", 0., None, 2., (5., 0., 1., 4., 0.)),
        _candidate("right", 6., None, 3., (1., 5e-7, 0., 0., 2.)),
    ]}
    return {**selection, "population": "uniform", "frame_index": 0}, prediction


def _candidate(key, y, fast_overlap, gap, margins):
    methods = ("trained", "untrained", "no_camera", "no_motion", "shuffled_camera")
    return {
        "candidate_id": key,
        "dt_s": .5,
        "poses": [[5., y, 0.] for _ in range(5)],
        "fast_overlap_s": fast_overlap,
        "fast_axis_gap_m": gap,
        "margins": dict(zip(methods, margins)),
    }


def _measure(frames, selection=None, prediction=None):
    default_selection, default_prediction = _identity()
    row, = measure_recording(frames, [selection or default_selection], [prediction or default_prediction])
    return row


class TransferContactTests(TestCase):
    def test_vehicle_pose_transform_from_future_ego_and_method_ties(self):
        frames = _frames()
        frames[0]["ego2global_rotation"] = (2 ** -.5, 0., 0., 2 ** -.5)
        frames[2]["ego2global_translation"] = (100., 206., 0.)
        _track(frames[2], VEHICLE, "vehicle", (0., 0., .8), (4., 1.6, 1.6))
        result = _measure(frames)
        self.assertEqual(result["window_status"], "complete")
        self.assertEqual(result["candidates"][0]["vehicle_contact_steps_s"], [1.])
        self.assertEqual(result["candidates"][1]["vehicle_contact_steps_s"], [])
        self.assertEqual(result["pairs"], [{
            "candidate_ids": ["left", "right"],
            "vehicle_outcome": "left_contact",
            "vulnerable_outcome": "neither_observed",
            "fast_binary_tied": True,
            "preferred": {
                "fast": "left", "fast_gap": "right", "trained": "right",
                "untrained": None, "no_camera": "right", "no_motion": "right",
                "shuffled_camera": "left",
            },
            "vulnerable_only": False,
        }])

    def test_vulnerable_contact_is_only_for_anchor_track_in_compatible_frustum(self):
        frames = _frames()
        _track(frames[0], PEDESTRIAN, "pedestrian", (12., 0., .8))
        _track(frames[0], BICYCLE, "bicycle", (7., 80., .8))
        _track(frames[1], PEDESTRIAN, "pedestrian", (6., 0., .8))
        _track(frames[1], BICYCLE, "bicycle", (6., 6., .8))
        _track(frames[2], "5" * 16, "pedestrian", (6., 6., .8))
        result = _measure(frames)
        self.assertEqual(result["eligible_vulnerable_anchor_count"], 1)
        self.assertEqual(result["annotated_eligible_vulnerable_count_by_sample"], [1, 1, 0, 0, 0, 0])
        self.assertEqual(result["pairs"][0]["vehicle_outcome"], "neither_observed")
        self.assertEqual(result["pairs"][0]["vulnerable_outcome"], "left_contact")
        self.assertTrue(result["pairs"][0]["vulnerable_only"])

    def test_vehicle_yaw_is_transformed_in_addition_to_center(self):
        frames = _frames()
        frames[0]["ego2global_rotation"] = (2 ** -.5, 0., 0., 2 ** -.5)
        frames[1]["ego2global_translation"] = (100., 205.55, 0.)
        _track(frames[1], VEHICLE, "vehicle", (0., 0., .8), (4., 1.6, 1.6))
        selection, prediction = _identity()
        prediction["candidates"][0]["poses"] = [[.8, 0., 0.] for _ in range(5)]
        prediction["candidates"][1]["poses"] = [[1.3, 0., 0.] for _ in range(5)]
        result = _measure(frames, selection, prediction)
        self.assertEqual(result["pairs"][0]["vehicle_outcome"], "right_contact")

    def test_projection_uses_translation_and_all_six_cameras(self):
        frames = _frames()
        for name, camera in frames[0]["cams"].items():
            if name != "CAM_L2":
                camera["sensor2lidar_translation"] = (20., 0., 0.)
        _track(frames[0], BICYCLE, "bicycle", (7., 0., .8))
        _track(frames[1], BICYCLE, "bicycle", (6., 6., .8))
        self.assertEqual(_measure(frames)["pairs"][0]["vulnerable_outcome"], "right_contact")
        frames[0]["cams"]["CAM_L2"]["sensor2lidar_translation"] = (20., 0., 0.)
        self.assertEqual(_measure(frames)["pairs"][0]["vulnerable_outcome"], "neither_observed")

    def test_vulnerable_only_excludes_mixed_exclusive_vehicle_and_vulnerable_contacts(self):
        frames = _frames()
        _track(frames[0], PEDESTRIAN, "pedestrian", (9., 0., .8))
        _track(frames[1], PEDESTRIAN, "pedestrian", (6., 0., .8))
        _track(frames[1], VEHICLE, "vehicle", (6., 6., .8), (4., 1.6, 1.6))
        pair = _measure(frames)["pairs"][0]
        self.assertEqual(pair["vulnerable_outcome"], "left_contact")
        self.assertEqual(pair["vehicle_outcome"], "right_contact")
        self.assertFalse(pair["vulnerable_only"])

    def test_concordant_sampled_contacts_stay_in_pair_support(self):
        frames = _frames()
        _track(frames[1], VEHICLE, "vehicle", (6., 3., .8), (8., 8., 2.))
        selection, prediction = _identity()
        prediction["candidates"][0]["fast_overlap_s"] = 1.
        prediction["candidates"][0]["fast_axis_gap_m"] = None
        pair = _measure(frames, selection, prediction)["pairs"][0]
        self.assertEqual(pair["vehicle_outcome"], "both_contact")
        self.assertFalse(pair["fast_binary_tied"])
        self.assertIsNone(pair["preferred"]["fast_gap"])

    def test_shared_t0_vulnerable_contact_is_not_credited_as_clearance(self):
        frames = _frames()
        _track(frames[0], PEDESTRIAN, "pedestrian", (1.5, 0., .4))
        result = _measure(frames)
        self.assertEqual(result["window_status"], "shared_observed_initial_contact")
        self.assertEqual(result["pairs"], [])

    def test_unknown_windows_retain_status_and_candidate_support_counts(self):
        frames = _frames()
        drift = deepcopy(frames)
        drift[2]["timestamp"] += 125_000
        gap = deepcopy(frames)
        gap[2]["timestamp"] += 275_000
        missing = deepcopy(frames)
        del missing[3]["anns"]["track_tokens"]
        crossed = deepcopy(frames)
        crossed[4]["log_token"] = "wrong-recording"
        for sample, expected in (
            (frames[:-1], "missing_future_samples"),
            (drift, "future_time_mismatch"),
            (gap, "noncontinuous_future"),
            (missing, "missing_annotations"),
            (crossed, "recording_boundary"),
        ):
            with self.subTest(expected=expected):
                result = _measure(sample)
                self.assertEqual(result["window_status"], expected)
                self.assertEqual(result["candidate_pairs"], 1)
                self.assertEqual(result["pairs"], [])

    def test_mismatched_anchor_identity_and_candidate_grid_are_rejected(self):
        selection, prediction = _identity()
        for update, message in (
            ({"ref_time_us": STAMP + 1}, "identities differ"),
            ({"frame_index": 1}, "exact metadata anchor"),
            ({"population": "other"}, "Unknown transfer population"),
        ):
            with self.subTest(update=update), self.assertRaisesRegex(ValueError, message):
                _measure(_frames(), {**selection, **update}, prediction)
        prediction["candidates"][0]["dt_s"] = .25
        with self.assertRaisesRegex(ValueError, "0.5 s annotation grid"):
            _measure(_frames(), selection, prediction)

    def test_out_of_envelope_retains_frame_without_inspecting_future(self):
        selection, prediction = _identity()
        result = _measure(_frames()[:1], selection, {**prediction, "status": "out_of_envelope", "candidates": []})
        self.assertEqual(result["window_status"], "out_of_envelope")
        self.assertEqual(result["candidate_pairs"], 0)
        self.assertEqual(result["pairs"], [])
