from copy import deepcopy
from unittest import TestCase

import numpy as np

from hybrid_drive.experiments.box_motion_data import CAMERAS
from hybrid_drive.experiments.visual_vru_geometry import exclusive_corridors, probe_pair, visible_vru_cells


PEDESTRIAN = "a" * 16
BICYCLE = "b" * 16
VEHICLE = "c" * 16


def _timeline(*, speed=4., turned=False):
    camera = {
        "cam_intrinsic": ((700., 0., 960.), (0., 700., 540.), (0., 0., 1.)),
        "sensor2lidar_rotation": ((0., 0., 1.), (-1., 0., 0.), (0., -1., 0.)),
        "sensor2lidar_translation": (0., 0., 0.),
    }
    frames = []
    for step in range(-1, 6):
        advance = step * .5 * speed
        translation = (100., 200. + advance, 0.) if turned else (100. + advance, 200., 0.)
        rotation = (2**-.5, 0., 0., 2**-.5) if turned else (1., 0., 0., 0.)
        frames.append({
            "timestamp": 10_000_000 + step*500_000, "log_token": "same", "log_name": "same-log",
            "ego2global_translation": translation, "ego2global_rotation": rotation,
            "anns": {"gt_names": [], "track_tokens": [], "gt_boxes": []},
            "cams": {name: deepcopy(camera) for name in CAMERAS},
        })
    return frames


def _track(frame, token, name, x, y, *, size=(.5, .5, 1.6)):
    anns = frame["anns"]
    anns["track_tokens"].append(token)
    anns["gt_names"].append(name)
    anns["gt_boxes"].append((x, y, .8, *size, 0.))


def _persist(frames, future_side=2.2):
    _track(frames[1], PEDESTRIAN, "pedestrian", 8.5, 2.2)
    for step, frame in enumerate(frames[2:], start=1):
        _track(frame, PEDESTRIAN, "pedestrian", 8.5 - step*2., future_side)


class VisualVruGeometryTests(TestCase):
    def test_positive_cell_deduplicates_and_uses_only_in_grid_vrus_and_real_camera_space(self):
        frame = _timeline()[1]
        for name in CAMERAS:
            if name != "CAM_L2":
                frame["cams"][name]["sensor2lidar_translation"] = (30., 0., 0.)
        _track(frame, PEDESTRIAN, "pedestrian", 8.3, 2.2)
        _track(frame, BICYCLE, "bicycle", 8.7, 2.8)
        _track(frame, VEHICLE, "vehicle", 10., 0.)
        _track(frame, "f"*16, "bicycle", 27., 0.)
        self.assertEqual(visible_vru_cells(frame), ((6, 9),))
        frame["cams"]["CAM_L2"]["sensor2lidar_translation"] = (30., 0., 0.)
        self.assertEqual(visible_vru_cells(frame), ())

    def test_missing_annotation_or_required_rig_not_mislabeled_negative(self):
        frames = _timeline()
        with self.assertRaisesRegex(ValueError, "annotations"):
            visible_vru_cells({})
        frame = frames[1]
        _track(frame, PEDESTRIAN, "pedestrian", 8.3, 2.2)
        del frame["cams"]["CAM_L2"]
        with self.assertRaisesRegex(ValueError, "Six calibrated"):
            visible_vru_cells(frame)
        del frame["anns"]
        self.assertIsNone(probe_pair(frames, 1))

    def test_pose_mirrors_and_contacts_use_past_speed_in_rotated_ego_axes(self):
        frames = _timeline(turned=True)
        _persist(frames)
        poses, blocked, future = probe_pair(frames, 1)
        left, right = np.array(poses)
        np.testing.assert_allclose(left[:, 0], [2., 4., 6., 8., 10.])
        np.testing.assert_allclose(right, left * [1., -1., -1.])
        np.testing.assert_allclose(left[-1], [10., 2., 0.])
        self.assertEqual((blocked, future), (0, 0))
        mask = exclusive_corridors(poses)
        self.assertEqual(mask.shape, (2, 25, 14))
        self.assertEqual(mask.dtype, np.bool_)
        self.assertFalse(np.any(mask[0] & mask[1]))
        np.testing.assert_array_equal(mask[0], mask[1, :, ::-1])
        self.assertEqual(mask[:, 6, 9].tolist(), [True, False])

    def test_future_measure_handles_reversal_without_changing_static_path(self):
        positive = _timeline()
        opposite = _timeline()
        _persist(positive, future_side=2.2)
        _persist(opposite, future_side=-2.2)
        poses0, label0, future0 = probe_pair(positive, 1)
        poses1, label1, future1 = probe_pair(opposite, 1)
        self.assertEqual((poses0, label0), (poses1, label1))
        self.assertEqual((future0, future1), (0, 1))

    def test_future_measure_abstains_on_bad_alignment_or_absent_track_or_mixed_contact(self):
        complete = _timeline()
        _persist(complete)
        short = complete[:-1]
        drift = deepcopy(complete)
        drift[3]["timestamp"] += 125_000
        missing = deepcopy(complete)
        missing[3]["anns"] = {"gt_names": [], "track_tokens": [], "gt_boxes": []}
        mixed = deepcopy(complete)
        _track(mixed[3], VEHICLE, "vehicle", 2., -.4, size=(4., 1.8, 1.6))
        for window in (short, drift, missing, mixed):
            with self.subTest(window=window):
                _, blocked, future = probe_pair(window, 1)
                self.assertEqual(blocked, 0)
                self.assertIsNone(future)

    def test_static_probe_rejects_initial_overlaps_mixed_vehicles_and_two_blocked_arms(self):
        annotated = _timeline()
        _persist(annotated)
        initial = deepcopy(annotated)
        _track(initial[1], BICYCLE, "bicycle", 1.8, 0.)
        mixed = deepcopy(annotated)
        _track(mixed[1], VEHICLE, "vehicle", 8.5, -2.2, size=(4., 1.8, 1.6))
        two_sides = deepcopy(annotated)
        _track(two_sides[1], BICYCLE, "bicycle", 8.5, -2.2)
        for window in (initial, mixed, two_sides):
            self.assertIsNone(probe_pair(window, 1))

    def test_label_does_not_credit_target_in_shared_or_other_exclusive_corridor(self):
        frames = _timeline()
        _track(frames[1], PEDESTRIAN, "pedestrian", 8.5, 2.2)
        # This is almost out of contact with the right path, yet its grid cell
        # is within the receiver's padded sweep; the visual training case is ambiguous.
        _track(frames[1], BICYCLE, "bicycle", 8.5, -3.9, size=(.15, .15, 1.6))
        self.assertIsNone(probe_pair(frames, 1))

    def test_out_of_envelope_or_missing_history_has_no_proposal(self):
        slow = _timeline(speed=.2)
        fast = _timeline(speed=9.5)
        complete = _timeline()
        other = deepcopy(complete)
        other[0]["log_name"] = "other-log"
        for frames in (slow, fast, complete, other):
            _track(frames[1], PEDESTRIAN, "pedestrian", 8.5, 2.2)
        for frames in (slow, fast, other):
            self.assertIsNone(probe_pair(frames, 1))
        self.assertIsNone(probe_pair(complete, 0))

    def test_identical_arms_leave_no_exclusive_area_and_invalid_shapes_rejected(self):
        path = [[0., 0., 0.]]*5
        self.assertFalse(exclusive_corridors([path, path]).any())
        for poses in ([path], [path, [[float("nan"), 0., 0.]]*5]):
            with self.assertRaises(ValueError):
                exclusive_corridors(poses)
