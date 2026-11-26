from copy import deepcopy
from unittest import TestCase

from hybrid_drive.experiments.vru_fusion_holdout_metrics import ARMS, summarize_fusion_holdout


_ZERO = (0., 0.)


def _object(key, x, y=2., *, speed=None):
    missing = {"status": "missing", "previous_object_id": None, "velocity_xy_mps": _ZERO,
               "radial_unit_xy": None, "common_cameras": [], "past_xy_m": None}
    velocity = _ZERO if speed is None else (0., speed)
    appearance = missing if speed is None else {
        "status": "estimated", "previous_object_id": key + "-previous", "velocity_xy_mps": velocity,
        "radial_unit_xy": (1., 0.), "common_cameras": ["front"], "past_xy_m": (x, y - .5 * speed),
    }
    return {"object_id": key, "kind": "person", "xy_m": (x, y), "confidence": .8, "cameras": ["front"],
            "motions": {"geometry": missing, "appearance": appearance},
            "confirmation": {"status": "no_initial_link" if speed is None else "confirmed",
                             "older_object_id": None if speed is None else key + "-older",
                             "common_cameras": [] if speed is None else ["front"],
                             "trend_velocity_xy_mps": velocity},
            "forecasts": {name: _ZERO if name == "stationary" else velocity for name in (
                "stationary", "two_frame", "confirmed_two_frame", "three_frame_trend")}}


def _track(x, movement=0.):
    return {"centers_xy_m": [(x, 2.), (x, 2. + movement / 2), (x, 2. + movement)]}


def _window(key="anchor", recording="a", *, older=True, ssd=(), faster=(), fused=(), tracks=(), elapsed=(.5, 1.)):
    common = {"sample_token": key, "previous_token": key + "-previous", "older_token": key + "-older" if older else None,
              "recording": recording, "ref_time_us": 1_000_000, "current_sha256": "b" * 64, "history_s": .5,
              "older_history_s": .5 if older else None}
    return ({name: {**common, "objects": objs} for name, objs in zip(ARMS, (ssd, faster, fused), strict=True)},
            {"sample_token": key, "tracks": tracks, "target_dt_s": elapsed})


def _arrays(*windows):
    return {name: [r[name] for r, _ in windows] for name in ARMS}, [label for _, label in windows]


def _score(*windows):
    return summarize_fusion_holdout(*_arrays(*windows))


def _study(recordings="abcd", *, common=3, extra=(2, 1, 1, 1), offset=.2):
    windows = []
    for recording, end in zip(recordings, extra):
        centers = [5. + 5 * index for index in range(common + end)]
        baseline = [_object(f"old-{i}", x) for i, x in enumerate(centers[:common])]
        fused = [_object(f"new-{i}", x, 2. + offset) for i, x in enumerate(centers)]
        windows.append(_window(recording, recording, ssd=baseline, faster=baseline, fused=fused,
                               tracks=[_track(x) for x in centers]))
    return windows


class TestVruFusionHoldoutMetrics(TestCase):
    def test_paired_motion_scores_only_joint_targets_at_the_observed_interval(self):
        result = _score(_window(
            ssd=[_object("short", 5.)],
            faster=[_object("both-moving", 5., 2.2, speed=.5), _object("lost", 15.), _object("still", 35.)],
            fused=[_object("other-id", 5., 2.4, speed=.8), _object("new", 25.), _object("still", 35., speed=.4)],
            tracks=[_track(5., 1.05), _track(15., 1.), _track(25., 1.), _track(35.)], elapsed=(.5, 1.05),
        ))
        self.assertEqual(result["best_single"], "faster_rcnn")
        self.assertEqual(result["counts"]["annotated_tracks"], 4)
        coverage = result["coverage"]["vs_best_single"]["all"]
        self.assertEqual((coverage["both"], coverage["only_reference"], coverage["only_candidate"]), (2, 1, 1))
        tracking = result["tracking"]
        self.assertEqual(tracking["by_arm"]["ssd_primary"]["moving_matched"], 2)
        self.assertEqual(tracking["by_arm"]["ssd_primary"]["moving_confirmed"], 1)
        self.assertEqual(tracking["by_arm"]["ssd_primary"]["quiet_nonzero_velocity"], 1)
        joint = tracking["paired_against_best_single"]
        self.assertEqual(joint["horizon_s"], 1.)
        self.assertEqual((joint["all"]["pairs"], joint["moving"]["pairs"], joint["quiet"]["pairs"]), (2, 1, 1))
        left, right = (joint["moving"]["arms"][arm] for arm in ("faster_rcnn", "ssd_primary"))
        self.assertAlmostEqual(left["displacement_mean_m"], .525)
        self.assertAlmostEqual(left["future_center_mean_m"], .325)
        self.assertAlmostEqual(right["displacement_mean_m"], .21)
        self.assertAlmostEqual(right["future_center_mean_m"], .19)
        self.assertEqual(joint["quiet"]["arms"]["ssd_primary"]["nonzero_velocity"], 1)

    def test_reference_is_stronger_single_not_always_faster_rcnn(self):
        result = _score(_window(ssd=[_object("keep", 5.), _object("extra", 20.)], faster=[_object("base", 5.)],
                                fused=[_object("same", 5.)], tracks=[_track(5.), _track(20.)]))
        self.assertEqual(result["best_single"], "ssdlite")
        recording = result["coverage"]["by_recording"]["a"]
        self.assertEqual(recording, {"targets": 2, "matched": {"ssdlite": 2, "faster_rcnn": 1, "ssd_primary": 1},
                                     "only_single": 1, "only_fusion": 0})
        self.assertEqual(set(result["tracking"]["paired_against_best_single"]["all"]["arms"]), {"ssdlite", "ssd_primary"})

    def test_empty_labels_and_empty_detections_do_not_disappear_from_coverage(self):
        result = _score(_window("unmatched", "a", tracks=[_track(5., 1.)]),
                        _window("no-labels", "b", fused=[_object("unknown", 5.)]))
        self.assertEqual(result["counts"]["windows"], 2)
        self.assertEqual(result["counts"]["annotated_tracks"], 1)
        self.assertEqual(result["coverage"]["arms"]["ssd_primary"]["unmatched_detections"], 1)
        self.assertEqual(result["coverage"]["vs_best_single"]["all"]["neither"], 1)
        self.assertEqual(result["gate"]["status"], "insufficient_common_support")
        arm = result["tracking"]["paired_against_best_single"]["all"]["arms"]["ssd_primary"]
        self.assertEqual(arm, {"displacement_mean_m": None, "future_center_mean_m": None,
                               "nonzero_velocity": 0, "confirmed": 0})

    def test_missing_third_frame_keeps_the_paired_target_with_zero_displacement(self):
        baseline = _object("old", 5.)
        fused = _object("new", 5., 2.3)
        result = _score(_window(older=False, ssd=[baseline], faster=[baseline], fused=[fused], tracks=[_track(5., 1.)]))
        self.assertEqual(result["best_single"], "faster_rcnn")
        counts = result["tracking"]["by_arm"]["ssd_primary"]["counts"]
        self.assertEqual((counts["matched_tracks"], counts["windows_without_older"], counts["confirmed_three_frame"]), (1, 1, 0))
        row = result["tracking"]["paired_against_best_single"]["moving"]
        self.assertEqual(row["pairs"], 1)
        self.assertEqual(row["arms"]["ssd_primary"]["displacement_mean_m"], 1.)
        self.assertAlmostEqual(row["arms"]["ssd_primary"]["future_center_mean_m"], .7)

    def test_gate_requires_the_frozen_gain_spread_support_and_center_bounds(self):
        gate = _score(*_study())["gate"]
        self.assertEqual(gate["status"], "coverage_signal")
        self.assertEqual((gate["additional_matched"], gate["improved_recordings"], gate["common_matches"]), (5, 4, 12))
        self.assertAlmostEqual(gate["common_mean_error_change_m"], .2)
        for study in (_study(extra=(1, 1, 1, 1)), _study(extra=(3, 2, 0, 0)), _study(offset=.21)):
            self.assertEqual(_score(*study)["gate"]["status"], "coverage_gain_not_established")
        self.assertEqual(_score(*_study(common=2))["gate"]["status"], "insufficient_common_support")

    def test_rejects_cross_profile_camera_and_time_conflicts(self):
        rows, labels = _arrays(_window())
        for key, bad in (("sample_token", "other"), ("previous_token", "other"), ("older_token", None),
                         ("recording", "other"), ("current_sha256", "c" * 64), ("history_s", .55),
                         ("older_history_s", .55), ("ref_time_us", 1_000_001), ("older_sha256", "d" * 64)):
            changed = deepcopy(rows)
            changed["ssd_primary"][0][key] = bad
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "observations or history timestamps differ"):
                summarize_fusion_holdout(changed, labels)
        del rows["ssdlite"][0]["ref_time_us"]
        with self.assertRaisesRegex(ValueError, "integer reference timestamp"):
            summarize_fusion_holdout(rows, labels)

    def test_dropped_windows_or_confirmation_without_older_frame_are_rejected(self):
        rows, labels = _arrays(_window())
        rows["ssd_primary"].clear()
        with self.assertRaisesRegex(ValueError, "same nonempty window set"):
            summarize_fusion_holdout(rows, labels)
        with self.assertRaisesRegex(ValueError, "requires a third observation"):
            _score(_window(older=False, fused=[_object("impossible", 5., speed=1.)], tracks=[_track(5., 1.)]))
