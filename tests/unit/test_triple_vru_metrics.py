from unittest import TestCase

from hybrid_drive.experiments.object_vru_metrics import summarize_object_motion
from hybrid_drive.experiments.triple_vru_metrics import summarize_three_frame


_ZERO = (0., 0.)
_GAINS = {"radial_gain": 0., "tangential_gain": 1.}


def _object(key="actor", xy=(5., 2.), *, two=(0., 1.), trend=None, reason="below_motion_floor"):
    absent = {"status": "missing", "previous_object_id": None, "velocity_xy_mps": _ZERO,
              "radial_unit_xy": None, "common_cameras": [], "past_xy_m": None}
    first = absent if two is None else {
        "status": "estimated", "previous_object_id": key + "-past", "velocity_xy_mps": (2., two[1]),
        "radial_unit_xy": (1., 0.), "common_cameras": ["front", "side"], "past_xy_m": (4., 2.),
    }
    return {"object_id": key, "kind": "person", "xy_m": xy, "confidence": .8,
            "motions": {"geometry": absent, "appearance": first},
            "confirmation": {"status": "confirmed" if trend is not None else reason,
                             "older_object_id": key + "-old" if trend is not None else None,
                             "common_cameras": ["front"] if trend is not None else [],
                             "trend_velocity_xy_mps": trend if trend is not None else _ZERO},
            "forecasts": {"stationary": _ZERO, "two_frame": two if two is not None else _ZERO,
                          "confirmed_two_frame": two if trend is not None else _ZERO,
                          "three_frame_trend": trend if trend is not None else _ZERO}}


def _window(token="a", recording="a", *, objects=(), tracks=(), elapsed=(.5, 1.), older=True):
    row = {"sample_token": token, "previous_token": token + "-past", "recording": recording,
           "older_token": token + "-old" if older else None, "history_s": .5,
           "older_history_s": .5 if older else None, "objects": objects}
    label = {"sample_token": token, "target_dt_s": elapsed,
             "tracks": [{"centers_xy_m": track} for track in tracks]}
    return row, label


def _score(*windows):
    return summarize_three_frame(*zip(*windows))


def _study(recordings, n_moving, *, current_y=2.):
    windows = []
    for recording in recordings:
        for i in range(n_moving):
            current = _object(xy=(5., current_y), trend=(0., 1.))
            windows.append(_window(f"{recording}-move-{i}", recording, objects=[current],
                                   tracks=[[(5., 2.), (5., 2.5), (5., 3.)]]))
        quiet = _object(two=(0., .4))
        windows.append(_window(recording + "-quiet", recording, objects=[quiet],
                               tracks=[[(5., 2.), (5., 2.), (5., 2.)]]))
    return windows


class TestTripleVruMetrics(TestCase):
    def test_uses_frozen_current_matches_calibrated_baseline_and_observed_time(self):
        active = _object(xy=(5., 2.2), two=(0., 1.), trend=(0., .8))
        window = _window(objects=[active], tracks=[[(5., 2.), (5., 2.45), (5., 3.05)]], elapsed=(.45, 1.05))
        score = _score(window)
        legacy = summarize_object_motion([window[0]], [window[1]], _GAINS)
        moved = score["horizons"]["1.0s"]["all_matched"]["moving"]
        self.assertEqual(score["counts"]["matched_tracks"], legacy["counts"]["matched_tracks"])
        self.assertEqual(moved["pairs"], 1)
        self.assertEqual(score["horizons"]["0.5s"]["observed_target_dt_s"], {"minimum": .45, "maximum": .45})
        self.assertEqual(score["horizons"]["0.5s"]["all_matched"]["intermediate"]["pairs"], 1)
        for name, value in legacy["horizons"]["1.0s"]["all_matched"]["moving"]["arms"]["calibrated"].items():
            self.assertEqual(moved["arms"]["two_frame"][name], value)
        self.assertEqual(moved["arms"]["confirmed_two_frame"], moved["arms"]["two_frame"])
        self.assertAlmostEqual(moved["arms"]["three_frame_trend"]["displacement_mean_m"], .21)
        self.assertAlmostEqual(moved["arms"]["three_frame_trend"]["future_center_mean_m"], .01)

    def test_missing_third_or_initial_link_does_not_drop_current_matches(self):
        linked = _window("linked", "left", objects=[_object(reason="no_older_observation")], older=False,
                         tracks=[[(5., 2.), (5., 2.5), (5., 3.)]])
        unlinked = _window("no-link", "right", objects=[_object(two=None, reason="no_initial_link")],
                           tracks=[[(5., 2.), (5., 2.), (5., 2.)], [(20., 2.)] * 3])
        score = _score(linked, unlinked)
        counts = score["counts"]
        self.assertEqual((counts["windows_with_older"], counts["windows_without_older"]), (1, 1))
        self.assertEqual((counts["matched_tracks"], counts["unmatched_tracks"], counts["accepted_two_frame"]), (2, 1, 1))
        self.assertEqual(counts["confirmed_three_frame"], 0)
        self.assertEqual(score["matched_confirmation_statuses"], {"no_initial_link": 1, "no_older_observation": 1})
        self.assertEqual(score["by_recording"]["left"]["counts"]["matched_tracks"], 1)
        moving = score["horizons"]["1.0s"]["all_matched"]["moving"]
        self.assertEqual(moving["arms"]["confirmed_two_frame"], moving["arms"]["stationary"])
        self.assertEqual(moving["arms"]["three_frame_trend"], moving["arms"]["stationary"])
        self.assertAlmostEqual(moving["arms"]["two_frame"]["displacement_mean_m"], 0.)
        self.assertIsNone(score["horizons"]["1.0s"]["confirmed"]["moving"]["arms"]["three_frame_trend"]["displacement_mean_m"])

    def test_only_mutual_matches_count_and_unmatched_reasons_are_separate(self):
        detections = [_object("near-a", xy=(5., 2.)), _object("near-b", xy=(5.1, 2.)),
                      _object("far", xy=(30., 2.), trend=(0., 1.))]
        score = _score(_window(objects=detections, tracks=[[(5.03, 2.)] * 3]))
        self.assertEqual(score["counts"]["detections"], 3)
        self.assertEqual(score["counts"]["detections_with_confirmed_three_frame"], 1)
        self.assertEqual(score["counts"]["matched_tracks"], 0)
        self.assertEqual(score["matched_confirmation_statuses"], {})
        self.assertEqual(score["confirmation_statuses"], {"below_motion_floor": 2, "confirmed": 1})
        self.assertEqual(score["gate"]["status"], "insufficient_support")
        self.assertIsNone(score["gate"]["future_center_gain_m"])

    def test_confirmed_chain_needs_both_links_and_shared_cameras(self):
        with self.assertRaisesRegex(ValueError, "third observation"):
            _score(_window(older=False, objects=[_object(trend=(0., 1.))]))
        with self.assertRaisesRegex(ValueError, "accepted two-frame link"):
            _score(_window(objects=[_object(two=None, trend=(0., 1.))]))
        row = _object(trend=(0., 1.))
        row["confirmation"]["common_cameras"] = ["other"]
        with self.assertRaisesRegex(ValueError, "shared with"):
            _score(_window(objects=[row]))

    def test_never_reuses_older_detection_within_one_window(self):
        first = _object("near", trend=(0., 1.))
        second = _object("far", xy=(12., 2.), trend=(0., 1.))
        second["confirmation"]["older_object_id"] = first["confirmation"]["older_object_id"]
        with self.assertRaisesRegex(ValueError, "reuse one older detection"):
            _score(_window(objects=[first, second]))

    def test_validation_rejects_time_and_identity_mismatch(self):
        row, label = _window()
        row["older_token"] = row["previous_token"]
        with self.assertRaisesRegex(ValueError, "third observation must be different"):
            _score((row, label))
        row, label = _window()
        row["older_history_s"] = .8
        with self.assertRaisesRegex(ValueError, "older history interval"):
            _score((row, label))
        row, label = _window(older=False)
        row["older_history_s"] = .5
        with self.assertRaisesRegex(ValueError, "absent third observation"):
            _score((row, label))
        with self.assertRaisesRegex(ValueError, "nonempty window set"):
            summarize_three_frame([row], [])
        row, label = _window()
        label["sample_token"] = "wrong"
        with self.assertRaisesRegex(ValueError, "window identities"):
            _score((row, label))

    def test_rejects_abstention_velocities_and_incoherent_forecasts(self):
        obj = _object()
        obj["confirmation"]["trend_velocity_xy_mps"] = (0., .1)
        with self.assertRaisesRegex(ValueError, "abstaining confirmation"):
            _score(_window(objects=[obj]))
        obj = _object(trend=(0., .8))
        obj["forecasts"]["confirmed_two_frame"] = (0., 1.5)
        with self.assertRaisesRegex(ValueError, "exact retained link"):
            _score(_window(objects=[obj]))
        obj = _object()
        obj["forecasts"]["three_frame_trend"] = (0., float("nan"))
        with self.assertRaisesRegex(ValueError, "bounded finite"):
            _score(_window(objects=[obj]))
        obj = _object(two=None, reason="no_initial_link")
        obj["forecasts"]["two_frame"] = (0., .1)
        with self.assertRaisesRegex(ValueError, "unavailable two-frame"):
            _score(_window(objects=[obj]))

    def test_synthetic_study_gate_preserves_correct_motion_and_rejects_false_quiet_speed(self):
        score = _score(*_study("abcd", 5))
        gate = score["gate"]
        self.assertEqual(gate["status"], "confirmation_signal")
        self.assertEqual(gate["confirmed_moving_pairs"], 20)
        self.assertEqual(gate["confirmed_moving_recordings"], 4)
        self.assertEqual(gate["quiet_nonzero_velocity_reduction"], 4)
        self.assertAlmostEqual(gate["quiet_displacement_gain_vs_two_frame_m"], .4)
        self.assertAlmostEqual(gate["moving_displacement_gain_m"], 1.)
        self.assertEqual(score["horizons"]["1.0s"]["all_matched"]["all"]["pairs"], 24)
        self.assertEqual(score["horizons"]["1.0s"]["confirmed"]["all"]["pairs"], 20)
        self.assertEqual(_score(*_study("abc", 7))["gate"]["status"], "insufficient_support")

    def test_good_displacement_cannot_hide_a_future_center_regression(self):
        gate = _score(*_study("abcd", 5, current_y=3.2))["gate"]
        self.assertAlmostEqual(gate["moving_displacement_gain_m"], 1.)
        self.assertEqual(gate["quiet_nonzero_velocity_reduction"], 4)
        self.assertLess(gate["future_center_gain_m"], 0.)
        self.assertEqual(gate["status"], "motion_gain_not_established")
