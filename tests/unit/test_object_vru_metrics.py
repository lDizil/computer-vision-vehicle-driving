from unittest import TestCase

from hybrid_drive.experiments.object_vru_metrics import (
    fit_motion_gains,
    match_current_centers,
    summarize_object_motion,
)


_ZERO = {"radial_gain": 0., "tangential_gain": 0.}


def _motion(velocity=None, radial=(1., 0.), previous="prior"):
    if velocity is None:
        return {"status": "missing", "previous_object_id": None, "velocity_xy_mps": [0., 0.],
                "radial_unit_xy": None, "common_cameras": [], "past_xy_m": None}
    return {"status": "estimated", "previous_object_id": previous, "velocity_xy_mps": velocity,
            "radial_unit_xy": radial, "common_cameras": ["front"], "past_xy_m": [5., 2.]}


def _object(key="actor", xy=(5.2, 2.), appearance=None, geometry=None, radial=(1., 0.), previous="prior"):
    return {"object_id": key, "kind": "person", "xy_m": xy, "confidence": .8,
            "motions": {"geometry": _motion(geometry, radial, previous),
                        "appearance": _motion(appearance, radial, previous)}}


def _window(token="a", recording="a", objects=(), tracks=(), elapsed=(.5, 1.)):
    observed = {"sample_token": token, "recording": recording, "history_s": .5, "objects": objects}
    labels = {"sample_token": token, "target_dt_s": elapsed, "tracks": [{"centers_xy_m": track} for track in tracks]}
    return observed, labels


def _score(windows, gains=_ZERO):
    return summarize_object_motion(*zip(*windows), gains)


def _fit(windows):
    return fit_motion_gains(*zip(*windows))


def _study(recordings, moving_per_recording, quiet_velocity=(0., 0.)):
    windows = []
    for recording in recordings:
        for time in range(moving_per_recording):
            current = _object(xy=(5., 2.), appearance=(2., .5))
            windows.append(_window(f"{recording}-moving-{time}", recording, [current],
                                   [[(5., 2.), (5.5, 2.25), (6., 2.5)]]))
        for time in range(2):
            current = _object(xy=(5., 2.), appearance=quiet_velocity)
            windows.append(_window(f"{recording}-quiet-{time}", recording, [current],
                                   [[(5., 2.), (5., 2.), (5., 2.)]]))
    return windows


class TestObjectVruMetrics(TestCase):
    def test_mutual_match_refuses_ambiguous_or_distant_centers(self):
        detections = [(0., 0.), (10., 0.), (10.1, 0.)]
        truth = [(0.2, 0.), (10.15, 0.), (40., 0.)]
        self.assertEqual(match_current_centers(detections, truth), ((0, 0),))
        self.assertEqual(match_current_centers([(1., 0.)], [(0., 0.), (2., 0.)]), ())
        self.assertEqual(match_current_centers([(0., 0.)], [(2.01, 0.)]), ())
        self.assertEqual(match_current_centers([(0., 0.)], [(2., 0.)]), ((0, 0),))

    def test_same_current_pair_for_all_arms_and_observed_time(self):
        window = _window(objects=[_object(xy=(5.2, 2.), appearance=(1., 0.))],
                         tracks=[[(5., 2.), (5.45, 2.), (6.05, 2.)]], elapsed=(.45, 1.05))
        scored = _score([window], gains={"radial_gain": 1., "tangential_gain": 1.})
        first = scored["horizons"]["0.5s"]
        second = scored["horizons"]["1.0s"]
        self.assertEqual(first["all_matched"]["intermediate"]["pairs"], 1)
        self.assertEqual(second["all_matched"]["moving"]["pairs"], 1)
        self.assertEqual(first["observed_target_dt_s"], {"minimum": .45, "maximum": .45})
        metrics = second["all_matched"]["moving"]["arms"]
        self.assertAlmostEqual(metrics["stationary"]["displacement_mean_m"], 1.05)
        self.assertAlmostEqual(metrics["stationary"]["future_center_mean_m"], .85)
        self.assertEqual(metrics["geometry"], metrics["stationary"])
        self.assertEqual(metrics["calibrated"], metrics["appearance"])
        self.assertAlmostEqual(metrics["appearance"]["displacement_mean_m"], 0.)
        self.assertAlmostEqual(metrics["appearance"]["future_center_mean_m"], .2)
        self.assertAlmostEqual(scored["current_center_error_m"]["mean"], .2)

    def test_radial_direction_is_not_the_ego_x_axis(self):
        actor = _object(xy=(5., 2.), appearance=(2., 1.), radial=(.6, .8))
        window = _window(objects=[actor], tracks=[[(5., 2.), (5.45, 1.975), (5.9, 1.95)]])
        scored = _score([window], gains={"radial_gain": .25, "tangential_gain": .75})
        metrics = scored["horizons"]["1.0s"]["all_matched"]["moving"]["arms"]
        self.assertAlmostEqual(metrics["calibrated"]["displacement_mean_m"], 0.)
        self.assertAlmostEqual(metrics["calibrated"]["future_center_mean_m"], 0.)
        self.assertGreater(metrics["appearance"]["displacement_mean_m"], 1.)

    def test_abstention_and_unassigned_detections_keep_separate_denominators(self):
        window = _window(objects=[_object(), _object("other", xy=(25., 3.), appearance=(2., 0.))],
                         tracks=[[(5., 2.), (6., 2.), (7., 2.)], [(14., 0.), (14., 0.), (14., 0.)]])
        scored = _score([window], gains={"radial_gain": 1., "tangential_gain": 1.})
        self.assertEqual(scored["counts"]["matched_tracks"], 1)
        self.assertEqual(scored["counts"]["unmatched_tracks"], 1)
        self.assertEqual(scored["counts"]["unmatched_detections"], 1)
        self.assertEqual(scored["counts"]["accepted_appearance"], 0)
        self.assertEqual(scored["association_statuses"]["appearance"], {"estimated": 1, "missing": 1})
        horizon = scored["horizons"]["1.0s"]
        self.assertEqual(horizon["all_matched"]["moving"]["pairs"], 1)
        self.assertEqual(horizon["accepted_appearance"]["moving"]["pairs"], 0)
        self.assertIsNone(horizon["accepted_appearance"]["moving"]["arms"]["calibrated"]["future_center_median_m"])
        self.assertEqual(scored["by_motion_status"]["missing"]["1.0s"]["all"]["pairs"], 1)
        self.assertEqual(scored["by_motion_status"]["estimated"]["1.0s"]["all"]["pairs"], 0)
        self.assertEqual(horizon["all_matched"]["moving"]["arms"]["calibrated"],
                         horizon["all_matched"]["moving"]["arms"]["stationary"])

    def test_empty_scene_reports_no_numeric_error(self):
        scored = _score([_window(objects=[_object()])])
        self.assertEqual(scored["counts"]["annotated_tracks"], 0)
        self.assertIsNone(scored["counts"]["matched_annotated_fraction"])
        self.assertIsNone(scored["current_center_error_m"]["mean"])
        self.assertEqual(scored["gate"]["status"], "insufficient_support")

    def test_fixed_gain_grid_learns_separate_directional_gains(self):
        fitted = _fit(_study("ab", 4))
        self.assertEqual(fitted["status"], "calibrated_on_development")
        self.assertEqual((fitted["radial_gain"], fitted["tangential_gain"]), (.5, 1.))
        self.assertEqual(fitted["counts"]["moving_pairs"], 8)
        self.assertEqual(len(fitted["candidates"]), 25)
        self.assertAlmostEqual(fitted["selected"]["moving_displacement_mean_m"], 0.)

    def test_quiet_regression_blocks_positive_gains(self):
        fitted = _fit(_study("ab", 4, quiet_velocity=(2., 2.)))
        self.assertEqual(fitted["status"], "stationary_selected")
        self.assertEqual((fitted["radial_gain"], fitted["tangential_gain"]), (0., 0.))
        self.assertEqual(sum(candidate["eligible"] for candidate in fitted["candidates"]), 1)

    def test_no_fit_when_motion_is_only_supported_in_one_recording(self):
        windows = _study("a", 10)
        windows.extend(_study("b", 0))
        fitted = _fit(windows)
        self.assertEqual(fitted["status"], "insufficient_support")
        self.assertEqual((fitted["radial_gain"], fitted["tangential_gain"]), (0., 0.))
        self.assertEqual(fitted["counts"]["moving_pairs"], 10)
        self.assertEqual(fitted["counts"]["moving_recordings"], 1)

    def test_heldout_gate_uses_input_gains_and_requires_recording_support(self):
        study = _study("abcd", 5)
        good = _score(study, gains={"radial_gain": .5, "tangential_gain": 1.})
        self.assertEqual(good["gate"]["status"], "motion_signal")
        self.assertEqual(good["gate"]["moving_recordings"], 4)
        self.assertEqual(good["gate"]["moving_pairs"], 20)
        self.assertGreater(good["gate"]["future_center_gain_m"], 0.)
        self.assertEqual(_score(study)["gate"]["status"], "motion_gain_not_established")
        short = _score(_study("abc", 7), gains={"radial_gain": .5, "tangential_gain": 1.})
        self.assertEqual(short["gate"]["status"], "insufficient_support")

    def test_wrong_wire_fails_instead_of_fabricating_motion(self):
        active = _object(appearance=(1., 0.))
        active["motions"]["appearance"]["status"] = "missing"
        with self.assertRaisesRegex(ValueError, "abstaining branch"):
            _score([_window(objects=[active])])
        active = _object(appearance=(1., 0.), radial=(4., 0.))
        with self.assertRaisesRegex(ValueError, "unit radial"):
            _score([_window(objects=[active])])
        with self.assertRaisesRegex(ValueError, "reused a previous detection"):
            _score([_window(objects=[_object(appearance=(1., 0.)), _object("other", appearance=(0., 0.))])])
        with self.assertRaisesRegex(ValueError, "window identities"):
            _score([_window(), _window()])
        with self.assertRaisesRegex(ValueError, "fixed development grid"):
            _score([_window()], gains={"radial_gain": .41, "tangential_gain": .25})
