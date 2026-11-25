from hashlib import sha256
from unittest import TestCase

from hybrid_drive.experiments.vru_detector_coverage_metrics import summarize_detector_coverage


def _object(object_id, x, y=2., *, kind="person", confidence=.8):
    return {"object_id": object_id, "kind": kind, "xy_m": (x, y), "confidence": confidence}


def _track(x, movement=0.):
    return {"centers_xy_m": ((x, 2.), (x, 2. + movement / 2), (x, 2. + movement))}


def _window(token="sample-a", recording="recording-a", *, reference=(), candidate=(), tracks=()):
    binding = {"sample_token": token, "previous_token": token + "-previous", "recording": recording,
               "current_sha256": sha256(token.encode()).hexdigest()}
    return ({**binding, "objects": reference}, {**binding, "objects": candidate},
            {"sample_token": token, "target_dt_s": (.47, 1.04), "tracks": tracks})


def _score(*windows):
    return summarize_detector_coverage(*zip(*windows))


def _study(recordings="abcd", common=3, extra=4, offset=.10):
    windows = []
    for name in recordings:
        xs = [3. * i for i in range(1, common + extra + 1)]
        windows.append(_window(name, name,
                               reference=[_object(f"ref-{i}", x) for i, x in enumerate(xs[:common])],
                               candidate=[_object(f"cand-{i}", x, 2. + offset) for i, x in enumerate(xs)],
                               tracks=[_track(x) for x in xs]))
    return windows


class TestVruDetectorCoverage(TestCase):
    def test_reports_shared_truth_and_common_center_errors_not_independent_means(self):
        comparison = _window(
            reference=[_object("r-shared", 5, 2.2), _object("r-only", 11, 3.6), _object("r-unassigned", 31)],
            candidate=[_object("c-shared", 5, 2.4), _object("c-only", 18, 2.0, kind="bicycle")],
            tracks=[_track(5, .75), _track(11, .25), _track(18, .5), _track(24, 1.)],
        )
        summary = _score(comparison)
        self.assertEqual(summary["counts"], {"windows": 1, "recordings": 1, "annotated_tracks": 4,
                                             "moving_tracks": 2, "quiet_tracks": 1, "intermediate_tracks": 1})
        self.assertEqual(summary["arms"]["reference"], {
            "matched_tracks": 2, "matched_annotated_fraction": .5, "detections": 3,
            "unmatched_tracks": 2, "unmatched_detections": 1, "matched_windows": 1,
        })
        self.assertEqual(summary["arms"]["candidate"]["matched_tracks"], 2)
        whole = summary["slices"]["all"]
        self.assertEqual([whole[k] for k in ("both", "only_reference", "only_candidate", "neither")], [1, 1, 1, 1])
        paired = whole["paired_center_error"]
        self.assertEqual(paired["common_tracks"], 1)
        self.assertAlmostEqual(paired["reference_mean_m"], .2)
        self.assertAlmostEqual(paired["reference_median_m"], .2)
        self.assertAlmostEqual(paired["candidate_mean_m"], .4)
        self.assertAlmostEqual(paired["candidate_median_m"], .4)
        self.assertAlmostEqual(paired["candidate_minus_reference_mean_m"], .2)
        self.assertEqual(summary["slices"]["moving"]["arms"]["candidate"]["matched_tracks"], 1)
        self.assertEqual(summary["slices"]["quiet"]["only_reference"], 1)
        self.assertEqual(summary["slices"]["intermediate"]["only_candidate"], 1)
        pairs = summary["rows"][0]["pairs"]
        self.assertEqual([row["target_index"] for row in pairs], [0, 1, 2, 3])
        self.assertEqual(pairs[0]["reference"]["object_id"], "r-shared")
        self.assertIsNone(pairs[1]["candidate"])
        self.assertEqual(set(pairs[0]), {"target_index", "category", "reference", "candidate"})

    def test_frozen_matcher_abstains_on_ambiguity_regardless_of_detection_order(self):
        a = _object("first", 5, 1.9, confidence=.8)
        b = _object("second", 5, 2.1, confidence=.99)
        c = _object("candidate", 5, 2.)
        s1 = _score(_window(reference=[a, b], candidate=[c], tracks=[_track(5)]))
        s2 = _score(_window(reference=[b, a], candidate=[c], tracks=[_track(5)]))
        self.assertEqual(s1, s2)
        self.assertEqual(s1["slices"]["all"]["only_candidate"], 1)
        self.assertEqual(s1["arms"]["reference"]["matched_tracks"], 0)
        self.assertEqual(s1["arms"]["reference"]["unmatched_detections"], 2)

    def test_empty_detections_and_empty_labels_both_stay_in_the_denominator(self):
        lost = _window("empty-detections", "a", tracks=[_track(5, 1.)])
        unknown = _window("empty-labels", "b", candidate=[_object("c", 5)])
        result = _score(lost, unknown)
        self.assertEqual(result["counts"]["windows"], 2)
        self.assertEqual(result["counts"]["annotated_tracks"], 1)
        self.assertEqual(result["slices"]["moving"]["neither"], 1)
        self.assertEqual(result["arms"]["candidate"]["unmatched_detections"], 1)
        self.assertEqual(result["policy"]["unmatched_detections"], "not_precision_incompletely_labeled_scene")
        self.assertIsNone(result["by_recording"]["b"]["arms"]["candidate"]["matched_annotated_fraction"])
        self.assertIsNone(result["slices"]["all"]["paired_center_error"]["reference_mean_m"])
        self.assertEqual(result["gate"]["status"], "insufficient_common_support")

    def test_gate_is_positive_only_with_shared_support_spread_and_bounded_center_error(self):
        good = _score(*_study(offset=.20))["gate"]
        self.assertEqual(good["status"], "coverage_signal")
        self.assertEqual(good["additional_matched"], 16)
        self.assertEqual(good["improved_recordings"], 4)
        self.assertEqual(good["common_matches"], 12)
        self.assertAlmostEqual(good["common_mean_error_change_m"], .20)
        self.assertEqual(_score(*_study(offset=.21))["gate"]["status"], "coverage_gain_not_established")
        wide_but_low_gain = _score(*_study(extra=3))["gate"]
        self.assertEqual(wide_but_low_gain["additional_matched"], 12)
        self.assertEqual(wide_but_low_gain["status"], "coverage_gain_not_established")
        concentrated = _score(*_study("abc", common=4, extra=5))["gate"]
        self.assertEqual((concentrated["additional_matched"], concentrated["common_matches"]), (15, 12))
        self.assertEqual(concentrated["status"], "coverage_gain_not_established")
        incomparable = _score(*_study(common=2, extra=4))["gate"]
        self.assertEqual(incomparable["additional_matched"], 16)
        self.assertEqual(incomparable["status"], "insufficient_common_support")

    def test_window_bindings_must_be_exact_and_unique(self):
        reference, candidate, label = _window()
        for key, value in (("sample_token", "other"), ("previous_token", "other"),
                           ("recording", "other"), ("current_sha256", "0" * 64)):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "observations differ"):
                _score((reference, {**candidate, key: value}, label))
        with self.assertRaisesRegex(ValueError, "observations differ"):
            _score((reference, candidate, {**label, "sample_token": "other"}))
        with self.assertRaisesRegex(ValueError, "repeat a window"):
            _score((reference, candidate, label), (reference, candidate, label))
        with self.assertRaisesRegex(ValueError, "same nonempty window set"):
            summarize_detector_coverage([reference], [candidate], [])
        with self.assertRaisesRegex(ValueError, "canonical current camera SHA256"):
            _score(({**reference, "current_sha256": "invalid"}, candidate, label))

    def test_invalid_detection_and_target_records_are_not_silently_dropped(self):
        reference, candidate, label = _window()
        bad_objects = (
            ([_object("repeat", 5), _object("repeat", 8)], "Duplicate detection"),
            ([_object("bad-class", 5, kind="car")], "unexpected object class"),
            ([_object("bad-center", float("nan"))], "bounded finite"),
            ([_object("bad-confidence", 5, confidence=1.01)], "Detection confidence"),
        )
        for objects, pattern in bad_objects:
            with self.subTest(pattern=pattern), self.assertRaisesRegex(ValueError, pattern):
                _score((reference, {**candidate, "objects": objects}, label))
        with self.assertRaisesRegex(ValueError, "around 0.5 s and 1.0 s"):
            _score((reference, candidate, {**label, "target_dt_s": (.5, 1.2)}))
        with self.assertRaisesRegex(ValueError, "current and two future centers"):
            _score((reference, candidate, {**label, "tracks": [{"centers_xy_m": ((5., 2.),)}]}))
