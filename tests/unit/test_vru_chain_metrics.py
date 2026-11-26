from copy import deepcopy
from unittest import TestCase

from hybrid_drive.experiments.vru_chain_metrics import ARMS, BASELINE, summarize_chain_replay


_ZERO = (0., 0.)


def _detection(key, xy):
    return {"object_id": key, "xy_m": xy, "kind": "person", "confidence": .7, "cameras": ["CAM_F0"]}


def _source(arm, key, xy, *, confirmed, has_older, speed):
    older_id = key + "-t2" if confirmed else None
    velocity = (0., speed) if confirmed else _ZERO
    status = "confirmed" if confirmed else "no_older_link" if has_older else "no_older_observation"
    return {**_detection(key, xy), "forecasts": {"three_frame_trend": velocity},
            "confirmation": {"status": status, "older_object_id": older_id, "trend_velocity_xy_mps": velocity},
            "trace": {"origin_arm": arm, "origin_object_id": key, "rescue": None,
                      "initial": {"status": "estimated", "previous_object_id": key + "-t1"},
                      "older": ({"status": "estimated" if confirmed else "no_reliable_history", "previous_object_id": older_id}
                                if has_older else None)}}


def _copy_evidence(kept, donor, rescue):
    return {**kept, **{key: donor[key] for key in ("confirmation", "forecasts")},
            "trace": {**donor["trace"], "rescue": rescue}}


def _window(token="sample", recording="recording-a", *, rescue=None, has_older=True, movement=1.,
            speed=1., dt=1., kept=(10., 0.), donor=(10.1, 0.)):
    older = token + "-t2" if has_older else None
    base = {"sample_token": token, "previous_token": token + "-t1", "older_token": older, "recording": recording,
            "current_sha256": "b" * 64, "ref_time_us": 1_000_000, "history_s": .5,
            "older_history_s": .5 if has_older else None}
    profiles = {}
    for arm, key, center in (("ssdlite", "ssd:n", kept), ("faster_rcnn", "rcnn:n", donor), (BASELINE, "ssd:n", kept)):
        confirmed = (arm, rescue) in (("ssdlite", "native"), ("faster_rcnn", "same_snapshot"))
        item = _source(arm, key, center, confirmed=confirmed, has_older=has_older, speed=speed)
        previous, earlier = (_detection(key + f"-t{j}", (center[0], -.5*j)) for j in (1, 2))
        profiles[arm] = {**base, "objects": [item], "history": {"previous": [previous], "older": [earlier] if has_older else None}}
    fused = profiles[BASELINE]
    native = _copy_evidence(fused["objects"][0], profiles["ssdlite"]["objects"][0], "native") if rescue == "native" else fused["objects"][0]
    bridge = (_copy_evidence(native, profiles["faster_rcnn"]["objects"][0], "same_snapshot")
              if rescue == "same_snapshot" else native)
    profiles.update(native_rescue={**fused, "objects": [native]}, bridge_rescue={**fused, "objects": [bridge]})
    label = {"sample_token": token, "target_dt_s": [.5, dt],
             "tracks": [{"centers_xy_m": [(10., 0.), (10., movement/2), (10., movement)]}]}
    truth = {"sample_token": token, "previous_token": token + "-t1", "older_token": older,
             "previous_status": "available", "older_status": "available" if has_older else "missing_frame",
             "tracks": [{"previous_xy_m": (10., -.5), "older_xy_m": (10., -1.) if has_older else None}]}
    return deepcopy((profiles, label, truth))


def _score(*windows):
    return summarize_chain_replay({name: [profiles[name] for profiles, _, _ in windows] for name in ARMS},
                                  [label for _, label, _ in windows], [truth for _, _, truth in windows])


def _supported(first=(), second=(), **changes):
    return [_window(f"sample-{i}", "a" if i < 6 else "b",
                    rescue="native" if i in first else "same_snapshot" if i in second else None, **changes) for i in range(12)]


class TestVruChainMetrics(TestCase):
    def test_snapshot_donor_history_is_audited_in_its_own_arm(self):
        result = _score(_window(rescue="same_snapshot", dt=1.05, movement=1.05, kept=(10., .2)))
        example = result["cases"][0]["arms"]
        bridge = example["bridge_rescue"]
        self.assertEqual(bridge["object_id"], "ssd:n")
        self.assertEqual(bridge["origin_object_id"], "rcnn:n")
        self.assertEqual(bridge["history"]["older"]["linked_object_id"], "rcnn:n-t2")
        self.assertEqual(bridge["history"]["older"]["linked_xy_m"], (10.1, -1.))
        self.assertEqual(bridge["cause"], "confirmation_confirmed")
        self.assertEqual(bridge["displacement_error_m"], 0.)
        self.assertAlmostEqual(bridge["future_center_error_m"], .2)
        self.assertEqual(example[BASELINE]["cause"], "older_no_reliable_history")
        moving = result["comparisons"]["bridge_rescue"]["moving"]
        self.assertEqual((moving["pairs"], moving["confirmed_gained"], moving["verified_gained"]), (1, 1, 1))
        self.assertEqual(result["comparisons"]["bridge_rescue"]["gate"]["status"], "insufficient_support")

    def test_matched_donor_different_target_is_not_verified_gain(self):
        window = _window("two-track", rescue="same_snapshot", donor=(10.7, 0.))
        _, labels, truth = window
        labels["tracks"].append({"centers_xy_m": [(10.8, 0.), (10.8, .5), (10.8, 1.)]})
        truth["tracks"].append({"previous_xy_m": (10.8, -.5), "older_xy_m": (10.8, -1.)})
        result = _score(*_supported(first=(0, 6)), window)
        rescue = result["cases"][-2]["arms"]["bridge_rescue"]
        self.assertEqual(rescue["cause"], "origin_other_target")
        self.assertTrue(rescue["wrong_link"])
        self.assertFalse(rescue["verified"])
        comparison = result["comparisons"]["bridge_rescue"]
        self.assertEqual(comparison["moving"]["confirmed_gained"], 3)
        self.assertEqual(comparison["moving"]["verified_gained"], 2)
        self.assertEqual(comparison["all"]["wrong_gained"], 1)
        self.assertFalse(comparison["gate"]["checks"]["no_added_wrong_links"])

    def test_later_link_to_another_matched_target_is_distinct_from_current_donor(self):
        profiles, label, truth = _window(rescue="same_snapshot")
        label["tracks"].append({"centers_xy_m": [(10.8, 0.), (10.8, .5), (10.8, 1.)]})
        truth["tracks"].append({"previous_xy_m": (10.8, -.5), "older_xy_m": (10.8, -1.)})
        profiles["faster_rcnn"]["history"]["older"][0]["xy_m"] = (10.7, -1.)
        row = _score((profiles, label, truth))["cases"][0]["arms"]["bridge_rescue"]
        self.assertEqual(row["origin_assignment"], "verified")
        self.assertEqual(row["cause"], "older_other_target")
        self.assertTrue(row["wrong_link"])

    def test_separates_absence_of_frame_annotation_center_and_unassigned_center(self):
        missing_frame = _window("no-packet", has_older=False)
        no_annotation = _window("no-label")
        no_annotation[2]["older_status"] = "missing_annotations"
        no_annotation[2]["tracks"][0]["older_xy_m"] = None
        no_center = _window("no-detection")
        no_center[0][BASELINE]["history"]["older"].clear()
        ambiguous = _window("ambiguous")
        ambiguous[0][BASELINE]["history"]["older"].append(_detection("rcnn:other", (10., -.9)))
        cases = _score(missing_frame, no_annotation, no_center, ambiguous)["cases"]
        self.assertEqual([case["arms"][BASELINE]["cause"] for case in cases],
                         ["older_missing_frame", "older_missing_annotations", "older_missing_center", "older_ambiguous_or_nonmutual_center"])
        self.assertTrue(all(not case["arms"][BASELINE]["wrong_link"] for case in cases))

    def test_unassigned_donor_does_not_count_as_conflicting_annotated_identity(self):
        w = _window(rescue="same_snapshot")
        obj = _source("faster_rcnn", "rcnn:other", (10., -.1), confirmed=False, has_older=True, speed=0.)
        w[0]["faster_rcnn"]["objects"].append(obj)
        w[0]["faster_rcnn"]["history"]["previous"].append(_detection("rcnn:other-t1", (30., 30.)))
        result = _score(w)
        row = result["cases"][0]["arms"]["bridge_rescue"]
        self.assertEqual(row["cause"], "origin_unverified_link")
        self.assertTrue(row["matched"])
        self.assertFalse(row["wrong_link"])
        self.assertFalse(row["verified"])

    def test_gate_requires_multiple_recordings_and_forbids_added_quiet_motion(self):
        passed = _score(*_supported(first=(0, 6)))["comparisons"]["native_rescue"]["gate"]
        self.assertEqual(passed["status"], "rescue_signal")
        one_recording = _score(*_supported(first=(0, 1)))["comparisons"]["native_rescue"]["gate"]
        self.assertEqual(one_recording["status"], "gain_not_established")
        self.assertFalse(one_recording["checks"]["verified_gain_spread"])
        quiet = _window("quiet-added", rescue="native", movement=.1)
        changed = _score(*_supported(first=(0, 6)), quiet)["comparisons"]["native_rescue"]
        self.assertEqual(changed["quiet"]["nonzero_gained"], 1)
        self.assertFalse(changed["gate"]["checks"]["no_added_quiet_motion"])

    def test_displacement_gain_cannot_hide_future_center_regression(self):
        rows = _supported(first=(0, 6), kept=(10., .9))
        result = _score(*rows)["comparisons"]["native_rescue"]
        self.assertLess(result["moving"]["displacement_change_m"], -.03)
        self.assertGreater(result["moving"]["future_center_change_m"], .05)
        self.assertEqual(result["gate"]["status"], "gain_not_established")

    def test_rejects_geometry_changes_cross_frames_and_substituted_source_links(self):
        geometry = _window(rescue="same_snapshot")
        geometry[0]["bridge_rescue"]["objects"][0]["xy_m"] = (10., .1)
        with self.assertRaisesRegex(ValueError, "keep exact current geometry"):
            _score(geometry)
        cross_frames = _window()
        cross_frames[2]["previous_token"] = "incorrect"
        with self.assertRaisesRegex(ValueError, "observation bindings differ"):
            _score(cross_frames)
        substituted = _window(rescue="same_snapshot")
        substituted[0]["bridge_rescue"]["objects"][0]["trace"]["initial"] = {"status": "estimated", "previous_object_id": "ssd:n-t1"}
        with self.assertRaisesRegex(ValueError, "preserve the exact source links"):
            _score(substituted)

    def test_no_annotations_preserves_detections_without_claiming_false_positives(self):
        profiles, labels, history = _window("empty", rescue="same_snapshot")
        labels["tracks"].clear()
        history["tracks"].clear()
        result = _score((profiles, labels, history))
        self.assertEqual(result["counts"]["windows"], 1)
        self.assertEqual(result["cases"], [])
        bridge = result["arms"]["bridge_rescue"]
        self.assertEqual(bridge["detections"], 1)
        self.assertEqual(bridge["rescue_detections"], {"same_snapshot": 1})
        self.assertIsNone(bridge["all"]["displacement_mean_m"])

    def test_rescue_cannot_replace_an_already_confirmed_chain(self):
        window = _window(rescue="same_snapshot")
        profiles, _, _ = window
        kept = _source(BASELINE, "ssd:n", (10., 0.), confirmed=True, has_older=True, speed=.8)
        for arm in (BASELINE, "native_rescue"):
            profiles[arm]["objects"] = [kept]
        with self.assertRaisesRegex(ValueError, "cannot replace motion"):
            _score(window)
