import json
from unittest import TestCase

import numpy as np

from hybrid_drive.experiments.temporal_vru_metrics import ARMS, summarize_motion
from hybrid_drive.experiments.visual_vru_geometry import GRID_COLS, GRID_ROWS


def _episode(recordings, tracks):
    anchors = [{"sample_token": f"current-{i}", "previous_token": f"past-{i}", "recording": recording}
               for i, recording in enumerate(recordings)]
    targets = [{"sample_token": row["sample_token"], "tracks": [{"centers_xy_m": track} for track in window],
                "target_dt_s": [.5, 1.]} for row, window in zip(anchors, tracks)]
    scores = {arm: np.zeros((len(anchors), 3, GRID_ROWS, GRID_COLS), np.float32) for arm in ARMS}
    return anchors, targets, scores, np.ones((len(anchors), GRID_ROWS, GRID_COLS), np.float32)


def _straight(y=-5.5):
    return [[3.5, y], [4.5, y], [5.5, y]]


def _working_gate():
    recording = [f"log-{i}" for i in range(4) for _ in range(10)]
    data = _episode(recording, [[_straight()] for _ in recording])
    scores = data[2]
    data[3][:] = 0.
    data[3][:, (1, 3), 1] = 1.
    for i in range(len(recording)):
        for arm in ARMS:
            correct = arm == "temporal" or i % 10 < (6 if arm == "single_frame" else 5)
            scores[arm][i, 2, 3 if correct else 1, 1] = 1.
    return data


class TemporalVruMetricsTests(TestCase):
    def test_covered_cells_only_ranked_and_future_timestamps_are_reported(self):
        data = _episode(["moving", "quiet"], [[_straight()], [[[8.5, 2.5]] * 3]])
        _, targets, scores, coverage = data
        targets[0]["target_dt_s"] = [.6, .9]
        coverage[0, 2, 1] = 0.
        scores["temporal"][0, 1:, 2, 1] = 50.  # Unknown false alarm cannot affect metrics.
        scores["temporal"][0, 2, 3, 1] = 2.
        scores["temporal"][1, :, 6, 9] = 3.
        report = summarize_motion(*data)
        h05, h10 = (report["horizons"][key] for key in ("0.5s", "1.0s"))
        self.assertEqual(h05["observed_target_dt_s"], {"minimum": .5, "maximum": .6})
        self.assertEqual(h10["observed_target_dt_s"], {"minimum": .9, "maximum": 1.})
        self.assertEqual(h05["counts"]["annotated_tracks"], 2)
        self.assertEqual(h05["counts"]["covered_tracks"], 1)
        self.assertIsNone(h05["cell_metrics"]["temporal"]["by_recording"]["moving"])
        self.assertEqual(h05["displacement"]["exclusions"]["uncovered"], 1)
        self.assertEqual(h10["cell_metrics"]["temporal"]["recording_macro_ap"], 1.)
        self.assertEqual(h10["localization_1m"]["temporal"]["hits"], 2)
        self.assertEqual(h10["displacement"]["arms"]["temporal"]["correct"], 1)
        self.assertEqual(h10["displacement"]["arms"]["single_frame"]["tied"], 1)
        self.assertEqual(report["temporal_gate"]["status"], "insufficient_support")
        self.assertTrue(json.dumps(report, allow_nan=False))

    def test_moving_tracks_exclude_same_cell_unknown_and_cross_track_separately(self):
        tracks = [
            [[3.5, -6.5], [3.6, -6.5], [3.75, -6.5]],       # quiet at the threshold
            [[4.5, -5.5], [4.9, -5.5], [5., -5.5]],        # intermediate
            [[6.05, -4.5], [6.45, -4.5], [6.9, -4.5]],    # moved, same cell
            _straight(-3.5),                               # uncovered destination
            [[8.5, -2.5], [9.5, -2.5], [10.5, -2.5]],    # two actors swapped cells
            [[10.5, -2.5], [9.5, -2.5], [8.5, -2.5]],
            [[11.25, -.5], [11.75, -.5], [12., -.5]],      # moved exactly .75 m, changed cell
        ]
        data = _episode(["one"], [tracks])
        data[3][0, 3, 3] = 0.
        motion = summarize_motion(*data)["horizons"]["1.0s"]["displacement"]
        self.assertEqual(motion["exclusions"], {"quiet": 1, "intermediate": 1, "same_cell": 1,
                                                "uncovered": 1, "cross_track": 2})
        self.assertEqual((motion["pairs"], motion["windows"], motion["recordings"]), (1, 1, 1))
        self.assertEqual(motion["arms"]["temporal"]["accuracy"], 0.)

    def test_equal_scores_are_one_ap_group_and_one_peak_never_credits_two_tracks(self):
        position = [[2.5, -6.5]] * 3
        data = _episode(["one"], [[position, position]])
        result = summarize_motion(*data)["horizons"]["now"]
        self.assertEqual(result["counts"]["annotated_tracks"], 2)
        self.assertEqual(result["counts"]["annotated_cells"], 1)
        self.assertAlmostEqual(result["cell_metrics"]["temporal"]["micro_ap"], 1. / (GRID_ROWS * GRID_COLS))
        self.assertEqual(result["localization_1m"]["temporal"]["hits"], 1)
        self.assertEqual(result["localization_1m"]["temporal"]["recall"], .5)

    def test_macro_accuracy_does_not_let_the_bigger_recording_dominate(self):
        data = _episode(["big"] * 9 + ["small"], [[_straight()] for _ in range(10)])
        scores = data[2]["temporal"]
        scores[:9, 2, 3, 1] = 1.
        scores[9, 2, 1, 1] = 1.
        stats = summarize_motion(*data)["horizons"]["1.0s"]["displacement"]["arms"]["temporal"]
        self.assertEqual((stats["pairs"], stats["correct"], stats["accuracy"]), (10, 9, .9))
        self.assertEqual(stats["recording_macro_accuracy"], .5)
        self.assertEqual(stats["by_recording"]["small"]["accuracy"], 0.)

    def test_numerically_tied_displacements_stay_in_denominator(self):
        data = _episode(["one", "one"], [[_straight()], [_straight()]])
        data[2]["temporal"][0, 2, 3, 1] = .5e-6
        data[2]["temporal"][1, 2, 3, 1] = 2.e-6
        arms = summarize_motion(*data)["horizons"]["1.0s"]["displacement"]["arms"]
        self.assertEqual((arms["temporal"]["correct"], arms["temporal"]["tied"], arms["temporal"]["accuracy"]), (1, 1, .5))

    def test_no_annotated_tracks_do_not_invent_perfect_accuracy(self):
        result = summarize_motion(*_episode(["empty"], [[]]))
        h10 = result["horizons"]["1.0s"]
        self.assertEqual(h10["counts"]["annotated_tracks"], 0)
        self.assertIsNone(h10["cell_metrics"]["temporal"]["micro_ap"])
        self.assertIsNone(h10["localization_1m"]["temporal"]["recall"])
        self.assertIsNone(h10["displacement"]["arms"]["temporal"]["accuracy"])
        self.assertEqual(result["temporal_gate"]["status"], "insufficient_support")

    def test_gate_requires_both_history_controls_and_actual_localization(self):
        data = _working_gate()
        accepted = summarize_motion(*data)["temporal_gate"]
        self.assertEqual(accepted["status"], "temporal_signal")
        self.assertEqual(accepted["recordings_beating_both_temporal_controls"], 4)
        self.assertAlmostEqual(accepted["gain_vs_single_frame"], .4)
        self.assertAlmostEqual(accepted["gain_vs_shuffled_past"], .5)

        data[2]["shuffled_past"][:] = data[2]["temporal"]
        self.assertEqual(summarize_motion(*data)["temporal_gate"]["status"], "temporal_gain_not_established")

        data = _working_gate()
        data[3][:] = 1.
        data[2]["temporal"][:, 2, 0, :] = 2.  # Direction still right, serious false-positive AP cost.
        gate = summarize_motion(*data)["temporal_gate"]
        self.assertEqual(gate["status"], "conflicting_metrics")
        self.assertLess(gate["ap_delta_vs_single_frame"], -.02)

    def test_gate_needs_several_recordings_even_with_many_valid_pairs(self):
        data = _working_gate()
        for row in data[0]:
            row["recording"] = "one"
        result = summarize_motion(*data)
        self.assertEqual(result["horizons"]["1.0s"]["displacement"]["pairs"], 40)
        self.assertEqual(result["temporal_gate"]["status"], "insufficient_support")

    def test_alignment_and_prediction_domains_fail_closed(self):
        data = _episode(["one"], [[_straight()]])
        anchors, targets, scores, coverage = data
        with self.assertRaisesRegex(ValueError, "Coverage"):
            summarize_motion(anchors, targets, scores, coverage.astype(bool))
        bad = coverage.copy()
        bad[0, 0, 0] = -1.
        with self.assertRaisesRegex(ValueError, "Coverage"):
            summarize_motion(anchors, targets, scores, bad)
        wrong_arm = {**scores, "temporal": scores["temporal"][:, 1:]}
        with self.assertRaisesRegex(ValueError, "finite float prediction grids"):
            summarize_motion(anchors, targets, wrong_arm, coverage)
        scores["temporal"][0, 0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite float prediction grids"):
            summarize_motion(*data)
        targets[0]["sample_token"] = "wrong"
        with self.assertRaisesRegex(ValueError, "identities"):
            summarize_motion(*data)
        targets[0]["sample_token"] = "current-0"
        targets[0]["target_dt_s"] = [.399, 1.]
        with self.assertRaisesRegex(ValueError, "sample horizons"):
            summarize_motion(*data)

    def test_half_open_cells_use_absolute_edges_even_near_floating_point_boundaries(self):
        point = [np.nextafter(27., -np.inf), np.nextafter(7., -np.inf)]
        other = [5.5, np.nextafter(0., -np.inf)]
        data = _episode(["one"], [[[point] * 3, [other] * 3]])
        data[2]["temporal"][0, :, -1, -1] = 1.
        data[2]["temporal"][0, :, 3, 6] = 1.
        stats = summarize_motion(*data)["horizons"]["1.0s"]
        self.assertEqual(stats["counts"]["covered_annotated_cells"], 2)
        self.assertEqual(stats["cell_metrics"]["temporal"]["micro_ap"], 1.)
        data[1][0]["tracks"][0]["centers_xy_m"] = [[26.5, 7.]] * 3
        with self.assertRaisesRegex(ValueError, "outside the declared grid"):
            summarize_motion(*data)
