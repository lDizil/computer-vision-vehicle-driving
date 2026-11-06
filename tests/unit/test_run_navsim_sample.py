import csv
import json
from pathlib import Path
import tempfile
import unittest

from scripts.run_navsim_sample import (
    build_hybrid_agent, compute_hybrid_prediction, render_trajectory, validate_scores, write_score_rows,
)


class ValidateScoresTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "scores.csv"

    def write(self, rows):
        with self.path.open("w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=["token", "valid", "score", "ego_progress"])
            writer.writeheader()
            writer.writerows(rows)

    def rows(self, score="0.7"):
        return [
            {"token": "real-token", "valid": "True", "score": score, "ego_progress": "0.8"},
            {"token": "average", "valid": "True", "score": score, "ego_progress": "0.8"},
        ]

    def test_accepts_valid_single_scene_and_average(self):
        self.write(self.rows())
        self.assertEqual(validate_scores(self.path, "real-token"), {"score": 0.7, "ego_progress": 0.8})

    def test_rejects_missing_or_duplicate_scenarios(self):
        for rows in ([], self.rows()[:1], self.rows()[1:], self.rows() + self.rows()[:1]):
            with self.subTest(rows=rows):
                self.write(rows)
                with self.assertRaises(ValueError):
                    validate_scores(self.path, "real-token")

    def test_rejects_unexpected_token(self):
        self.write(self.rows())
        with self.assertRaises(ValueError):
            validate_scores(self.path, "other-token")

    def test_rejects_invalid_scenario_even_when_process_succeeded(self):
        rows = self.rows()
        rows[0]["valid"] = "False"
        self.write(rows)
        with self.assertRaises(ValueError):
            validate_scores(self.path, "real-token")

    def test_rejects_nonfinite_and_out_of_range_scores(self):
        for value in ("nan", "inf", "-0.1", "1.1", ""):
            with self.subTest(value=value):
                self.write(self.rows(value))
                with self.assertRaises(ValueError):
                    validate_scores(self.path, "real-token")

    def test_rejects_inconsistent_average(self):
        rows = self.rows()
        rows[1]["score"] = "0.6"
        self.write(rows)
        with self.assertRaises(ValueError):
            validate_scores(self.path, "real-token")

    def test_rejects_other_nonfinite_metrics(self):
        for index in (0, 1):
            with self.subTest(index=index):
                rows = self.rows()
                rows[index]["ego_progress"] = "nan"
                self.write(rows)
                with self.assertRaises(ValueError):
                    validate_scores(self.path, "real-token")

    def test_rejects_inconsistent_average_component(self):
        rows = self.rows()
        rows[1]["ego_progress"] = "0.2"
        self.write(rows)
        with self.assertRaises(ValueError):
            validate_scores(self.path, "real-token")


class HybridEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.csv_path = self.root / "scores.csv"
        self.agent = build_hybrid_agent()
        self.agent.initialize()

    def agent_input(self, acceleration=0):
        import numpy as np
        from navsim.common.dataclasses import AgentInput, EgoStatus

        return AgentInput(ego_statuses=[EgoStatus(
            ego_pose=np.zeros(3), ego_velocity=np.array([5.0, 0.0]),
            ego_acceleration=np.array([acceleration, 0.0]), driving_command=np.array([1, 0, 0]),
        )], cameras=[], lidars=[])

    def evidence(self):
        return json.loads((self.root / "decision.json").read_text())

    def test_records_actual_native_selection_with_runner_scene_token(self):
        prediction = compute_hybrid_prediction(
            self.agent, self.agent_input(), "selected-real-token", self.root, self.csv_path,
        )
        evidence = self.evidence()
        self.assertEqual(evidence["scene_token"], "selected-real-token")
        self.assertFalse(evidence["failed"])
        self.assertEqual(evidence["selection_status"], "candidate")
        self.assertEqual(evidence["candidate_id"], "fast")
        self.assertEqual(evidence["slow_status"], "UNKNOWN")
        self.assertEqual(evidence["slow_disposition"], "not_requested")
        self.assertEqual(evidence["native_trajectory"]["poses"], prediction.poses.tolist())
        self.assertEqual(evidence["native_trajectory"]["interval_seconds"], 0.5)
        self.assertTrue(evidence["candidate_kinematics"][0]["within_limits"])
        self.assertFalse(evidence["policy"]["safety_calibrated"])

    def test_successful_braking_records_rejection_and_fallback_kinematics(self):
        compute_hybrid_prediction(
            self.agent, self.agent_input(acceleration=4), "braking-token", self.root, self.csv_path,
        )
        evidence = self.evidence()
        self.assertEqual(evidence["selection_status"], "fallback")
        self.assertFalse(evidence["failed"])
        self.assertIsNone(evidence["candidate_id"])
        self.assertEqual(evidence["fallback_reason"], "no_eligible_candidate")
        self.assertTrue(evidence["fallback"]["succeeded"])
        self.assertTrue(evidence["fallback"]["kinematics"]["within_limits"])
        self.assertIn("jerk", evidence["candidate_kinematics"][0]["violations"])

    def test_finite_native_placeholder_remains_failed_and_invalid_in_csv(self):
        with self.assertRaisesRegex(ValueError, "native placeholder"):
            compute_hybrid_prediction(
                self.agent, self.agent_input(acceleration=20), "failed-token", self.root, self.csv_path,
            )
        evidence = self.evidence()
        self.assertTrue(evidence["failed"])
        self.assertEqual(evidence["selection_status"], "failure")
        self.assertEqual(evidence["fallback_reason"], "initial_acceleration")
        self.assertIn("acceleration", evidence["candidate_kinematics"][0]["violations"])
        self.assertTrue(evidence["native_trajectory"]["poses"])
        with self.assertRaisesRegex(ValueError, "invalid scenario"):
            validate_scores(self.csv_path, "failed-token")
        with self.csv_path.open(newline="") as source:
            rows = list(csv.DictReader(source))
        self.assertEqual([row["valid"] for row in rows], ["False", "False"])
        self.assertEqual([row["score"] for row in rows], ["", ""])

    def test_input_error_does_not_reuse_previous_successful_decision(self):
        compute_hybrid_prediction(self.agent, self.agent_input(), "first-token", self.root, self.csv_path)
        invalid = self.agent_input()
        invalid.ego_statuses.clear()
        with self.assertRaises(ValueError):
            compute_hybrid_prediction(self.agent, invalid, "second-token", self.root, self.csv_path)
        evidence = self.evidence()
        self.assertTrue(evidence["failed"])
        self.assertEqual(evidence["scene_token"], "second-token")
        self.assertIsNone(evidence["selection_status"])
        self.assertIsNone(evidence["native_trajectory"])
        self.assertEqual(evidence["candidate_kinematics"], [])

    def test_renderer_uses_supplied_native_prediction_without_recomputing_fast(self):
        from types import SimpleNamespace
        from unittest.mock import Mock

        prediction = compute_hybrid_prediction(
            self.agent, self.agent_input(), "selected-token", self.root, self.csv_path,
        )
        loader = Mock()
        loader.get_agent_input_from_token.side_effect = AssertionError("Prediction must not be recomputed")
        loader.get_scene_from_token.return_value.get_future_trajectory.return_value = SimpleNamespace(poses=prediction.poses)
        evidence = render_trajectory(loader, "selected-token", self.root, 0.7, prediction, "HybridDriveAgent")
        self.assertEqual(evidence["poses"], prediction.poses.tolist())
        self.assertTrue((self.root / "trajectory.png").is_file())

    def test_direct_score_csv_has_valid_single_scene_and_matching_average(self):
        metrics = {"score": 0.75, "ego_progress": 0.5}
        write_score_rows(self.csv_path, "selected-token", metrics)
        self.assertEqual(validate_scores(self.csv_path, "selected-token"), metrics)


if __name__ == "__main__":
    unittest.main()
