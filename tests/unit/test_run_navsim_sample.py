import csv
from pathlib import Path
import tempfile
import unittest

from scripts.run_navsim_sample import validate_scores


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


if __name__ == "__main__":
    unittest.main()
