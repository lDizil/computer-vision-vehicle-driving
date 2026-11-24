from copy import deepcopy
from hashlib import sha256
from math import pi
from pathlib import Path
import pickle
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import numpy as np

from hybrid_drive.experiments import paired_slow_natural as natural
from hybrid_drive.experiments.box_motion_data import _pose
from hybrid_drive.experiments.io import write_json
from hybrid_drive.geometry.rigid import invert_pose


_TOKEN = "3" * 16
_DIGEST = "e" * 64
_RECORDING = "holdout"
_REF = 12_000_000


def _frames():
    return [
        {
            "token": _TOKEN if not step else f"{step + 9:016x}",
            "timestamp": _REF + 500_000 * step,
            "log_token": "physical-log",
            "log_name": "physical-log-with-anchor",
            "ego2global_translation": (100, 200, 0),
            "ego2global_rotation": (2 ** -.5, 0, 0, 2 ** -.5),
            "anns": {"gt_names": [], "track_tokens": [], "gt_boxes": []},
        }
        for step in range(6)
    ]


def _annotate(frame, center=(6, 0, .8), *, yaw=0., size_lwh=(4., 1.6, 1.6)):
    frame["anns"] = {
        "gt_names": ["vehicle"],
        "track_tokens": ["c" * 16],
        "gt_boxes": [(*center, *size_lwh, yaw)],
    }


def _item(candidate, *, x, y, slow, random):
    return {
        "candidate_id": candidate,
        "dt_s": .5,
        "poses": [[x, y, 0.] for _ in range(5)],
        "fast_overlap_s": None,
        "slow_margin": slow,
        "untrained_margin": random,
    }


def _prediction():
    return {
        "sample_token": _TOKEN,
        "snapshot_sha256": _DIGEST,
        "recording": _RECORDING,
        "ref_time_us": _REF,
        "candidates": [
            _item("first", x=5., y=0., slow=5., random=0.),
            _item("second", x=5., y=4., slow=1., random=5e-7),
        ],
    }


def _index():
    return {
        "schema": "hybrid-drive-box-motion-inputs-v1",
        "source": {"source_sha256": natural.SOURCES_SHA256},
        "anchors": [{"sample_token": _TOKEN, "recording": _RECORDING}],
        "frames": {_TOKEN: {"recording": _RECORDING, "ref_time_us": _REF}},
    }


def _heldout():
    return {"recording": _RECORDING, "anchor_indices": [0]}


class TestNaturalScore(TestCase):
    def test_actual_future_box_with_different_ego_yaw_and_translation_sets_pair_outcome(self):
        frames = _frames()
        # The same moving annotation is measured in the second ego frame. Its
        # future center is at (6, 0) in the anchor frame, not its local origin.
        frames[2]["ego2global_translation"] = (100, 206, 0)
        frames[2]["ego2global_rotation"] = (1, 0, 0, 0)
        _annotate(frames[2], center=(0, 0, .8))
        entry, = natural._bind_predictions([_prediction()], _index(), _heldout())
        row, = natural._score_recording(frames, _heldout(), (entry,))
        self.assertEqual(row["candidates"][0]["observed_contact_sample_times_s"], [1.])
        self.assertEqual(row["candidates"][1]["observed_contact_sample_times_s"], [])
        self.assertEqual(row["pairs"][0]["candidate_without_observed_contact"], "second")
        counts = natural._summarize([row])
        self.assertEqual(counts["comparable_pairs"], 1)
        self.assertEqual(counts["method_counts"]["fast_first"]["incorrect"], 1)
        self.assertEqual(counts["method_counts"]["slow"]["correct"], 1)
        self.assertEqual(counts["method_counts"]["untrained"]["tied"], 1)

    def test_box_orientation_transforms_with_frame_not_just_center(self):
        anchor, following = _frames()[:2]
        following["ego2global_rotation"] = (1, 0, 0, 0)
        following["ego2global_translation"] = (100, 205.55, 0)
        _annotate(following, center=(0, 0, .8))
        boxes = natural._boxes_in_anchor(following, invert_pose(_pose(anchor)))
        center, yaw, extent = boxes["c" * 16]
        np.testing.assert_allclose(center, (5.55, 0.), atol=1e-12)
        self.assertAlmostEqual(yaw, -pi / 2)
        self.assertEqual(extent, (2., .8))
        self.assertEqual(natural._overlapping_ids((0, 0, 0), boxes), [])
        self.assertEqual(natural._overlapping_ids((1.2, 0, 0), boxes), ["c" * 16])
        self.assertEqual(natural._overlapping_ids((0, 0, 0), {"wrong-yaw": (center, 0, extent)}), ["wrong-yaw"])

    def test_window_rejects_missing_future_broken_gaps_and_association_lags(self):
        frames = _frames()
        entry, = natural._bind_predictions([_prediction()], _index(), _heldout())

        changed_log = deepcopy(frames)
        changed_log[3]["log_token"] = "different-physical-log"
        gap = deepcopy(frames)
        gap[3]["timestamp"] += 300_000
        drift = deepcopy(frames)
        drift[3]["timestamp"] += 150_000
        missing_annotations = deepcopy(frames)
        del missing_annotations[3]["anns"]["gt_boxes"]
        for window, expected in (
            (frames[:-1], "missing_future_samples"),
            (changed_log, "recording_boundary"),
            (gap, "noncontinuous_future"),
            (drift, "future_time_mismatch"),
            (missing_annotations, "missing_annotations"),
        ):
            with self.subTest(expected=expected):
                row = natural._score_window(window, entry)
                self.assertEqual(row["window_status"], expected)
                result = natural._summarize([row])
                self.assertEqual(result["unscorable_window_pairs"], 1)
                self.assertEqual(result["comparable_pairs"], 0)
                self.assertIsNone(result["method_counts"]["slow"]["accuracy_including_ties"])

    def test_observed_shared_initial_contact_is_not_a_future_clearance(self):
        frames = _frames()
        _annotate(frames[0], center=(1.8, 0, .8))
        entry, = natural._bind_predictions([_prediction()], _index(), _heldout())
        row = natural._score_window(frames, entry)
        self.assertEqual(row["window_status"], "shared_observed_initial_contact")
        self.assertEqual(row["pairs"], [])

    def test_concordant_observations_are_counted_but_do_not_rank(self):
        r = _prediction()
        r["candidates"][1]["poses"] = [[6.0, 0.0, 0.0] for _ in range(5)]
        entry, = natural._bind_predictions([r], _index(), _heldout())
        frames = _frames()
        neither = natural._score_window(frames, entry)
        _annotate(frames[-1], center=(7., 0., .8))
        both = natural._score_window(frames, entry)
        summary = natural._summarize([neither, both])
        self.assertEqual(summary["complete_window_pairs"], 2)
        self.assertEqual(summary["both_contact_pairs"], 1)
        self.assertEqual(summary["neither_contact_observed_pairs"], 1)
        self.assertEqual(summary["comparable_pairs"], 0)

    def test_refuses_to_deserialize_on_mismatch_of_pinned_metadata_member(self):
        payload = pickle.dumps(_frames())
        heldout = {"member": "selected", "sha256": "0" * 64, "size_bytes": len(payload)}
        with (
            patch.object(natural, "_read_selected", return_value=iter([("selected", payload)])),
            patch.object(natural.pickle, "loads") as deserialize,
        ):
            with self.assertRaisesRegex(ValueError, "differs from its size or SHA256"):
                natural._load_verified_recording(Path("/never-read"), {}, heldout)
            deserialize.assert_not_called()

    def test_prediction_identity_or_candidate_grid_changes_fail(self):
        wrong_cohort = _prediction()
        wrong_cohort["recording"] = "train"
        wrong_time = _prediction()
        wrong_time["ref_time_us"] += 1
        wrong_grid = _prediction()
        wrong_grid["candidates"][0]["dt_s"] = .25
        for r, expected in (
            (wrong_cohort, "unique held-out observation"),
            (wrong_time, "unique held-out observation"),
            (wrong_grid, "sampled at 0.5 s"),
        ):
            with self.subTest(expected=expected), self.assertRaisesRegex(ValueError, expected):
                natural._bind_predictions([r], _index(), _heldout())

    def test_saved_predictions_sealed_before_opening_outcome_or_choice_of_scope(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            predictions, absent, output = root / "pred.json", root / "absent", root / "out"
            write_json(predictions, {"schema": "different payload"})
            with patch.object(natural, "_load_verified_recording") as load_truth:
                with self.assertRaisesRegex(ValueError, "differ from the seal"):
                    natural.score_natural(predictions, "0" * 64, absent, absent, absent, output)
                with self.assertRaisesRegex(ValueError, "without labels"):
                    natural.score_natural(
                        predictions, sha256(predictions.read_bytes()).hexdigest(),
                        absent, absent, absent, output,
                    )
                load_truth.assert_not_called()
            self.assertFalse(output.exists())

    def test_score_saves_one_report_bound_to_pinned_prediction_and_outcome(self):
        frames = _frames()
        _annotate(frames[-1])
        payload = pickle.dumps(frames)
        heldout = {
            **_heldout(),
            "member": "heldout.pkl",
            "size_bytes": len(payload),
            "sha256": sha256(payload).hexdigest(),
            "log_name": frames[0]["log_name"],
            "frame_count": len(frames),
        }
        source = {"metadata_archive": {"sha256": "4" * 64}, "logs": [{}, heldout]}
        manifest_digest = sha256(natural.canonical_json(source)).hexdigest()
        with TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            predictions, index, sources, output = (root / k for k in ("pred", "index", "sources", "out"))
            write_json(sources, source)
            fixture_index = _index()
            fixture_index["source"]["source_sha256"] = manifest_digest
            write_json(index, fixture_index)
            write_json(predictions, {
                "schema": natural._PREDICTION_SCHEMA,
                "revision": "2" * 40,
                "heldout_targets_opened": False,
                "rows": [_prediction()],
            })
            seal = sha256(predictions.read_bytes()).hexdigest()
            with (
                patch.object(natural, "SOURCES_SHA256", manifest_digest),
                patch.object(natural, "sys", SimpleNamespace(platform="linux")),
                patch.object(natural, "_read_selected", return_value=iter([("heldout.pkl", payload)])) as read,
            ):
                result = natural.score_natural(predictions, seal, root / "archive", sources, index, output)
            read.assert_called_once_with(
                root / "archive", source["metadata_archive"], {"heldout.pkl"}, natural._METADATA_LIMIT,
            )
            self.assertEqual(result["prediction_seal"], seal)
            self.assertEqual(result["metadata_member_sha256"], heldout["sha256"])
            self.assertEqual(result["summary"]["method_counts"]["slow"]["correct"], 1)
            self.assertEqual([p.name for p in output.iterdir()], ["natural-score.json"])
