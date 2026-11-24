from dataclasses import replace
from math import cos, sin
import unittest

import numpy as np

from hybrid_drive.contracts.planning import Candidate, CandidateSet
from hybrid_drive.contracts.trajectory import Pose2D, Trajectory
from hybrid_drive.experiments.paired_slow_controls import crossing_controls
from hybrid_drive.fast.forecast_proposals import ForecastObject
from hybrid_drive.fast.scene_payload import FastScene, RoadEvidence, SceneGrid


def _straight(candidate_id, positions, *, interval=.5):
    return Candidate(candidate_id, Trajectory([(x, 0., 0.) for x in positions], interval))


def _scene(candidates=None):
    if candidates is None:
        candidates = (
            _straight("ease_off", (3.4, 6.5, 9.15, 11.6, 13.75)),
            _straight("brake", (3.25, 5.95, 7.98, 9.9, 11.)),
            _straight("accelerate", (3.7, 7.6, 11.6, 16., 20.4)),
        )
    plan = CandidateSet("plan", "observed", 7.5, Pose2D(14, -6, .2), candidates)
    grid = SceneGrid(0, -1, 1, 2, 2)
    road = RoadEvidence(grid, np.full((2, 2), .4), np.full((2, 2), .7), np.eye(2, dtype=np.bool_))
    actor = ForecastObject("actual-fast-box", (12., -4., 1.), (1.9, 4., 1.8), .2, (1., 2.), .8)
    return FastScene("observed", "a" * 64, 7_500_000, (actor,), plan, road)


def _summary(controls):
    return tuple((item.candidate_ids, item.blocked, item.gaps_m, tuple(s.boxes[0] for s in item.worlds))
                 for item in controls)


def _rectangle(center, half_length, half_width, yaw):
    corners = np.array([[-half_length, -half_width], [half_length, -half_width],
                        [half_length, half_width], [-half_length, half_width]])
    rotation = np.array([[cos(yaw), sin(yaw)], [-sin(yaw), cos(yaw)]])
    return corners @ rotation + center


def _polygon_separation(first, second):
    gaps = []
    for polygon in (first, second):
        for delta in np.diff(polygon[:3], axis=0):
            perpendicular = np.array([-delta[1], delta[0]]) / np.linalg.norm(delta)
            a, b = first @ perpendicular, second @ perpendicular
            gaps.append(max(a.min() - b.max(), b.min() - a.max()))
    return max(gaps)


def _straight_candidate_gaps(candidate, actor):
    gaps = []
    for tick in range(51):
        t = tick / 20.
        poses = [0., *candidate.trajectory.poses[:, 0]]
        left = min(4, tick // 10)
        alpha = (tick - 10 * left) / 10.
        x = (1. - alpha) * poses[left] + alpha * poses[left + 1]
        ego = _rectangle(np.array([x + 1.25, 0.]), 2.25, .95, 0.)
        center = np.asarray(actor.center_xyz[:2]) + t * np.asarray(actor.velocity_xy)
        obstacle = _rectangle(center, actor.size_wlh[1] / 2., actor.size_wlh[0] / 2., actor.yaw)
        gaps.append(_polygon_separation(ego, obstacle))
    return np.array(gaps)


class CrossingControlsTest(unittest.TestCase):
    def test_keeps_real_snapshot_road_and_plan_while_changing_only_hypothetical_velocity(self):
        source = _scene()
        controls = crossing_controls(source, seed=163, count=16)

        self.assertEqual(len(controls), 16)
        self.assertEqual(source.boxes[0].object_id, "actual-fast-box")
        self.assertEqual({item.blocked[0] for item in controls}, {"accelerate", "brake"})
        self.assertEqual({int(np.sign(item.worlds[0].boxes[0].center_xyz[1])) for item in controls}, {-1, 1})
        for item in controls:
            a, b = item.worlds
            self.assertNotEqual(a.content_sha256, b.content_sha256)
            self.assertEqual(set(item.blocked), set(item.candidate_ids))
            for world in item.worlds:
                self.assertIs(world.plan, source.plan)
                self.assertIs(world.road, source.road)
                self.assertEqual(world.snapshot_id, source.snapshot_id)
                self.assertEqual(world.ref_time_us, source.ref_time_us)
                self.assertEqual(world.snapshot_sha256, source.snapshot_sha256)
                self.assertEqual(len(world.boxes), 1)
            actor, other_actor = a.boxes[0], b.boxes[0]
            self.assertNotEqual(actor.velocity_xy, other_actor.velocity_xy)
            self.assertEqual(replace(actor, velocity_xy=other_actor.velocity_xy), other_actor)

    def test_independent_corner_projection_reproduces_margins_and_no_initial_overlap(self):
        source = _scene()
        candidates = {c.candidate_id: c for c in source.plan.candidates}
        for case in crossing_controls(source, seed=611, count=5):
            for arm, world in enumerate(case.worlds):
                for slot, key in enumerate(case.candidate_ids):
                    gaps = _straight_candidate_gaps(candidates[key], world.boxes[0])
                    self.assertAlmostEqual(float(gaps.min()), case.gaps_m[arm][slot], places=11)
                    self.assertGreaterEqual(gaps[0], 1.)
                    if key == case.blocked[arm]:
                        self.assertLessEqual(gaps[10::10].min(), -.4)
                    else:
                        self.assertGreaterEqual(gaps.min(), .6)

    def test_deterministic_by_seed_and_candidate_preference_does_not_change_case_geometry(self):
        scene = _scene()
        reversed_plan = replace(scene.plan, candidates=tuple(reversed(scene.plan.candidates)))
        reversed_scene = replace(scene, plan=reversed_plan)
        sample = _summary(crossing_controls(scene, seed=77, count=8))

        self.assertEqual(sample, _summary(crossing_controls(scene, seed=77, count=8)))
        self.assertEqual(sample, _summary(crossing_controls(reversed_scene, seed=77, count=8)))
        self.assertNotEqual(sample, _summary(crossing_controls(scene, seed=78, count=8)))

    def test_same_sign_changes_are_deliberate_and_span_both_sides(self):
        controls = crossing_controls(_scene(), seed=611, count=16)

        hard = []
        for case in controls:
            a, b = (world.boxes[0] for world in case.worlds)
            if a.velocity_xy[0] * b.velocity_xy[0] > 0:
                hard.append((np.sign(a.velocity_xy[0]), np.sign(a.center_xyz[1])))
        self.assertEqual(len(controls), 16)
        self.assertGreaterEqual(len(hard), 8)
        self.assertEqual({sign for sign, _ in hard}, {-1, 1})
        self.assertEqual({side for _, side in hard}, {-1, 1})

    def test_uses_other_original_candidates_when_named_longitudinal_pair_is_absent(self):
        candidates = (
            _straight("middle", (3.4, 6.5, 9.15, 11.6, 13.75)),
            _straight("slow", (3.25, 5.95, 7.98, 9.9, 11.)),
            _straight("fast", (3.7, 7.6, 11.6, 16., 20.4)),
        )
        controls = crossing_controls(_scene(candidates), seed=163, count=3)

        self.assertEqual(len(controls), 3)
        self.assertEqual({c.candidate_ids for c in controls}, {("fast", "slow")})

    def test_empty_when_candidates_cannot_support_a_control(self):
        examples = (
            (_straight("only", (1., 2., 3., 4., 5.)),),
            (_straight("first", (1., 2., 3., 4., 5.)), _straight("second", (1., 2., 3., 4., 5.))),
            (_straight("first", (1., 2., 3., 4., 5.)), _straight("second", (1., 3., 4., 5., 7.))),
            (_straight("first", (1., 2., 3., 4., 5.), interval=.6),
             _straight("second", (1., 3., 6., 10., 11.), interval=.6)),
            (_straight("first", (1., 2., 3., 4., 5.)), _straight("second", (1., 3., 6., 10., 1e12))),
        )
        for candidates in examples:
            with self.subTest(candidates=candidates):
                self.assertEqual(crossing_controls(_scene(candidates), seed=163, count=3), ())

    def test_rejects_invalid_seed_or_unbounded_budget(self):
        scene = _scene()
        for seed, count in ((True, 1), (-1, 4), (2**64, 1), (4, True), (4, 0), (4, 33)):
            with self.subTest(seed=seed, count=count), self.assertRaises(ValueError):
                crossing_controls(scene, seed=seed, count=count)
        with self.assertRaises(ValueError):
            crossing_controls(object(), seed=0, count=1)
