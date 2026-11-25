import math
from dataclasses import FrozenInstanceError
from unittest import TestCase

import numpy as np

from hybrid_drive.experiments.object_vru_geometry import (
    GroundHit, GroundObject, align_ground_objects, fuse_ground_hits, project_detection,
)
from hybrid_drive.geometry.camera import CameraCalibration


def _camera(*, skew=0, near=.1, rear=False):
    pose = np.array([[0., -1., 0., 0.], [0., 0., -1., 1.], [1., 0., 0., 0.], [0., 0., 0., 1.]])
    if rear:
        pose = pose @ np.diag([-1., -1., 1., 1.])
    return CameraCalibration("front", 201, 101, [[100, skew, 100], [0, 100, 50], [0, 0, 1]], pose, near)


def _rgb():
    return np.full((101, 201, 3), (240, 10, 130), dtype=np.uint8)


def _hit(detection_id, camera, xy, *, kind="person", confidence=.9, sensitivity=.4):
    return GroundHit(detection_id, kind, camera, xy, (1., 2., 1.8), confidence,
                     sensitivity, (1.,) + (0.,) * 35)


class ObjectVruGeometryTest(TestCase):
    def test_calibrated_footpoint_consumes_full_intrinsic_and_reports_pixel_sensitivity(self):
        calibration = _camera(skew=14)
        hit = project_detection(calibration, _rgb(), [75.4, 26., 87.4, 60.], "person", .87, detection_id="front:3")
        assert hit is not None
        np.testing.assert_allclose(hit.xy, (10, 2), atol=1e-12)
        assert hit.camera_xyz == (0., 0., 1.)
        # At v=58.5, x=100/(58.5-50), y=-(81.4-100-14*.085)*x/100.
        x = 100 / 8.5
        y = (100 + 14 * .085 - 81.4) * x / 100
        self.assertAlmostEqual(hit.uncertainty_m, math.dist((10, 2), (x, y)))
        assert hit.detection_id == "front:3" and hit.camera_name == "front"

    def test_inner_crop_descriptor_is_channelwise_and_owns_an_immutable_snapshot(self):
        rgb = _rgb()
        rgb[15:25, 70:88] = 0
        hit = project_detection(_camera(), rgb, [74, 25, 86, 60], "person", .8, detection_id="box")
        assert hit is not None
        rgb[:] = 0
        assert len(hit.appearance) == 36
        assert [i for i, weight in enumerate(hit.appearance) if weight] == [11, 12, 30]
        self.assertAlmostEqual(sum(hit.appearance), 1.)
        with self.assertRaises(FrozenInstanceError):
            hit.xy = (4., 2.)

    def test_uncertain_footpoints_abstain_without_returning_a_zero_position(self):
        boxes = [
            [0, 25, 86, 60],    # right/left clip means the bottom-centre ray shifted
            [74, 0, 86, 60],    # vertically clipped object
            [74, 25, 86, 101],  # bbox clipped at the bottom
            [74, 25, 77, 60],   # insufficient width
            [74, 56, 86, 60],   # insufficient height
            [74, 24, 86, 48],   # ray points above the pinhole horizon
            [94, 26, 106, 54],  # central point is in bounds, perturbation is not stable
            [4, 26, 16, 56],    # ground intersection outside the lateral range
        ]
        for box in boxes:
            with self.subTest(box=box):
                assert project_detection(_camera(), _rgb(), box, "person", .9, detection_id="box") is None

    def test_projection_rejects_behind_ego_and_near_plane(self):
        box = [74, 26, 86, 60]
        assert project_detection(_camera(rear=True), _rgb(), box, "bicycle", .9, detection_id="box") is None
        assert project_detection(_camera(near=11), _rgb(), box, "bicycle", .9, detection_id="box") is None

    def test_invalid_detection_protocol_is_an_error_not_geometric_abstention(self):
        with self.assertRaisesRegex(ValueError, "rectified"):
            project_detection(_camera(), _rgb()[:, :-1], [74, 26, 86, 60], "person", .9, detection_id="box")
        with self.assertRaisesRegex(ValueError, "XYXY order"):
            project_detection(_camera(), _rgb(), [86, 26, 74, 60], "person", .9, detection_id="box")
        with self.assertRaisesRegex(ValueError, "confidence"):
            project_detection(_camera(), _rgb(), [74, 26, 86, 60], "person", True, detection_id="box")
        with self.assertRaisesRegex(ValueError, "person or bicycle"):
            project_detection(_camera(), _rgb(), [74, 26, 86, 60], "truck", .9, detection_id="box")

    def test_two_camera_fusion_preserves_individual_appearance_and_quality_weights(self):
        front = _hit("first", "front", (7., 1.), confidence=.9, sensitivity=.2)
        side = _hit("second", "side", (7.6, 1.), confidence=.6, sensitivity=.4)
        fused, = fuse_ground_hits([side, front])
        weight = .9 / .2**2
        other = .6 / .4**2
        assert fused.object_id == "d00" and fused.confidence == .9
        np.testing.assert_allclose(fused.xy, ((7 * weight + 7.6 * other) / (weight + other), 1.))
        assert fused.hits == (front, side)
        assert fuse_ground_hits([front, side]) == (fused,)

    def test_fusion_forbids_chain_bridges_same_camera_or_different_class(self):
        a = _hit("a", "front", (7., 0.), confidence=.95)
        b = _hit("b", "left", (7.75, 0.), confidence=.94)
        c = _hit("c", "right", (8.5, 0.), confidence=.93)
        duplicate_camera = _hit("different", "front", (7., 0.), confidence=.92)
        bicycle = _hit("bicycle", "back", (7.3, 0.), kind="bicycle", confidence=.91)
        instances = fuse_ground_hits([bicycle, c, b, duplicate_camera, a])
        assert tuple(obj.hits for obj in instances) == ((a, b), (c,), (duplicate_camera,), (bicycle,))
        assert [item.object_id for item in instances] == ["d00", "d01", "d02", "d03"]
        with self.assertRaisesRegex(ValueError, "unique"):
            fuse_ground_hits([a, a])

    def test_alignment_moves_ground_and_camera_origin_in_different_planes(self):
        frame, = fuse_ground_hits([_hit("box", "front", (4., 2.))])
        # (x, y, z) -> (-y, -z, x) + (4,3,0.5), a proper rotation plus translation.
        pose = [[0, -1, 0, 4], [0, 0, -1, 3], [1, 0, 0, .5], [0, 0, 0, 1]]
        aligned, = align_ground_objects([frame], pose)
        assert isinstance(aligned, GroundObject) and aligned.object_id == frame.object_id
        assert aligned.xy == aligned.hits[0].xy == (2., 3.)
        np.testing.assert_allclose(aligned.hits[0].camera_xyz, (2., 1.2, 1.5))
        assert aligned.hits[0].detection_id == frame.hits[0].detection_id
        assert aligned.hits[0].appearance == frame.hits[0].appearance
        assert aligned.hits[0].uncertainty_m == frame.hits[0].uncertainty_m
        assert frame.xy == (4., 2.) and frame.hits[0].camera_xyz == (1., 2., 1.8)
        with self.assertRaisesRegex(ValueError, "rotation"):
            align_ground_objects([frame], np.diag([2., 1., 1., 1.]))
