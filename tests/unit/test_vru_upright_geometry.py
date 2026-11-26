from itertools import product
from math import cos, dist, sin
from types import SimpleNamespace

import numpy as np
import pytest

from hybrid_drive.contracts.sensors import SurroundView, ViewDirection
from hybrid_drive.experiments.object_vru_geometry import project_detection
from hybrid_drive.experiments.vru_upright_geometry import typical_heights, upright_hit
from hybrid_drive.geometry.camera import CameraCalibration


def _view(*, pitch=0., roll=0., origin=(1.8, -.4, 1.6), focal=600., near=.1):
    straight = np.array([[0., -1., 0.], [0., 0., -1.], [1., 0., 0.]])
    around_x = np.array([[1., 0., 0.], [0., cos(pitch), -sin(pitch)], [0., sin(pitch), cos(pitch)]])
    around_z = np.array([[cos(roll), -sin(roll), 0.], [sin(roll), cos(roll), 0.], [0., 0., 1.]])
    rotation = around_z @ around_x @ straight
    ego_to_camera = np.eye(4)
    ego_to_camera[:3, :3] = rotation
    ego_to_camera[:3, 3] = -rotation @ origin
    calibration = CameraCalibration("front", 768, 432, [[focal, 0, 384], [0, focal, 216], [0, 0, 1]], ego_to_camera, near)
    rgb = np.full((432, 768, 3), (240, 10, 130), dtype=np.uint8)
    return SurroundView(ViewDirection.FRONT, calibration, rgb)


def _project(view, position):
    calibrated = view.calibration
    ray = calibrated.intrinsic @ (calibrated.ego_to_camera @ (*position, 1.))[:3]
    return ray[:2] / ray[2], ray[2]


def _box(view, x, y, z, height, *, kind="person"):
    top, top_depth = _project(view, (x, y, z + height))
    bottom, bottom_depth = _project(view, (x, y, z))
    u = float((top[0] + bottom[0]) / 2.)
    return {"kind": kind, "confidence": .88, "xyxy": (u - 8., float(top[1]), u + 8., float(bottom[1]))}, top_depth, bottom_depth


def test_upright_recovers_an_elevated_object_with_pitch_and_unequal_optical_depths():
    view = _view(pitch=.16)
    box, top, bottom = _box(view, 15., -.4, .55, 1.75)
    assert top != pytest.approx(bottom)

    hit = upright_hit(view, SimpleNamespace(**box), 1.75, detection_id="v0-det0")

    assert hit is not None
    assert hit.xy == pytest.approx((15., -.4))
    assert hit.camera_xyz == pytest.approx((1.8, -.4, 1.6))
    assert hit.detection_id == "v0-det0" and hit.kind == "person" and hit.confidence == .88
    assert hit.camera_name == view.calibration.name
    baseline = project_detection(view.calibration, view.rgb, box["xyxy"], "person", .88, detection_id="baseline")
    assert baseline is not None
    assert abs(hit.xy[0] - baseline.xy[0]) > 4.


def test_level_camera_range_and_uncertainty_agree_with_pixel_height_formula():
    focal, height = 600., 1.7
    origin = (1.8, -.4, 1.6)
    view = _view(origin=origin, focal=focal)
    point = (12., 1.2)
    box, _, _ = _box(view, *point, 0., height)
    hit = upright_hit(view, box, height, detection_id="box")
    assert hit is not None
    assert hit.xy == pytest.approx(point)
    left, top, right, bottom = box["xyxy"]
    u = (left + right) / 2.

    def closed_form(du, dt, db):
        depth = focal * height / ((bottom + db) - (top + dt))
        return origin[0] + depth, origin[1] - depth * (u + du - 384.) / focal

    expected = max(dist(point, closed_form(*deltas)) for deltas in product((-1.5, 1.5), repeat=3))
    assert hit.uncertainty_m == pytest.approx(expected)
    baseline = project_detection(view.calibration, view.rgb, box["xyxy"], "person", .88, detection_id="baseline")
    assert baseline is not None and hit.appearance == baseline.appearance


def test_small_angular_separation_abstains_even_with_a_resolved_pixel_box():
    view = _view(origin=(-15., 0., 1.2), focal=2000.)
    box, top, bottom = _box(view, 16., 0., 0., .60, kind="bicycle")
    assert 0 < top <= 48 and 0 < bottom <= 48
    assert box["xyxy"][3] - box["xyxy"][1] > 30

    assert upright_hit(view, box, .60, detection_id="far-short") is None


def test_large_vertical_mismatch_caused_by_roll_is_not_reported_as_a_center():
    view = _view(roll=.65)
    box, _, _ = _box(view, 14., -.4, 0., 1.7)

    assert upright_hit(view, box, 1.7, detection_id="rolled") is None


def test_small_box_that_would_move_more_than_four_metres_after_pixel_jitter_abstains():
    view = _view(origin=(0., 0., 1.6), focal=100.)
    box, _, _ = _box(view, 20., 0., 0., 1.7)
    assert box["xyxy"][3] - box["xyxy"][1] == pytest.approx(8.5)

    assert upright_hit(view, box, 1.7, detection_id="unstable") is None


def test_clip_edges_and_optical_near_plane_are_geometric_abstentions():
    view = _view()
    box, _, _ = _box(view, 14., -.4, 0., 1.7)
    clipped = {**box, "xyxy": (box["xyxy"][0], 0., *box["xyxy"][2:])}

    assert upright_hit(view, clipped, 1.7, detection_id="clipped") is None
    assert upright_hit(_view(near=20.), box, 1.7, detection_id="behind-near") is None


def _target(kind, dz):
    return {"kind": kind, "box_ego": (8., -2., .7, .8, .5, dz, .1)}


def test_typical_height_uses_separate_medians_and_never_invents_absent_class_height():
    rows = [{"targets": [_target("person", 1.4), _target("bicycle", .9), _target("unsupported", 8.)]},
            {"targets": [_target("person", 1.8), _target("person", 5.2), _target("bicycle", 1.1)]}]

    assert typical_heights(rows) == {"person": 1.8, "bicycle": 1.}
    assert typical_heights([{"targets": [_target("person", 1.65)]}]) == {"person": 1.65}
    with pytest.raises(ValueError, match="positive truth box dimensions"):
        typical_heights([{"targets": [_target("person", 0.)]}])


def test_invalid_explicit_height_is_an_input_error_not_missing_detection():
    view = _view()
    box, _, _ = _box(view, 14., -.4, 0., 1.7)

    with pytest.raises(ValueError, match="plausible person height"):
        upright_hit(view, box, 4.2, detection_id="wrong-units")
