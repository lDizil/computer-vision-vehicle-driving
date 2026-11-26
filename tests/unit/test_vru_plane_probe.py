import pytest

from hybrid_drive.experiments.object_vru_geometry import GroundHit
from hybrid_drive.experiments.vru_plane_probe import _fit_height, _gate, _move_hit, score_plane


def test_fit_uses_equal_recording_weight_and_excludes_one_target_recordings():
    dense = [-.4] * 60
    small = [0.] * 3
    samples = {"a": dense, "b": small, "c": [-.2] * 4, "d": [.2] * 6, "rare": [.9]}
    height, recorded = _fit_height(samples)
    assert height == -.1
    assert set(recorded) == {"a", "b", "c", "d"}
    assert recorded["a"]["targets"] == 60
    with pytest.raises(ValueError, match="independent"):
        _fit_height({"a": dense, "rare": [.9]})


def test_projection_moves_from_each_camera_origin_along_the_same_ray():
    hit = GroundHit("first", "person", "CAM_F0", (12., 2.), (2., 1., 1.6),
                    .8, .50, (1.,) + (0.,)*35)
    moved = _move_hit(hit, -.2)
    assert moved.xy == pytest.approx((13.25, 2.125))
    assert moved.uncertainty_m == pytest.approx(.5625)
    assert moved.detection_id == hit.detection_id and moved.appearance == hit.appearance
    assert _move_hit(hit, 0.) is hit


def test_more_total_matches_cannot_hide_a_local_recall_regression():
    labels = [{"tracks": [{"centers_xy_m": [(4., 1.)]}]}]
    score = {"gate": {"additional_matched": 6, "common_matches": 20, "improved_recordings": 4,
                      "common_mean_error_change_m": -.02}, "rows": [{"pairs": [
                          {"target_index": 0, "reference": {"object_id": "near"}, "candidate": None},
                      ]}]}
    result = _gate(score, labels)
    assert result["status"] == "gain_not_established"
    assert not result["checks"]["near_coverage_preserved"]


def test_control_score_rejects_wrong_seal_without_reading_labels(tmp_path):
    absent = tmp_path / "unmounted"
    wrong = tmp_path / "forecast.json"
    wrong.write_text("{}")
    with pytest.raises(ValueError, match="SHA fixed"):
        score_plane(absent, wrong, absent, tmp_path / "out", sealed_sha256="0"*64)
