from dataclasses import asdict
from types import SimpleNamespace

import pytest

from hybrid_drive.experiments.object_vru_geometry import GroundHit, GroundObject
from hybrid_drive.experiments.vru_projection_audit import _trace_window, score_projection_audit
from hybrid_drive.experiments.vru_projection_metrics import summarize_projection_audit


RECT = (20., 20., 35., 60.)
CAMERA = "CAM_F0"


def trace(state, rect=RECT, *, reason=None, x=None, key=None, score=.7, kind="person"):
    return {"kind": kind, "confidence": score, "xyxy": rect, "tile": 0, "state": state, "projection_reason": reason,
            "xy_m": None if x is None else (x, 0.), "detection_id": key}


def true_target(x, rect=RECT, *, bottom=0., kind="person"):
    return {"kind": kind, "box_ego": (x, 0., bottom + .8, .6, .6, 1.6, 0.),
            "views": [{"camera_name": CAMERA, "xyxy": rect}]}


def _profile(traces):
    objects = []
    for raw in traces:
        if raw["state"] == "projected":
            hit = GroundHit(raw["detection_id"], raw["kind"], CAMERA, raw["xy_m"], (0., 0., 1.6),
                            raw["confidence"], .15, (1.,) + (0.,) * 35)
            obj = GroundObject("d" + str(len(objects)), hit.kind, hit.xy, hit.confidence, (hit,))
            objects.append(asdict(obj))
    return {"objects": objects, "cameras": [{"camera_name": CAMERA, "boxes": traces}]}


def _inputs(targets, *, ssd=(), faster=()):
    images = {"t": {"views": [{"camera": CAMERA, "camera_to_ego_translation": (0., 0., 1.6)}]}}
    row = {"sample_token": "t", "recording": "r", "profiles": {"ssdlite": _profile(ssd), "faster_rcnn": _profile(faster)}}
    labels = {"sample_token": "t", "target_dt_s": (.5, 1.),
              "tracks": [{"centers_xy_m": [(target["box_ego"][0], dy) for dy in (0., .5, 1.)]} for target in targets]}
    truth = {"sample_token": "t", "targets": targets}
    return [row], [truth], [labels], images


@pytest.mark.parametrize(("boxes", "cause"), [
    ([], "no_reliable_camera_box"),
    ([trace("score_floor", score=.22)], "camera_box_filtered"),
    ([trace("projection_rejected", reason="unstable")], "camera_projection_rejected"),
    ([trace("projected", x=15., key="v0-b0")], "camera_projection_far"),
    ([trace("projected", x=10.2, key="v0-b0")], "matched"),
])
def test_buckets_distinguish_detector_floor_projection_failure_and_center_match(boxes, cause):
    score = summarize_projection_audit(*_inputs([true_target(10.)], faster=boxes))
    assert score["moving"]["targets"] == 1
    assert score["moving"]["causes"] == {cause: 1}
    if cause == "camera_projection_rejected":
        assert score["moving"]["projection_rejection_reasons"] == {"unstable": 1}


def test_annotation_height_oracle_moves_a_ray_without_changing_the_actual_match():
    scores = summarize_projection_audit(*_inputs([true_target(10., bottom=.5)],
                                                 faster=[trace("projected", x=15., key="v0-b0")]))
    case, = scores["cases"]
    assert case["cause"] == "camera_projection_far"
    assert case["plane_probe"]["best_before_m"] == 5.
    assert case["plane_probe"]["best_oracle_m"] == pytest.approx(.3125)
    assert scores["moving"]["diagnostic_plane_oracle"]["misses_moved_within_radius"] == 1


def test_an_accepted_camera_hit_can_be_lost_only_after_fusion():
    ssd = [trace("projected", (200., 20., 215., 60.), x=12.25, key="v0-b0")]
    faster = [trace("projected", x=11.80, key="v0-b0")]
    result = summarize_projection_audit(*_inputs([true_target(10.)], ssd=ssd, faster=faster))
    assert result["moving"]["causes"] == {"lost_after_camera_projection": 1}


def test_tied_camera_envelopes_do_not_create_two_box_witnesses():
    rows = _inputs([true_target(10.), true_target(14.)], faster=[trace("projection_rejected", reason="outside_roi")])
    result = summarize_projection_audit(*rows)
    assert result["moving"]["causes"] == {"camera_box_ambiguous": 2}
    assert result["moving"]["no_near_with_camera_witness"] == {"0.2": 0, "0.3": 0, "0.45": 0}


def test_bad_projected_id_is_not_treated_as_a_different_matched_target():
    rows = _inputs([true_target(10.)], faster=[trace("projected", x=10., key="v0-b0")])
    rows[0][0]["profiles"]["faster_rcnn"]["cameras"][0]["boxes"][0]["detection_id"] = "wrong"
    with pytest.raises(ValueError, match="exact projected hit"):
        summarize_projection_audit(*rows)


def test_capture_shares_one_current_shot_and_never_loads_history(tmp_path):
    calls = []
    shot = SimpleNamespace(content_sha256="a"*64, ref_time_us=25)

    def load(root, token, frame, *, rectified_wh):
        assert (root, token, frame, rectified_wh) == (tmp_path, "now", "index-row", (768, 432))
        calls.append("load")
        return shot

    def detector(snapshot):
        assert snapshot is shot
        calls.append("detect")
        return (), []

    anchor = {"sample_token": "now", "previous_token": "before", "older_token": "old", "recording": "r"}
    _trace_window(tmp_path, {"frames": {"now": "index-row"}}, anchor,
                  dict.fromkeys(("ssdlite", "faster_rcnn"), SimpleNamespace(audit=detector)), load, (768, 432))
    assert calls == ["load", "detect", "detect"]


def test_invalid_trace_seal_fails_before_mounting_or_loading_any_annotation(tmp_path):
    trace_file = tmp_path / "trace"
    trace_file.write_text("{}")
    absent = tmp_path / "not-mounted"
    with pytest.raises(ValueError, match="SHA fixed"):
        score_projection_audit(absent, trace_file, absent, absent, absent, tmp_path / "out",
                               trace_sha256="0"*64, reference_sha256="0"*64)
