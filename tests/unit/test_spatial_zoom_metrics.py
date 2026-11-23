from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from hybrid_drive.experiments.spatial_decision_policy import ABSENT, PRESENT, UNCERTAIN
from hybrid_drive.experiments.spatial_track_metrics import TrackingTarget, match_tracking_frame
from hybrid_drive.experiments.spatial_zoom_metrics import apply_zoom_boxes, evaluate_zoom_boxes, select_zoom_candidate
from hybrid_drive.experiments.temporal_spatial_types import FrameDecision, FrameObservation


BOXES = ((.10, .2, .18, .4), (.26, .2, .43, .5), (.54, .1, .70, .4), (.81, .2, .93, .6))
SMALL = (.10, .20, .18, .28)


def _decision(*, sid="frame", rec="rec", index=0, states=(PRESENT, ABSENT, UNCERTAIN, ABSENT),
              boxes=BOXES):
    source = FrameObservation(rec, rec + "-clip", sid, index, 1_000_000 + index * 500_000,
                              (1., -1., 0., -1.), boxes)
    return FrameDecision(source, states, source.scores, boxes, ("raw",) * 4, "initial" if not index else None)


def _target(*pairs):
    labels, boxes = [None] * 4, [None] * 4
    for zone, box in pairs:
        labels[zone], boxes[zone] = f"gt-{zone}", box
    return TrackingTarget(tuple(labels), tuple(boxes))


def test_apply_preserves_policy_provenance_order_and_exact_inactive_coordinates():
    first = _decision(sid="frame-1")
    second = _decision(sid="frame-2", index=1, states=(ABSENT, PRESENT, ABSENT, UNCERTAIN))
    candidate = np.array((first.boxes, second.boxes), dtype=np.float32)
    candidate[0, 0] = [.11, .21, .19, .41]
    candidate[1, 1] = [.27, .24, .44, .55]
    output = apply_zoom_boxes((first, second), candidate)
    assert tuple(x.observation.sample_id for x in output) == ("frame-1", "frame-2")
    for source, changed, f32 in zip((first, second), output, candidate):
        assert changed.observation is source.observation
        assert (changed.scores, changed.states, changed.reasons, changed.reset_reason) == (
            source.scores, source.states, source.reasons, source.reset_reason,
        )
        for band, state in enumerate(source.states):
            expected = tuple(map(float, f32[band])) if state == PRESENT else source.boxes[band]
            assert changed.boxes[band] == expected
    assert first.boxes == second.boxes == BOXES


def test_accepts_model_fp32_roundtrip_without_replacing_original_decision():
    d = _decision()
    assert apply_zoom_boxes([d], np.asarray([d.boxes], dtype=np.float32))[0] is d


def test_optional_torch_input():
    torch = pytest.importorskip("torch")
    d = _decision()
    value = torch.tensor([d.boxes], dtype=torch.float32)
    value[0, 0, 0] = .14
    got = apply_zoom_boxes([d], value)
    assert got[0].boxes[0][0] == pytest.approx(.14)
    assert got[0].boxes[1:] == d.boxes[1:]


@pytest.mark.parametrize("band", [1, 2, 3])
def test_attempted_inactive_change_is_rejected_including_uncertain(band):
    first = _decision()
    candidate = np.asarray([first.boxes], dtype=np.float32)
    candidate[0, band, 1] += np.float32(.02)
    with pytest.raises(ValueError, match="must not change inactive"):
        apply_zoom_boxes([first], candidate)
    assert first.boxes == BOXES


@pytest.mark.parametrize("new_box", [
    (.32, .2, .42, .4),  # Wrong quarter.
    (.125, .2, .125 + 1e-10, .4),  # Collapses at canonical FP32 precision.
    (.14, .3, .14, .5),
    (-.04, .1, .18, .4),
    (.01, .2, np.nan, .4),
    (False, .2, .14, .4),
])
def test_malformed_or_wrong_band_box_is_never_admitted(new_box):
    first = _decision()
    with pytest.raises(ValueError):
        apply_zoom_boxes([first], [[new_box, *first.boxes[1:]]])


@pytest.mark.parametrize("bad", [[], [BOXES, BOXES], "foo", [[BOXES[0]]], [BOXES[:3]]])
def test_wrong_frame_count_or_shape_is_rejected(bad):
    with pytest.raises(ValueError, match="CPU numeric array"):
        apply_zoom_boxes([_decision()], bad)


def test_counts_iou_precision_recall_ceiling_and_true_projected_area_by_recording():
    a1 = _decision(sid="a1", states=(PRESENT, PRESENT, ABSENT, ABSENT),
                   boxes=(SMALL, *BOXES[1:]))
    a2 = _decision(sid="a2", index=1, states=(UNCERTAIN, ABSENT, ABSENT, ABSENT))
    b = _decision(sid="b", rec="second", states=(ABSENT, ABSENT, PRESENT, ABSENT))
    truth = {"a1": _target((0, SMALL)), "a2": _target((0, SMALL)), "b": _target((2, BOXES[2]))}
    report = evaluate_zoom_boxes([a1, a2, b], truth, include_tracking=True)
    assert report["frames"] == 3
    assert report["labeled_boxes"] == report["output_boxes"] == 3
    assert report["detections_matched"] == 2
    assert report["precision"] == report["recall"] == report["idf1_ceiling"] == pytest.approx(2 / 3)
    assert report["mean_matched_iou"] == pytest.approx(1)
    assert report["projected_area"] == {
        "small_lt_0p01": {"labeled_boxes": 2, "detections_matched": 1, "recall": .5},
        "large_ge_0p01": {"labeled_boxes": 1, "detections_matched": 1, "recall": 1.},
    }
    assert report["per_recording"]["rec"]["detections_matched"] == 1
    assert report["per_recording"]["second"]["detections_matched"] == 1
    assert set(report["tracking"]) == {"zone", "motion_strict"}
    assert all(m["detections_matched"] == 2 for m in report["tracking"].values())
    assert all(m["fixed_match_idf1"] <= report["idf1_ceiling"] for m in report["tracking"].values())


def test_measurement_uses_maximum_cardinality_match_instead_of_greedy_highest_iou():
    # P0 can explain either target; P1 only the first one. Greedy P0→G0
    # gives IoU 1.0 but loses a usable match.
    gt0, gt1 = (.06, .18, .36, .58), (.11, .18, .48, .58)
    p1 = (.17, .18, .35, .58)
    row = _decision(states=(PRESENT, PRESENT, ABSENT, ABSENT), boxes=(gt0, p1, *BOXES[2:]))
    truth = _target((0, gt0), (1, gt1))
    assert match_tracking_frame(row, truth) == ((0, 1), (1, 0))
    report = evaluate_zoom_boxes([row], {"frame": truth})
    assert report["detections_matched"] == 2
    assert report["mean_matched_iou"] == pytest.approx((.25 / .42 + .18 / .30) / 2)


def test_identity_binds_order_source_scores_states_and_truth_but_not_refined_boxes():
    a, b = _decision(sid="a"), _decision(sid="b", index=1, rec="second")
    truth = {"a": _target((0, BOXES[0])), "b": _target((0, BOXES[0]))}
    baseline = evaluate_zoom_boxes([a, b], truth)
    moved = apply_zoom_boxes([a, b], [[(.02, .2, .07, .4), *a.boxes[1:]], b.boxes])
    changed = evaluate_zoom_boxes(moved, truth)
    assert changed["evaluation_identity"] == baseline["evaluation_identity"]
    assert changed["detections_matched"] == baseline["detections_matched"] - 1
    assert changed["output_boxes"] == baseline["output_boxes"]
    assert evaluate_zoom_boxes([b, a], truth)["evaluation_identity"] != baseline["evaluation_identity"]
    modified_truth = {**truth, "a": TrackingTarget(("other", None, None, None), truth["a"].boxes)}
    assert evaluate_zoom_boxes([a, b], modified_truth)["evaluation_identity"] != baseline["evaluation_identity"]
    for item in (replace(a, scores=(2., -1., 0., -1.)),
                 replace(a, states=(UNCERTAIN, ABSENT, UNCERTAIN, ABSENT)),
                 replace(a, observation=replace(a.observation, timestamp_us=1_000_001))):
        assert evaluate_zoom_boxes([item, b], truth)["evaluation_identity"] != baseline["evaluation_identity"]


def test_empty_support_reports_none_and_ambiguous_labels_fail():
    a = _decision(states=(UNCERTAIN, ABSENT, ABSENT, ABSENT))
    report = evaluate_zoom_boxes([a], {"frame": _target()})
    assert report["labeled_boxes"] == report["output_boxes"] == report["detections_matched"] == 0
    for key in ("precision", "recall", "mean_matched_iou", "idf1_ceiling"):
        assert report[key] is None
    with pytest.raises(ValueError, match="exactly one"):
        evaluate_zoom_boxes([a], {"not-frame": _target()})
    with pytest.raises(ValueError, match="Duplicate"):
        evaluate_zoom_boxes([a, a], {"frame": _target()})
    with pytest.raises(ValueError, match="TrackingTarget"):
        evaluate_zoom_boxes([a], {"frame": a})


def _metrics(a, b, identity="fixed"):
    return {
        "frames": 16, "labeled_boxes": 40, "output_boxes": 40,
        "detections_matched": a + b, "evaluation_identity": identity,
        "per_recording": {
            "alpha": {"frames": 8, "labeled_boxes": 20, "output_boxes": 20, "detections_matched": a},
            "beta": {"frames": 8, "labeled_boxes": 20, "output_boxes": 20, "detections_matched": b},
        },
    }


def _dev():
    return {
        "baseline": _metrics(5, 5),
        "geometry_0": _metrics(5, 4), "geometry_1": _metrics(5, 5),
        "global_0": _metrics(6, 6), "global_1": _metrics(5, 7),
        "zoom_0": _metrics(7, 7), "zoom_1": _metrics(8, 10),
    }


def test_selects_seed_zero_for_a_qualifying_family_even_if_seed_one_is_better():
    result = select_zoom_candidate(_dev())
    assert result["name"] == "zoom_0"
    assert result["summary"]["chosen_seed"] == 0
    assert result["eligible_families"] == ["global", "zoom"]
    assert result["candidates"]["zoom_0"]["matched_changes"] == {
        "baseline": 4, "geometry_0": 5, "global_0": 2,
    }


def test_requires_both_seeds_to_beat_global_for_close_up_claim():
    dev = _dev()
    dev["zoom_1"] = _metrics(6, 7)
    result = select_zoom_candidate(dev)
    assert result["name"] == "global_0"
    assert result["eligible_families"] == ["global"]
    assert result["candidates"]["zoom_1"]["reasons"] == ["insufficient_gain_vs_global_1"]


def test_no_hiding_recording_losses_in_better_pooled_box_count():
    dev = _dev()
    dev["zoom_0"] = _metrics(3, 20)
    result = select_zoom_candidate(dev)
    assert result["name"] == "global_0"
    assert result["candidates"]["zoom_0"]["recording_drops"] == {"alpha": 2}
    assert "recording_drop:alpha" in result["candidates"]["zoom_0"]["reasons"]
    assert select_zoom_candidate(dev, max_recording_drop=2)["name"] == "zoom_0"


def test_mismatched_control_inputs_reject_even_if_visual_arms_look_good():
    dev = _dev()
    dev["geometry_1"] = _metrics(5, 5, identity="different")
    result = select_zoom_candidate(dev)
    assert result["name"] == "baseline"
    assert result["summary"]["chosen_seed"] is None
    for arm in ("global_1", "zoom_1"):
        assert "geometry_1:different_frozen_inputs_or_labels" in result["candidates"][arm]["reasons"]


def test_per_recording_counts_are_bound_not_just_equal_totals():
    dev = _dev()
    dev["geometry_1"]["per_recording"]["alpha"]["output_boxes"] += 1
    dev["geometry_1"]["per_recording"]["beta"]["output_boxes"] -= 1
    r = select_zoom_candidate(dev)
    assert r["name"] == "baseline"
    assert r["candidates"]["global_1"]["reasons"] == ["geometry_1:different_fixed_counts"]


def test_missing_metrics_and_counts_that_do_not_add_up_cannot_qualify():
    dev = _dev()
    dev.pop("geometry_1")
    dev["zoom_0"]["detections_matched"] += 1
    result = select_zoom_candidate(dev)
    assert result["name"] == "baseline"
    assert result["candidates"]["zoom_0"]["reasons"] == ["zoom_0:invalid_metrics"]
    assert "geometry_1:missing_metrics" in result["candidates"]["global_1"]["reasons"]
    broken = deepcopy(dev)
    broken["baseline"]["frames"] += 1
    with pytest.raises(ValueError, match="baseline"):
        select_zoom_candidate(broken)


@pytest.mark.parametrize("kwargs", [{"min_gain":0},{"min_gain":True},{"max_recording_drop":-1}])
def test_invalid_guard_parameters(kwargs):
    with pytest.raises(ValueError):
        select_zoom_candidate(_dev(), **kwargs)
