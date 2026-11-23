from copy import deepcopy
from dataclasses import replace

import pytest

from hybrid_drive.experiments.spatial_decision_policy import ABSENT, PRESENT, UNCERTAIN
from hybrid_drive.experiments.spatial_track_metrics import TrackingTarget, match_tracking_frame
from hybrid_drive.experiments.spatial_zoom_guard_metrics import (
    GUARD_MODES,
    check_guard_holdout,
    evaluate_guard_changes,
    oracle_guard_match_ceiling,
    select_guard_mode,
)
from hybrid_drive.experiments.spatial_zoom_metrics import evaluate_zoom_boxes
from hybrid_drive.experiments.temporal_spatial_types import FrameDecision, FrameObservation


# Binary fractions keep the IoU boundary test exact.
_BOXES = (
    (1/16, 1/8, 3/16, 3/8), (5/16, 1/8, 7/16, 3/8),
    (9/16, 1/8, 11/16, 3/8), (13/16, 1/8, 15/16, 3/8),
)
_CROSS_QUARTER = (1/8, 1/8, 11/32, 3/8)
_NEIGHBOR_LABEL = (1/8, 1/8, 7/16, 3/8)


def _decision(sid="frame", *, rec="recording", index=0, states=(PRESENT, ABSENT, UNCERTAIN, ABSENT), boxes=_BOXES):
    observation = FrameObservation(rec, rec + "-clip", sid, index, 500_000 * index, (1., -1., 0., -1.), boxes)
    return FrameDecision(observation, states, observation.scores, boxes, ("raw",) * 4, None)


def _target(*pairs):
    ids, boxes = [None] * 4, [None] * 4
    for slot, box in pairs:
        ids[slot], boxes[slot] = f"target-{slot}", box
    return TrackingTarget(tuple(ids), tuple(boxes))


def _move(row, slot, box):
    boxes = list(row.boxes)
    boxes[slot] = box
    return replace(row, boxes=boxes)


def _lower(box):
    return box[0], 1/2, box[2], 3/4


def test_counts_both_targets_in_an_equal_count_quarter_boundary_flip():
    first = _decision("first")
    second = _decision("second", rec="other")
    changed = _move(first, 0, _CROSS_QUARTER)
    labels = {"first": _target((0, _BOXES[0]), (1, _NEIGHBOR_LABEL)), "second": _target()}
    assert match_tracking_frame(first, labels["first"]) == ((0, 0),)
    assert match_tracking_frame(changed, labels["first"]) == ((0, 1),)
    before, after = [first, second], [changed, second]
    delta = evaluate_guard_changes(before, after, labels)
    assert delta == {
        "frames": 2, "gained_targets": 1, "lost_targets": 1, "changed_present_slots": 1,
        "per_recording": {
            "recording": {"frames": 1, "gained_targets": 1, "lost_targets": 1, "changed_present_slots": 1},
            "other": {"frames": 1, "gained_targets": 0, "lost_targets": 0, "changed_present_slots": 0},
        },
    }
    n_before = evaluate_zoom_boxes(before, labels)["detections_matched"]
    n_after = evaluate_zoom_boxes(after, labels)["detections_matched"]
    assert delta["gained_targets"] - delta["lost_targets"] == n_after - n_before == 0


@pytest.mark.parametrize("top, recovered", [(1/4, 1), (1/4 - 2**-20, 0)])
def test_inclusive_iou_half_boundary(top, recovered):
    row = _decision(boxes=(_lower(_BOXES[0]), *_BOXES[1:]))
    left, bottom, right, _ = _BOXES[0]
    adjusted = _move(row, 0, (left, bottom, right, top))
    labels = {"frame": _target((0, _BOXES[0]))}
    changes = evaluate_guard_changes((row,), (adjusted,), labels)
    assert not changes["lost_targets"]
    assert changes["gained_targets"] == recovered
    assert changes["changed_present_slots"] == 1


def test_reassigning_targets_to_different_slots_does_not_count_as_losing_them():
    gt0, gt1 = (.06, .18, .36, .58), (.11, .18, .48, .58)
    before = _decision(states=(PRESENT, PRESENT, ABSENT, ABSENT),
                       boxes=(gt0, (.17, .18, .35, .58), *_BOXES[2:]))
    after = _move(before, 1, gt1)
    target = _target((0, gt0), (1, gt1))
    assert match_tracking_frame(before, target) == ((0, 1), (1, 0))
    assert match_tracking_frame(after, target) == ((0, 0), (1, 1))
    changes = evaluate_guard_changes([before], [after], {"frame": target})
    assert changes["gained_targets"] == changes["lost_targets"] == 0
    assert changes["changed_present_slots"] == 1


@pytest.mark.parametrize("mutate, reason", [
    (lambda row: replace(row, states=(PRESENT, PRESENT, UNCERTAIN, ABSENT)), "preserve occupancy"),
    (lambda row: replace(row, scores=(2., -1., 0., -1.)), "preserve occupancy"),
    (lambda row: replace(row, reasons=("changed", *row.reasons[1:])), "policy reasons"),
    (lambda row: replace(row, reset_reason="gap"), "policy reasons"),
    (lambda row: replace(row, observation=replace(row.observation, timestamp_us=1)), "ordered observations"),
    (lambda row: replace(row, observation=replace(row.observation, scores=(2., -1., 0., -1.))),
     "ordered observations"),
    (lambda row: _move(row, 1, _lower(row.boxes[1])), "inactive boxes"),
    (lambda row: _move(row, 2, _lower(row.boxes[2])), "inactive boxes"),
])
def test_cannot_modify_any_frozen_input_or_inactive_box(mutate, reason):
    row = _decision()
    changed = mutate(row)
    target = {"frame": _target((0, _BOXES[0]))}
    with pytest.raises(ValueError, match=reason):
        evaluate_guard_changes((row,), (changed,), target)
    with pytest.raises(ValueError, match=reason):
        oracle_guard_match_ceiling((row,), (row,), (changed,), target)


def test_requires_same_frame_order_count_and_exact_labels():
    rows = (_decision("a"), _decision("b", index=1))
    labels = {sid: _target() for sid in ("a", "b")}
    with pytest.raises(ValueError, match="ordered observations"):
        evaluate_guard_changes(rows, rows[::-1], labels)
    with pytest.raises(ValueError, match="ordered frame count"):
        evaluate_guard_changes(rows, rows[:1], labels)
    with pytest.raises(ValueError, match="Duplicate"):
        evaluate_guard_changes(rows, rows[:1] * 2, labels)
    with pytest.raises(ValueError, match="exactly one"):
        evaluate_guard_changes(rows, rows, {**labels, "elsewhere": _target()})
    with pytest.raises(ValueError, match="TrackingTarget"):
        oracle_guard_match_ceiling(rows, rows, rows, {**labels, "a": rows[0]})


def test_oracle_combines_slot_choices_while_old_one_to_one_assignment_still_limits_them():
    base = _decision("mixed", states=(PRESENT,) * 4,
                     boxes=(_BOXES[0], *tuple(map(_lower, _BOXES[1:]))))
    half = replace(base, boxes=(_lower(_BOXES[0]), _BOXES[1], _BOXES[2], _lower(_BOXES[3])))
    full = replace(base, boxes=(*tuple(map(_lower, _BOXES[:3])), _BOXES[3]))
    narrow = _decision("one-slot", rec="contention")
    cross = _move(narrow, 0, _CROSS_QUARTER)
    targets = {
        "mixed": _target(*enumerate(_BOXES)),
        "one-slot": _target((0, _BOXES[0]), (1, _NEIGHBOR_LABEL)),
    }
    got = oracle_guard_match_ceiling([base, narrow], [half, narrow], [full, cross], targets)
    assert got["scope"] == "labels_only_fixed_present_box_choices"
    assert got["min_iou"] == 0.5
    assert got["frames"] == 2
    assert got["labeled_boxes"] == 6
    assert got["output_boxes"] == 5
    assert got["baseline_detections_matched"] == 2
    assert got["zoom_half_detections_matched"] == 3
    assert got["zoom_full_detections_matched"] == 2
    assert got["oracle_detections_matched"] == 5
    assert got["recoverable_matches"] == 3
    assert got["per_recording"]["recording"]["oracle_detections_matched"] == 4
    # Label union has size 2, but there is only one movable prediction.
    assert got["per_recording"]["contention"]["labeled_boxes"] == 2
    assert got["per_recording"]["contention"]["oracle_detections_matched"] == 1


def test_oracle_does_not_invent_boxes_or_headroom():
    rows = (_decision("empty", states=(ABSENT, UNCERTAIN, ABSENT, ABSENT)),
            _decision("saturated", rec="saturated"),
            _decision("unsupported", rec="unsupported"))
    labels = {
        "empty": _target((0, _BOXES[0])),
        "saturated": _target((0, _BOXES[0])),
        "unsupported": _target((1, _BOXES[1])),
    }
    got = oracle_guard_match_ceiling(rows, rows, rows, labels)
    assert got["oracle_detections_matched"] == got["baseline_detections_matched"] == 1
    assert got["recoverable_matches"] == 0


def _stats(role, gains=(0, 0), losses=(0, 0), moved=None):
    groups, changes = {}, {}
    moved = tuple(a + b for a, b in zip(gains, losses)) if moved is None else moved
    for index, (gain, loss, changed) in enumerate(zip(gains, losses, moved)):
        name = f"{role}-{index}"
        groups[name] = {"frames": 8, "labeled_boxes": 20, "output_boxes": 19,
                        "detections_matched": 5 + gain - loss}
        changes[name] = {"frames": 8, "gained_targets": gain, "lost_targets": loss, "changed_present_slots": changed}
    return {
        **{key: sum(item[key] for item in groups.values()) for key in next(iter(groups.values()))},
        "per_recording": groups,
        "evaluation_identity": role,
        "changes": {
            **{key: sum(item[key] for item in changes.values()) for key in next(iter(changes.values()))},
            "per_recording": changes,
        },
    }


def _modes(role):
    return {mode: _stats(role) for mode in GUARD_MODES}


def test_dev_requires_no_role_regression_and_no_more_than_one_net_drop_per_recording():
    roles = {role: _modes(role) for role in ("alpha", "beta")}
    roles["alpha"]["zoom_full"] = _stats("alpha", gains=(5, 2))
    roles["beta"]["zoom_full"] = _stats("beta", losses=(1, 0))
    roles["alpha"]["zoom_half"] = _stats("alpha", gains=(0, 5), losses=(2, 0))
    roles["alpha"]["gate_supported"] = _stats("alpha", gains=(2, 3), losses=(3, 0))
    got = select_guard_mode(roles)
    assert got["name"] == "gate_supported"
    assert got["candidates"]["zoom_full"]["matched_gain"] == 6
    assert got["candidates"]["zoom_full"]["reasons"] == ["role_regression:beta"]
    assert got["candidates"]["zoom_half"]["reasons"] == ["recording_drop:alpha:alpha-0"]
    # Actual lost targets are reported, but the gate uses the recording's net change.
    assert got["candidates"]["gate_supported"]["lost_targets"] == 3
    assert got["candidates"]["gate_supported"]["per_role"]["alpha"]["recording_drops"] == {"alpha-0": 1}


def test_stable_ties_use_exact_gain_then_fewer_moved_slots_then_lexical_name():
    roles = {role: _modes(role) for role in ("west", "east")}
    roles["west"]["zoom_full"] = _stats("west", gains=(2, 1), moved=(5, 1))
    roles["west"]["gate"] = _stats("west", gains=(1, 2))
    roles["west"]["gate_supported"] = _stats("west", gains=(1, 2))
    roles["west"]["zoom_half"] = _stats("west", gains=(1, 1))
    roles["west"]["gate"]["tracking"] = {"zone": {"fixed_match_idf1": 0.2}}
    roles["west"]["gate_supported"]["tracking"] = {"zone": {"fixed_match_idf1": 0.9}}
    got = select_guard_mode(roles)
    assert got["name"] == "gate"
    assert got == select_guard_mode(dict(reversed(list(roles.items()))))
    assert all(row["eligible"] for row in got["candidates"].values())


def test_incompatible_metrics_are_never_eligible_even_when_match_count_is_high():
    roles = {"dev": _modes("dev")}
    roles["dev"]["zoom_full"] = {**_stats("dev", gains=(4, 4)), "evaluation_identity": "other"}
    switched = _stats("dev", gains=(4, 4))
    switched["per_recording"]["dev-0"]["output_boxes"] += 1
    switched["per_recording"]["dev-1"]["output_boxes"] -= 1
    roles["dev"]["zoom_half"] = switched
    roles["dev"]["gate"] = deepcopy(_stats("dev", gains=(3, 3)))
    roles["dev"]["gate"]["changes"]["lost_targets"] = 1
    roles["dev"]["gate_supported"] = _stats("dev", gains=(1, 0))
    got = select_guard_mode(roles)
    assert got["name"] == "baseline"
    assert {mode: result["reasons"] for mode, result in got["candidates"].items()} == {
        "zoom_full": ["different_frozen_inputs_or_labels:dev"],
        "zoom_half": ["different_fixed_counts:dev"],
        "gate": ["invalid_change_counts:dev"],
        "gate_supported": ["insufficient_matched_gain"],
    }


def test_requires_valid_baseline():
    with pytest.raises(ValueError, match="named development roles"):
        select_guard_mode({})
    with pytest.raises(ValueError, match="Invalid frozen baseline"):
        select_guard_mode({"dev": {"gate": _stats("dev", gains=(2, 2))}})
    broken = _stats("dev", gains=(3, 3), losses=(3, 3))
    with pytest.raises(ValueError, match="Invalid frozen baseline"):
        select_guard_mode({"dev": {"baseline": broken}})


def test_holdout_checks_only_preselected_mode_and_does_not_find_a_replacement():
    holdout = _modes("holdout")
    holdout["zoom_full"] = _stats("holdout", gains=(4, 4))
    holdout["gate"] = _stats("holdout", gains=(1, 0))
    assert check_guard_holdout("gate", holdout) == {
        "evaluated_recipe": "gate", "outcome": "no_replacement", "reasons": ["insufficient_matched_gain"],
    }
    holdout["gate"] = _stats("holdout", gains=(3, 0), losses=(0, 1))
    assert check_guard_holdout("gate", holdout) == {
        "evaluated_recipe": "gate", "outcome": "preselected_passed", "reasons": [],
    }
    holdout["gate"] = _stats("holdout", gains=(4, 0), losses=(0, 2))
    assert check_guard_holdout("gate", holdout) == {
        "evaluated_recipe": "gate", "outcome": "no_replacement", "reasons": ["recording_drop:holdout:holdout-1"],
    }
    assert check_guard_holdout("baseline", {}) == {
        "evaluated_recipe": "baseline", "outcome": "baseline_retained_on_dev", "reasons": [],
    }
    with pytest.raises(ValueError, match="Unknown frozen"):
        check_guard_holdout("oracle", holdout)
