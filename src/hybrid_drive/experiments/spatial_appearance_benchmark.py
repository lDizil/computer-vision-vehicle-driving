"""Blind comparison of appearance, geometry and within-frame permutation controls."""

from dataclasses import replace
import hashlib

from .bounded_io import decode_json, read_bounded, require_hex
from .io import write_json
from .proposal_alignment import _new_result_directory
from .protocol import content_hash
from .spatial_appearance_capture import APPEARANCE_ROLE, INPUT_ROLES
from .spatial_appearance_features import APPEARANCE_KINDS
from .spatial_appearance_report import write_appearance_gallery, write_appearance_report
from .spatial_appearance_signal import measure_appearance_signal
from .spatial_track_metrics import evaluate_tracking, select_tracking_recipe
from .spatial_tracking_benchmark import TrackingTruthFile, _targets
from .spatial_vehicle_tracks import TRACK_RECIPES, replay_vehicle_tracks


SCOPE = "openscene_box_appearance_tracking_v1"
_MOTION = "motion_strict"
_ZONE = "zone"
_WEIGHT = 0.4
_MIN_SHUFFLE_GAIN = 0.005
_MATCH_IOU = 0.5
_CONTROL_SEED = "mv-appearance-permutation-v1"
_GATE = "development-gate.json"


def run_appearance_benchmark(capture, truth, output_root, *, revision, input_roots=()):
    require_hex(revision, 40, "appearance benchmark revision")
    if set(truth) != set(INPUT_ROLES) or any(not isinstance(t, TrackingTruthFile) for t in truth.values()):
        raise ValueError("Expected five bound track-truth inputs")
    root = _new_result_directory(
        output_root, (*capture.inputs.protected_roots, *(t.path for t in truth.values()), *input_roots),
    )
    protocol = {
        "capture_sha256": capture.sha256, "capture_revision": capture.manifest["revision"],
        "observation_anchors": capture.manifest["observation_anchors"], "model_hash": capture.inputs.model_hash,
        "truth_hashes": {role: item.sha256 for role, item in truth.items()},
        "fixed_frame_match_min_iou": _MATCH_IOU, "appearance_weight": _WEIGHT,
        "profiles": list(APPEARANCE_KINDS), "descriptor": capture.manifest["descriptor_protocol"],
        "permutation": {"scope": "within_a_frame_only", "seed": _CONTROL_SEED, "preserve_missing_mask": True,
                        "identity_permutation_allowed": True},
        "min_gain_vs_shuffled": _MIN_SHUFFLE_GAIN, "physical_control": False, "geometry_changed": False,
    }
    result = {"scope": SCOPE, "status": "running", "revision": revision, "protocol": protocol}
    write_json(root / "result.json", result)
    try:
        _compare(root, capture, truth, result)
        result["status"] = "passed"
    except KeyboardInterrupt:
        result.update(status="interrupted", error="Interrupted by operator")
    except Exception as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    try:
        write_appearance_report(root, result)
    except Exception as exc:
        result["report_error"] = f"{type(exc).__name__}: {exc}"
        if result["status"] == "passed":
            result.update(status="failed", error="Could not save appearance report")
    path = root / "result.json"
    write_json(path, result)
    return root, result, hashlib.sha256(read_bounded(path, 3 * 1024 * 1024)).hexdigest()


def _compare(root, capture, truth, result):
    development = tuple(role for role in INPUT_ROLES if role != APPEARANCE_ROLE)
    dev_fixed, dev_looks, dev_targets = _collect(capture, truth, development)
    dev, _ = _evaluate(dev_fixed, dev_looks, dev_targets)
    selection = _pick(dev["tracking"], dev["shuffle"])
    gate = {
        "scope": SCOPE, "protocol_hash": content_hash(result["protocol"]),
        "development_hash": content_hash(dev), "selection": selection,
    }
    path = root / _GATE
    write_json(path, gate)
    frozen = decode_json(read_bounded(path, 128 * 1024))
    if content_hash(frozen) != content_hash(gate):
        raise ValueError("Development appearance selection did not survive the freeze")
    chosen = frozen["selection"]["name"]
    result["development_gate_hash"] = content_hash(frozen)

    # The holdout ID file is opened only below the exact development recipe
    # gate. Alternatives are reported, but not selectable from these labels.
    final_fixed, final_looks, final_targets = _collect(capture, truth, (APPEARANCE_ROLE,))
    if set(dev_targets).intersection(final_targets):
        raise ValueError("Appearance benchmark roles contain overlapping frames")
    final, predicted = _evaluate(final_fixed, final_looks, final_targets)
    matrix = {
        "development": dev, "holdout": final, "selection": frozen["selection"],
        "holdout_diagnostic": _final_check(chosen, final["tracking"], final["shuffle"]),
        "frame_counts": {"development": len(dev_fixed), "holdout": len(final_fixed)},
        "configurations": len(dev["tracking"]) + len(final["tracking"]),
        "raw_prediction_inputs_identical": True,
    }
    result.update(matrix=matrix, matrix_hash=content_hash(matrix))
    gallery = root / "appearance.html"
    write_appearance_gallery(
        gallery, capture.inputs.packs[APPEARANCE_ROLE], final_targets, predicted, selected=chosen,
    )
    result["gallery"] = gallery.name


def _collect(capture, refs, roles):
    frames, features, labels = [], [], {}
    for role in roles:
        targets = _targets(capture.inputs.packs[role], refs[role])
        if set(labels) & set(targets):
            raise ValueError("A source frame appears twice in appearance evaluation")
        labels.update(targets)
        frames.extend(capture.decisions[role])
        features.extend(capture.appearances[role])
    if tuple(d.observation.sample_id for d in frames) != tuple(r.sample_id for r in features):
        raise ValueError("Descriptors must remain aligned to exact source predictions")
    return tuple(frames), tuple(features), labels


def permute_frame_appearance(frames):
    """Randomize only the available slots in that frame, deterministically.

    The ordering depends on the sample ID, rather than a fixed swap in every
    frame, which could otherwise preserve a consistent wrong identity.
    Identity permutations (notably singletons) are intentionally allowed.
    """
    altered = {kind: {"eligible_frames": 0, "changed_slots": 0} for kind in APPEARANCE_KINDS}
    shuffled = []
    for frame in frames:
        replacement = {}
        for kind in APPEARANCE_KINDS:
            values = getattr(frame, kind)
            slots = [index for index, value in enumerate(values) if value is not None]
            if len(slots) > 1:
                altered[kind]["eligible_frames"] += 1
            def key(index):
                return hashlib.sha256(f"{_CONTROL_SEED}\0{frame.sample_id}\0{index}".encode()).digest()
            ordered = sorted(slots, key=key)
            result = list(values)
            for dest, source in zip(slots, ordered):
                result[dest] = values[source]
                altered[kind]["changed_slots"] += result[dest] != values[dest]
            replacement[kind] = tuple(result)
        shuffled.append(replace(frame, **replacement))
    return tuple(shuffled), altered


def _evaluate(fixed, features, truth):
    permuted, mask = permute_frame_appearance(features)
    modes = {
        _ZONE: replay_vehicle_tracks(fixed, TRACK_RECIPES[_ZONE]),
        _MOTION: replay_vehicle_tracks(fixed, TRACK_RECIPES[_MOTION]),
    }
    for kind in APPEARANCE_KINDS:
        for name, frames in ((kind, features), (kind + "_shuffled", permuted)):
            modes[name] = replay_vehicle_tracks(
                fixed, TRACK_RECIPES[_MOTION],
                appearance_by_sample={f.sample_id: getattr(f, kind) for f in frames},
                appearance_weight=_WEIGHT,
            )
    results, reference = {}, None
    for name, rows in modes.items():
        if len(rows) != len(fixed) or any(row.decision is not d for row, d in zip(rows, fixed)):
            raise ValueError("Appearance must not replace a box or presence decision")
        report = evaluate_tracking(rows, truth, min_iou=_MATCH_IOU)
        comparable = tuple(report[k] for k in (
            "evaluation_identity", "frames", "labeled_boxes", "output_boxes", "detections_matched",
            "adjacent_links", "cross_zone_links",
        ))
        if reference is None:
            reference = comparable
        elif comparable != reference:
            raise ValueError("Tracking modes require identical original detections and frame matches")
        results[name] = report
    signal = measure_appearance_signal(fixed, truth, features, permuted)
    return {"tracking": results, "signal": signal, "shuffle": mask}, modes


def _assess(name, metrics, shuffle):
    reasons = []
    for comparator in (_ZONE, _MOTION):
        against = select_tracking_recipe(
            {comparator: metrics[comparator], name: metrics[name]}, baseline_name=comparator,
        )
        reasons.extend(f"{comparator}:{why}" for why in against["candidates"][name]["reasons"])
    actual, control = metrics[name]["fixed_match_idf1"], metrics[name + "_shuffled"]["fixed_match_idf1"]
    effect = None if actual is None or control is None else actual - control
    if not shuffle[name]["changed_slots"]:
        reasons.append("no_permutable_distinct_features")
    if effect is None or effect < _MIN_SHUFFLE_GAIN - 1e-12:
        reasons.append("insufficient_gain_vs_shuffled")
    return {"eligible": not reasons, "reasons": reasons, "idf1_gain_vs_shuffled": effect}


def _pick(metrics, shuffle):
    checks = {name: _assess(name, metrics, shuffle) for name in APPEARANCE_KINDS}
    eligible = [
        (-metrics[name]["fixed_match_idf1"], metrics[name]["wrong_links"], metrics[name]["adjacent_switches"], name)
        for name, check in checks.items() if check["eligible"]
    ]
    return {"name": min(eligible)[-1] if eligible else _ZONE, "candidates": checks, "scope": "development_only"}


def _final_check(name, metrics, shuffle):
    if name == _ZONE:
        return {"evaluated_recipe": name, "outcome": "baseline_retained_on_dev", "reasons": []}
    # Check only the already chosen candidate; test data never select another.
    check = _assess(name, metrics, shuffle)
    return {"evaluated_recipe": name, "outcome": "preselected_passed" if check["eligible"] else "no_replacement",
            "reasons": check["reasons"]}
