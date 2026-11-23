"""Blind comparison of identity assignment on fixed camera predictions."""

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path

from hybrid_drive.assets.spatial_manifest import read_spatial_targets
from hybrid_drive.assets.spatial_track_truth import read_track_truth
from .bounded_io import decode_json, read_bounded, require_hex
from .io import write_json
from .proposal_alignment import _new_result_directory
from .protocol import content_hash
from .spatial_decision_policy import ZoneDecisionPolicy
from .spatial_track_metrics import TrackingTarget, evaluate_tracking, select_tracking_recipe
from .spatial_tracking_capture import TRACKING_ROLE
from .spatial_tracking_report import write_tracking_gallery, write_tracking_report
from .spatial_vehicle_tracks import TRACK_RECIPES, replay_vehicle_tracks
from .temporal_spatial_capture import ROLES
from .temporal_spatial_filter import RECIPES
from .temporal_spatial_replay import replay_temporal


SCOPE = "openscene_camera_object_association_v1"
_GATE_FILE = "development-gate.json"
_BASELINE = "zone"
_MATCH_IOU = 0.5


@dataclass(frozen=True)
class TrackingTruthFile:
    path: Path
    sha256: str

    def __post_init__(self):
        require_hex(self.sha256, 64, "truth sidecar byte hash")


def run_tracking_benchmark(
    previous, final_pack, current, truth_files, output_root, *,
    revision, tracking_capture_sha256, protected_roots=(),
):
    require_hex(revision, 40, "tracking benchmark revision")
    require_hex(tracking_capture_sha256, 64, "tracking capture byte hash")
    if set(truth_files) != {*ROLES, TRACKING_ROLE} or any(
        not isinstance(v, TrackingTruthFile) for v in truth_files.values()
    ):
        raise ValueError("Tracking benchmark needs bound truth from all video roles")
    if (
        current.role != TRACKING_ROLE or current.pack_hash != final_pack.manifest_hash
        or current.model_hash != previous.manifest["model_hash"] or len(current.frames) != len(final_pack.frames)
    ):
        raise ValueError("Tracking observations must be bound to the same camera model and final pack")
    root = _new_result_directory(output_root, (
        final_pack.root, *(pack.root for pack in previous.packs.values()),
        *(ref.path for ref in truth_files.values()), *protected_roots,
    ))
    policy = ZoneDecisionPolicy(**previous.manifest["model_receipt"]["calibrated_policy"])
    protocol = {
        "model_hash": previous.manifest["model_hash"], "video_capture_sha256": previous.manifest_sha256,
        "tracking_capture_sha256": tracking_capture_sha256,
        "pack_hashes": {
            **{role: p.manifest_hash for role, p in previous.packs.items()},
            TRACKING_ROLE: final_pack.manifest_hash,
        },
        "truth_hashes": {role: f.sha256 for role, f in truth_files.items()},
        "camera_policy": asdict(policy), "recipes": {name: asdict(v) for name, v in TRACK_RECIPES.items()},
        "fixed_frame_match_min_iou": _MATCH_IOU, "baseline": _BASELINE,
        "prior_video_holdout_reused_for_development": True,
        "geometry_changed": False, "driving_claim": False,
    }
    result = {"scope": SCOPE, "status": "running", "revision": revision, "protocol": protocol}
    write_json(root / "result.json", result)
    try:
        _compare(root, previous, final_pack, current, truth_files, policy, result)
        result["status"] = "passed"
    except KeyboardInterrupt:
        result.update(status="interrupted", error="Interrupted by operator")
    except Exception as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    try:
        write_tracking_report(root, result)
    except Exception as exc:
        result["report_error"] = f"{type(exc).__name__}: {exc}"
        if result["status"] == "passed":
            result.update(status="failed", error="Could not save tracking report")
    path = root / "result.json"
    write_json(path, result)
    return root, result, hashlib.sha256(read_bounded(path, 2 * 1024 * 1024)).hexdigest()


def _compare(root, previous, final_pack, current, truth_files, policy, result):
    development, dev_truth = [], {}
    for role in ROLES:
        targets = _targets(previous.packs[role], truth_files[role])
        if set(dev_truth) & set(targets):
            raise ValueError("Development roles have duplicate source frames")
        dev_truth.update(targets)
        development.extend(previous.traces[role].frames)
    dev_base = replay_temporal(development, policy, RECIPES["framewise"]).decisions
    dev_metric, _ = _evaluate(dev_base, dev_truth)
    dev_selection = select_tracking_recipe(dev_metric, baseline_name=_BASELINE)
    gate = {
        "scope": SCOPE, "protocol_hash": content_hash(result["protocol"]),
        "development_hash": content_hash(dev_metric), "selection": dev_selection,
    }
    path = root / _GATE_FILE
    write_json(path, gate)
    frozen = decode_json(read_bounded(path, 128 * 1024))
    if content_hash(frozen) != content_hash(gate):
        raise ValueError("Saved dev identity selection did not match the frozen gate")
    selected = frozen["selection"]["name"]
    if selected not in TRACK_RECIPES:
        raise ValueError("Frozen selection contains an unknown tracking recipe")
    result["development_gate_hash"] = content_hash(frozen)

    # This is the first read of final track IDs or 2D target boxes. The
    # acceptance check below compares only the preselected recipe, not alternatives.
    final_truth = _targets(final_pack, truth_files[TRACKING_ROLE])
    if set(dev_truth) & set(final_truth):
        raise ValueError("Final benchmark frames overlap development")
    final_base = replay_temporal(current.frames, policy, RECIPES["framewise"]).decisions
    final_metric, predictions = _evaluate(final_base, final_truth)
    matrix = {
        "development": dev_metric, "holdout": final_metric,
        "selection": dev_selection,
        "holdout_diagnostic": _holdout_check(selected, final_metric),
        "configurations": len(dev_metric) + len(final_metric),
        "development_frames": len(development), "holdout_frames": len(final_base),
        "inference_reused_development": True,
        "identical_input_predictions_in_each_arm": True,
    }
    result.update(matrix=matrix, matrix_hash=content_hash(matrix))
    gallery = root / "tracking.html"
    write_tracking_gallery(gallery, final_pack, final_truth, predictions, selected=selected)
    result["gallery"] = gallery.name


def _targets(pack, source):
    aliases = read_track_truth(pack, source.path, expected_sha256=source.sha256)
    rows = read_spatial_targets(pack.root, pack.manifest, "holdout")
    if tuple(row["sample_id"] for row in rows) != tuple(frame.sample_id for frame in pack.frames):
        raise ValueError("Target rows differ from clip order")
    if set(aliases) != {row["sample_id"] for row in rows}:
        raise ValueError("Each source frame must have tracking labels")
    return {
        row["sample_id"]: TrackingTarget(aliases[row["sample_id"]], row["boxes_2d"])
        for row in rows
    }


def _evaluate(fixed, truth):
    snapshots, outcomes = {}, {}
    expected = None
    for name, recipe in TRACK_RECIPES.items():
        linked = replay_vehicle_tracks(fixed, recipe)
        if any(row.decision is not before for row, before in zip(linked, fixed)) or len(linked) != len(fixed):
            raise ValueError("Tracking must not replace, alter or omit an input detection")
        stats = evaluate_tracking(linked, truth, min_iou=_MATCH_IOU)
        comparable = tuple(stats[k] for k in (
            "evaluation_identity", "frames", "output_boxes", "labeled_boxes", "detections_matched",
            "adjacent_links", "cross_zone_links",
        ))
        if expected is None:
            expected = comparable
        elif expected != comparable:
            raise ValueError("All tracking arms must share the same frame matches and raw predictions")
        snapshots[name], outcomes[name] = stats, linked
    return snapshots, outcomes


def _holdout_check(selected, metrics):
    if selected == _BASELINE:
        return {"evaluated_recipe": selected, "outcome": "baseline_retained_on_dev", "reasons": []}
    assessment = select_tracking_recipe(
        {_BASELINE: metrics[_BASELINE], selected: metrics[selected]}, baseline_name=_BASELINE,
    )
    reasons = assessment["candidates"][selected]["reasons"]
    return {
        "evaluated_recipe": selected,
        "outcome": "preselected_passed" if assessment["name"] == selected else "no_replacement",
        "reasons": reasons,
    }
