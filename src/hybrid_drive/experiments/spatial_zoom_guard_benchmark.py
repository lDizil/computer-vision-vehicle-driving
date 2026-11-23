"""Cross-fit a conservative box guard, then evaluate once on new recordings."""

from dataclasses import dataclass
import hashlib
from pathlib import Path

import torch

from hybrid_drive.neural.spatial_zoom_guard import ZoomGuard, apply_zoom_guard, fit_zoom_guard, mix_zoom_boxes, zoom_guard_protocol
from hybrid_drive.neural.spatial_zoom_refiner import ZoomBoxRefiner, fit_zoom_refiner, zoom_head_protocol
from .bounded_io import decode_json, read_bounded, require_hex
from .io import write_json
from .neural_checkpoint import load_neural_weights, module_state_hash, save_neural_weights
from .proposal_alignment import _new_result_directory
from .protocol import content_hash
from .spatial_decision_policy import PRESENT
from .spatial_tracking_benchmark import TrackingTruthFile, _targets
from .spatial_zoom_capture import ZOOM_ROLES
from .spatial_zoom_guard_capture import FRESH_ROLE
from .spatial_zoom_guard_metrics import (
    GUARD_MODES, check_guard_holdout, evaluate_guard_changes, oracle_guard_match_ceiling, select_guard_mode,
)
from .spatial_zoom_guard_report import write_guard_gallery, write_guard_report
from .spatial_zoom_metrics import apply_zoom_boxes, evaluate_zoom_boxes


SCOPE = "openscene_box_guard_crossfit_fresh_holdout_v1"
FOLD_ROLES = ("development", "stress", "holdout")
DEV_ROLES = ("tracking_holdout", "appearance_holdout")
TRUTH_ROLES = (*ZOOM_ROLES, FRESH_ROLE)
SEED = 0
EPOCHS = 8
BATCH = 16


@dataclass(frozen=True)
class _Rows:
    decisions: tuple
    targets: dict
    boxes: torch.Tensor
    scores: torch.Tensor
    active: torch.Tensor
    windows: torch.Tensor
    features: torch.Tensor

    @classmethod
    def of(cls, decisions, grids, targets):
        return cls(
            decisions, targets, torch.tensor([d.boxes for d in decisions], dtype=torch.float32),
            torch.tensor([d.scores for d in decisions], dtype=torch.float32),
            torch.tensor([[s == PRESENT for s in d.states] for d in decisions], dtype=torch.bool),
            grids.windows, grids.zoom_features,
        )


def run_guarded_zoom(inputs, fresh, truth, output_root, *, revision, extra_inputs=()):
    require_hex(revision, 40, "guard benchmark revision")
    if set(truth) != set(TRUTH_ROLES) or any(not isinstance(x, TrackingTruthFile) for x in truth.values()):
        raise ValueError("Guarded zoom needs six bound target sidecars")
    root = _new_result_directory(output_root, (*inputs.protected_roots, *(f.path for f in truth.values()), *extra_inputs))
    revision_data = {
        "known_capture_sha256": inputs.known.sha256, "fresh_capture_sha256": fresh.sha256,
        "source_model_hash": inputs.known.inputs.source_model_hash,
        "feature_protocol": inputs.known.manifest["protocol"], "feature_dim": inputs.known.manifest["feature_dim"],
        "truth_hashes": {role: ref.sha256 for role, ref in truth.items()},
        "fold_roles": list(FOLD_ROLES), "dev_roles": list(DEV_ROLES), "fresh_role": FRESH_ROLE,
        "fold_policy": "hold_out_one_whole_two-recording_pack",
        "seed": SEED, "epochs": EPOCHS, "batch_size": BATCH,
        "refiner": zoom_head_protocol(), "guard": zoom_guard_protocol(), "modes": list(GUARD_MODES),
        "visible_decisions_changed": False, "association_changed": False, "physical_control": False,
    }
    result = {"scope":SCOPE, "status":"running", "revision":revision, "protocol":revision_data}
    write_json(root / "result.json", result)
    try:
        _compare(root, inputs, fresh, truth, result)
        result["status"] = "passed"
    except KeyboardInterrupt:
        result.update(status="interrupted", error="Interrupted by operator")
    except Exception as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    try:
        write_guard_report(root, result)
    except Exception as exc:
        result["report_error"] = f"{type(exc).__name__}: {exc}"
        if result["status"] == "passed":
            result.update(status="failed", error="Could not save guarded box report")
    path = root / "result.json"
    write_json(path, result)
    return root, result, hashlib.sha256(read_bounded(path, 4 * 1024 * 1024)).hexdigest()


def _compare(root, inputs, fresh, truth, result):
    known = inputs.known
    train = {role: _load_known(known, truth, role) for role in FOLD_ROLES}
    _check_group_boundary(train.values())
    joined = _join(tuple(train.values()))
    oof = {}
    oof_proposals = []
    for held in FOLD_ROLES:
        fit_roles = tuple(r for r in FOLD_ROLES if r != held)
        portion = _join(tuple(train[r] for r in fit_roles))
        print(f"Cross-fit for {held} on {len(portion.decisions)} other frames", flush=True)
        model, info = _fit(root, f"oof-{held}", portion, fit_roles, result)
        proposal = _propose(model, train[held])
        oof_proposals.append(proposal)
        oof[held] = {
            **info, "held_role": held, "held_frames": len(train[held].decisions),
            "held_prediction_hash": content_hash(proposal.tolist()),
        }
    annotated, presence = _targets_as_tensors(joined)
    guard = fit_zoom_guard(joined.boxes, torch.cat(oof_proposals), joined.active, annotated, presence, joined.scores)
    state = guard.to_dict()
    guard_path = root / "zoom-guard.json"
    write_json(guard_path, state)
    guard_raw = read_bounded(guard_path, 128 * 1024)
    guard_sha = hashlib.sha256(guard_raw).hexdigest()
    if content_hash(decode_json(guard_raw)) != content_hash(state):
        raise ValueError("Saved guard differs from the fitted artifact")
    final_model, final = _fit(root, "full-zoom", joined, FOLD_ROLES, result)

    development = {role: _load_known(known, truth, role) for role in DEV_ROLES}
    _check_group_boundary((*train.values(), *development.values()))
    dev_scores = {role: _score(rows, final_model, guard, track=False)[0] for role, rows in development.items()}
    selection = select_guard_mode(dev_scores)
    gate = {
        "scope": SCOPE, "protocol_hash": content_hash(result["protocol"]),
        "development_hash": content_hash(dev_scores), "selection": selection, "guard_sha256": guard_sha,
        "final_model_sha256": final["checkpoint"]["sha256"], "final_model_state": final["state_hash"],
        "folds": {role: {"trained_roles": item["trained_roles"], "held_prediction_hash": item["held_prediction_hash"],
                          "checkpoint_sha256": item["checkpoint"]["sha256"]} for role, item in oof.items()},
    }
    gate_path = root / "development-gate.json"
    write_json(gate_path, gate)
    frozen = decode_json(read_bounded(gate_path, 256 * 1024))
    if content_hash(frozen) != content_hash(gate):
        raise ValueError("Saved selection differs from the frozen development gate")
    guard_raw = read_bounded(guard_path, 128 * 1024)
    if hashlib.sha256(guard_raw).hexdigest() != frozen["guard_sha256"]:
        raise ValueError("Saved guard byte hash differs")
    guard = ZoomGuard.from_dict(decode_json(guard_raw))
    loaded = ZoomBoxRefiner(known.manifest["feature_dim"]).eval()
    load_neural_weights(
        root / "full-zoom.safetensors", {"zoom":loaded}, expected_sha256=frozen["final_model_sha256"],
        expected_metadata=final["metadata"],
    )
    if module_state_hash(loaded) != frozen["final_model_state"]:
        raise ValueError("Restored proposal model differs from the frozen state")
    result.update(development_gate_hash=content_hash(frozen), training={"folds":oof,"final":final,"guard":state})

    # Only now may the blind sidecar or its ground-truth boxes be opened.
    actual_targets = _targets(inputs.fresh_pack, truth[FRESH_ROLE])
    holdout = _Rows.of(fresh.decisions, fresh.grids, actual_targets)
    _check_group_boundary((*train.values(),*development.values(),holdout))
    scores, forecasts = _score(holdout, loaded, guard, track=True)
    chosen = frozen["selection"]["name"]
    diagnostic = check_guard_holdout(chosen, scores)
    oracle = oracle_guard_match_ceiling(
        forecasts["baseline"], forecasts["zoom_half"], forecasts["zoom_full"], holdout.targets,
    )
    matrix = {
        "development":dev_scores, "holdout":scores, "selection":frozen["selection"],
        "holdout_diagnostic":diagnostic,"oracle":oracle,
        "frame_counts":{"oof":len(joined.decisions),
                        "development":{k:len(d.decisions) for k,d in development.items()}, "holdout":len(holdout.decisions)},
        "holdout_previously_seen":False, "detector_and_visibility_frozen":True,
        "tracking_is_secondary":True,
    }
    result.update(matrix=matrix, matrix_hash=content_hash(matrix))
    gallery = root / "guard.html"
    write_guard_gallery(gallery, inputs.fresh_pack, holdout.targets, forecasts, selected=chosen)
    result["gallery"] = gallery.name


def _load_known(known, truth, role):
    return _Rows.of(known.inputs.decisions[role], known.grids[role], _targets(known.inputs.packs[role], truth[role]))


def _check_group_boundary(groups):
    recordings, samples = set(), set()
    for part in groups:
        ids = {f.observation.sample_id for f in part.decisions}
        rec = {f.observation.recording_id for f in part.decisions}
        if len(ids) != len(part.decisions) or ids != set(part.targets) or samples & ids or recordings & rec:
            raise ValueError("Frames or recordings overlap or do not match their target pack")
        recordings.update(rec)
        samples.update(ids)


def _join(parts):
    _check_group_boundary(parts)
    return _Rows(
        decisions=tuple(x for p in parts for x in p.decisions),
        targets={k:v for p in parts for k,v in p.targets.items()},
        **{name: torch.cat([getattr(p, name) for p in parts], dim=0)
           for name in ("boxes","scores","active","windows","features")},
    )


def _fit(root, name, rows, roles, result):
    boxes, presence = _targets_as_tensors(rows)
    model, trace = fit_zoom_refiner(
        rows.boxes, rows.windows, rows.features, rows.active, boxes, presence,
        seed=SEED, epochs=EPOCHS, batch_size=BATCH,
    )
    metadata = {
        "scope":SCOPE,"name":name,"revision":result["revision"],"protocol_hash":content_hash(result["protocol"]),
        "trained_roles":list(roles),
    }
    stored = save_neural_weights(root / f"{name}.safetensors", {"zoom":model}, metadata=metadata)
    return model, {
        "metadata":metadata,"checkpoint":stored,"state_hash":module_state_hash(model),
        "trained_roles":list(roles),"train_frames":len(rows.decisions),"trace":trace,
    }


def _targets_as_tensors(data):
    absent = (float("nan"),) * 4
    labels = [data.targets[d.observation.sample_id] for d in data.decisions]
    boxes = torch.tensor([[absent if b is None else b for b in row.boxes] for row in labels], dtype=torch.float32)
    present = torch.tensor([[key is not None for key in row.track_ids] for row in labels], dtype=torch.bool)
    return boxes,present


@torch.no_grad()
def _propose(model, data):
    return model(data.boxes, data.windows, data.features, data.active)


@torch.no_grad()
def _score(data, model, guard, *, track):
    proposal = _propose(model, data)
    half = mix_zoom_boxes(data.boxes, proposal, data.active, .5)
    gated, fractions = apply_zoom_guard(guard, data.boxes, proposal, data.active, data.scores)
    support, supported = apply_zoom_guard(guard, data.boxes, proposal, data.active, data.scores, require_support=True)
    forecasts = {
        "baseline":data.decisions, "zoom_full":apply_zoom_boxes(data.decisions, proposal),
        "zoom_half":apply_zoom_boxes(data.decisions, half), "gate":apply_zoom_boxes(data.decisions, gated),
        "gate_supported":apply_zoom_boxes(data.decisions, support),
    }
    if set(forecasts) != set(GUARD_MODES):
        raise ValueError("Unexpected forecast families")
    scores = {}
    for name, frames in forecasts.items():
        m = evaluate_zoom_boxes(frames, data.targets, include_tracking=track)
        m["changes"] = evaluate_guard_changes(data.decisions, frames, data.targets)
        if name in ("gate","gate_supported"):
            choices = fractions if name == "gate" else supported
            vals = choices[data.active]
            m["guard_actions"] = {key:int((vals == value).sum()) for key,value in (("keep",0.),("half",.5),("full",1.))}
        scores[name] = m
    return scores,forecasts
