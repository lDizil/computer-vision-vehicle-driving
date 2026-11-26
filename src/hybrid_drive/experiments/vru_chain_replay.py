"""Compare two chain repairs on frozen cameras without changing current coverage."""

from dataclasses import asdict
from hashlib import sha256

from hybrid_drive.experiments.bounded_io import read_bounded, require_hex
from hybrid_drive.experiments.io import write_json
from hybrid_drive.experiments.object_vru_geometry import align_ground_objects
from hybrid_drive.experiments.vru_chain_capture import (
    TIMES, TOKENS, ensure_new_output, index_sha, load_chain_observations,
    read_ground_objects, save_sealed, sealed_read, validate_anchor,
)
from hybrid_drive.experiments.vru_chain_rescue import (
    ARMS, EARLY, MAX_BRIDGE_APPEARANCE_L1, MAX_BRIDGE_DISTANCE_M, MAX_VELOCITY_DELTA_MPS, RESCUES,
    SINGLES, pair_velocity, rescue_chains, track_chains,
)
from hybrid_drive.experiments.vru_fusion_holdout_data import LABEL_LIMIT, load_fusion_inputs, load_fusion_labels
from hybrid_drive.experiments.vru_observation_fusion import profile_vru_observations
from hybrid_drive.geometry.rigid import invert_pose, rigid_transform


PREDICTION_SCHEMA = "hybrid-drive-vru-chain-predictions-v1"
SCORE_SCHEMA = "hybrid-drive-vru-chain-score-v1"
POLICY = {
    "current_centers": "unchanged_early_fusion",
    "eligible": "early_no_older_link",
    "native": "exact_origin_and_previous_link",
    "bridge": "unique_suppressed_same_class_and_corroborating_camera",
    "max_bridge_distance_m": MAX_BRIDGE_DISTANCE_M,
    "max_bridge_appearance_l1": MAX_BRIDGE_APPEARANCE_L1,
    "max_recent_velocity_delta_mps": MAX_VELOCITY_DELTA_MPS,
}


def forecast_chain_replay(input_root, capture_path, output, *, capture_sha256, revision):
    require_hex(revision, 40, "chain forecast revision")
    ensure_new_output(output)
    index, input_sha = load_fusion_inputs(input_root), index_sha(input_root)
    capture = load_chain_observations(index, input_sha, capture_path, capture_sha256)
    windows = [_window(index, row) for row in capture["windows"]]
    seal = save_sealed(output, "predictions.json", {
        "schema": PREDICTION_SCHEMA, "revision": revision, "capture_revision": capture["revision"],
        "capture_sha256": capture_sha256, "input_sha256": input_sha, "labels_mounted": False, "policy": POLICY,
        "arms": {name: [window[name] for window in windows] for name in ARMS},
    })
    return {"windows": len(windows), "prediction_sha256": seal}


def _window(index, row):
    observations = row["observations"]
    frames = {phase: index["frames"][row[key]] if row[key] is not None else None
              for phase, key in zip(TIMES, TOKENS, strict=True)}
    times = {phase: frame["ref_time_us"] if frame is not None else None for phase, frame in frames.items()}
    inverse = invert_pose(frames["current"]["ego_to_world"])
    transforms = {phase: inverse @ rigid_transform(frame["ego_to_world"], "observed ego pose")
                  for phase, frame in frames.items() if phase != "current" and frame is not None}
    profiles = {phase: profile_vru_observations(*(read_ground_objects(obs["objects"][name]) for name in SINGLES))
                if obs is not None else None for phase, obs in observations.items()}
    dt = (times["current"] - times["previous"]) / 1e6
    older_dt = None if times["older"] is None else (times["previous"] - times["older"]) / 1e6
    header = {**{key: value for key, value in row.items() if key != "observations"}, "ref_time_us": times["current"],
              "history_s": dt, "older_history_s": older_dt,
              **{phase + "_sha256": item["sha256"] if item is not None else None
                 for phase, item in observations.items()}}
    histories, chains = {}, {}
    for name in (*SINGLES, EARLY):
        role = "combined" if name == EARLY else name
        current = getattr(profiles["current"], role)
        past, old = (align_ground_objects(getattr(profiles[phase], role), transforms[phase])
                     if profiles[phase] is not None else None for phase in TIMES[1:])
        histories[name] = (current, past, old)
        chains[name] = track_chains(name, current, past, old, dt, older_dt)
    exact, bridged = rescue_chains(profiles["current"], chains[EARLY], *(chains[name] for name in SINGLES))
    return {
        **{name: _profile_row(header, histories[name], chains[name], chains[name]) for name in (*SINGLES, EARLY)},
        **{name: _profile_row(header, histories[EARLY], selected, chains[EARLY])
           for name, selected in zip(RESCUES, (exact, bridged), strict=True)},
    }


def _profile_row(header, frames, selected, original):
    current, previous, older = frames
    return {**header, "history": {"previous": [_projected(obj) for obj in previous],
                                  "older": None if older is None else [_projected(obj) for obj in older]},
            "objects": [_forecast(obj, selected[obj.object_id], original[obj.object_id]) for obj in current]}


def _projected(obj):
    return {"object_id": obj.object_id, "kind": obj.kind, "xy_m": obj.xy, "confidence": obj.confidence,
            "cameras": [hit.camera_name for hit in obj.hits]}


def _forecast(obj, selected, baseline):
    pair = pair_velocity(baseline)
    confirmation = selected.confirmation
    return {
        **_projected(obj),
        "motions": {"appearance": asdict(baseline.initial), "geometry": asdict(baseline.geometry)},
        "confirmation": asdict(confirmation),
        "forecasts": {"stationary": (0., 0.), "two_frame": pair,
                      "confirmed_two_frame": pair_velocity(selected) if confirmation.status == "confirmed" else (0., 0.),
                      "three_frame_trend": confirmation.trend_velocity_xy_mps},
        "trace": {"origin_arm": selected.origin_arm, "origin_object_id": selected.origin_object_id,
                  "initial": _link(selected.initial), "older": _link(selected.older), "rescue": selected.rescue},
    }


def _link(motion):
    return None if motion is None else {"status": motion.status, "previous_object_id": motion.previous_object_id}


def score_chain_replay(input_root, prediction_path, labels_path, truth_path, output, *, prediction_sha256):
    from hybrid_drive.experiments.vru_chain_metrics import summarize_chain_replay
    from hybrid_drive.experiments.vru_chain_truth import load_vru_chain_truth
    from hybrid_drive.experiments.vru_fusion_holdout_metrics import summarize_fusion_holdout

    ensure_new_output(output)
    pred = sealed_read(prediction_path, prediction_sha256)
    index = load_fusion_inputs(input_root)
    if (pred["schema"] != PREDICTION_SCHEMA or pred["input_sha256"] != index_sha(input_root)
            or pred["labels_mounted"] is not False or pred["policy"] != POLICY or set(pred["arms"]) != set(ARMS)):
        raise ValueError("Expected an input-only sealed forecast for the fixed experiment")
    require_hex(pred["revision"], 40, "chain forecast revision")
    require_hex(pred["capture_revision"], 40, "detector capture revision")
    require_hex(pred["capture_sha256"], 64, "detector capture seal")
    arms = pred["arms"]
    for rows in arms.values():
        for anchor, row in zip(index["anchors"], rows, strict=True):
            validate_anchor(anchor, row)
    labels = load_fusion_labels(input_root, labels_path)["rows"]
    truth = load_vru_chain_truth(input_root, labels_path, truth_path)
    summary = summarize_chain_replay(arms, labels, truth["rows"])
    baseline = summarize_fusion_holdout({name: arms[name] for name in (*SINGLES, EARLY)}, labels)
    output.mkdir(parents=True)
    write_json(output / "score.json", {
        "schema": SCORE_SCHEMA, "revision": pred["revision"], "capture_revision": pred["capture_revision"],
        "input_sha256": pred["input_sha256"], "capture_sha256": pred["capture_sha256"],
        "prediction_sha256": prediction_sha256, "labels_sha256": truth["labels_sha256"],
        "history_truth_sha256": sha256(read_bounded(truth_path, LABEL_LIMIT)).hexdigest(),
        "baseline": baseline, "summary": summary,
    })
    return {"windows": len(labels), "prediction_sha256": prediction_sha256}
