"""Score the independent balanced development set without any control artifacts."""

from hybrid_drive.experiments.io import write_json
from hybrid_drive.experiments.vru_chain_capture import (
    ensure_new_output, index_sha, sealed_read, validate_anchor,
)
from hybrid_drive.experiments.vru_clip_control import _candidate_rows
from hybrid_drive.experiments.vru_clip_development_data import load_development_inputs, load_development_labels
from hybrid_drive.experiments.vru_clip_metrics import summarize_vru_clips
from hybrid_drive.experiments.vru_clip_replay import (
    DEVELOPMENT_SCORE, MODEL_ARMS, _bind_capture, _bound_depthpro, _validate_depthpro_cameras, replay_clip,
)


SPLIT = "balanced-development"


def score_balanced_development(root, candidate_path, labels_path, output, *,
                               candidate_sha256, depthpro_path=None, depthpro_sha256=None):
    ensure_new_output(output)
    candidate = sealed_read(candidate_path, candidate_sha256)
    index, digest = load_development_inputs(root), index_sha(root)
    _bind_capture(candidate, digest, SPLIT)
    depth = _bound_depthpro(depthpro_path, depthpro_sha256, candidate_sha256, digest, SPLIT, len(index["anchors"]))
    arms = (*MODEL_ARMS, "dfine_calibrated") + (("dfine_depthpro",) if depthpro_path is not None else ())
    forecasts = {arm: [] for arm in arms}
    for anchor, window, apple in zip(index["anchors"], candidate["observations"], depth, strict=True):
        validate_anchor(anchor, window)
        if apple is not None:
            validate_anchor(anchor, apple)
            _validate_depthpro_cameras(window, apple)
        for arm in arms:
            forecasts[arm].append(replay_clip(anchor, index, _candidate_rows(arm, window, apple)))
    labels = load_development_labels(root, labels_path)["rows"]
    scores = {arm: summarize_vru_clips(rows, labels, expected_count=60, required_pass=40)
              for arm, rows in forecasts.items()}
    output.mkdir(parents=True)
    write_json(output/"score.json", {"schema": DEVELOPMENT_SCORE, "input_sha256": digest,
               "candidate_sha256": candidate_sha256, "candidate_profile": candidate["profile"],
               "depthpro_sha256": depthpro_sha256, "purpose": "balanced_development_only",
               "scores": scores})
    return {name: score["counts"] for name, score in scores.items()}
