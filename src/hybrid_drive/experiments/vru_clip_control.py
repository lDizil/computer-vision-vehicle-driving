"""Run and score a candidate selected without the fixed control answers."""

from dataclasses import asdict
from math import ceil

from hybrid_drive.experiments.bounded_io import require_hex
from hybrid_drive.experiments.io import write_json
from hybrid_drive.experiments.object_vru_geometry import fuse_ground_hits
from hybrid_drive.experiments.vru_chain_capture import (
    ensure_new_output, index_sha, read_ground_objects, save_sealed, sealed_read, validate_anchor,
)
from hybrid_drive.experiments.vru_clip_data import load_clip_inputs, load_clip_labels
from hybrid_drive.experiments.vru_clip_metrics import summarize_vru_clips
from hybrid_drive.experiments.vru_clip_replay import (
    DEVELOPMENT_SCORE, FITTED_PLANE_Z_M, FRAMES, FRAME_SIZE, MODEL_ARMS, TOKENS,
    _bind_capture, _bound_depthpro, _validate_comparable_cameras, _validate_depthpro_cameras, replay_clip,
)
from hybrid_drive.experiments.vru_observation_fusion import fuse_vru_observations


PLAN_SHA256 = "5d8f9954e5185ea59e612cb323b255bb2aa70a07f9077da9cd410360c10faacc"
CONTROL_CLIPS = 120
REQUIRED_PASS = 80
BASELINE_CAPTURE = "hybrid-drive-vru-120-baseline-v1"
CONTROL_RELEASE = "hybrid-drive-vru-120-release-v1"
CONTROL_SCORE = "hybrid-drive-vru-120-score-v1"
ARMS = (*MODEL_ARMS, "dfine_calibrated", "dfine_depthpro")


def control_index(root):
    index = load_clip_inputs(root)
    if index["plan_sha256"] != PLAN_SHA256 or len(index["anchors"]) != CONTROL_CLIPS:
        raise ValueError("Expected the precommitted 120-clip plan, not an equivalent-looking selection")
    return index


def capture_baseline(root, ssd_weights, faster_weights, output, *, revision):
    from hybrid_drive.experiments.box_motion_inputs import load_motion_snapshot
    from hybrid_drive.experiments.object_vru_detector import LocalVruDetector
    from hybrid_drive.experiments.vru_coverage_detector import CoverageDetector

    ensure_new_output(output)
    require_hex(revision, 40, "clip baseline revision")
    index = control_index(root)
    ssd, faster = LocalVruDetector(ssd_weights), CoverageDetector(faster_weights)
    windows = []
    for anchor in index["anchors"]:
        observations = {}
        for role, slot in zip(FRAMES, TOKENS, strict=True):
            token = anchor[slot]
            shot = load_motion_snapshot(root, token, index["frames"][token], rectified_wh=FRAME_SIZE)
            first, _ = ssd(shot)
            second, _ = faster(shot)
            objects = fuse_vru_observations(first, second)
            observations[role] = {"sha256": shot.content_sha256, "fused_objects": [asdict(obj) for obj in objects]}
        windows.append({**anchor, "observations": observations})
        if len(windows) % 8 == 0:
            print(f"Original baseline: {len(windows)}/{CONTROL_CLIPS} clips", flush=True)
    seal = save_sealed(output, "baseline.json", {
        "schema": BASELINE_CAPTURE, "revision": revision, "input_sha256": index_sha(root), "plan_sha256": PLAN_SHA256,
        "labels_read": False, "windows": windows, "profiles": {"ssdlite": ssd.provenance, "faster_rcnn": faster.provenance},
    })
    return {"clips": len(windows), "sha256": seal}


def freeze_release(dev_path, control_root, output, *, dev_sha256, arm, revision):
    """Bind a chosen development arm and its actual evidence before control scoring."""
    ensure_new_output(output)
    require_hex(revision, 40, "clip release revision")
    development = sealed_read(dev_path, dev_sha256)
    if arm not in ARMS or development["schema"] != DEVELOPMENT_SCORE or arm not in development["scores"]:
        raise ValueError("Expected a candidate that was actually evaluated on development")
    result = development["scores"][arm]["counts"]
    count, passed = result["clips"], result["passed"]
    if type(count) is not int or type(passed) is not int or count < 30 or not ceil(count*REQUIRED_PASS/CONTROL_CLIPS) <= passed <= count:
        raise ValueError("Save the untouched control until a development candidate reaches the release bar")
    depthpro_profile = None
    if arm == "dfine_depthpro":
        from hybrid_drive.experiments.vru_depthpro2 import validate_depthpro_profile

        require_hex(development.get("depthpro_sha256"), 64, "development depth artifact SHA256")
        depthpro_profile = validate_depthpro_profile(development.get("depthpro_profile"))
    control_index(control_root)
    control_sha = index_sha(control_root)
    if control_sha == development["input_sha256"]:
        raise ValueError("Control cannot be used to select the release")
    seal = save_sealed(output, "release.json", {
        "schema": CONTROL_RELEASE, "revision": revision, "arm": arm, "plan_sha256": PLAN_SHA256,
        "input_sha256": control_sha, "development_score_sha256": dev_sha256,
        "development_clips": count, "development_passed": passed,
        "candidate_profile": development["candidate_profile"],
        "depthpro_profile": depthpro_profile,
        "threshold": {"clips": CONTROL_CLIPS, "passed": REQUIRED_PASS},
    })
    return {"arm": arm, "release_sha256": seal}


def score_control(root, baseline_path, candidate_path, release_path, labels_path, output, *,
                  baseline_sha256, candidate_sha256, release_sha256, depthpro_path=None, depthpro_sha256=None):
    ensure_new_output(output)
    release = sealed_read(release_path, release_sha256)
    baseline = sealed_read(baseline_path, baseline_sha256)
    candidate = sealed_read(candidate_path, candidate_sha256)
    index, digest = control_index(root), index_sha(root)
    _bind_capture(candidate, digest, "control")
    if (release["schema"] != CONTROL_RELEASE or release["arm"] not in ARMS
            or release["input_sha256"] != digest or release["plan_sha256"] != PLAN_SHA256
            or release["candidate_profile"] != candidate["profile"]
            or release["threshold"] != {"clips": CONTROL_CLIPS, "passed": REQUIRED_PASS}
            or baseline["schema"] != BASELINE_CAPTURE or baseline["input_sha256"] != digest
            or baseline["plan_sha256"] != PLAN_SHA256 or baseline["labels_read"] is not False):
        raise ValueError("Expected a comparison locked to the fixed control with a preselected candidate")
    require_hex(baseline["revision"], 40, "original baseline revision")
    arm = release["arm"]
    if (depthpro_path is not None) != (arm == "dfine_depthpro"):
        raise ValueError("The frozen release determines which depth artifact can be evaluated")
    depth, depthpro_profile = _bound_depthpro(depthpro_path, depthpro_sha256, candidate_sha256, digest, "control", CONTROL_CLIPS)
    if ((arm == "dfine_depthpro" and depthpro_profile is None)
            or depthpro_profile != release.get("depthpro_profile")):
        raise ValueError("Control depth checkpoint, graph and worker must match the frozen development profile")
    before, after = [], []
    for anchor, old, new, apple in zip(index["anchors"], baseline["windows"], candidate["observations"], depth, strict=True):
        validate_anchor(anchor, old)
        validate_anchor(anchor, new)
        _validate_comparable_cameras(new, old)
        if apple is not None:
            validate_anchor(anchor, apple)
            _validate_depthpro_cameras(new, apple)
        baseline_rows = {phase: read_ground_objects(old["observations"][phase]["fused_objects"]) for phase in FRAMES}
        before.append(replay_clip(anchor, index, baseline_rows))
        after.append(replay_clip(anchor, index, _candidate_rows(arm, new, apple)))
    labels = load_clip_labels(root, labels_path)["rows"]
    scores = {name: summarize_vru_clips(rows, labels) for name, rows in (("fused_baseline", before), (arm, after))}
    output.mkdir(parents=True)
    write_json(output/"score.json", {"schema": CONTROL_SCORE, "input_sha256": digest, "plan_sha256": PLAN_SHA256,
               "baseline_sha256": baseline_sha256, "candidate_sha256": candidate_sha256, "release_sha256": release_sha256,
               "depthpro_sha256": depthpro_sha256, "chosen_arm": arm, "scores": scores})
    return {name: value["counts"] for name, value in scores.items()}


def _candidate_rows(arm, row, apple):
    if arm == "dfine_depthpro":
        return {phase: read_ground_objects(apple["observations"][phase]["objects"]) for phase in FRAMES}
    if arm != "dfine_calibrated":
        return {phase: read_ground_objects(row["observations"][phase]["objects"][arm]) for phase in FRAMES}
    from hybrid_drive.experiments.vru_plane_probe import _move_hit

    return {phase: fuse_ground_hits(tuple(
        _move_hit(hit, FITTED_PLANE_Z_M)
        for obj in read_ground_objects(row["observations"][phase]["objects"]["dfine_plane"]) for hit in obj.hits
    )) for phase in FRAMES}
