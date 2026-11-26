"""Freeze input-only detector observations so new chain policies need no inference."""

from dataclasses import asdict
from hashlib import sha256

from hybrid_drive.experiments.bounded_io import canonical_json, decode_json, read_bounded, require_hex
from hybrid_drive.experiments.box_motion_inputs import load_motion_snapshot
from hybrid_drive.experiments.io import write_json
from hybrid_drive.experiments.object_vru_geometry import GroundHit, GroundObject
from hybrid_drive.experiments.vru_chain_rescue import SINGLES
from hybrid_drive.experiments.vru_fusion_holdout_data import load_fusion_inputs


CAPTURE_SCHEMA = "hybrid-drive-vru-chain-capture-v1"
TIMES = ("current", "previous", "older")
TOKENS = ("sample_token", "previous_token", "older_token")
_MAX_ARTIFACT_BYTES = 32 * 1024**2


def capture_chain_observations(input_root, ssd_weights, faster_weights, output, *, revision):
    from hybrid_drive.experiments.object_vru_detector import LocalVruDetector, RECTIFIED_WH
    from hybrid_drive.experiments.vru_coverage_detector import CoverageDetector

    require_hex(revision, 40, "detector capture revision")
    ensure_new_output(output)
    index = load_fusion_inputs(input_root)
    models = dict(zip(SINGLES, (LocalVruDetector(ssd_weights), CoverageDetector(faster_weights)), strict=True))

    def observe(anchor):
        return _observe(input_root, index, anchor, models, RECTIFIED_WH)

    windows = []
    for anchor in index["anchors"]:
        windows.append(observe(anchor))
        if len(windows) % 8 == 0:
            print(f"Captured VRU windows: {len(windows)}/{len(index['anchors'])}", flush=True)
    rechecks = []
    for i in repeat_indices(windows):
        if canonical_json(observe(index["anchors"][i])) != canonical_json(windows[i]):
            raise ValueError("Detector capture did not repeat on identical camera inputs")
        rechecks.append(recheck(windows[i], i))
    seal = save_sealed(output, "observations.json", {
        "schema": CAPTURE_SCHEMA, "revision": revision, "input_sha256": index_sha(input_root),
        "labels_mounted": False, "windows": windows, "rechecks": rechecks,
        "detectors": {name: model.provenance for name, model in models.items()},
    })
    return {"windows": len(windows), "rechecked": len(rechecks), "capture_sha256": seal}


def _observe(root, index, anchor, models, size):
    result = {}
    for phase, key in zip(TIMES, TOKENS, strict=True):
        token = anchor[key]
        if token is None:
            result[phase] = None
            continue
        shot = load_motion_snapshot(root, token, index["frames"][token], rectified_wh=size)
        result[phase] = {"sha256": shot.content_sha256,
                         "objects": {name: [asdict(obj) for obj in detector(shot)[0]]
                                     for name, detector in models.items()}}
    return {**anchor, "observations": result}


def load_chain_observations(index, digest, path, seal):
    from hybrid_drive.experiments.object_vru_detector import WEIGHTS_SHA256 as SSD
    from hybrid_drive.experiments.vru_coverage_detector import WEIGHTS_SHA256 as FASTER

    capture = sealed_read(path, seal)
    if (capture["schema"] != CAPTURE_SCHEMA or capture["input_sha256"] != digest
            or capture["labels_mounted"] is not False or set(capture["detectors"]) != set(SINGLES)
            or capture["detectors"]["ssdlite"]["weights_sha256"] != SSD
            or capture["detectors"]["faster_rcnn"]["weights_sha256"] != FASTER):
        raise ValueError("Expected the sealed capture from the pinned detector profiles")
    require_hex(capture["revision"], 40, "detector capture revision")
    windows = capture["windows"]
    if capture["rechecks"] != [recheck(windows[i], i) for i in repeat_indices(windows)]:
        raise ValueError("Capture rechecks do not match the scored windows")
    for anchor, row in zip(index["anchors"], windows, strict=True):
        validate_anchor(anchor, row)
        observations = row["observations"]
        if set(observations) != set(TIMES):
            raise ValueError("Expected all three observation slots")
        for phase, key in zip(TIMES, TOKENS, strict=True):
            item = observations[phase]
            if (anchor[key] is None) != (item is None):
                raise ValueError("Detector capture changed the availability of a camera bundle")
            if item is not None:
                require_hex(item["sha256"], 64, "camera observation")
                if set(item["objects"]) != set(SINGLES):
                    raise ValueError("Detector capture is missing a source profile")
    return capture


def read_ground_objects(rows):
    return tuple(GroundObject(**{**row, "hits": tuple(GroundHit(**hit) for hit in row["hits"])}) for row in rows)


def validate_anchor(anchor, row):
    if any(row[key] != value for key, value in anchor.items()):
        raise ValueError("Artifact observations differ from the fixed input window")


def repeat_indices(windows):
    if len(windows) < 3:
        raise ValueError("Capture requires at least three windows")
    return 0, (len(windows) - 1) // 2, len(windows) - 1


def recheck(row, i):
    return {"index": i, "sha256": sha256(canonical_json(row)).hexdigest()}


def index_sha(root):
    return sha256(read_bounded(root / "index.json", 16 * 1024**2)).hexdigest()


def sealed_read(path, seal):
    require_hex(seal, 64, "artifact seal")
    raw = read_bounded(path, _MAX_ARTIFACT_BYTES)
    if sha256(raw).hexdigest() != seal:
        raise ValueError("Artifact does not match the SHA fixed before evaluation")
    return decode_json(raw)


def ensure_new_output(path):
    if path.exists() or path.is_symlink():
        raise ValueError("Expected a new artifact output directory")


def save_sealed(output, name, data):
    output.mkdir(parents=True)
    path = output / name
    write_json(path, data)
    seal = sha256(read_bounded(path, _MAX_ARTIFACT_BYTES)).hexdigest()
    write_json(output / "seal.json", {"sha256": seal})
    return seal
