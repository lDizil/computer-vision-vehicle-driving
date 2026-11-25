"""Recover one additional camera observation without changing the #50 episodes."""

from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

from hybrid_drive.assets.files import PreparationError, check_components
from hybrid_drive.contracts.sensors import timestamp_us
from hybrid_drive.experiments.bounded_io import decode_json, read_bounded, require_hex, require_keys
from hybrid_drive.experiments.box_motion_data import CAMERAS, _observation, _read_selected
from hybrid_drive.experiments.frozen_transfer_data import _validate_frame
from hybrid_drive.experiments.io import write_json
from hybrid_drive.experiments.temporal_vru_data import INDEX_LIMIT, SOURCE_SHA256, _load_source, _read_index, _read_recordings


EXTENSION_SCHEMA = "hybrid-drive-triple-vru-extension-v1"
SPLITS = ("development", "heldout")
_STATUSES = ("ready", "missing_older", "older_cadence", "missing_older_cameras")
_STEP_US = (400_000, 600_000)
_IMAGE_LIMIT = 2 * 1024**2


def prepare_third_observations(metadata_archive: Path, camera_root: Path, sources_path: Path,
                               base_root: Path, output: Path) -> dict:
    """Copy only the older JPEGs, in a new atomic observation-only extension."""
    if sys.platform != "linux":
        raise PreparationError("Third observation preparation requires the isolated Linux image")
    source = _load_source(sources_path)
    output = Path(output)
    check_components(output)
    if output.exists():
        raise PreparationError("Third observation output must not already exist")
    output.parent.mkdir(parents=True, exist_ok=True)

    base = {split: _load_base(Path(base_root) / split / "inputs", split) for split in SPLITS}
    base_tokens, base_jpegs = _claim_base_observations(base.values())
    selected_source = {**source, "logs": [row for row in source["logs"] if row["split"] in SPLITS]}
    archives = {str(spec["camera_shard"]): source["camera_archives"][str(spec["camera_shard"])]
                for spec in selected_source["logs"]}
    wanted = {shard: {} for shard in archives}
    manifests = _collect_third_observations(metadata_archive, selected_source, base, archives, wanted, base_tokens)

    with TemporaryDirectory(dir=output.parent, prefix=".triple-vru-") as temporary:
        stage = Path(temporary) / "observations"
        _extract_third_images(Path(camera_root), archives, wanted, stage, manifests, base_jpegs)
        summaries = {}
        for split, manifest in manifests.items():
            base_raw, base_index = base[split]
            _validate_extension(manifest, base_index, sha256(base_raw).hexdigest())
            root = stage / split
            root.mkdir(parents=True, exist_ok=True)
            index_path = root / "index.json"
            write_json(index_path, manifest)
            statuses = Counter(row["status"] for row in manifest["anchors"])
            summaries[split] = {"windows": len(manifest["anchors"]), "ready": len(manifest["frames"]),
                                "camera_jpegs": len(manifest["frames"]) * len(CAMERAS),
                                "statuses": {name: statuses[name] for name in _STATUSES},
                                "base_input_sha256": manifest["base_input_sha256"],
                                "input_sha256": sha256(read_bounded(index_path, INDEX_LIMIT)).hexdigest()}
        check_components(output)
        if output.exists():
            raise PreparationError("Third observation output appeared during preparation")
        stage.rename(output)
    return {**{name: sum(d[name] for d in summaries.values()) for name in ("windows", "ready", "camera_jpegs")},
            "splits": summaries}


def load_third_observations(base_input: Path, extension_root: Path) -> dict:
    """Bind an observation-only extension to the exact bytes of its base index."""
    raw, base = _load_base(base_input)
    index = decode_json(read_bounded(Path(extension_root) / "index.json", INDEX_LIMIT))
    _validate_extension(index, base, sha256(raw).hexdigest())
    return index


def _load_base(root, required_split=None):
    raw, base = _read_index(Path(root))
    if base["split"] not in SPLITS or (required_split is not None and base["split"] != required_split):
        raise ValueError("Expected the designated development or heldout observation split")
    return raw, base


def _claim_base_observations(base_indexes):
    tokens, jpegs = set(), set()
    for _, index in base_indexes:
        for token, frame in index["frames"].items():
            if token in tokens:
                raise PreparationError("Base episodes share an observation token")
            tokens.add(token)
            for view in frame["views"]:
                _claim_jpeg(jpegs, view["sha256"])
    return tokens, jpegs


def _claim_jpeg(seen, digest):
    if digest in seen:
        raise PreparationError("Third observation may not share JPEG bytes within or across episodes and splits")
    seen.add(digest)


def _collect_third_observations(metadata_archive, source, base, archives, wanted, base_tokens):
    by_recording = defaultdict(list)
    for _, index in base.values():
        for anchor in index["anchors"]:
            by_recording[anchor["recording"]].append(anchor)
    rows, observed = {}, {split: {} for split in SPLITS}
    taken = set(base_tokens)
    seen_recordings = set()
    for spec, frames in _read_recordings(metadata_archive, source):
        split, recording, shard = spec["split"], spec["recording"], str(spec["camera_shard"])
        if recording not in by_recording or recording in seen_recordings:
            raise PreparationError("Third observations must come from the exact base recordings")
        seen_recordings.add(recording)
        positions = {frame["token"]: i for i, frame in enumerate(frames)}
        base_frames = base[split][1]["frames"]
        prefix = archives[shard]["archive_prefix"]
        for anchor in by_recording[recording]:
            current_token = anchor["sample_token"]
            history_token = anchor["previous_token"]
            current_i = positions.get(current_token)
            if current_i is None or current_i < 1 or positions.get(history_token) != current_i - 1:
                raise PreparationError("Pinned recording does not contain the base pair as consecutive observations")
            history, current = frames[current_i - 1:current_i + 1]
            for frame in (history, current):
                _check_base_metadata(frame, base_frames[frame["token"]], recording, spec["log_name"], prefix)
            older = frames[current_i - 2] if current_i >= 2 else None
            status = _third_status(older, history)
            older_token = None
            if status == "ready":
                older_token = older["token"]
                if older_token in taken:
                    raise PreparationError("The older observation token is already used by another episode")
                taken.add(older_token)
                observed[split][older_token] = _observation(older, recording, spec["log_name"], prefix, wanted[shard])
            rows[current_token] = {**anchor, "older_token": older_token, "status": status}
    if seen_recordings != by_recording.keys():
        raise PreparationError("Missing a pinned base recording for third observation preparation")
    return {split: {"schema": EXTENSION_SCHEMA, "source_sha256": SOURCE_SHA256,
                    "base_input_sha256": sha256(raw).hexdigest(), "split": split,
                    "anchors": [rows[a["sample_token"]] for a in index["anchors"]], "frames": observed[split]}
            for split, (raw, index) in base.items()}


def _check_base_metadata(frame, baseline, recording, log, prefix):
    actual = _observation(frame, recording, log, prefix, {})
    # The base JPEG hashes bind its input bytes. All other metadata must
    # independently match the original trusted recording before extending it.
    for fresh, known in zip(actual["views"], baseline["views"], strict=True):
        fresh["sha256"] = known["sha256"]
    if actual != baseline:
        raise PreparationError("Base observation metadata differs from the pinned recording")


def _third_status(older, history):
    if (older is None or older["log_name"] != history["log_name"]
            or older["log_token"] != history["log_token"]):
        return "missing_older"
    older_time = timestamp_us(older["timestamp"], "older observation timestamp")
    history_time = timestamp_us(history["timestamp"], "base previous timestamp")
    if not _STEP_US[0] <= history_time - older_time <= _STEP_US[1]:
        return "older_cadence"
    cams = older.get("cams")
    if not isinstance(cams, dict) or any(not isinstance(cams.get(name), dict)
                                         or not isinstance(cams[name].get("data_path"), str)
                                         or not cams[name]["data_path"] for name in CAMERAS):
        return "missing_older_cameras"
    return "ready"


def _extract_third_images(root, archives, wanted, stage, manifests, already_seen):
    lookup = {row["recording"]: split for split, manifest in manifests.items() for row in manifest["anchors"]}
    seen = set(already_seen)
    for shard, spec in sorted(archives.items()):
        images = wanted[shard]
        if not images:
            continue
        for member, blob in _read_selected(root / spec["filename"], spec, set(images), _IMAGE_LIMIT):
            relative, view, recording = images[member]
            if not blob.startswith(b"\xff\xd8\xff"):
                raise PreparationError("Third observation must be a JPEG archive member")
            digest = sha256(blob).hexdigest()
            _claim_jpeg(seen, digest)
            view["sha256"] = digest
            path = stage / lookup[recording] / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(blob)


def _validate_extension(index, base, expected_sha):
    require_keys(index, {"schema", "source_sha256", "base_input_sha256", "split", "anchors", "frames"}, "Third observation index")
    anchors, frames = index["anchors"], index["frames"]
    if (index["schema"] != EXTENSION_SCHEMA or index["source_sha256"] != SOURCE_SHA256 or index["split"] != base["split"]
            or index["base_input_sha256"] != expected_sha or not isinstance(anchors, list)
            or len(anchors) != len(base["anchors"]) or not isinstance(frames, dict) or len(frames) > len(anchors)):
        raise ValueError("Third observation index does not bind the exact base observation split")
    base_tokens, jpegs = _claim_base_observations(((None, base),))
    referenced = set()
    for row, original in zip(anchors, base["anchors"], strict=True):
        require_keys(row, {"sample_token", "previous_token", "recording", "older_token", "status"}, "Third observation anchor")
        if any(row[key] != value for key, value in original.items()) or row["status"] not in _STATUSES:
            raise ValueError("Third observation order or identity differs from the base")
        older = row["older_token"]
        if row["status"] != "ready":
            if older is not None:
                raise ValueError("Missing third observation must not point at a frame")
            continue
        require_hex(older, 16, "older observation token")
        if older not in frames or older in base_tokens or older in referenced:
            raise ValueError("Missing or duplicated older observation")
        referenced.add(older)
        frame = frames[older]
        stamp = _validate_frame(frame, older, row["recording"], {})
        history = base["frames"][row["previous_token"]]["ref_time_us"]
        if not _STEP_US[0] <= history - stamp <= _STEP_US[1]:
            raise ValueError("Older observation must precede the original previous frame by 0.4–0.6 seconds")
        for view in frame["views"]:
            _claim_jpeg(jpegs, view["sha256"])
    if referenced != frames.keys():
        raise ValueError("Third observation index contains an unreferenced frame")
