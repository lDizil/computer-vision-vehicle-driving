"""Verified original JPEGs for the frozen six-scene OpenScene debug cohort."""
from __future__ import annotations

from contextlib import contextmanager
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tarfile
import tempfile
import zlib

from .cameras import BoundedTarReader, MAX_MEMBER_BYTES, MAX_MEMBERS, MAX_TAR_BYTES, validate_jpeg
from .fetch import AssetSpec, CHUNK_SIZE
from .files import AtomicFile, PreparationError, check_components, destination, read_manifest, safe_relative, verify_file, write_manifest


SOURCE_SCHEMA = "hybrid-drive-real-cohort-sources-v1"
MANIFEST_SCHEMA = "hybrid-drive-real-cohort-assets-v1"
MANIFEST_NAME = "cohort-manifest.json"
CAMERA_ORDER = ("CAM_L0", "CAM_F0", "CAM_R0")
LOG_NAME = "2021.10.06.17.43.07_veh-28_00508_00877"
_NAVSIM_REVISION = "3e8291bfa89ff247231e0227778840cd0a036896"
_NUPLAN_REVISION = "ce3c323af01c0d7ec5672f7832ef53f9c679aab0"
_DATA_REVISION = "a76f840b65e972bc45e56c2adced897498e9a026"
_PINNED_LOG = {
    "url": f"https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/{_DATA_REVISION}/openscene-v1.1/openscene_metadata_mini.tgz",
    "revision": _DATA_REVISION,
    "archive_size_bytes": 534344190,
    "archive_sha256": "c2e08867e062ba916eccd56df191e728d42fb94de854c28f2226eed14cecd8be",
    "prefix_size_bytes": 2097152,
    "prefix_sha256": "ebd39e42489626866205ab81fad029c67c99ec359f128a89de9b66c38e5b9cfe",
    "member": f"openscene-v1.1/meta_datas/mini/{LOG_NAME}.pkl",
    "path": f"navsim_logs/mini/{LOG_NAME}.pkl",
    "size_bytes": 4832776,
    "sha256": "d2a9b373da30670aaf327c2ab4ccc332e2b441be6cd5e3f72d77debd3adbcff3",
}
_PINNED_ARCHIVE = {
    "url": f"https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/{_DATA_REVISION}/openscene-v1.1/openscene_sensor_mini_camera/openscene_sensor_mini_camera_16.tgz",
    "revision": _DATA_REVISION,
    "size_bytes": 2903678609,
    "sha256": "09fc68bca920dced27bf513eb9867f08f5f7d34e73733c99e9aaa31471bebc50",
    "etag": "9f6293b69d2202e9df798e6ebf3fabdb144faa3ba540c4bc72b72f95ffae24dc",
    "archive_prefix": "openscene-v1.1/sensor_blobs/mini",
}


def load_cohort_sources(path):
    sources = read_manifest(Path(path))
    if sources is None:
        raise PreparationError("Cohort sources are missing")
    validate_cohort_sources(sources)
    return sources


def validate_cohort_sources(sources):
    try:
        _validate_sources(sources)
    except PreparationError:
        raise
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise PreparationError("Invalid cohort sources") from exc


def _validate_sources(sources):
    if (set(sources) != {"schema", "navsim_revision", "nuplan_revision", "log", "camera_archive", "selection", "cases"}
            or sources["schema"] != SOURCE_SCHEMA
            or sources["navsim_revision"] != _NAVSIM_REVISION
            or sources["nuplan_revision"] != _NUPLAN_REVISION
            or sources["log"] != _PINNED_LOG
            or sources["camera_archive"] != _PINNED_ARCHIVE):
        raise PreparationError("Cohort must use the pinned OpenScene log and camera archive")
    AssetSpec.from_dict(sources["camera_archive"]).validate()
    selected = _validate_selection(sources["selection"])
    cases = sources["cases"]
    if not isinstance(cases, list) or len(cases) != 6:
        raise PreparationError("Expected exactly six frozen cohort cases")
    tokens, members, paths = set(), set(), set()
    previous_timestamp = -1
    for index, (case, selected_index) in enumerate(zip(cases, selected)):
        if set(case) != {"case_id", "source_list_index", "metadata_frame_idx", "metadata_scene_token", "scene_token", "frame_timestamp_us", "cameras"}:
            raise PreparationError("Invalid cohort case fields")
        if case["case_id"] != f"r{index:02d}" or case["source_list_index"] != selected_index:
            raise PreparationError("Cohort case order differs from temporal selection")
        if not _integer(case["metadata_frame_idx"], minimum=3) or not _hex(case["metadata_scene_token"], 16):
            raise PreparationError("Invalid source chunk identity")
        token, timestamp = case["scene_token"], case["frame_timestamp_us"]
        if not _hex(token, 16) or token in tokens or not _integer(timestamp) or timestamp <= previous_timestamp:
            raise PreparationError("Duplicate or unordered cohort frame identity")
        tokens.add(token)
        previous_timestamp = timestamp
        cameras = case["cameras"]
        if not isinstance(cameras, list) or len(cameras) != 3:
            raise PreparationError("Expected exactly three cameras per case")
        for camera, entry in zip(CAMERA_ORDER, cameras):
            if set(entry) != {"camera", "member", "path"} or entry["camera"] != camera:
                raise PreparationError("Cohort camera order differs from model input order")
            safe_relative(entry["member"])
            safe_relative(entry["path"])
            suffix = f"{LOG_NAME}/{camera}/"
            filename = entry["path"].removeprefix("sensor_blobs/" + suffix)
            if (not re.fullmatch(r"[0-9a-f]{16}\.jpg", filename)
                    or entry["path"] != "sensor_blobs/" + suffix + filename
                    or entry["member"] != _PINNED_ARCHIVE["archive_prefix"] + "/" + suffix + filename
                    or entry["member"] in members or entry["path"] in paths):
                raise PreparationError("Invalid or duplicate cohort camera identity")
            members.add(entry["member"])
            paths.add(entry["path"])


def _validate_selection(selection):
    expected = {
        "rule": "uniform-anchors-nearest-temporal-valid-v1", "num_source_frames": 738,
        "num_history_frames": 4, "num_future_frames": 10,
        "tie_break": "lower-source-list-index", "timestamp_step_us": [499000, 501000],
        "same_metadata_scene_token": True, "metadata_frame_idx_step": 1,
        "unique_frame_tokens": True, "frozen_before_inference": True,
    }
    if (set(selection) != set(expected) | {"anchors", "selected_indices", "temporal_valid_ranges", "temporal_valid_count"}
            or any(selection[key] != value for key, value in expected.items())):
        raise PreparationError("Unknown cohort temporal selection rule")
    anchors = [3 + k * (738 - 14) // 5 for k in range(6)]
    if selection["anchors"] != anchors:
        raise PreparationError("Cohort anchors differ from the uniform time grid")
    valid = []
    for bounds in selection["temporal_valid_ranges"]:
        if (not isinstance(bounds, list) or len(bounds) != 2
                or not all(_integer(value, minimum=3) and value <= 727 for value in bounds)
                or bounds[0] > bounds[1] or (valid and bounds[0] <= valid[-1])):
            raise PreparationError("Invalid temporal-valid source ranges")
        valid.extend(range(bounds[0], bounds[1] + 1))
    if not valid or selection["temporal_valid_count"] != len(valid):
        raise PreparationError("Temporal-valid source count differs from ranges")
    selected = [min(valid, key=lambda value: (abs(value - anchor), value)) for anchor in anchors]
    if selection["selected_indices"] != selected or any(right - left < 14 for left, right in zip(selected, selected[1:])):
        raise PreparationError("Cohort windows overlap or differ from temporal selection")
    return selected


def prepare_cohort(root, sources, archive_path):
    validate_cohort_sources(sources)
    root = Path(os.path.abspath(root))
    check_components(root)
    with _verified_archive(archive_path, sources["camera_archive"]) as archive:
        if root.exists() and (root / MANIFEST_NAME).exists():
            return verify_cohort(root, sources)
        if root.exists() and (not root.is_dir() or any(root.iterdir())):
            raise PreparationError("Cohort preparation requires a new empty root; legacy assets are not overwritten")
        check_components(root.parent)
        root.parent.mkdir(parents=True, exist_ok=True)
        check_components(root.parent)
        with tempfile.TemporaryDirectory(prefix=".cohort-", dir=root.parent) as temporary:
            stage = Path(temporary)
            records = _extract_cameras(archive, stage, sources)
            manifest = {
                "schema": MANIFEST_SCHEMA, "sources_hash": _hash_json(sources),
                "archive": _archive_identity(sources), "files": records,
            }
            manifest["manifest_hash"] = _hash_json(manifest)
            write_manifest(stage / MANIFEST_NAME, manifest)
            check_components(root)
            # A directory rename publishes the complete cohort, never a partial set of JPEGs.
            os.rename(stage, root)
    return manifest


@contextmanager
def _verified_archive(path, spec):
    path = Path(os.path.abspath(path))
    check_components(path)
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as archive:
        info = os.fstat(archive.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size != spec["size_bytes"]:
            raise PreparationError("Camera archive failed pinned size verification")
        digest = hashlib.sha256()
        for block in iter(lambda: archive.read(CHUNK_SIZE), b""):
            digest.update(block)
        if digest.hexdigest() != spec["sha256"]:
            raise PreparationError("Camera archive failed pinned SHA256 verification")
        archive.seek(0)
        yield archive


def _extract_cameras(archive, root, sources):
    entries = _entries(sources)
    allowlist = {entry["member"]: entry for entry in entries}
    records, seen = {}, set()
    try:
        with gzip.GzipFile(fileobj=archive, mode="rb") as stream:
            tar = BoundedTarReader(stream)
            while True:
                header = tar.read_exact(512)
                if header == b"\0" * 512:
                    if tar.read_exact(512) != b"\0" * 512:
                        raise PreparationError("Invalid tar end markers")
                    _read_padding(tar)
                    break
                member = tarfile.TarInfo.frombuf(header, "utf-8", "strict")
                safe_relative(member.name)
                if member.name in seen:
                    raise PreparationError("Duplicate tar member")
                seen.add(member.name)
                if len(seen) > MAX_MEMBERS:
                    raise PreparationError("Tar member count exceeds budget")
                if not (member.isreg() or member.isdir()):
                    raise PreparationError("Tar links and extended/special entries are forbidden")
                if member.size < 0 or member.size > MAX_MEMBER_BYTES or (member.isdir() and member.size):
                    raise PreparationError("Tar member exceeds its size limit")
                entry = allowlist.get(member.name)
                if entry is None:
                    tar.skip(member.size)
                else:
                    if not member.isreg() or not member.size:
                        raise PreparationError("Allowlisted camera is not a nonempty regular file")
                    records[member.name] = _write_camera(tar, root, entry, member.size)
                tar.skip((-member.size) % 512)
        if set(records) != set(allowlist):
            raise PreparationError(f"Camera archive does not contain all {len(allowlist)} allowlisted JPEGs")
    except (tarfile.TarError, EOFError, gzip.BadGzipFile, zlib.error, UnicodeError, OSError) as exc:
        raise PreparationError("Invalid or truncated camera archive") from exc
    return [records[entry["member"]] for entry in entries]


def _write_camera(tar, root, entry, size):
    digest = hashlib.sha256()
    with AtomicFile(destination(root, entry["path"])) as temporary:
        remaining = size
        while remaining:
            block = tar.read_exact(min(remaining, CHUNK_SIZE))
            temporary.out.write(block)
            digest.update(block)
            remaining -= len(block)
        temporary.out.flush()
        decoded = validate_jpeg(temporary.partial)
        temporary.commit()
    return {**entry, "size_bytes": size, "sha256": digest.hexdigest(), **decoded}


def _read_padding(tar):
    while True:
        block = tar.stream.read(CHUNK_SIZE)
        if not block:
            return
        tar.received += len(block)
        if tar.received > MAX_TAR_BYTES or any(block):
            raise PreparationError("Invalid or oversized trailing tar padding")


def verify_cohort(root, sources):
    validate_cohort_sources(sources)
    root = Path(os.path.abspath(root))
    check_components(root)
    if not root.is_dir():
        raise PreparationError("Prepared cohort root is missing")
    manifest = read_manifest(destination(root, MANIFEST_NAME))
    if manifest is None:
        raise PreparationError("Prepared cohort manifest is missing")
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    if (set(manifest) != {"schema", "sources_hash", "archive", "files", "manifest_hash"}
            or manifest["manifest_hash"] != _hash_json(unsigned)
            or manifest["schema"] != MANIFEST_SCHEMA
            or manifest["sources_hash"] != _hash_json(sources)
            or manifest["archive"] != _archive_identity(sources)):
        raise PreparationError("Prepared cohort manifest differs from pinned sources")
    entries, records = _entries(sources), manifest["files"]
    if not isinstance(records, list) or len(records) != 18:
        raise PreparationError("Prepared cohort must contain exactly 18 camera records")
    for entry, record in zip(entries, records):
        if (not isinstance(record, dict) or set(record) != set(entry) | {"size_bytes", "sha256", "width", "height", "format", "mode"}
                or any(record.get(key) != value for key, value in entry.items())
                or not _integer(record["size_bytes"], minimum=1) or record["size_bytes"] > MAX_MEMBER_BYTES
                or not _hex(record["sha256"], 64)):
            raise PreparationError("Prepared camera identity differs from cohort allowlist")
        path = destination(root, entry["path"])
        if verify_file(path, size_bytes=record["size_bytes"], sha256=record["sha256"]) is None:
            raise PreparationError("Prepared cohort camera is missing")
        if any(record.get(key) != value for key, value in validate_jpeg(path).items()):
            raise PreparationError("Prepared camera decoding differs from manifest")
    _check_contents(root, {entry["path"] for entry in entries} | {MANIFEST_NAME})
    return manifest


def _check_contents(root, files):
    directories = {str(parent) for name in files for parent in Path(name).parents if str(parent) != "."}
    found = set()
    for current, children, names in os.walk(root, followlinks=False):
        for name in children + names:
            path = Path(current) / name
            check_components(path)
            relative = path.relative_to(root).as_posix()
            if path.is_dir() and relative in directories:
                continue
            if relative not in files or not path.is_file():
                raise PreparationError("Unexpected file or directory in prepared cohort")
            found.add(relative)
    if found != files:
        raise PreparationError("Prepared cohort files are incomplete")


def _entries(sources):
    return [{"case_id": case["case_id"], **camera} for case in sources["cases"] for camera in case["cameras"]]


def _archive_identity(sources):
    return {key: sources["camera_archive"][key] for key in ("size_bytes", "sha256")} | {"full_archive_sha256_verified": True}


def _hash_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _integer(value, minimum=0):
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def _hex(value, length):
    return isinstance(value, str) and re.fullmatch("[0-9a-f]{" + str(length) + "}", value) is not None
