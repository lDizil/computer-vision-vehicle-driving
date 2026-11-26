"""Freeze 120 real three-observation clips before mounting any model outputs."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from hashlib import sha256
from itertools import combinations
from math import dist
from pathlib import Path
import pickle
import re
import sys
from tempfile import TemporaryDirectory

from hybrid_drive.assets.files import AtomicFile, PreparationError, check_components
from hybrid_drive.contracts.sensors import timestamp_us
from hybrid_drive.experiments.bounded_io import canonical_json, decode_json, read_bounded, require_hex, require_keys
from hybrid_drive.experiments.box_motion_data import CAMERAS, _chronological, _observation, _pose, _read_selected
from hybrid_drive.experiments.frozen_transfer_contacts import _eligible_vulnerable, _tracked_boxes
from hybrid_drive.experiments.frozen_transfer_data import _validate_frame
from hybrid_drive.experiments.temporal_vru_data import INDEX_LIMIT, LABEL_LIMIT, _has_annotations, _has_cameras, _in_grid
from hybrid_drive.experiments.visual_vru_geometry import GRID_X_MIN
from hybrid_drive.experiments.vru_fusion_holdout_data import _extract_cameras
from hybrid_drive.geometry.rigid import invert_pose, matrix


SOURCE_SHA256 = "139c7eb94125824d8a538bd052d1544bd12d2753be84713f7d156cdca543a474"
PLAN_SCHEMA = "hybrid-drive-vru-clip-plan-v1"
INPUT_SCHEMA = "hybrid-drive-vru-clip-inputs-v1"
LABEL_SCHEMA = "hybrid-drive-vru-clip-targets-v1"
ANCHOR_KEYS = ("sample_token", "previous_token", "older_token", "recording")
_SOURCE_SCHEMA = "hybrid-drive-vru-clip-sources-v1"
_RECORDING = re.compile(r"(\d{4}\.\d{2}\.\d{2})\.\d{2}\.\d{2}\.\d{2}_(veh-\d+)")
_BANDS = ((GRID_X_MIN, 8.0), (8.0, 16.0), (16.0, 27.0))
_PER_BAND, _MIN_GROUPS, _MAX_PER_GROUP = 40, 6, 24
_TOTAL = len(_BANDS) * _PER_BAND
_MIN_GAP_US = 3_000_000
_SLOT_ALTERNATIVES_US = 500_000
_MIN_STEP_US, _MAX_STEP_US = 400_000, 600_000
_META_LIMIT = 48 * 1024**2
_PLAN_LIMIT = 256 * 1024
_CLASS_MAP = {"pedestrian": "person", "bicycle": "bicycle"}
_PHASES = ("older", "previous", "current")


@dataclass(frozen=True)
class _Primary:
    band: int
    ident: str
    motion: str


@dataclass(frozen=True)
class _Clip:
    shard: str
    log_name: str
    recording: str
    stamp: int
    tokens: tuple[str, str, str]
    options: tuple[_Primary, ...]

    @property
    def group(self):
        return _group(self.recording)


@dataclass(frozen=True)
class _Selection:
    per_band: int
    min_groups: int
    max_per_group: int

    @property
    def total(self):
        return len(_BANDS) * self.per_band


_CONTROL_SELECTION = _Selection(_PER_BAND, _MIN_GROUPS, _MAX_PER_GROUP)


def plan_clip_corpus(metadata_archive: Path, source_path: Path, output: Path) -> dict:
    """Publish a metadata-only plan; its path must not already exist."""
    _require_linux()
    output = _new_plan_path(output)
    source = _load_source(source_path)
    plan = _plan(metadata_archive, source)
    return _publish_plan(plan, output)


def _new_plan_path(path):
    output = Path(path)
    check_components(output)
    if output.exists():
        raise PreparationError("Clip plan output must not already exist")
    return output


def _publish_plan(plan, output):
    payload = canonical_json(plan) + b"\n"
    if len(payload) > _PLAN_LIMIT:
        raise PreparationError("Clip plan exceeds its artifact byte budget")
    with AtomicFile(output) as staged:
        staged.out.write(payload)
        staged.commit()
    return {**plan["summary"], "plan_sha256": sha256(payload).hexdigest(), "camera_archives": plan["camera_archives"]}


def prepare_clip_corpus(metadata_archive: Path, camera_root: Path, source_path: Path,
                        plan_path: Path, output: Path) -> dict:
    """Recompute the frozen plan, then publish separate images and annotations."""
    _require_linux()
    output = Path(output)
    check_components(output)
    if output.exists():
        raise PreparationError("Clip corpus output must not already exist")
    output.parent.mkdir(parents=True, exist_ok=True)
    source = _load_source(source_path)
    plan_bytes = read_bounded(Path(plan_path), _PLAN_LIMIT)
    plan = _plan(metadata_archive, source)
    if plan_bytes != canonical_json(plan) + b"\n":
        raise PreparationError("Frozen clip plan differs from the pinned metadata selection")
    plan_sha = sha256(plan_bytes).hexdigest()
    frames, wanted, rows = _materialize(metadata_archive, source, plan)
    anchors = [{key: row[key] for key in ANCHOR_KEYS} for row in plan["clips"]]
    with TemporaryDirectory(dir=output.parent, prefix=".vru-clips-") as tmp:
        stage = Path(tmp) / "corpus"
        inputs = stage / "inputs"
        _extract_cameras(camera_root, plan["camera_archives"], wanted, inputs)
        index = {"schema": INPUT_SCHEMA, "source_sha256": SOURCE_SHA256, "plan_sha256": plan_sha,
                 "anchors": anchors, "frames": frames}
        _validate_index(index)
        raw = canonical_json(index) + b"\n"
        if len(raw) > INDEX_LIMIT:
            raise PreparationError("Clip inputs exceed their artifact byte budget")
        (inputs / "index.json").write_bytes(raw)
        digest = sha256(raw).hexdigest()
        labels = {"schema": LABEL_SCHEMA, "input_sha256": digest, "rows": rows}
        _validate_labels(labels, index, digest)
        truth = canonical_json(labels) + b"\n"
        if len(truth) > LABEL_LIMIT:
            raise PreparationError("Clip labels exceed their artifact byte budget")
        label_path = stage / "labels" / "labels.json"
        label_path.parent.mkdir()
        label_path.write_bytes(truth)
        check_components(output)
        if output.exists():
            raise PreparationError("Clip corpus output appeared during preparation")
        stage.rename(output)
    return {**plan["summary"], "plan_sha256": plan_sha, "input_sha256": digest,
            "labels_sha256": sha256(truth).hexdigest(), "camera_jpegs": len(frames) * len(CAMERAS)}


def load_clip_inputs(input_root: Path) -> dict:
    """Read only the sensor index; never open the plan or the labels."""
    return _read_index(input_root)[1]


def load_clip_labels(input_root: Path, labels_path: Path) -> dict:
    raw, index = _read_index(input_root)
    labels = decode_json(read_bounded(Path(labels_path), LABEL_LIMIT))
    _validate_labels(labels, index, sha256(raw).hexdigest())
    return labels


def _require_linux():
    if sys.platform != "linux":
        raise PreparationError("Clip corpus metadata preparation requires isolated Linux")


def _load_source(path):
    value = decode_json(read_bounded(Path(path), 32 * 1024))
    if (not isinstance(value, dict) or value.get("schema") != _SOURCE_SCHEMA
            or sha256(canonical_json(value)).hexdigest() != SOURCE_SHA256):
        raise PreparationError("Expected the checked-in clip corpus source")
    groups = {_group(row["recording"]) for row in value["logs"]}
    if len(groups) != 12 or groups & {_group(name) for name in value["previous_recordings"]}:
        raise PreparationError("Clip sources overlap a prior recording group")
    return value


def _group(name):
    match = _RECORDING.fullmatch(name) if isinstance(name, str) else None
    if not match:
        raise ValueError("Invalid clip recording identity")
    return "_".join(match.groups())


def _read_recordings(metadata_archive, source, only_logs=None):
    _require_linux()
    specs = {row["member"]: row for row in source["logs"] if only_logs is None or row["log_name"] in only_logs}
    for member, payload in _read_selected(metadata_archive, source["metadata_archive"], set(specs), _META_LIMIT):
        spec = specs[member]
        if len(payload) != spec["size_bytes"] or sha256(payload).hexdigest() != spec["sha256"]:
            raise PreparationError("Clip metadata member differs from its pinned size or digest")
        # Only after checking the entire archive and this exact member, in isolated Linux.
        frames = pickle.loads(payload)
        if (not isinstance(frames, list) or not 3 <= len(frames) <= 4096
                or any(not isinstance(f, dict) or f.get("log_name") != spec["log_name"] for f in frames)):
            raise PreparationError("Clip metadata recording does not match its pinned log")
        yield spec, _chronological(frames)


def _plan(metadata_archive, source, *, selection=_CONTROL_SELECTION, schema=PLAN_SCHEMA, source_sha256=SOURCE_SHA256):
    options = [clip for spec, frames in _read_recordings(metadata_archive, source)
               for clip in _candidates(spec, frames)]
    selected = _select(options, source["camera_archives"], selection=selection)
    clips = [{"sample_token": clip.tokens[2], "previous_token": clip.tokens[1], "older_token": clip.tokens[0],
              "recording": clip.recording, "log_name": clip.log_name, "primary_id": primary.ident}
             for clip, primary in sorted(selected, key=lambda pair: (pair[0].group, pair[0].stamp, pair[0].log_name))]
    archives = {shard: source["camera_archives"][shard] for shard in sorted({clip.shard for clip, _ in selected}, key=int)}
    groups = Counter(clip.group for clip, _ in selected)
    motion = Counter(primary.motion for _, primary in selected)
    return {"schema": schema, "source_sha256": source_sha256, "clips": clips, "camera_archives": archives,
            "summary": {"clips": len(clips), "camera_shards": list(archives),
                        "archive_bytes": sum(spec["size_bytes"] for spec in archives.values()),
                        "groups": dict(sorted(groups.items())), "bands": _band_counts(selected),
                        "motion": {key: motion[key] for key in ("moving", "still", "intermediate")}}}


def _band_counts(selected):
    counts = Counter(primary.band for _, primary in selected)
    return {f"[{lo:g},{hi:g})": counts[i] for i, (lo, hi) in enumerate(_BANDS)}


def _candidates(spec, frames):
    stamps = [timestamp_us(frame["timestamp"], "clip association stamp") for frame in frames]
    tracks = [_tracks(frame) for frame in frames]
    for i in range(2, len(frames)):
        if not all(_MIN_STEP_US <= stamps[j] - stamps[j - 1] <= _MAX_STEP_US for j in (i - 1, i)):
            continue
        matches = {ident: item for ident, item in tracks[i].items()
                   if _in_grid(item[1][:2]) and all(
                       ident in tracks[j] and tracks[j][ident][0] == item[0] for j in (i - 2, i - 1))}
        if not matches or not all(_has_cameras(frame) for frame in frames[i - 2:i + 1]):
            continue
        primary_views = _eligible_vulnerable(frames[i], matches)
        if not primary_views:
            continue
        current_from_older = invert_pose(_pose(frames[i])) @ _pose(frames[i - 2])
        candidates = []
        for band, (lo, hi) in enumerate(_BANDS):
            possible = [(abs(box[1]), box[0], ident) for ident, (_, box) in matches.items()
                        if ident in primary_views and lo <= box[0] < hi]
            if possible:
                _, _, ident = min(possible)
                current = matches[ident][1][:2]
                old = _center(current_from_older, tracks[i - 2][ident][1])
                moved = dist(current, old)
                motion = "moving" if moved >= .75 else "still" if moved <= .25 else "intermediate"
                candidates.append(_Primary(band, ident, motion))
        if candidates:
            yield _Clip(str(spec["camera_shard"]), spec["log_name"], spec["recording"], stamps[i],
                        tuple(frame["token"] for frame in frames[i - 2:i + 1]), tuple(candidates))


def _tracks(frame):
    return {key: pair for key, pair in _tracked_boxes(frame["anns"]).items()
            if pair[0] in _CLASS_MAP} if _has_annotations(frame) else {}


def _center(transform, box):
    return (transform[:3, :3] @ box[:3] + transform[:3, 3])[:2].tolist()


def _select(candidates, camera_archives, *, selection=_CONTROL_SELECTION):
    by_shard = defaultdict(list)
    for clip in candidates:
        by_shard[clip.shard].append(clip)
    possible_shards = sorted(by_shard, key=int)
    subsets = [group for count in range(1, len(possible_shards) + 1) for group in combinations(possible_shards, count)]
    subsets.sort(key=lambda group: (sum(camera_archives[k]["size_bytes"] for k in group), len(group), group))
    for subset in subsets:
        available = [clip for shard in subset for clip in by_shard[shard]]
        if len(available) < selection.total or len({clip.group for clip in available}) < selection.min_groups:
            continue
        # Compress nearby alternatives into one slot. Any assignment of one clip
        # per slot preserves the three-second gap in every recording group.
        slots = _slots(available)
        selected = _assign(slots, selection=selection)
        if selected is not None:
            return selected
    by_band = Counter(option.band for clip in candidates for option in clip.options)
    raise PreparationError(f"No exact VRU-{selection.total} selection: viable anchor-option counts before spacing "
                           f"{[by_band[i] for i in range(len(_BANDS))]}")


def _slots(available):
    group_clips = defaultdict(list)
    for clip in available:
        group_clips[clip.group].append(clip)
    slots = {}
    for group, values in sorted(group_clips.items()):
        rows = sorted(values, key=lambda clip: (clip.stamp, clip.log_name, clip.tokens))
        i = 0
        selected = []
        while i < len(rows):
            first = rows[i].stamp
            slot = {}
            while i < len(rows) and rows[i].stamp <= first + _SLOT_ALTERNATIVES_US:
                slot.update({option.band: (rows[i], option) for option in rows[i].options if option.band not in slot})
                i += 1
            selected.append(slot)
            threshold = rows[i - 1].stamp + _MIN_GAP_US
            while i < len(rows) and rows[i].stamp < threshold:
                i += 1
        slots[group] = selected
    return slots


def _assign(slots, *, selection=_CONTROL_SELECTION):
    if (sum(min(len(rows), selection.max_per_group) for rows in slots.values()) < selection.total
            or any(sum(b in row for rows in slots.values() for row in rows) < selection.per_band for b in range(len(_BANDS)))):
        return None
    source, sink = ("source",), ("sink",)
    network = defaultdict(dict)
    pools = {}
    for group, rows in slots.items():
        _add_edge(network, source, ("group", group), selection.max_per_group)
        for row in rows:
            bucket = ("options", group, tuple(sorted(row)))
            pools.setdefault(bucket, []).append(row)
    for bucket, rows in pools.items():
        _, group, bands = bucket
        _add_edge(network, ("group", group), bucket, len(rows))
        for band in bands:
            _add_edge(network, bucket, ("band", band), len(rows))
    for band in range(len(_BANDS)):
        _add_edge(network, ("band", band), sink, selection.per_band)
    if _flow(network, source, sink) != selection.total:
        return None
    answer = []
    for bucket, rows in pools.items():
        bins = [band for band in bucket[2] for _ in range(network[("band", band)][bucket])]
        for j, band in enumerate(bins):
            # Spread chosen anchors uniformly over their compatible pool.
            answer.append(rows[((2*j + 1) * len(rows)) // (2 * len(bins))][band])
    used = {clip.group for clip, _ in answer}
    while len(used) < selection.min_groups:
        spare = next((rows[0] for group, rows in slots.items() if group not in used), None)
        if spare is None:
            return None
        band = min(spare)
        counts = Counter(clip.group for clip, _ in answer)
        i = next(i for i, (clip, primary) in enumerate(answer) if primary.band == band and counts[clip.group] > 1)
        answer[i] = spare[band]
        used.add(answer[i][0].group)
    return answer


def _add_edge(graph, a, b, limit):
    graph[a][b] = limit
    graph[b][a] = 0


def _flow(graph, source, sink):
    sent = 0
    while True:
        queue, parent = deque((source,)), {source: None}
        while queue and sink not in parent:
            a = queue.popleft()
            for b, remaining in graph[a].items():
                if remaining and b not in parent:
                    parent[b] = a
                    queue.append(b)
        if sink not in parent:
            return sent
        path, b = [], sink
        while (a := parent[b]) is not None:
            path.append((a, b))
            b = a
        capacity = min(graph[a][b] for a, b in path)
        for a, b in path:
            graph[a][b] -= capacity
            graph[b][a] += capacity
        sent += capacity


def _materialize(metadata_archive, source, plan):
    by_log = defaultdict(list)
    for row in plan["clips"]:
        by_log[row["log_name"]].append(row)
    frames, labels = {}, {}
    wanted = {shard: {} for shard in plan["camera_archives"]}
    for spec, rows in _read_recordings(metadata_archive, source, set(by_log)):
        log, shard = spec["log_name"], str(spec["camera_shard"])
        include = plan["camera_archives"][shard]
        by_token = {frame["token"]: frame for frame in rows}
        for clip in by_log[log]:
            tokens = (clip["older_token"], clip["previous_token"], clip["sample_token"])
            window = tuple(by_token[token] for token in tokens)
            for token, frame in zip(tokens, window, strict=True):
                if token in frames:
                    raise PreparationError("Frozen clips share a camera observation")
                frames[token] = _observation(frame, clip["recording"], log, include["archive_prefix"], wanted[shard])
            labels[clip["sample_token"]] = _label_row(clip, window)
    return frames, wanted, [labels[clip["sample_token"]] for clip in plan["clips"]]


def _label_row(clip, window):
    current_from_world = invert_pose(_pose(window[-1]))
    observed = {}
    for phase, frame in zip(_PHASES, window, strict=True):
        transform = current_from_world @ _pose(frame) if phase != "current" else None
        observed[phase] = [{"track_id": ident, "kind": _CLASS_MAP[kind],
                            "xy_m": box[:2].tolist() if transform is None else _center(transform, box)}
                           for ident, (kind, box) in sorted(_tracks(frame).items())]
    return {"sample_token": clip["sample_token"], "recording": clip["recording"],
            "primary_id": clip["primary_id"], "frames": observed}


def _read_index(root):
    raw = read_bounded(Path(root) / "index.json", INDEX_LIMIT)
    index = decode_json(raw)
    _validate_index(index)
    return raw, index


def _validate_index(index):
    require_keys(index, {"schema", "source_sha256", "plan_sha256", "anchors", "frames"}, "Clip sensor index")
    require_hex(index["plan_sha256"], 64, "clip plan checksum")
    anchors, frames = index["anchors"], index["frames"]
    if (index["schema"] != INPUT_SCHEMA or index["source_sha256"] != SOURCE_SHA256
            or not isinstance(anchors, list) or len(anchors) != _TOTAL or not isinstance(frames, dict)
            or len(frames) != 3 * _TOTAL):
        raise ValueError("Unexpected clip sensor index shape or provenance")
    used, images, counts, previous = set(), {}, Counter(), {}
    for row in anchors:
        require_keys(row, set(ANCHOR_KEYS), "Clip anchor")
        group = _group(row["recording"])
        stamps = []
        for key in ("older_token", "previous_token", "sample_token"):
            token = row[key]
            require_hex(token, 16, "clip observation token")
            if token in used or token not in frames:
                raise ValueError("Clip corpus shares or misses an observation")
            used.add(token)
            stamps.append(_validate_frame(frames[token], token, row["recording"], images))
        if (not all(_MIN_STEP_US <= b - a <= _MAX_STEP_US for a, b in zip(stamps, stamps[1:]))
                or group in previous and stamps[-1] - previous[group] < _MIN_GAP_US):
            raise ValueError("Clip cadence or spacing changed from the frozen corpus")
        previous[group] = stamps[-1]
        counts[group] += 1
    if (used != frames.keys() or len(images) != len(CAMERAS) * len(frames)
            or len(counts) < _MIN_GROUPS or any(n > _MAX_PER_GROUP for n in counts.values())):
        raise ValueError("Unexpected clip grouping or duplicate sensor inputs")


def _validate_labels(labels, index, digest):
    require_keys(labels, {"schema", "input_sha256", "rows"}, "Clip labels")
    if (labels["schema"] != LABEL_SCHEMA or labels["input_sha256"] != digest
            or not isinstance(labels["rows"], list) or len(labels["rows"]) != _TOTAL):
        raise ValueError("Clip labels do not bind the exact frozen index")
    counts = Counter()
    for row, anchor in zip(labels["rows"], index["anchors"], strict=True):
        require_keys(row, {"sample_token", "recording", "primary_id", "frames"}, "Clip annotation row")
        if any(row[key] != anchor[key] for key in ("sample_token", "recording")):
            raise ValueError("Clip labels differ from their observation anchors")
        require_keys(row["frames"], set(_PHASES), "Clip phase labels")
        primary = []
        for phase in _PHASES:
            seen = set()
            tracks = row["frames"][phase]
            if not isinstance(tracks, list):
                raise ValueError("Expected an ordered clip track array")
            for track in tracks:
                require_keys(track, {"track_id", "kind", "xy_m"}, "Clip track")
                ident, kind = track["track_id"], track["kind"]
                require_hex(ident, 16, "clip track identity")
                xy = matrix(track["xy_m"], (2,), "clip track position")
                if ident in seen or kind not in _CLASS_MAP.values():
                    raise ValueError("Unexpected clip track identity or class")
                seen.add(ident)
                if ident == row["primary_id"]:
                    primary.append((kind, xy))
        if len(primary) != 3 or len({kind for kind, _ in primary}) != 1 or not _in_grid(primary[-1][1]):
            raise ValueError("Clip primary annotation is not stable or in the current ROI")
        x = primary[-1][1][0]
        counts[next(i for i, (lo, hi) in enumerate(_BANDS) if lo <= x < hi)] += 1
    if counts != {i: _PER_BAND for i in range(len(_BANDS))}:
        raise ValueError("Clip labels no longer have the fixed distance balance")
