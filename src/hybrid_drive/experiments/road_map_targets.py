"""Prepare road geometry labels from local nuScenes maps, apart from inference."""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
from math import cos, isfinite, sin
import os
from pathlib import Path
import re
import tempfile
from zipfile import ZipFile

import numpy as np

from hybrid_drive.assets.files import check_components
from hybrid_drive.experiments.bounded_io import decode_json, read_bounded, require_hex
from hybrid_drive.experiments.fast_mini_data import (
    _ARCHIVE_SHA256,
    _archive_signature,
    _selected_members,
    _token_index,
    _verify_archive,
)
from hybrid_drive.experiments.io import write_json


SCHEMA = "hybrid-drive-road-map-targets-v1"
SPLITS = ("train", "dev", "test")
_LOCATIONS = frozenset(("boston-seaport", "singapore-hollandvillage", "singapore-onenorth", "singapore-queenstown"))
_MAP_RELEASE = "1.3"
_MAP_BYTES = 398_535_531
_MAX_JSON = 32 * 1024**2
_MAX_CAPTURE_JSON = 8 * 1024**2


@dataclass(frozen=True)
class _Frame:
    token: str
    location: str
    pose: tuple[float, float, float]


@dataclass(frozen=True)
class _VectorMap:
    canvas: tuple[float, float]
    areas: object
    dividers: object
    repaired_polygons: int


def prepare_road_map_targets(capture: Path, metadata_archive: Path, map_archive: Path,
                             map_sha256: str, output: Path) -> dict:
    """Write separately mountable target packs; never decode sensor or object labels."""
    require_hex(map_sha256, 64, "map_sha256")
    output = Path(output).expanduser().absolute()
    check_components(output)
    if output.exists():
        raise ValueError("Road target output already exists")
    capture_raw = read_bounded(Path(capture) / "capture.json", _MAX_CAPTURE_JSON)
    captured = decode_json(capture_raw)
    if (captured.get("schema") != "hybrid-drive-fast-capture-v1" or captured.get("targets_opened") is not False
            or captured.get("raw_source_sha256") != _ARCHIVE_SHA256 or set(captured["splits"]) != set(SPLITS)):
        raise ValueError("Expected a nuScenes Mini observation-only capture")

    from hybrid_drive.fast.metric_bev import BevGrid

    grid_data = dict(captured["grid"])
    grid_data["heights_m"] = tuple(grid_data["heights_m"])
    grid = BevGrid(**grid_data)
    tables = _read_sample_tables(metadata_archive)
    frames, observed_sha = _capture_frames(Path(capture), captured, tables)
    labels, known, map_stats = _rasterize_batch(map_archive, map_sha256, frames, grid)
    summary = {
        "schema": SCHEMA,
        "capture_sha256": sha256(capture_raw).hexdigest(),
        "map_sha256": map_sha256,
        "map_version": _MAP_RELEASE,
        "mini_sha256": _ARCHIVE_SHA256,
        "maps": map_stats,
        "world_patch_overlap": _world_patch_overlap(frames, grid),
        "splits": {},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".road-map-", dir=output.parent) as temp:
        staging = Path(temp) / "ready"
        for split in SPLITS:
            out = staging / split
            out.mkdir(parents=True)
            target = out / "targets.npz"
            np.savez_compressed(target, labels=labels[split], known=known[split])
            manifest = {
                "schema": SCHEMA,
                "capture_sha256": summary["capture_sha256"],
                "observation_sha256": observed_sha[split],
                "target_sha256": sha256(target.read_bytes()).hexdigest(),
                "map_sha256": map_sha256,
                "tokens": [frame.token for frame in frames[split]],
                "locations": [frame.location for frame in frames[split]],
                "grid": captured["grid"],
                "channels": ["drivable_area", "lane_divider"],
            }
            write_json(out / "targets.json", manifest)
            summary["splits"][split] = {
                "frames": len(frames[split]),
                "locations": dict(sorted(Counter(manifest["locations"]).items())),
                "known_cells": int(known[split].sum()),
                "drivable_cells": int(labels[split][:, 0].sum()),
                "divider_cells": int(labels[split][:, 1].sum()),
            }
        write_json(staging / "summary.json", summary)
        check_components(output)
        if output.exists():
            raise ValueError("Road target output already exists")
        staging.rename(output)
    return summary


@contextmanager
def _open_archive(path):
    path = Path(path).expanduser().absolute()
    check_components(path)
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as source:
        yield source


def _read_sample_tables(archive):
    names = {f"v1.0-mini/{table}.json": table for table in ("sample", "scene", "log")}
    with _open_archive(archive) as source:
        verified = _verify_archive(source)
        tables = {
            names[path]: _token_index(decode_json(data), names[path])
            for path, data in _selected_members(source, names, _MAX_JSON)
        }
        if _archive_signature(source) != verified:
            raise ValueError("Mini metadata archive changed during preparation")
    return tables


def _capture_frames(capture, captured, tables):
    out, digests = {}, {}
    token_roles, recording_roles = {}, {}
    for split in SPLITS:
        data = read_bounded(capture / split / "observations.json", _MAX_CAPTURE_JSON)
        digest = sha256(data).hexdigest()
        entry = captured["splits"][split]
        if entry["artifacts"]["observations.json"] != digest:
            raise ValueError("Capture observations differ from capture manifest")
        rows = decode_json(data)
        if not isinstance(rows, list) or not 1 <= len(rows) == entry["count"] <= 100:
            raise ValueError("Unexpected captured observation count")
        out[split] = []
        for row in rows:
            token = row["sample_token"]
            scene = tables["scene"][tables["sample"][token]["scene_token"]]
            recording = scene["log_token"]
            if row["scene"] != scene["name"] or row["recording"] != recording:
                raise ValueError("Captured sample no longer belongs to the recorded scene/log")
            if token in token_roles or recording_roles.setdefault(recording, split) != split:
                raise ValueError("Duplicate samples or recordings across roles")
            token_roles[token] = split
            location = tables["log"][recording]["location"]
            if location not in _LOCATIONS:
                raise ValueError(f"Unsupported nuScenes vector map: {location}")
            pose = _real_values(row["ego_world_xy_yaw"], 3, "ego_world_xy_yaw")
            out[split].append(_Frame(token, location, pose))
        digests[split] = digest
    return out, digests


def _verify_maps(source, expected_sha):
    signature = _archive_signature(source)
    if signature[0] != _MAP_BYTES:
        raise ValueError("nuScenes map ZIP size differs from v1.3 release")
    digest = sha256()
    for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
        digest.update(block)
    if digest.hexdigest() != expected_sha or _archive_signature(source) != signature:
        raise ValueError("nuScenes map ZIP differs from admitted SHA256")
    return signature


def _rasterize_batch(archive, expected_sha, frames, grid):
    labels = {role: np.zeros((len(frames[role]), 2, grid.rows, grid.cols), dtype=np.bool_) for role in SPLITS}
    known = {role: np.zeros((len(frames[role]), grid.rows, grid.cols), dtype=np.bool_) for role in SPLITS}
    stats = {}
    with _open_archive(archive) as source:
        verified = _verify_maps(source, expected_sha)
        with ZipFile(source) as zip_file:
            for location in sorted({row.location for rows in frames.values() for row in rows}):
                world_map = _load_vector_map(_read_vector_member(zip_file, location))
                stats[location] = {"canvas_m": list(world_map.canvas), "repaired_polygons": world_map.repaired_polygons}
                for role in SPLITS:
                    for i, frame in enumerate(frames[role]):
                        if frame.location == location:
                            labels[role][i], known[role][i] = _rasterize(world_map, grid, frame.pose)
                            if not known[role][i].any():
                                raise ValueError(f"Observation outside publisher map canvas: {frame.token}")
                del world_map
        if _archive_signature(source) != verified:
            raise ValueError("nuScenes map ZIP changed during preparation")
    return labels, known, stats


def _read_vector_member(zip_file, location):
    if location not in _LOCATIONS:
        raise ValueError("Invalid nuScenes map location")
    name = f"expansion/{location}.json"
    entries = [member for member in zip_file.infolist() if member.filename == name]
    if len(entries) != 1 or entries[0].is_dir() or not 0 < entries[0].file_size <= _MAX_JSON:
        raise ValueError(f"Missing, ambiguous or oversized map vector member: {name}")
    with zip_file.open(entries[0]) as stream:
        raw = stream.read(_MAX_JSON + 1)
    if len(raw) != entries[0].file_size:
        raise ValueError(f"Truncated map vector member: {name}")
    return decode_json(raw)


def _load_vector_map(obj):
    from shapely import LineString, Polygon, STRtree, make_valid, union_all

    if obj.get("version") != _MAP_RELEASE:
        raise ValueError("Expected a nuScenes 1.3 vector map")
    canvas = _real_values(obj["canvas_edge"], 2, "canvas_edge")
    if min(canvas) <= 0:
        raise ValueError("Expected a positive map canvas")
    nodes = _map_token_index(obj["node"], "node")
    polygons = _map_token_index(obj["polygon"], "polygon")
    lines = _map_token_index(obj["line"], "line")

    def coords(tokens):
        return [_real_values((nodes[t]["x"], nodes[t]["y"]), 2, "map node") for t in tokens]

    drivable, repaired = [], 0
    tokens = {token for rec in obj["drivable_area"] for token in rec["polygon_tokens"]}
    if tokens - polygons.keys():
        raise ValueError("Drivable layer references missing polygons")
    for token in sorted(tokens, key=lambda value: "" if value is None else value):
        record = polygons[token]
        exterior = coords(record["exterior_node_tokens"])
        holes = [coords(rec["node_tokens"]) for rec in record["holes"] if rec["node_tokens"]]
        polygon = Polygon(exterior, holes)
        if not polygon.is_valid:
            polygon = make_valid(polygon)
            repaired += 1
        parts = list(_polygon_components(polygon))
        if not parts:
            raise ValueError(f"Drivable polygon has no area: {token}")
        drivable.extend(parts)
    dividers = []
    for token in sorted({record["line_token"] for record in obj["lane_divider"]}):
        vertices = coords(lines[token]["node_tokens"])
        if len(vertices) >= 2:
            dividers.append(LineString(vertices))
    if not drivable or not dividers:
        raise ValueError("Missing road or divider map geometry")
    # Dissolve shared borders: a centre on an internal seam is still
    # inside the drivable surface. Real polygon holes stay excluded.
    dissolved = list(_polygon_components(union_all(drivable)))
    return _VectorMap(canvas, STRtree(dissolved), STRtree(dividers), repaired)


_MAP_UUID = re.compile(r"[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12}")


def _map_token_index(records, layer):
    result = {}
    for record in records:
        token = record["token"]
        # v1.3 includes a unique null polygon referenced by real drivable
        # records in Queenstown and Holland Village. The empty placeholder
        # in One North has no geometry and is not a usable polygon.
        if (layer == "polygon" and token == "" and not record["exterior_node_tokens"]
                and not record["holes"]):
            continue
        if not (layer == "polygon" and token is None) and (
            not isinstance(token, str) or _MAP_UUID.fullmatch(token) is None
        ):
            raise ValueError(f"Invalid {layer} identifier in the nuScenes vector map")
        if token in result:
            raise ValueError(f"Duplicate {layer} identifier in the nuScenes vector map")
        result[token] = record
    return result


def _polygon_components(geometry):
    if geometry.geom_type == "Polygon" and not geometry.is_empty and geometry.area > 0:
        yield geometry
    elif geometry.geom_type in ("GeometryCollection", "MultiPolygon"):
        for part in geometry.geoms:
            yield from _polygon_components(part)


def _real_values(value, count, name):
    if not isinstance(value, (list, tuple)) or len(value) != count or any(
        type(x) not in (int, float) or not isfinite(x) for x in value
    ):
        raise ValueError(f"{name} must contain {count} finite real values")
    return tuple(float(x) for x in value)


def _to_world(points, pose):
    px, py, yaw = pose
    c, s = cos(yaw), sin(yaw)
    return points @ np.array([[c, s], [-s, c]]) + (px, py)


def _rasterize(world_map, grid, pose):
    from shapely import points, polygons

    centres = np.stack(grid.xy(), axis=-1).reshape(-1, 2)
    corners = centres[:, None, :] + grid.step_m / 2 * np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
    fitted = _to_world(corners, pose)
    known = ((0 <= fitted) & (fitted <= world_map.canvas)).all(axis=(1, 2))
    labels = np.zeros((2, len(centres)), dtype=np.bool_)
    if known.any():
        indices = np.flatnonzero(known)
        areal = world_map.areas.query(points(_to_world(centres[known], pose)), predicate="within")[0]
        linear = world_map.dividers.query(polygons(fitted[known]), predicate="intersects")[0]
        labels[0, indices[areal]] = True
        labels[1, indices[linear]] = True
    return labels.reshape(2, grid.rows, grid.cols), known.reshape(grid.rows, grid.cols)


def _world_patch_overlap(frames, grid):
    from shapely import Polygon, STRtree

    x1, y1 = grid.x_min + grid.rows * grid.step_m, grid.y_min + grid.cols * grid.step_m
    patch = np.array([[grid.x_min, grid.y_min], [x1, grid.y_min], [x1, y1], [grid.x_min, y1]])
    outlines = {role: [Polygon(_to_world(patch, frame.pose)) for frame in frames[role]] for role in SPLITS}
    result = {}
    for first, later in (("train", "dev"), ("train", "test"), ("dev", "test")):
        overlapping = set()
        for location in sorted({frame.location for frame in frames[first]}):
            index = STRtree([g for f, g in zip(frames[first], outlines[first]) if f.location == location])
            for i, (frame, outline) in enumerate(zip(frames[later], outlines[later])):
                if frame.location == location and any(outline.intersection(index.geometries[j]).area > 0
                                                       for j in index.query(outline)):
                    overlapping.add(i)
        result[f"{later}_frames_overlapping_{first}"] = len(overlapping)
    return result
