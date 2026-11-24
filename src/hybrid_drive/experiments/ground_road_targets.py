"""Evaluation-only nuPlan vector targets for the frozen OpenScene #43 anchors."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from math import atan2, cos, sin
import os
from pathlib import Path
import pickle
import stat
import sys
from tempfile import TemporaryDirectory
from zlib import crc32

import numpy as np

from hybrid_drive.assets.files import check_components
from hybrid_drive.experiments.bounded_io import canonical_json, decode_json, read_bounded, require_hex
from hybrid_drive.experiments.box_motion_data import SOURCES_SHA256, _chronological, _pose, _read_selected
from hybrid_drive.experiments.io import write_json
from hybrid_drive.geometry.rigid import matrix, rigid_transform


SCHEMA = "hybrid-drive-ground-external-targets-v1"
GRID = {"x_min": -16.0, "y_min": -32.0, "step_m": 2.0, "rows": 32, "cols": 32}
_OBSERVATION_SHA256 = "e04c01e8394d675d7a3fd1412658d714b4a8cdeb10e22ab093c5e64fcc704d21"
_NAVSIM_SOURCE_SHA256 = "92a85f97591d9c45fd8e27edc2e79184f5fdb69cd8471be74dad90b4b1b2b9f5"
_MAP_VERSION_PATH = "maps/nuplan-maps-v1.0.json"
_LOCATION = "us-nv-las-vegas-strip"
_MAP_FILE = f"maps/{_LOCATION}/9.15.1915/map.gpkg"
# From pinned nuPlan devkit: raster dimensions are (rows, columns).
_RASTER_SHAPE = (69820, 30120)
_DRIVABLE_LAYERS = ("road_segments", "intersections", "generic_drivable_areas", "carpark_areas")
_LANE_COLUMNS = ("lane_group_fid", "lane_index", "left_boundary_fid", "right_boundary_fid")
_NUPLAN_VERSION = "ce3c323af01c0d7ec5672f7832ef53f9c679aab0"


@dataclass(frozen=True)
class _Frame:
    token: str
    recording: str
    location: str
    pose: tuple[float, float, float]


@dataclass(frozen=True)
class _VectorMap:
    areas: object
    dividers: object
    world_to_pixel: np.ndarray
    shape: tuple[int, int]
    projected_crs: str
    repaired_areas: int


def prepare_ground_road_targets(input_root: Path, metadata_archive: Path, maps_root: Path,
                                output: Path, *, motion_sources: Path, navsim_sources: Path) -> dict:
    """Build labels after blind prediction; inputs need only the original index."""
    if sys.platform != "linux":
        raise ValueError("Prepare trusted OpenScene pickle metadata only in the offline Linux image")
    output = Path(output).expanduser().absolute()
    check_components(output)
    if output.exists():
        raise ValueError("External road target output already exists")
    index_bytes = read_bounded(Path(input_root) / "index.json", 8 * 1024**2)
    observation_sha256 = sha256(index_bytes).hexdigest()
    if observation_sha256 != _OBSERVATION_SHA256:
        raise ValueError("Expected the exact #43 OpenScene observation index")
    index = decode_json(index_bytes)
    openscene = decode_json(read_bounded(Path(motion_sources), 16 * 1024))
    if sha256(canonical_json(openscene)).hexdigest() != SOURCES_SHA256:
        raise ValueError("Unexpected OpenScene source manifest")
    frames = _load_frames(index, Path(metadata_archive), openscene)

    navsim_raw = read_bounded(Path(navsim_sources), 16 * 1024)
    if sha256(navsim_raw).hexdigest() != _NAVSIM_SOURCE_SHA256:
        raise ValueError("Unexpected pinned nuPlan map source manifest")
    spec = decode_json(navsim_raw)
    if spec["nuplan_revision"] != _NUPLAN_VERSION:
        raise ValueError("Unexpected nuPlan map API revision")
    map_root = Path(maps_root).expanduser().absolute()
    receipt_bytes = read_bounded(map_root / "maps-receipt.json", 32 * 1024)
    inputs = _verify_maps(map_root, decode_json(receipt_bytes), spec["maps"])
    version = decode_json(read_bounded(map_root / _MAP_VERSION_PATH, 64 * 1024))[_LOCATION]
    if version["version"] != "9.15.1915":
        raise ValueError("Expected the Las Vegas map version from the publisher manifest")
    raster = version["layers"]["drivable_area"]
    if tuple(raster["shape"]) != _RASTER_SHAPE:
        raise ValueError("Las Vegas raster dimensions drift from the pinned devkit")
    world_to_pixel = matrix(raster["transform_matrix"], (4, 4), "world_to_pixel")
    world = _load_vector_map(map_root / _MAP_FILE, world_to_pixel)
    for path, data in inputs.items():
        if _signature((map_root / path).stat()) != data["signature"]:
            raise ValueError(f"Map input changed during preparation: {path}")

    values = [_rasterize(world, GRID, frame.pose) for frame in frames]
    labels = np.stack([row[0] for row in values])
    known = np.stack([row[1] for row in values])
    if not known.any(axis=(1, 2)).all():
        raise ValueError("An OpenScene anchor was outside the official mapped extent")
    from shapely import points
    origins = points(np.array([frame.pose[:2] for frame in frames]))
    road_origins = len(set(world.areas.query(origins, predicate="within")[0].tolist()))
    if not road_origins:
        raise ValueError("No ego origin lies in a drivable area; check the cohort's coordinate system")
    summary = {
        "schema": SCHEMA,
        "observation_sha256": observation_sha256,
        "metadata_sha256": openscene["metadata_archive"]["sha256"],
        "map_receipt_sha256": sha256(receipt_bytes).hexdigest(),
        "map_inputs": [data["receipt"] for data in inputs.values()],
        "grid": GRID,
        "tokens": [frame.token for frame in frames],
        "recordings": [frame.recording for frame in frames],
        "locations": [frame.location for frame in frames],
        "channels": ["drivable_area", "lane_divider"],
        "semantics": {
            "nuplan_devkit_revision": _NUPLAN_VERSION,
            "projected_crs": world.projected_crs,
            "drivable_area": {"layers": list(_DRIVABLE_LAYERS),
                              "condition": "BEV center within any official vector drivable feature"},
            "lane_divider": "cell intersects a shared nuPlan lane boundary of adjacent lanes in one lane group",
            "known": "all four cell corners within official drivable raster pixel extent",
            "repaired_area_features": world.repaired_areas,
        },
        "frames": len(frames),
        "ego_origins_in_drivable_area": road_origins,
        "known_cells": int(known.sum()),
        "drivable_cells": int(labels[:, 0].sum()),
        "divider_cells": int(labels[:, 1].sum()),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".ground-road-targets-", dir=output.parent) as tmp:
        stage = Path(tmp) / "ready"
        stage.mkdir()
        target = stage / "targets.npz"
        np.savez_compressed(target, labels=labels, known=known)
        summary["target_sha256"] = sha256(target.read_bytes()).hexdigest()
        write_json(stage / "targets.json", summary)
        check_components(output)
        if output.exists():
            raise ValueError("External road target output already exists")
        stage.rename(output)
    return summary


def _load_frames(index, archive, sources):
    if (index["schema"] != "hybrid-drive-box-motion-inputs-v1"
            or index["source"]["source_sha256"] != SOURCES_SHA256):
        raise ValueError("Unknown external observation cohort")
    selected = {log["member"]: log for log in sources["logs"]}
    available = {}
    for member, raw in _read_selected(archive, sources["metadata_archive"], set(selected), 48 * 1024**2):
        spec = selected[member]
        if len(raw) != spec["size_bytes"] or sha256(raw).hexdigest() != spec["sha256"]:
            raise ValueError("External OpenScene metadata member differs from its pin")
        rows = pickle.loads(raw)
        if (not isinstance(rows, list) or len(rows) != spec["frame_count"]
                or any(frame["log_name"] != spec["log_name"] for frame in rows)):
            raise ValueError("Metadata is not the expected original recording")
        log = _chronological(rows)
        for i in spec["anchor_indices"]:
            current = log[i]
            token = current["token"]
            if token in available or current["map_location"] != _LOCATION:
                raise ValueError("Unexpected map location or duplicate anchor")
            available[token] = (
                spec["recording"], [row["token"] for row in log[i-3:i]],
                current["timestamp"], _pose(current),
            )
    frames = []
    for anchor in index["anchors"]:
        token = anchor["sample_token"]
        recording, history, ref_time, pose = available.pop(token)
        observed = index["frames"][token]
        if (anchor["recording"] != recording or anchor["history_tokens"] != history
                or observed["recording"] != recording or observed["ref_time_us"] != ref_time
                or not np.allclose(rigid_transform(observed["ego_to_world"], "observed ego_to_world"),
                                   pose, rtol=0, atol=1e-9)):
            raise ValueError("OpenScene anchor identity or pose differs from the observed packet")
        frames.append(_Frame(token, recording, _LOCATION,
                             (float(pose[0, 3]), float(pose[1, 3]), atan2(pose[1, 0], pose[0, 0]))))
    if len(frames) != 40 or available:
        raise ValueError("Expected the complete two-recording holdout")
    return frames


def _signature(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _verify_maps(root, receipt, spec):
    if (receipt.get("schema") != "hybrid-drive-ground-nuplan-maps-v1" or receipt.get("url") != spec["url"]
            or receipt.get("archive_etag") != spec["etag"]):
        raise ValueError("Map download receipt does not reference the pinned publisher archive")
    pinned = {entry["path"]: entry for entry in spec["files"]}
    received = {entry["path"]: entry for entry in receipt["files"]}
    if len(receipt["files"]) != 2 or set(received) != {_MAP_FILE, _MAP_VERSION_PATH}:
        raise ValueError("Expected exactly the required Las Vegas files in the map receipt")
    result = {}
    for path, info in received.items():
        entry = pinned[path]
        if any(info[k] != entry[k] for k in ("member", "size_bytes", "crc32")):
            raise ValueError(f"Map download receipt drift: {path}")
        require_hex(info["sha256"], 64, "observed member SHA256")
        file = root / path
        check_components(file)
        with os.fdopen(os.open(file, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as stream:
            st = os.fstat(stream.fileno())
            signature = _signature(st)
            if not stat.S_ISREG(st.st_mode) or st.st_size != entry["size_bytes"]:
                raise ValueError(f"Map file has the wrong type or length: {path}")
            observed, crc = sha256(), 0
            for block in iter(lambda: stream.read(8 * 1024**2), b""):
                observed.update(block)
                crc = crc32(block, crc)
            if (f"{crc:08x}" != entry["crc32"] or observed.hexdigest() != info["sha256"]
                    or ("sha256" in entry and observed.hexdigest() != entry["sha256"])
                    or _signature(os.fstat(stream.fileno())) != signature):
                raise ValueError(f"Map file fails publisher pin or observed SHA256: {path}")
        result[path] = {"signature": signature, "receipt": info}
    return result


def _fid(value):
    if isinstance(value, str) and value.isascii() and value.isdecimal():
        return int(value)
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (float, np.floating)) and np.isfinite(value) and value.is_integer():
        return int(value)
    raise ValueError(f"Expected an integer map index, got {value!r}")


def _shared_divider_fids(rows):
    lanes = {}
    for group, index, left, right in rows:
        key = _fid(group), _fid(index)
        sides = _fid(left), _fid(right)
        if key in lanes and lanes[key] != sides:
            raise ValueError("Ambiguous lane positions inside a nuPlan lane group")
        lanes[key] = sides
    return {
        right for (group, index), (_, right) in lanes.items()
        if right > 0 and (group, index + 1) in lanes and lanes[(group, index + 1)][0] == right
    }


def _load_vector_map(path, world_to_pixel):
    import pyogrio
    from pyproj import CRS
    from shapely import STRtree, make_valid

    # Same metres/axis transform as GPKGMapsDB.load_vector_layer, without
    # instantiating MapsDB (its constructor preloads all four locations).
    meta = pyogrio.read_dataframe(path, layer="meta", columns=["key", "value"], read_geometry=False)
    projected, = meta.loc[meta["key"] == "projectedCoordSystem", "value"]
    crs = CRS.from_user_input(projected)
    if not crs.is_projected or any(not np.isclose(v.unit_conversion_factor, 1, rtol=0, atol=1e-12)
                                   for v in crs.axis_info[:2]):
        raise ValueError("nuPlan published projected coordinates must be in metres")

    def geometry(layer):
        rows = pyogrio.read_dataframe(path, layer=layer, columns=[], fid_as_index=True)
        if rows.crs is None or not rows.index.is_unique:
            raise ValueError(f"Invalid coordinate system or FIDs for nuPlan layer {layer}")
        return rows.geometry.to_crs(crs)

    areas, repaired = [], 0
    for layer in _DRIVABLE_LAYERS:
        for geom in geometry(layer):
            if geom is None or geom.is_empty:
                raise ValueError(f"Missing geometry in nuPlan vector layer {layer}")
            if not geom.is_valid:
                geom = make_valid(geom)
                repaired += 1
            polygons = list(_area_parts(geom))
            if not polygons:
                raise ValueError(f"No nonempty polygon in nuPlan vector layer {layer}")
            areas.extend(polygons)
    lanes = pyogrio.read_dataframe(path, layer="lanes_polygons", columns=list(_LANE_COLUMNS), read_geometry=False)
    fids = _shared_divider_fids(lanes.loc[:, list(_LANE_COLUMNS)].itertuples(index=False, name=None))
    boundaries = geometry("boundaries")
    if fids - set(boundaries.index):
        raise ValueError("A nuPlan shared lane boundary does not exist in the boundaries layer")
    dividers = list(boundaries.loc[sorted(fids)])
    if not areas or not dividers or any(
        v is None or v.is_empty or v.geom_type not in ("LineString", "MultiLineString") for v in dividers
    ):
        raise ValueError("Expected drivable polygons and shared lane boundaries")
    # Use interior of at least one feature, matching nuPlan is_in_layer;
    # do not fill holes or label the exterior of the published map.
    return _VectorMap(STRtree(areas), STRtree(dividers), world_to_pixel, _RASTER_SHAPE, crs.to_string(), repaired)


def _area_parts(geom):
    if geom.geom_type == "Polygon" and geom.area > 0:
        yield geom
    elif geom.geom_type in ("MultiPolygon", "GeometryCollection"):
        for part in geom.geoms:
            yield from _area_parts(part)


def _world_grid(grid, pose):
    x0, y0, yaw = pose
    step = grid["step_m"]
    xs = grid["x_min"] + (np.arange(grid["rows"]) + .5) * step
    ys = grid["y_min"] + (np.arange(grid["cols"]) + .5) * step
    centres = np.stack(np.meshgrid(xs, ys, indexing="ij"), axis=-1).reshape(-1, 2)
    corners = centres[:, None] + step / 2 * np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
    c, s = cos(yaw), sin(yaw)
    to_world = np.array([[c, s], [-s, c]])
    return centres @ to_world + (x0, y0), corners @ to_world + (x0, y0)


def _map_known(corners, transform, shape):
    if (not np.array_equal(transform[3], [0, 0, 0, 1])
            or abs(np.linalg.det(transform[:2, :2])) <= 1e-12):
        raise ValueError("Expected an invertible affine world-to-pixel transform")
    pixels = corners @ transform[:2, :2].T + transform[:2, 3]
    # Conservative extent: don't round individual corners to extend bounds.
    rows, cols = shape
    return ((pixels >= 0) & (pixels <= np.array([cols - 1, rows - 1]))).all(axis=(1, 2))


def _rasterize(world, grid, pose):
    from shapely import points, polygons
    centres, corners = _world_grid(grid, pose)
    known = _map_known(corners, world.world_to_pixel, world.shape)
    labels = np.zeros((2, len(centres)), dtype=np.bool_)
    labels[0, world.areas.query(points(centres), predicate="within")[0]] = True
    labels[1, world.dividers.query(polygons(corners), predicate="intersects")[0]] = True
    labels[:, ~known] = False
    shape = (grid["rows"], grid["cols"])
    return labels.reshape(2, *shape), known.reshape(shape)
