from hashlib import sha256
from math import pi
from zlib import crc32

import numpy as np
import pytest

from hybrid_drive.experiments import ground_road_targets as targets


_GRID = {"x_min": 0.0, "y_min": 0.0, "step_m": 2.0, "rows": 2, "cols": 2}


def _tiny_map(shape=(10, 10)):
    g = pytest.importorskip("shapely")
    road = g.Polygon([(2, 2), (6.5, 2), (6.5, 6.5), (2, 6.5)],
                     holes=[[(2.5, 4.5), (3.5, 4.5), (3.5, 5.5), (2.5, 5.5)]])
    line = g.LineString([(5.5, 2.1), (5.5, 5.9)])
    # Vertically flipped y axis, as in a nuPlan raster matrix.
    matrix = np.diag([1.0, -1.0, 1.0, 1.0])
    matrix[1, 3] = shape[0] - 1
    return targets._VectorMap(g.STRtree([road]), g.STRtree([line]), matrix, shape, "synthetic", 0)


def test_bev_rows_move_forward_columns_left_and_polygon_holes_remain_unknown_to_road():
    labels, known = targets._rasterize(_tiny_map(), _GRID, (6.0, 2.0, pi / 2))
    np.testing.assert_array_equal(known, [[True, True], [True, True]])
    np.testing.assert_array_equal(labels[0], [[True, True], [True, False]])
    np.testing.assert_array_equal(labels[1], [[True, False], [True, False]])


def test_world_cells_with_only_centre_inside_raster_are_not_map_known():
    labels, known = targets._rasterize(_tiny_map((6, 8)), _GRID, (6.0, 2.0, pi / 2))
    # Row 1 reaches beyond the official footprint despite having centres in it.
    np.testing.assert_array_equal(known, [[True, True], [False, False]])
    np.testing.assert_array_equal(labels[:, 1], False)


def test_only_shared_boundary_of_adjacent_lanes_inside_one_group_is_a_divider():
    rows = [
        ("17", 0, 101, 102), (17, 1, 102, 110),
        (18, 0, 110, 115), (18, 2, 115, 116),  # Not adjacent lane indices.
        (19, 0, 116, 117), (19, 1, 987, 118),  # Do not infer a divider from proximity.
        (20, 0, 119, -1), (20, 1, -1, 120),    # Invalid/missing upstream FID.
    ]
    assert targets._shared_divider_fids(rows) == {102}
    with pytest.raises(ValueError, match="Ambiguous"):
        targets._shared_divider_fids([*rows, (17, 1, 903, 904)])


def test_download_receipt_cannot_replace_published_crc_with_self_reported_sha(tmp_path):
    received, pinned = [], []
    for path, data in ((targets._MAP_FILE, b"map"), (targets._MAP_VERSION_PATH, b"version")):
        p = tmp_path / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        spec = {"path": path, "member": "archive/" + path, "size_bytes": len(data),
                "crc32": f"{crc32(data):08x}"}
        if path == targets._MAP_VERSION_PATH:
            spec["sha256"] = sha256(data).hexdigest()
        pinned.append(spec)
        received.append({**spec, "sha256": sha256(data).hexdigest()})
    spec = {"url": "https://publisher.invalid/maps.zip", "etag": "pinned", "files": pinned}
    receipt = {"schema": "hybrid-drive-ground-nuplan-maps-v1", "url": spec["url"],
               "archive_etag": spec["etag"], "files": received}
    assert set(targets._verify_maps(tmp_path, receipt, spec)) == {targets._MAP_FILE, targets._MAP_VERSION_PATH}
    (tmp_path / targets._MAP_FILE).write_bytes(b"bad")
    received[0]["sha256"] = sha256(b"bad").hexdigest()
    with pytest.raises(ValueError, match="publisher pin"):
        targets._verify_maps(tmp_path, receipt, spec)


def test_gpkg_loader_projects_vectors_and_selects_a_shared_boundary_fid(tmp_path):
    pyogrio = pytest.importorskip("pyogrio")
    gpd = pytest.importorskip("geopandas")
    pd = pytest.importorskip("pandas")
    g = pytest.importorskip("shapely")
    from shapely.affinity import translate
    path = tmp_path / "example.gpkg"
    crs = "EPSG:32611"
    x, y = 665000, 3990000
    pyogrio.write_dataframe(pd.DataFrame({"key": ["projectedCoordSystem"], "value": [crs]}),
                             path, layer="meta")
    shape = translate(g.box(0.0, 0.0, 4.9, 4.9), xoff=x, yoff=y)
    for layer in targets._DRIVABLE_LAYERS:
        frame = gpd.GeoDataFrame({"geometry": [shape]}, crs=crs).to_crs("EPSG:4326")
        pyogrio.write_dataframe(frame, path, layer=layer)
    lanes = gpd.GeoDataFrame({
        "lane_group_fid": [10, 10], "lane_index": [0, 1],
        "left_boundary_fid": [1, 2], "right_boundary_fid": [2, 3],
        "geometry": [shape, shape],
    }, crs=crs).to_crs("EPSG:4326")
    pyogrio.write_dataframe(lanes, path, layer="lanes_polygons")
    offset_lines = [translate(g.LineString([(k, .25), (k, 3.75)]), xoff=x, yoff=y)
                    for k in (0.4, 3.2, 4.6)]
    lines = gpd.GeoDataFrame({"geometry": offset_lines}, crs=crs).to_crs("EPSG:4326")
    pyogrio.write_dataframe(lines, path, layer="boundaries")
    transform = np.diag([1.0, -1.0, 1.0, 1.0])
    transform[:2, 3] = [-x, y+5]
    world = targets._load_vector_map(path, transform)
    labels, known = targets._rasterize(world, _GRID, (x, y, 0.0))
    np.testing.assert_array_equal(known, True)
    np.testing.assert_array_equal(labels[0], True)
    np.testing.assert_array_equal(labels[1], [[False, False], [True, True]])
