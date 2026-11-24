from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
from math import pi
from zipfile import ZipFile

import numpy as np
import pytest

from hybrid_drive.experiments import road_map_targets as roads


_LOCATION = "boston-seaport"


@dataclass(frozen=True)
class Grid:
    x_min: float = 0.0
    y_min: float = 0.0
    step_m: float = 2.0
    rows: int = 2
    cols: int = 2

    def xy(self):
        return np.meshgrid([1.0, 3.0], [1.0, 3.0], indexing="ij")


def _map_id(n):
    return f"00000000-0000-4000-8000-{n:012x}"


def _world_map(*, canvas=(9, 9), lane=((5.5, 2.2), (5.5, 5.8)), polygon_token=_map_id(98)):
    points = [(2, 2), (6.5, 2), (6.5, 6), (2, 6), (2.5, 4.5), (2.5, 5.5), (3.5, 5.5), (3.5, 4.5), *lane]
    nodes = [{"token": _map_id(i), "x": x, "y": y} for i, (x, y) in enumerate(points)]
    return {
        "version": "1.3",
        "canvas_edge": list(canvas),
        "node": nodes,
        "polygon": [{
            "token": polygon_token, "exterior_node_tokens": [_map_id(i) for i in range(4)],
            "holes": [{"node_tokens": [_map_id(i) for i in range(4, 8)]}],
        }],
        "line": [{"token": _map_id(99), "node_tokens": [_map_id(i) for i in range(8, len(nodes))]}],
        "drivable_area": [{"polygon_tokens": [polygon_token]}],
        "lane_divider": [{"line_token": _map_id(99)}],
    }


def test_rotated_rows_forward_columns_left_and_hole_matches_map():
    pytest.importorskip("shapely")
    # After a +90 degree rotation, x_world decreases leftward across columns.
    # Row 1, column 1 is (3, 5), inside the hole. The 5.5 m lane intersects
    # column 0 although it passes through none of the sampled centres.
    labels, known = roads._rasterize(roads._load_vector_map(_world_map()), Grid(), (6.0, 2.0, pi / 2))
    np.testing.assert_array_equal(known, [[True, True], [True, True]])
    np.testing.assert_array_equal(labels[0], [[True, True], [True, False]])
    np.testing.assert_array_equal(labels[1], [[True, False], [True, False]])


def test_mask_requires_whole_rotated_cell_inside_map_canvas():
    pytest.importorskip("shapely")
    layers = roads._load_vector_map(_world_map(canvas=(6, 5.5)))
    labels, known = roads._rasterize(layers, Grid(), (6.0, 2.0, pi / 2))
    # The second row has both centres inside the map but exits the canvas.
    np.testing.assert_array_equal(known, [[True, True], [False, False]])
    np.testing.assert_array_equal(labels[:, 1], False)


def test_lane_touching_shared_cell_edge_marks_each_intersected_closed_cell():
    pytest.importorskip("shapely")
    obj = _world_map(lane=((4, 2.25), (4, 3.7)))
    labels, _ = roads._rasterize(roads._load_vector_map(obj), Grid(), (2.0, 2.0, 0.0))
    # The divider is along the border between metric rows 0 and 1.
    np.testing.assert_array_equal(labels[1], [[True, False], [True, False]])


def test_world_overlap_counts_frames_on_same_map_with_positive_area():
    pytest.importorskip("shapely")
    frame = roads._Frame
    frames = {
        "train": [frame("t", _LOCATION, (0, 0, 0))],
        "dev": [frame("d", _LOCATION, (4, 0, 0))],
        "test": [frame("hit", _LOCATION, (2, 0, 0)), frame("other-map", "singapore-onenorth", (0, 0, 0))],
    }
    assert roads._world_patch_overlap(frames, Grid()) == {
        "dev_frames_overlapping_train": 0, "test_frames_overlapping_train": 1,
        "test_frames_overlapping_dev": 1,
    }


def _observations(tmp_path):
    tables = {name: {} for name in ("sample", "scene", "log")}
    captured = {"splits": {}}
    for s in roads.SPLITS:
        sdir = tmp_path / s
        sdir.mkdir()
        j = f"anchor-{s}"
        row = {"sample_token": j, "recording": f"{s}-log", "scene": s, "ego_world_xy_yaw": [6.0, 2.0, pi/2]}
        tables["sample"][j] = {"scene_token": s}
        tables["scene"][s] = {"name": s, "log_token": row["recording"]}
        tables["log"][row["recording"]] = {"location": _LOCATION}
        data = json.dumps([row]).encode()
        (sdir / "observations.json").write_bytes(data)
        captured["splits"][s] = {"count": 1, "artifacts": {"observations.json": sha256(data).hexdigest()}}
    return tables, captured


def test_capture_is_bound_to_observation_digests_and_authoritative_log(tmp_path):
    tables, captured = _observations(tmp_path)
    frames, _ = roads._capture_frames(tmp_path, captured, tables)
    assert [frames[s][0].token for s in roads.SPLITS] == ["anchor-train", "anchor-dev", "anchor-test"]
    tables["scene"]["test"]["log_token"] = "train-log"
    with pytest.raises(ValueError, match="scene/log"):
        roads._capture_frames(tmp_path, captured, tables)
    p = tmp_path / "test" / "observations.json"
    p.write_bytes(p.read_bytes() + b" ")
    with pytest.raises(ValueError, match="manifest"):
        roads._capture_frames(tmp_path, captured, tables)


def test_zip_requires_one_exact_bounded_vector_location():
    archive = BytesIO()
    with ZipFile(archive, "w") as z:
        z.writestr("basemap/boston-seaport.json", "{}")
        z.writestr("expansion/not-the-map.json", b'{"version":"1.3"}')
    with ZipFile(archive) as z:
        with pytest.raises(ValueError, match="member"):
            roads._read_vector_member(z, _LOCATION)
        with pytest.raises(ValueError, match="location"):
            roads._read_vector_member(z, "../file")


def test_publish_mountable_labels_and_verify_npz_receipts(tmp_path, monkeypatch):
    pytest.importorskip("shapely")
    pytest.importorskip("torch")
    capture = tmp_path / "capture"
    capture.mkdir()
    tables, captured = _observations(capture)
    captured.update(schema="hybrid-drive-fast-capture-v1", targets_opened=False,
                    raw_source_sha256=roads._ARCHIVE_SHA256,
                    grid={"x_min": 0, "y_min": 0, "step_m": 2, "rows": 2, "cols": 2, "heights_m": [0.8]})
    raw = json.dumps(captured).encode()
    (capture / "capture.json").write_bytes(raw)
    maps = tmp_path / "maps.zip"
    with ZipFile(maps, "w") as z:
        z.writestr("expansion/boston-seaport.json", json.dumps(_world_map()))
    monkeypatch.setattr(roads, "_MAP_BYTES", maps.stat().st_size)
    monkeypatch.setattr(roads, "_read_sample_tables", lambda _: tables)
    output = tmp_path / "labels"
    sha = sha256(maps.read_bytes()).hexdigest()
    summary = roads.prepare_road_map_targets(capture, tmp_path / "mini.tgz", maps, sha, output)
    assert summary["capture_sha256"] == sha256(raw).hexdigest()
    for split in roads.SPLITS:
        tar = output / split / "targets.npz"
        manifest = json.loads((output / split / "targets.json").read_bytes())
        assert manifest["tokens"] == [f"anchor-{split}"]
        assert manifest["target_sha256"] == sha256(tar.read_bytes()).hexdigest()
        with np.load(tar, allow_pickle=False) as data:
            assert data["labels"].shape == (1, 2, 2, 2)
            assert data["labels"].dtype == data["known"].dtype == np.bool_
            assert data["known"].all()
            np.testing.assert_array_equal(data["labels"][0, 0], [[True, True], [True, False]])
    with pytest.raises(ValueError, match="already exists"):
        roads.prepare_road_map_targets(capture, tmp_path / "mini.tgz", maps, sha, output)



def test_provider_null_polygon_keeps_geometry_and_an_empty_placeholder_does_not():
    pytest.importorskip("shapely")
    obj = _world_map(polygon_token=None)
    obj["polygon"].extend([{"token": "", "exterior_node_tokens": [], "holes": []},
                           {**obj["polygon"][0], "token": _map_id(98)}])
    obj["drivable_area"].append({"polygon_tokens": [_map_id(98)]})
    labels, _ = roads._rasterize(roads._load_vector_map(obj), Grid(), (6.0, 2.0, pi / 2))
    np.testing.assert_array_equal(labels[0], [[True, True], [True, False]])


def test_vector_identifiers_are_not_mini_sample_ids_and_duplicate_null_is_rejected():
    with pytest.raises(ValueError, match="Invalid"):
        roads._map_token_index([{"token": "a"*32}],"node")
    with pytest.raises(ValueError, match="Duplicate"):
        roads._map_token_index([{"token":None},{"token":None}],"polygon")
