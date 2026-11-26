from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import pytest

from hybrid_drive.assets.files import PreparationError
from hybrid_drive.experiments import vru_clip_data as data


def _clip(group, number, bands, shard="1", *, time_us=None):
    name = f"2021.01.{group:02d}.01.02.03_veh-00"
    tokens = tuple(f"{group * 10000 + number * 10 + i:016x}" for i in range(3))
    return data._Clip(shard, name + "_00000_99999", name, number * 3_600_000 if time_us is None else time_us,
                      tokens, tuple(data._Primary(band, f"{group:016x}", "still") for band in bands))


def test_pinned_catalog_has_twelve_fresh_groups_and_rejects_changes(tmp_path):
    source = Path(__file__).resolve().parents[2] / "scripts" / "vru-120-sources.json"
    catalog = data._load_source(source)
    groups = {data._group(entry["recording"]) for entry in catalog["logs"]}
    assert len(groups) == 12
    assert not groups.intersection(map(data._group, catalog["previous_recordings"]))
    edited = tmp_path / "altered.json"
    edited.write_bytes(source.read_bytes().replace(b"OpenScene v1.1 Mini", b"Mini"))
    with pytest.raises(PreparationError, match="checked-in clip corpus source"):
        data._load_source(edited)


def test_flow_reroutes_flexible_slots_and_enforces_exact_clips_per_band():
    # If the first 20 are greedily assigned to band 0, the two exclusively
    # band-0 blocks cannot fit. The augmenting path must reassign them.
    patterns = [(0, 1), (0,), (0,), (1,), (2,), (2,)]
    options = [_clip(group, j, bands) for group, bands in enumerate(patterns, 1) for j in range(20)]
    result = data._assign(data._slots(options))
    assert result is not None
    assert Counter(option.band for _, option in result) == {0: 40, 1: 40, 2: 40}
    assert len({clip.group for clip, _ in result}) == 6
    assert all(option.band == 1 for clip, option in result if clip.group == options[0].group)


def test_lowest_bytes_feasible_shards_not_fewest_shards():
    cheap = [_clip(group, j, ((group-1)//2,), "1" if group < 4 else "2")
             for group in range(1, 7) for j in range(20)]
    pricey = [_clip(group, j, ((group-7)//2,), "3") for group in range(7, 13) for j in range(20)]
    archives = {"1": {"size_bytes": 4}, "2": {"size_bytes": 6}, "3": {"size_bytes": 11}}
    result = data._select(pricey + cheap, archives)
    assert {clip.shard for clip, _ in result} == {"1", "2"}
    assert data._select(list(reversed(cheap + pricey)), archives) == result


def test_cap_and_a_sixth_group_are_enforced_even_when_five_can_fill_the_bands():
    options = [_clip(group, j, (0, 1, 2)) for group in range(1, 8) for j in range(30)]
    result = data._assign(data._slots(options))
    assert result is not None
    counts = Counter(clip.group for clip, _ in result)
    assert len(counts) >= 6
    assert max(counts.values()) <= 24
    assert Counter(option.band for _, option in result) == {0: 40, 1: 40, 2: 40}


def test_spaced_slots_make_all_band_alternatives_nonoverlapping():
    options = [_clip(1, j, (j % 3,), time_us=stamp)
               for j, stamp in enumerate((0, 500_000, 600_000, 3_001_000, 3_500_000, 3_980_000, 7_000_000))]
    slots = next(iter(data._slots(options).values()))
    assert len(slots) == 3
    assert set(slots[0]) == {0, 1}
    assert all(right.stamp - left.stamp >= 3_000_000
               for a, b in combinations(slots, 2) for left, _ in a.values() for right, _ in b.values())


def test_clip_label_coordinates_align_history_to_current_ego(monkeypatch):
    ident, distractor = "a" * 16, "b" * 16
    monkeypatch.setattr(data, "_pose", lambda frame: frame["pose"])
    monkeypatch.setattr(data, "_tracks", lambda frame: frame["tracks"])
    frames = []
    for origin, current in zip((7.0, 8.0, 10.0), (8.0, 9.0, 10.0), strict=True):
        pose = np.eye(4)
        pose[0, 3] = origin
        frames.append({"pose": pose, "tracks": {ident: ("pedestrian", np.array([current, 1.0, 2.0])),
                                                 distractor: ("bicycle", np.array([12.0, 2.0, 1.0]))}})
    clip = {"sample_token": "t", "recording": "recording", "primary_id": ident}
    label = data._label_row(clip, frames)
    assert label["frames"]["older"][0] == {"track_id": ident, "kind": "person", "xy_m": [5.0, 1.0]}
    assert label["frames"]["previous"][0]["xy_m"] == [7.0, 1.0]
    assert label["frames"]["current"][0]["xy_m"] == [10.0, 1.0]
    assert label["frames"]["current"][1]["kind"] == "bicycle"


def test_labels_bind_anchor_and_require_same_primary_at_all_observations(monkeypatch):
    monkeypatch.setattr(data, "_TOTAL", 3)
    monkeypatch.setattr(data, "_PER_BAND", 1)
    index = {"anchors": [{"sample_token": str(i), "recording": "recording"} for i in range(3)]}
    token = "a" * 16
    rows = [{**row, "primary_id": token, "frames": {
        phase: [{"track_id": token, "kind": "person", "xy_m": [distance, 0.0]}] for phase in data._PHASES
    }} for row, distance in zip(index["anchors"], (4, 10, 20), strict=True)]
    labels = {"schema": data.LABEL_SCHEMA, "input_sha256": "bound", "rows": rows}
    data._validate_labels(labels, index, "bound")
    with pytest.raises(ValueError, match="exact frozen index"):
        data._validate_labels(labels, index, "different")
    rows[0]["frames"]["older"] = []
    with pytest.raises(ValueError, match="primary annotation"):
        data._validate_labels(labels, index, "bound")
