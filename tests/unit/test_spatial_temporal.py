from copy import deepcopy
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import pickle
import tarfile
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from hybrid_drive.assets import alignment_sample, spatial_manifest, spatial_sample, spatial_temporal
from hybrid_drive.assets.alignment_manifest import ARCHIVE_VERIFICATION, digest_json
from hybrid_drive.assets.files import PreparationError
from hybrid_drive.assets.spatial_temporal import read_temporal_spatial_pack, verify_independent_temporal_final


SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _spec(role):
    profile = role.replace("_", "-")
    return json.loads((SCRIPTS / f"spatial-temporal-{profile}-sources.json").read_text())


def _archive(path, declaration, entries):
    with path.open("wb") as out, gzip.GzipFile(fileobj=out, mode="wb", mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode="w", format=tarfile.USTAR_FORMAT) as archive:
            for name, data in entries:
                member = tarfile.TarInfo(name)
                member.size = len(data)
                archive.addfile(member, io.BytesIO(data))
    data = path.read_bytes()
    declaration.update(size_bytes=len(data), sha256=_sha(data))


def _persist_manifest(root, manifest):
    manifest["manifest_hash"] = digest_json({k: v for k, v in manifest.items() if k != "manifest_hash"})
    (root / spatial_manifest.MANIFEST_NAME).write_text(json.dumps(manifest))


@pytest.fixture()
def archives(tmp_path, monkeypatch, request):
    role = getattr(request, "param", "holdout")

    def camera(_, native, *args, width, height, **kwargs):
        # Rectification is previously covered; preserve the fixture pixels so
        # this test exercises the actual image export and archive admission.
        prepared = Image.fromarray(native).resize((width, height))
        return SimpleNamespace(
            frame=SimpleNamespace(rgb=np.asarray(prepared)),
            calibration=SimpleNamespace(intrinsic=np.eye(3), ego_to_camera=np.eye(4)),
        )

    monkeypatch.setattr(alignment_sample, "prepare_openscene_camera", camera)
    monkeypatch.setattr(spatial_sample, "sys", SimpleNamespace(platform="linux"))
    sources = _spec("development")
    sources["temporal_role"] = role
    shard = "4"
    prefix = sources["camera_archives"][shard]["archive_prefix"]
    entries, frames_by_recording = [], {}
    for number, log in enumerate(sources["logs"]["holdout"], start=1):
        log["frame_count"] = 90 + 2 * number
        selected = {idx for clip in spatial_manifest.temporal_clip_indices(log["frame_count"]) for idx in clip}
        frames = []
        for index in range(log["frame_count"]):
            token = f"{1000 * number + index:016x}"
            path = f"{log['log_name']}/CAM_F0/{token}.jpg"
            frames.append({
                "token": token, "timestamp": 1000000 + 500000 * index,
                "frame_idx": index, "scene_token": f"{number:016x}", "log_token": f"{number:016x}",
                "log_name": log["log_name"],
                "cams": {"CAM_F0": {
                    "data_path": path, "cam_intrinsic": np.eye(3),
                    "sensor2lidar_rotation": np.eye(3), "sensor2lidar_translation": [0, 0, 0],
                    "distortion": [0, 0, 0, 0, 0],
                }} if index in selected else {},
                "anns": {"gt_names": [], "gt_boxes": np.empty((0, 7))},
            })
            if index in selected:
                image = io.BytesIO()
                Image.new("RGB", (16, 12), (number * 60, index * 2, 90)).save(image, format="JPEG")
                entries.append((f"{prefix}/{path}", image.getvalue()))
        frames_by_recording[log["recording_id"]] = frames

    raw_metadata, raw_cameras = tmp_path / "metadata.tgz", tmp_path / "cameras.tgz"
    root = tmp_path / "pack"

    def prepare():
        logs = []
        for log in sources["logs"]["holdout"]:
            data = pickle.dumps(frames_by_recording[log["recording_id"]])
            log.update(size_bytes=len(data), sha256=_sha(data))
            logs.append((log["member"], data))
        _archive(raw_metadata, sources["metadata_archive"], logs)
        _archive(raw_cameras, sources["camera_archives"][shard], entries)
        monkeypatch.setitem(spatial_manifest._TEMPORAL_SOURCE_HASHES, role, digest_json(sources))
        return spatial_sample.prepare_spatial_sample(root, sources, raw_metadata, {shard: raw_cameras})

    return root, sources, frames_by_recording, entries, prepare


def test_temporal_source_profiles_pin_distinct_roles_but_reuse_spatial_ground_truth():
    expected = {
        "development": "4", "holdout": "22", "stress": "19", "tracking_holdout": "23",
        "appearance_holdout": "3", "zoom_gate_holdout": "1",
    }
    profiles = [
        spatial_manifest.load_spatial_sources(SCRIPTS / f"spatial-temporal-{role.replace('_', '-')}-sources.json")
        for role in expected
    ]
    assert tuple(expected) == spatial_manifest.TEMPORAL_ROLES
    assert [profile["temporal_role"] for profile in profiles] == list(expected)
    assert [list(profile["camera_archives"]) for profile in profiles] == [[shard] for shard in expected.values()]
    assert all(profile["rgb"] == profiles[0]["rgb"] and profile["loc"] == profiles[0]["loc"] for profile in profiles)
    assert all(len(profile["logs"]["holdout"]) == 2 for profile in profiles)
    assert all(log["sample_count"] == 72 for profile in profiles for log in profile["logs"]["holdout"])



def test_zoom_gate_holdout_sources_do_not_overlap_any_prior_spatial_pack():
    source = spatial_manifest.load_spatial_sources(SCRIPTS / "spatial-temporal-zoom-gate-holdout-sources.json")
    recordings = source["logs"]["holdout"]
    names = [recording["log_name"] for recording in recordings]
    assert names == ["2021.10.05.07.10.04_veh-52_01442_01802", "2021.10.01.19.16.42_veh-28_02011_02410"]
    assert source["publisher_mapping"]["selected_entries"] == {"openscene_sensor_mini_1": names}
    assert [recording["camera_shard"] for recording in recordings] == [1, 1]
    assert [spatial_manifest.spatial_frame_indices(source, recording) for recording in recordings] == [
        (*range(108, 132), *range(348, 372), *range(588, 612)),
        (*range(121, 145), *range(387, 411), *range(653, 677)),
    ]

    def groups(profile):
        return {
            (log["recording_id"][:10], log["recording_id"].rsplit("_", 1)[1])
            for split in profile["logs"].values() for log in split
        }

    previous = (
        "spatial-loc", "spatial-loc-evaluation", "spatial-loc-policy-evaluation",
        "spatial-temporal-development", "spatial-temporal-holdout", "spatial-temporal-stress",
        "spatial-temporal-tracking-holdout", "spatial-temporal-appearance-holdout",
    )
    scope = ("dataset", "openscene_revision", "navsim_revision", "nuplan_revision", "rgb", "loc")
    fresh_groups = groups(source)
    assert len(fresh_groups) == 2
    for name in previous:
        earlier = spatial_manifest.load_spatial_sources(SCRIPTS / f"{name}-sources.json")
        assert fresh_groups.isdisjoint(groups(earlier)), name
        assert all(source[key] == earlier[key] for key in scope), name


@pytest.mark.parametrize("change", ["role", "recording", "member", "archive", "selection"])
def test_zoom_gate_holdout_rejects_edits_to_the_frozen_source_facts(change):
    source = _spec("zoom_gate_holdout")
    if change == "role":
        source["temporal_role"] = "appearance_holdout"
    elif change == "recording":
        source["logs"]["holdout"][0] = _spec("appearance_holdout")["logs"]["holdout"][0]
    elif change == "member":
        source["logs"]["holdout"][0]["sha256"] = "0" * 64
    elif change == "archive":
        source["camera_archives"]["1"]["size_bytes"] -= 1
    else:
        source["selection"]["frames_per_clip"] -= 1
    with pytest.raises(PreparationError, match="pinned recordings"):
        spatial_manifest.validate_spatial_sources(source)


@pytest.mark.parametrize("change", ["role", "sampling", "archive"])
def test_temporal_source_role_and_payload_are_pinned(change):
    source = _spec("development")
    if change == "role":
        source["temporal_role"] = "holdout"
    elif change == "sampling":
        source["selection"]["frames_per_clip"] -= 1
    else:
        source["camera_archives"]["4"]["size_bytes"] += 1
    with pytest.raises(PreparationError, match="pinned recordings"):
        spatial_manifest.validate_spatial_sources(source)


def test_clips_are_exact_midpoint_windows_without_gaps_inside_each_clip():
    windows = spatial_manifest.temporal_clip_indices(172)
    assert [(window.start, window.stop) for window in windows] == [(16, 40), (74, 98), (131, 155)]
    shortest = spatial_manifest.temporal_clip_indices(72)
    assert [(window.start, window.stop) for window in shortest] == [(0, 24), (24, 48), (48, 72)]
    with pytest.raises(PreparationError, match="too short"):
        spatial_manifest.temporal_clip_indices(71)


def test_preparation_reuses_spatial_labels_and_records_exact_clip_memberships(archives):
    root, sources, _, _, prepare = archives
    manifest = prepare()
    assert prepare() == manifest
    assert manifest["archive_verification"] == ARCHIVE_VERIFICATION
    pack = read_temporal_spatial_pack(root, role="holdout")
    assert pack.manifest_hash == manifest["manifest_hash"]
    assert pack.root == root and len(pack.frames) == 144
    assert len({row.clip_id for row in pack.frames}) == 6
    assert len({row.sample_id for row in pack.frames}) == 144
    for offset, spec in enumerate(sources["logs"]["holdout"]):
        recording = pack.frames[offset * 72:(offset + 1) * 72]
        for clip_no, window in enumerate(spatial_manifest.temporal_clip_indices(spec["frame_count"])):
            clip = recording[clip_no * 24:(clip_no + 1) * 24]
            assert [row.source_index for row in clip] == list(window)
            assert {row.recording_id for row in clip} == {spec["recording_id"]}
            assert {row.clip_id for row in clip} == {f"{spec['recording_id']}-clip-{clip_no:02d}"}
            assert all(right.timestamp_us - left.timestamp_us == 500000 for left, right in zip(clip, clip[1:]))
    assert spatial_manifest.read_spatial_rgb(root, pack.frames[0].rgb).shape == (224, 224, 3)
    targets = spatial_manifest.read_spatial_targets(root, pack.manifest, "holdout")
    assert [target["sample_id"] for target in targets] == [row.sample_id for row in pack.frames]
    assert all(target["presence"] == [False] * 4 and target["boxes_2d"] == [None] * 4 for target in targets)
    assert not list(root.rglob("*.jpg")) and not list(root.rglob("*.pkl"))


def test_reading_timeline_and_checking_roles_never_opens_targets_or_images(archives, monkeypatch):
    root, _, _, _, prepare = archives
    prepare()
    original = os.open

    def guard(path, *args, **kwargs):
        if not isinstance(path, int) and Path(path).parent in (root / "targets", root / "rgb"):
            pytest.fail("Reading a temporal timeline opened a label or image")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(os, "open", guard)
    assert len(read_temporal_spatial_pack(root, role="holdout").frames) == 144
    with pytest.raises(PreparationError, match="pinned temporal spatial role"):
        read_temporal_spatial_pack(root, role="development")


@pytest.mark.parametrize("change", ["early", "late", "index", "order", "missing"])
def test_manifest_cannot_fake_timed_clips_even_with_recomputed_hashes(archives, change):
    root, _, _, _, prepare = archives
    manifest = prepare()
    samples = manifest["splits"]["holdout"]["recordings"][0]["samples"]
    if change == "early":
        samples[1]["timestamp_us"] = samples[0]["timestamp_us"] + 249999
    elif change == "late":
        samples[1]["timestamp_us"] = samples[0]["timestamp_us"] + 800001
    elif change == "index":
        samples[24]["source_list_index"] += 1
    elif change == "order":
        samples[23], samples[24] = samples[24], samples[23]
    else:
        samples.pop(2)
    _persist_manifest(root, manifest)
    with pytest.raises(PreparationError, match="timestamp delta|frame order|frame count|sample count"):
        read_temporal_spatial_pack(root, role="holdout")


def test_prepare_rejects_broken_cadence_without_export_or_substitution(archives, monkeypatch):
    root, sources, frames, _, prepare = archives
    first = sources["logs"]["holdout"][0]
    start = spatial_manifest.temporal_clip_indices(first["frame_count"])[0].start
    first_clip = frames[first["recording_id"]]
    first_clip[start + 1]["timestamp"] = first_clip[start]["timestamp"] + 25000

    def never_extract(*args, **kwargs):
        pytest.fail("Invalid clip must be rejected before camera extraction")

    monkeypatch.setattr(spatial_sample, "_extract_spatial_cameras", never_extract)
    with pytest.raises(PreparationError, match="timestamp delta"):
        prepare()
    assert not root.exists()


def test_missing_exact_clip_jpeg_does_not_select_a_nearby_frame(archives):
    root, _, _, entries, prepare = archives
    del entries[12]
    with pytest.raises(PreparationError, match="allowlisted JPEGs"):
        prepare()
    assert not root.exists() and not list(root.parent.glob(".spatial-loc-*"))


def _earlier(manifest):
    previous = deepcopy(manifest)
    previous["manifest_hash"] = "another-prepared-manifest"
    previous["sources"]["selection"] = {"rule": "old-sparse-selection"}
    for index, item in enumerate(previous["splits"]["holdout"]["recordings"], start=1):
        item["recording_id"] = f"2020.06.01.12.00.0{index}_veh-{index}"
        item["log_token"] = f"earlier-log-{index}"
        for row in item["samples"]:
            row["frame_token"] = "earlier-" + row["frame_token"]
            row["jpeg"]["member"] = "earlier/" + row["jpeg"]["member"]
            row["jpeg"]["sha256"] = "earlier-" + row["jpeg"]["sha256"]
            row["rgb"]["pixels_sha256"] = "earlier-" + row["rgb"]["pixels_sha256"]
    return previous


def test_independence_accepts_same_protocol_with_different_clip_selection(archives, monkeypatch):
    root, _, _, _, prepare = archives
    prepare()
    final = read_temporal_spatial_pack(root, role="holdout")
    earlier = _earlier(final.manifest)
    inspected = []

    def read(path):
        inspected.append(path)
        return earlier

    monkeypatch.setattr(spatial_temporal, "read_spatial_manifest", read)
    verify_independent_temporal_final(final, ("old-sparse", "old-stress", "old-dev"))
    assert inspected == ["old-sparse", "old-stress", "old-dev"]


@pytest.mark.parametrize("overlap", ["group", "log", "frame", "jpeg-member", "jpeg-sha", "rgb-sha", "protocol"])
def test_independence_rejects_overlap_with_any_earlier_pack(archives, monkeypatch, overlap):
    root, _, _, _, prepare = archives
    prepare()
    final = read_temporal_spatial_pack(root, role="holdout")
    first, another = _earlier(final.manifest), _earlier(final.manifest)
    earlier_recording = another["splits"]["holdout"]["recordings"][0]
    actual_recording = final.manifest["splits"]["holdout"]["recordings"][0]
    actual, earlier = actual_recording["samples"][0], earlier_recording["samples"][0]
    if overlap == "group":
        identity, vehicle = actual_recording["recording_id"].rsplit("_", 1)
        earlier_recording["recording_id"] = identity[:10] + ".23.59.59_" + vehicle
    elif overlap == "log":
        earlier_recording["log_token"] = actual_recording["log_token"]
    elif overlap == "frame":
        earlier["frame_token"] = actual["frame_token"]
    elif overlap == "jpeg-member":
        earlier["jpeg"]["member"] = actual["jpeg"]["member"]
    elif overlap == "jpeg-sha":
        earlier["jpeg"]["sha256"] = actual["jpeg"]["sha256"]
    elif overlap == "rgb-sha":
        earlier["rgb"]["pixels_sha256"] = actual["rgb"]["pixels_sha256"]
    else:
        another["sources"]["rgb"]["profile"] = "not-the-model-input"
    monkeypatch.setattr(spatial_temporal, "read_spatial_manifest", {"distinct": first, "other": another}.__getitem__)
    with pytest.raises(ValueError, match="overlaps the previously seen pack|target protocol"):
        verify_independent_temporal_final(final, ("distinct", "other"))



@pytest.mark.parametrize("change", ["substitute_role", "replace_recording", "archive_digest", "sampling"])
def test_appearance_pin_rejects_role_substitution_and_edited_source_facts(change):
    source = _spec("appearance_holdout")
    if change == "substitute_role":
        source["temporal_role"] = "tracking_holdout"
    elif change == "replace_recording":
        source["logs"]["holdout"][0] = _spec("tracking_holdout")["logs"]["holdout"][0]
    elif change == "archive_digest":
        source["camera_archives"]["3"]["sha256"] = "0" * 64
    else:
        source["selection"]["frames_per_clip"] = 23
    with pytest.raises(PreparationError, match="pinned recordings"):
        spatial_manifest.validate_spatial_sources(source)


@pytest.mark.parametrize(
    ("archives", "previous_count"), [("appearance_holdout", 7), ("zoom_gate_holdout", 8)], indirect=["archives"],
)
def test_additional_holdout_admission_checks_all_previous_packs(archives, previous_count, monkeypatch):
    root, sources, _, _, prepare = archives
    prepare()
    role = sources["temporal_role"]
    final = read_temporal_spatial_pack(root, role=role)
    assert len(final.frames) == 144 and len({frame.clip_id for frame in final.frames}) == 6
    with pytest.raises(PreparationError, match="pinned temporal spatial role"):
        read_temporal_spatial_pack(root, role="tracking_holdout")

    previous_roots = tuple(f"previous-{index}" for index in range(previous_count))
    manifests = {previous: _earlier(final.manifest) for previous in previous_roots}
    observed = []

    def read_previous(path):
        observed.append(path)
        return manifests[path]

    monkeypatch.setattr(spatial_temporal, "read_spatial_manifest", read_previous)
    with pytest.raises(PreparationError, match="pinned temporal holdout"):
        verify_independent_temporal_final(final, previous_roots)
    assert observed == []

    verify_independent_temporal_final(final, previous_roots, expected_role=role)
    assert observed == list(previous_roots)

    extra = manifests[previous_roots[-1]]["splits"]["holdout"]["recordings"][0]["samples"][0]
    extra["rgb"]["pixels_sha256"] = final.frames[0].rgb["pixels_sha256"]
    observed.clear()
    with pytest.raises(ValueError, match="overlaps the previously seen pack by RGB digest"):
        verify_independent_temporal_final(final, previous_roots, expected_role=role)
    assert observed == list(previous_roots)
