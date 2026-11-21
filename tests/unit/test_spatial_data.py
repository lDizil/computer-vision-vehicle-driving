from copy import deepcopy
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import pickle
import tarfile

import numpy as np
from PIL import Image
import pytest
import torch

from hybrid_drive.assets import alignment_sample, spatial_manifest
from hybrid_drive.assets.alignment_manifest import digest_json
from hybrid_drive.assets.files import PreparationError
from hybrid_drive.assets.spatial_manifest import MANIFEST_NAME, SPLITS, load_spatial_sources, sample_id
from hybrid_drive.assets.spatial_sample import prepare_spatial_sample
from hybrid_drive.neural.spatial_data import SpatialAlignmentDataset, collate_spatial_alignment


SOURCES = Path(__file__).resolve().parents[2] / "scripts" / "spatial-loc-sources.json"
ROTATION = [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]
INTRINSIC = [[112, 0, 111.5], [0, 112, 111.5], [0, 0, 1]]


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _archive(path, spec, entries):
    with path.open("wb") as output, gzip.GzipFile(fileobj=output, filename="", mode="wb", mtime=0) as stream:
        with tarfile.open(fileobj=stream, mode="w", format=tarfile.USTAR_FORMAT) as archive:
            for name, data in entries:
                member = tarfile.TarInfo(name)
                member.size = len(data)
                archive.addfile(member, io.BytesIO(data))
    data = path.read_bytes()
    spec.update(size_bytes=len(data), sha256=_sha(data))


@pytest.fixture()
def fixed_archives(tmp_path, monkeypatch):
    sources = load_spatial_sources(SOURCES)
    logs = {}
    images = {shard: [] for shard in sources["camera_archives"]}
    specs = [log for split in SPLITS for log in sources["logs"][split]]
    for offset, spec in enumerate(specs, 1):
        shard = str(spec["camera_shard"])
        prefix = sources["camera_archives"][shard]["archive_prefix"]
        frames = []
        for index in range(4):
            token = f"{1000 * offset + index:016x}"
            data_path = f"{spec['log_name']}/CAM_F0/{token}.jpg"
            frames.append({
                "token": token, "timestamp": 1000000 + 500000 * index,
                "frame_idx": index, "scene_token": f"{100+offset:016x}",
                "log_token": f"{offset:016x}", "log_name": spec["log_name"],
                "cams": {"CAM_F0": {
                    "data_path": data_path, "cam_intrinsic": INTRINSIC,
                    "sensor2lidar_rotation": ROTATION, "sensor2lidar_translation": [0, 0, 0],
                    "distortion": [0, 0, 0, 0, 0],
                }},
                "anns": {
                    "gt_names": ["vehicle", "vehicle"],
                    "gt_boxes": np.array([[10, 5, 0, 4, 2, 2, 0], [10, 0, 0, 4, 2, 2, 0]])
                    if index == 1 else np.array([[41, 0, 0, 4, 2, 2, 0], [8, -10, 0, 4, 2, 2, 0]]),
                },
            })
            picture = io.BytesIO()
            Image.new("RGB", (224, 224), (30 * offset, 40 + index, 89)).save(picture, format="JPEG")
            images[shard].append((f"{prefix}/{data_path}", picture.getvalue()))
        spec.update(frame_count=4, sample_count=2)
        logs[spec["recording_id"]] = frames

    metadata_path = tmp_path / "metadata.tgz"
    archive_paths = {shard: tmp_path / f"cameras-{shard}.tgz" for shard in images}
    root = tmp_path / "prepared"

    def prepare(path=root, *, bad_member=False):
        metadata = [("unused/not-a-pickle.pkl", b"this file must not be deserialized")]
        for spec in specs:
            content = pickle.dumps(logs[spec["recording_id"]])
            digest = "0" * 64 if bad_member and spec is specs[-1] else _sha(content)
            spec.update(size_bytes=len(content), sha256=digest)
            metadata.append((spec["member"], content))
        _archive(metadata_path, sources["metadata_archive"], metadata)
        for shard, camera_path in archive_paths.items():
            _archive(camera_path, sources["camera_archives"][shard], images[shard])
        monkeypatch.setattr(spatial_manifest, "_SOURCE_HASH", digest_json(sources))
        return prepare_spatial_sample(path, sources, metadata_path, archive_paths)

    return root, sources, logs, images, prepare


def _rewrite_manifest(root, manifest):
    manifest["manifest_hash"] = digest_json({key: value for key, value in manifest.items() if key != "manifest_hash"})
    (root / MANIFEST_NAME).write_text(json.dumps(manifest))


def _replace_targets(root, manifest, split, mutate):
    reference = manifest["splits"][split]["targets"]
    path = root / reference["path"]
    document = json.loads(path.read_bytes())
    mutate(document["rows"])
    data = json.dumps(document).encode()
    path.write_bytes(data)
    reference.update(size_bytes=len(data), sha256=_sha(data))
    _rewrite_manifest(root, manifest)


def test_production_plan_assigns_six_recordings_to_frozen_roles():
    sources = load_spatial_sources(SOURCES)
    plans = {split: [(log["recording_id"], log["camera_shard"], log["sample_count"])
                     for log in sources["logs"][split]] for split in SPLITS}
    assert plans == {
        "train": [("2021.06.14.18.42.45_veh-12", 16, 48), ("2021.10.06.17.43.07_veh-28", 16, 48)],
        "dev": [("2021.10.11.07.12.18_veh-50", 5, 24), ("2021.10.11.02.57.41_veh-50", 6, 24)],
        "holdout": [("2021.06.09.14.58.55_veh-35", 5, 32), ("2021.08.30.14.54.34_veh-40", 6, 32)],
    }


def test_alignment_contains_one_current_rgb_and_four_labels_per_frozen_frame(fixed_archives):
    root, sources, _, _, prepare = fixed_archives
    manifest = prepare()
    for split in SPLITS:
        dataset = SpatialAlignmentDataset(root, split)
        planned_ids = tuple(log["recording_id"] for log in sources["logs"][split])
        assert dataset.dataset_hash == manifest["manifest_hash"]
        assert dataset.recording_ids == tuple(identity for identity in planned_ids for _ in range(2))
        assert dataset.sample_ids == tuple(sample_id(split, identity, index) for identity in planned_ids for index in (1, 3))
        batch = collate_spatial_alignment([dataset[index] for index in (0, 3)])
        assert batch.sample_ids == (dataset.sample_ids[0], dataset.sample_ids[3])
        assert batch.recording_ids == planned_ids
        assert batch.images.shape == (2, 1, 3, 224, 224)
        assert batch.images.dtype == torch.float32 and 0 <= batch.images.min() <= batch.images.max() <= 1
        assert batch.presence.dtype == torch.bool
        assert batch.presence.tolist() == [[True, False, True, False], [False] * 4]
        assert batch.boxes_2d.shape == (2, 4, 4) and batch.boxes_2d.dtype == torch.float32
        assert torch.equal(torch.isnan(batch.boxes_2d).all(dim=-1), ~batch.presence)
        assert torch.isfinite(batch.boxes_2d[batch.presence]).all()
        assert batch.boxes_2d[0, 2].tolist() == [0.4375, 0.4375, 0.5625, 0.5625]
    assert not list(root.rglob("*.jpg")) and not list(root.rglob("*.pkl"))
    assert spatial_manifest.verify_spatial_sample(root) == manifest
    assert prepare() == manifest


def test_train_loads_neither_dev_nor_holdout_targets(fixed_archives, monkeypatch):
    root, _, _, _, prepare = fixed_archives
    prepare()
    excluded = {root / "targets" / f"{split}.json" for split in ("dev", "holdout")}
    original = os.open

    def guard(path, *args, **kwargs):
        if not isinstance(path, int) and Path(path) in excluded:
            pytest.fail("Train opened labels from another split")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(os, "open", guard)
    assert SpatialAlignmentDataset(root, "train")[0].presence.tolist() == [True, False, True, False]


def test_all_member_hashes_checked_before_deserializing_any_pickle(fixed_archives, monkeypatch):
    root, _, _, _, prepare = fixed_archives

    def fail_on_unverified_pickle(_):
        pytest.fail("Deserialized a pickle before verifying all selected metadata members")

    monkeypatch.setattr(alignment_sample.pickle, "loads", fail_on_unverified_pickle)
    with pytest.raises(PreparationError, match="member.*SHA256"):
        prepare(bad_member=True)
    assert not root.exists()


def test_missing_pinned_jpeg_cannot_be_replaced_and_preparation_is_atomic(fixed_archives):
    root, _, _, images, prepare = fixed_archives
    # The earlier archive and adjacent JPEGs exist, but the frozen index 1 in shard 6 does not.
    del images["6"][1]
    with pytest.raises(PreparationError, match="allowlisted JPEGs"):
        prepare()
    assert not root.exists()
    assert not list(root.parent.glob(".spatial-loc-*"))


def test_different_labels_change_no_selected_image(fixed_archives):
    root, sources, logs, _, prepare = fixed_archives
    prepare()
    before = SpatialAlignmentDataset(root, "train")
    logs[sources["logs"]["train"][0]["recording_id"]][1]["anns"]["gt_boxes"][0][0] = 41
    alternate = root.with_name("different-labels")
    prepare(alternate)
    after = SpatialAlignmentDataset(alternate, "train")
    assert before.dataset_hash != after.dataset_hash
    assert before.sample_ids == after.sample_ids and before.recording_ids == after.recording_ids
    assert before[0].presence.tolist() == [True, False, True, False]
    assert after[0].presence.tolist() == [False, False, True, False]
    torch.testing.assert_close(before[0].image, after[0].image, rtol=0, atol=0)


@pytest.mark.parametrize(("mutate", "error"), [
    (lambda rows: rows.reverse(), "target identity"),
    (lambda rows: rows[0]["presence"].__setitem__(0, False), "negative spatial target"),
    (lambda rows: rows[0]["boxes_2d"].__setitem__(0, rows[0]["boxes_2d"][2]), "another band"),
    (lambda rows: rows[0]["presence"].__setitem__(0, 1), "mask differs"),
])
def test_reader_rejects_inconsistent_targets_even_with_recomputed_hashes(fixed_archives, mutate, error):
    root, _, _, _, prepare = fixed_archives
    manifest = prepare()
    _replace_targets(root, manifest, "train", mutate)
    with pytest.raises(PreparationError, match=error):
        SpatialAlignmentDataset(root, "train")


def test_refuses_cross_split_group_and_duplicate_recording_even_if_source_plan_repinned(fixed_archives, monkeypatch):
    _, sources, _, _, prepare = fixed_archives
    prepare()
    overlap = deepcopy(sources)
    overlap["logs"]["holdout"][0], overlap["logs"]["dev"][0] = overlap["logs"]["dev"][0], overlap["logs"]["holdout"][0]
    monkeypatch.setattr(spatial_manifest, "_SOURCE_HASH", digest_json(overlap))
    with pytest.raises(PreparationError, match=r"date\+vehicle group"):
        spatial_manifest.validate_spatial_sources(overlap)
    duplicate = deepcopy(sources)
    duplicate["logs"]["dev"][0] = duplicate["logs"]["train"][0]
    monkeypatch.setattr(spatial_manifest, "_SOURCE_HASH", digest_json(duplicate))
    with pytest.raises(PreparationError, match="unique full recording"):
        spatial_manifest.validate_spatial_sources(duplicate)


@pytest.mark.parametrize(("record", "field", "error"), [
    ("jpeg", "sha256", "Identical source JPEG"),
    ("rgb", "pixels_sha256", "Identical decoded RGB pixels"),
])
def test_reader_rejects_cross_split_content_before_opening_targets(fixed_archives, monkeypatch, record, field, error):
    root, _, _, _, prepare = fixed_archives
    manifest = prepare()
    train = manifest["splits"]["train"]["recordings"][0]["samples"][0]
    holdout = manifest["splits"]["holdout"]["recordings"][0]["samples"][0]
    holdout[record][field] = train[record][field]
    _rewrite_manifest(root, manifest)

    def fail_if_opened(*_args):
        pytest.fail("Opened targets before verifying cross-split content")

    monkeypatch.setattr(spatial_manifest, "_read_verified", fail_if_opened)
    with pytest.raises(PreparationError, match=error):
        SpatialAlignmentDataset(root, "train")


def test_rejects_posthoc_symlink_traversal_and_rgb_corruption(fixed_archives):
    root, _, _, _, prepare = fixed_archives
    manifest = prepare()
    dataset = SpatialAlignmentDataset(root, "train")
    sample = manifest["splits"]["train"]["recordings"][0]["samples"][0]
    path = root / sample["rgb"]["path"]
    data = path.read_bytes()
    path.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
    with pytest.raises(PreparationError, match="SHA256"):
        dataset[0]

    outside = root.parent / "external.png"
    outside.write_bytes(data)
    path.unlink()
    path.symlink_to(outside)
    with pytest.raises(PreparationError, match="Symlink"):
        dataset[0]
    path.unlink()
    path.write_bytes(data)
    sample["rgb"]["path"] = "../external.png"
    _rewrite_manifest(root, manifest)
    with pytest.raises(PreparationError, match="file path"):
        SpatialAlignmentDataset(root, "train")
