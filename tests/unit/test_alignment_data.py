from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import pickle
import tarfile

import numpy as np
from PIL import Image
import pytest
import torch

from hybrid_drive.assets import alignment_manifest, alignment_sample
from hybrid_drive.assets.alignment_manifest import MANIFEST_NAME, digest_json, load_alignment_sources
from hybrid_drive.assets.files import PreparationError
from hybrid_drive.neural.alignment_data import AlignmentDataset, collate_alignment


SOURCES = Path(__file__).resolve().parents[2] / "scripts" / "alignment-data-sources.json"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def archive_bytes(entries):
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz", format=tarfile.USTAR_FORMAT) as archive:
        for name, body in entries:
            entry = tarfile.TarInfo(name)
            entry.size = len(body)
            archive.addfile(entry, io.BytesIO(body))
    return output.getvalue()


@pytest.fixture()
def fixture_archives(tmp_path, monkeypatch):
    sources = load_alignment_sources(SOURCES)
    prefix = sources["camera_archive"]["archive_prefix"]
    logs, images = {}, []
    for offset, split in enumerate(("train", "holdout"), 1):
        log = sources["logs"][split]
        frames = []
        for index in range(4):
            token = f"{offset * 1000 + index:016x}"
            data_path = f"{log['log_name']}/CAM_F0/{token}.jpg"
            frames.append({
                "token": token, "timestamp": 1000000 + index * 500000,
                "frame_idx": index, "scene_token": f"{offset+100:016x}",
                "log_token": f"{offset:016x}", "log_name": log["log_name"],
                "cams": {"CAM_F0": {
                    "data_path": data_path,
                    "cam_intrinsic": [[112, 0, 111.5], [0, 112, 111.5], [0, 0, 1]],
                    "sensor2lidar_rotation": [[0, 0, 1], [-1, 0, 0], [0, -1, 0]],
                    "sensor2lidar_translation": [0, 0, 0],
                    "distortion": [0, 0, 0, 0, 0],
                }},
                "anns": {"gt_names": ["vehicle"], "gt_boxes": np.array([[10 if index < 3 else 41, 0, 0, 4, 2, 2, 0]]),
                         "instance_tokens": [token], "track_tokens": [f"{offset+200:016x}"]},
            })
            buffer = io.BytesIO()
            Image.new("RGB", (224, 224), (40 * offset, 50 + index, 80)).save(buffer, format="JPEG")
            images.append((f"{prefix}/{data_path}", buffer.getvalue()))
        logs[split] = frames
        log.update(frame_count=4, sample_count=2)

    metadata_path, camera_path = tmp_path / "metadata.tgz", tmp_path / "camera.tgz"

    def prepare(root=None, *, broken_member=False, camera_entries=None):
        root = root or tmp_path / "prepared"
        entries = [("unused/unlisted.pkl", b"not a pickle")]
        for split, frames in logs.items():
            body = pickle.dumps(frames)
            spec = sources["logs"][split]
            spec.update(size_bytes=len(body), sha256=("0" * 64 if broken_member else sha(body)))
            entries.append((spec["member"], body))
        for path, key, entries in (
            (metadata_path, "metadata_archive", entries),
            (camera_path, "camera_archive", images if camera_entries is None else camera_entries),
        ):
            body = archive_bytes(entries)
            path.write_bytes(body)
            sources[key].update(size_bytes=len(body), sha256=sha(body))
        monkeypatch.setattr(alignment_manifest, "_SOURCE_HASH", digest_json(sources))
        return alignment_sample.prepare_alignment_sample(root, sources, metadata_path, camera_path)

    return tmp_path / "prepared", logs, images, prepare


def test_prepare_and_read_only_fixed_current_images_and_loc_targets(fixture_archives):
    root, _, _, prepare = fixture_archives
    manifest = prepare()
    train = AlignmentDataset(root, "train")
    holdout = AlignmentDataset(root, "holdout")
    assert train.dataset_hash == holdout.dataset_hash == manifest["manifest_hash"]
    assert train.recording_id != holdout.recording_id
    assert train.sample_ids == ("train-0001", "train-0003")
    assert holdout.sample_ids == ("holdout-0001", "holdout-0003")
    assert all(partition["positive_count"] == partition["negative_count"] == 1
               for partition in train.manifest["splits"].values())
    assert not list(root.rglob("*.jpg")) and not list(root.rglob("*.pkl"))

    batch = collate_alignment([train[0], train[1]])
    assert batch.images.shape == (2, 1, 3, 224, 224)
    assert batch.images.dtype == torch.float32 and 0 <= batch.images.min() <= batch.images.max() <= 1
    assert batch.presence.dtype == torch.bool and batch.presence.tolist() == [True, False]
    assert batch.boxes_3d.dtype == batch.boxes_2d.dtype == torch.float32
    assert batch.boxes_3d[0].tolist() == [10, 0, 0, 2, 4, 2, 0]
    assert batch.boxes_2d[0].tolist() == pytest.approx([0.4375, 0.4375, 0.5625, 0.5625])
    assert torch.isnan(batch.boxes_3d[1]).all() and torch.isnan(batch.boxes_2d[1]).all()
    assert alignment_manifest.verify_alignment_sample(root) == manifest


def test_labels_cannot_change_selection_or_rgb(fixture_archives):
    root, logs, _, prepare = fixture_archives
    prepare()
    first = AlignmentDataset(root, "train")
    before = first[0]
    logs["train"][1]["anns"]["gt_boxes"][0, 0] = 41
    other = root.with_name("other")
    prepare(other)
    second = AlignmentDataset(other, "train")
    after = second[0]
    assert first.dataset_hash != second.dataset_hash
    assert first.sample_ids == second.sample_ids and before.sample_id == after.sample_id
    torch.testing.assert_close(before.image, after.image, rtol=0, atol=0)
    assert before.presence and not after.presence


def test_pinned_member_hash_checked_before_pickle_load(fixture_archives, monkeypatch):
    root, _, _, prepare = fixture_archives

    def unexpected(_):
        raise AssertionError("pickle loaded before both source members passed verification")

    monkeypatch.setattr(alignment_sample.pickle, "loads", unexpected)
    with pytest.raises(PreparationError, match="Member|member.*SHA256"):
        prepare(broken_member=True)
    assert not root.exists()


def test_missing_fixed_frame_reads_no_adjacent_substitute(fixture_archives):
    root, _, images, prepare = fixture_archives
    # The adjacent source frames are present, but the pinned index 1 is not.
    with pytest.raises(PreparationError, match="allowlisted JPEGs"):
        prepare(camera_entries=images[:1] + images[2:])
    assert not root.exists()
    assert not list(root.parent.glob(".alignment-*"))


def test_jpeg_link_cannot_cross_recordings(fixture_archives):
    root, logs, _, prepare = fixture_archives
    logs["train"][1]["cams"]["CAM_F0"]["data_path"] = logs["holdout"][1]["cams"]["CAM_F0"]["data_path"]
    with pytest.raises(PreparationError, match="matching OpenScene log"):
        prepare()
    assert not root.exists()


def test_train_does_not_load_holdout_targets(fixture_archives):
    root, _, _, prepare = fixture_archives
    prepare()
    path = root / "targets" / "holdout.json"
    source = path.read_bytes()
    path.write_bytes(source.replace(b"true", b"TRUE"))
    assert AlignmentDataset(root, "train")[0].presence
    with pytest.raises(PreparationError, match="SHA256"):
        AlignmentDataset(root, "holdout")


def test_rgb_hash_checked_at_read(fixture_archives):
    root, _, _, prepare = fixture_archives
    prepare()
    train = AlignmentDataset(root, "train")
    path = root / "rgb" / f"{train.sample_ids[0]}.png"
    raw = path.read_bytes()
    path.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
    with pytest.raises(PreparationError, match="SHA256"):
        train[0]


def test_target_join_remains_strict_if_file_and_manifest_hashes_recalculated(fixture_archives):
    root, _, _, prepare = fixture_archives
    manifest = deepcopy(prepare())
    reference = manifest["splits"]["train"]["targets"]
    path = root / reference["path"]
    changed = json.loads(path.read_bytes())
    changed["rows"].reverse()
    raw = json.dumps(changed).encode()
    path.write_bytes(raw)
    reference.update(size_bytes=len(raw), sha256=sha(raw))
    manifest.pop("manifest_hash")
    manifest["manifest_hash"] = digest_json(manifest)
    (root / MANIFEST_NAME).write_text(json.dumps(manifest))
    with pytest.raises(PreparationError, match="target identity"):
        AlignmentDataset(root, "train")
