import hashlib
from types import SimpleNamespace

import pytest

from hybrid_drive.experiments import temporal_spatial_capture as capture
from hybrid_drive.experiments.io import write_json
from hybrid_drive.experiments.protocol import content_hash


def _setup(tmp_path, monkeypatch):
    roles = {role: tmp_path / role for role in capture.ROLES}
    previous = {role: tmp_path / role for role in capture.PRIOR_ROLES}
    references = {role: str(i) * 64 for i, role in enumerate(previous, 1)}
    packs = {name: SimpleNamespace(manifest_hash=str(i) * 64, frames=(name,)) for i, name in enumerate(roles, 4)}
    admissions = []

    def read_pack(root, *, role):
        assert root == roles[role]
        return packs[role]

    def verify(final, past):
        assert final is packs["holdout"]
        admissions.append(set(past))

    def prior_manifest(root):
        name = next(role for role, path in previous.items() if root == path)
        return {"manifest_hash": references[name]}

    monkeypatch.setattr(capture, "read_temporal_spatial_pack", read_pack)
    monkeypatch.setattr(capture, "verify_independent_temporal_final", verify)
    monkeypatch.setattr(capture, "read_spatial_manifest", prior_manifest)
    return roles, previous, references, packs, admissions


def test_admission_includes_all_previous_and_development_metadata(tmp_path, monkeypatch):
    roles, previous, refs, packs, admissions = _setup(tmp_path, monkeypatch)
    assert capture.open_video_packs(roles, previous) == (packs, refs)
    assert admissions == [set(previous.values()) | {roles["development"], roles["stress"]}]
    with pytest.raises(ValueError, match="Expected all"):
        capture.open_video_packs(roles, {"reference": previous["reference"]})


def test_reopen_hard_binds_predecessor_roles_and_trace_hashes(tmp_path, monkeypatch):
    roles, previous, refs, packs, _ = _setup(tmp_path, monkeypatch)
    model = {"accuracy_data_hashes": refs, "calibrated_policy": {"absent_below": 0.0, "present_at_or_above": 1.0}}
    traces = {k: {"pack_hash": p.manifest_hash, "trace_sha256": str(i) * 64, "frames": 1, "inference_wall_s": 0.5}
              for i, (k, p) in enumerate(packs.items(), 1)}
    manifest = {"schema": capture.SCHEMA, "revision": "a" * 40, "status": "captured",
                "model_receipt": model, "model_hash": content_hash(model), "captures": traces,
                "batch_size": 8, "labels_opened": False}
    opened = {}

    def stub(path, pack, *, expected_sha256, expected_model_hash):
        role = path.stem.removeprefix("observations-")
        assert pack == packs[role] and expected_sha256 == traces[role]["trace_sha256"]
        assert expected_model_hash == manifest["model_hash"]
        opened[role] = object()
        return SimpleNamespace(inference_wall_s=0.5)

    monkeypatch.setattr(capture, "read_spatial_trace", stub)
    path = tmp_path / capture.MANIFEST_NAME
    write_json(path, manifest)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    bound = capture.reopen_video_capture(tmp_path, roles, previous, expected_capture_sha256=digest)
    assert set(opened) == set(roles) and bound.manifest_sha256 == digest
    opened.clear()
    # Even with unchanged archive bytes, swapping two previously used roles
    # does not prove independence from the training/calibration datasets.
    other = {**previous, "evaluation": previous["previous_final_reused_as_dev"],
             "previous_final_reused_as_dev": previous["evaluation"]}
    with pytest.raises(ValueError, match="incompatible"):
        capture.reopen_video_capture(tmp_path, roles, other, expected_capture_sha256=digest)
    assert not opened
    with pytest.raises(ValueError, match="manifest byte hash"):
        capture.reopen_video_capture(tmp_path, roles, previous, expected_capture_sha256="f" * 64)
    assert not opened
