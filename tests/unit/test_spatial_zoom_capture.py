import hashlib
from types import SimpleNamespace

import pytest
import torch

from hybrid_drive.experiments import spatial_zoom_capture as capture
from hybrid_drive.experiments.io import write_json


def _grids():
    boxes = torch.tensor([[(.03,.2,.14,.49),(.3,.2,.43,.49),(.55,.2,.71,.51),(.79,.2,.92,.57)]] * 2)
    active = torch.tensor([[True,False,True,False],[False,False,False,True]])
    windows = capture.make_zoom_windows(boxes, active)
    global_ = torch.zeros(2,4,3,4,4)
    local = torch.zeros_like(global_)
    global_[active] = torch.arange(3*3*4*4,dtype=torch.float32).reshape(3,3,4,4)
    local[active] = -global_[active]
    return boxes,active,capture.ZoomGrids(windows,global_,local)


def test_compact_role_round_trip_and_mutability_guards(tmp_path):
    boxes,active,g=_grids()
    path=tmp_path/"one.safetensors"
    digest=capture._write_compact(path,g,boxes,active,3,"binding")
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    reopened=capture._read_compact(path,boxes,active,3,"binding",digest)
    assert torch.equal(reopened.windows,g.windows)
    assert torch.equal(reopened.global_features,g.global_features)
    assert torch.equal(reopened.zoom_features,g.zoom_features)
    with pytest.raises(ValueError,match="feature byte hash"):
        capture._read_compact(path,boxes,active,3,"binding","0"*64)
    with pytest.raises(ValueError,match="provenance"):
        capture._read_compact(path,boxes,active,3,"unrelated",digest)
    with pytest.raises(FileExistsError):
        capture._write_compact(path,g,boxes,active,3,"binding")


def test_writer_rejects_wrong_window_or_features_on_an_inactive_slot(tmp_path):
    boxes,active,g=_grids()
    path=tmp_path/"foo.safetensors"
    moved=g.windows.clone()
    moved[0,0,0]+=0.01
    with pytest.raises(ValueError,match="original visible boxes"):
        capture._write_compact(path,capture.ZoomGrids(moved,g.global_features,g.zoom_features),boxes,active,3,"binding")
    g.global_features[~active]=1
    with pytest.raises(ValueError,match="visible boxes"):
        capture._write_compact(path,g,boxes,active,3,"binding")
    assert not path.exists()


def test_admission_checks_every_pair_of_existing_clip_roles(monkeypatch):
    source=SimpleNamespace()
    ids={role:SimpleNamespace(sample_id=f"{role}_first") for role in capture.ZOOM_ROLES}
    prior=SimpleNamespace(
        inputs=SimpleNamespace(
            packs={role:SimpleNamespace(manifest=role,frames=(r,)) for role,r in ids.items()},
            model_hash="a"*64, model_receipt={"vision_hash":"b"*64}, protected_roots=("old",),
        ),
        decisions={role:(SimpleNamespace(observation=row),) for role,row in ids.items()}, sha256="d"*64,
    )
    inspected=[]
    monkeypatch.setattr(capture,"reopen_appearance_capture",lambda *a,**k:prior)
    monkeypatch.setattr(capture,"verify_disjoint_spatial_manifests",
                        lambda one,two,**k:inspected.append(frozenset((one,two))))
    inputs=capture.open_zoom_inputs(source,"capture",expected_capture_sha256="d"*64)
    assert len(set(inspected))==len(inspected)==10
    assert inputs.protected_roots==("old","capture")


def test_wrong_previous_decision_hash_rejected_before_reopening_tensor(tmp_path,monkeypatch):
    scope=SimpleNamespace(source_model_hash="a"*64,vision_hash="b"*64,source_capture_sha256="c"*64)
    header={
        "schema":"hybrid-drive-spatial-zoom-capture-v1","status":"captured","revision":"d"*40,
        "source_model_hash":scope.source_model_hash,"vision_hash":scope.vision_hash,
        "source_capture_sha256":"e"*64,"feature_dim":3,"protocol":capture.zoom_protocol(),
        "roles":{},"labels_opened":False,
    }
    f=tmp_path/"zoom-capture.json"
    write_json(f,header)
    monkeypatch.setattr(capture,"_read_compact",lambda *a,**k:pytest.fail("Tensor accessed without provenance"))
    with pytest.raises(ValueError,match="original frozen decisions"):
        capture.reopen_zoom_capture(tmp_path,scope,expected_sha256=hashlib.sha256(f.read_bytes()).hexdigest())



def test_role_larger_than_one_encoder_batch_is_verified_in_chunks(tmp_path):
    boxes, active, row = _grids()
    n = 17
    boxes = boxes[:1].expand(n, -1, -1).clone()
    active = active[:1].expand(n, -1).clone()
    g = capture.ZoomGrids(*(x[:1].expand(n, *x.shape[1:]).clone()
                            for x in (row.windows, row.global_features, row.zoom_features)))
    p = tmp_path / "whole-role.safetensors"
    digest = capture._write_compact(p, g, boxes, active, 3, "whole-role")
    restored = capture._read_compact(p, boxes, active, 3, "whole-role", digest)
    assert torch.equal(restored.windows, g.windows)
    assert torch.equal(restored.zoom_features, g.zoom_features)
