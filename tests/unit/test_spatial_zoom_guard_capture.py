import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from hybrid_drive.experiments import spatial_zoom_guard_capture as capture
from hybrid_drive.experiments.io import write_json
from hybrid_drive.experiments.protocol import content_hash


def _inputs():
    hashes = {f"path-{i}": str(i)*64 for i in range(1, 9)}
    receipt = {"accuracy_data_hashes": {str(i): hashes[f"path-{i}"] for i in range(1,4)}}
    inputs = SimpleNamespace(
        source_model_hash=content_hash(receipt), protected_roots=tuple(hashes),
        packs={str(i):SimpleNamespace(manifest_hash=hashes[f"path-{i}"]) for i in range(4,9)},
    )
    return hashes, receipt, inputs


def test_blind_admission_requires_exact_prior_packs_and_six_clip_role(monkeypatch):
    hashes, receipt, inputs = _inputs()
    called = []
    pack, known = object(), object()
    monkeypatch.setattr(capture, "read_spatial_manifest", lambda root: {"manifest_hash":hashes[root]})

    def load_pack(path, *, role):
        assert path == "new" and role == capture.FRESH_ROLE
        return pack

    def verify(actual, roots, *, expected_role):
        assert actual is pack and expected_role == capture.FRESH_ROLE
        called.append(tuple(roots))

    monkeypatch.setattr(capture, "read_temporal_spatial_pack", load_pack)
    monkeypatch.setattr(capture, "verify_independent_temporal_final", verify)
    monkeypatch.setattr(capture, "reopen_zoom_capture", lambda *a,**k:known)
    supplied = tuple(hashes)
    bundle = capture.open_guard_inputs(inputs,receipt,"old-capture","new",supplied,known_zoom_sha256="0"*64)
    assert bundle.known is known and bundle.fresh_pack is pack
    assert bundle.protected_roots == (*supplied,"old-capture","new")
    assert called == [supplied]
    hashes[supplied[-1]] = "a"*64
    with pytest.raises(ValueError,match="eight exact packs"):
        capture.open_guard_inputs(inputs,receipt,"old-capture","new",supplied,known_zoom_sha256="0"*64)
    assert called == [supplied]


@pytest.mark.parametrize("field,new",[("labels_opened",True),("known_zoom_sha256","f"*64),("pack_hash","e"*64)])
def test_valid_byte_hash_does_not_allow_wrong_holdout_provenance(tmp_path,monkeypatch,field,new):
    source = SimpleNamespace(source_model_hash="a"*64,vision_hash="b"*64)
    inputs = SimpleNamespace(known=SimpleNamespace(inputs=source,sha256="c"*64),
        fresh_pack=SimpleNamespace(manifest_hash="d"*64,frames=(object(),)))
    manifest={
        "schema":"hybrid-drive-zoom-guard-fresh-capture-v1","status":"captured","revision":"0"*40,
        "labels_opened":False,"known_zoom_sha256":"c"*64,"pack_hash":"d"*64,
        "source_model_hash":"a"*64,"vision_hash":"b"*64,"protocol":capture.zoom_protocol(),
        "observation_sha256":"e"*64,"feature_sha256":"f"*64,"frames":1,"visible_boxes":1,
    }
    manifest[field]=new
    path=tmp_path/"fresh-capture.json"
    write_json(path,manifest)
    monkeypatch.setattr(capture,"read_spatial_trace",lambda *a,**k:pytest.fail("Read frames without provenance"))
    with pytest.raises(ValueError,match="bound source"):
        capture.reopen_guard_holdout(tmp_path,inputs,expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def test_crop_role_rejects_reordered_source_before_reading_pixels(monkeypatch):
    from hybrid_drive.experiments import spatial_zoom_capture
    frames=tuple(SimpleNamespace(sample_id=n,rgb={}) for n in ("first","second"))
    decisions=tuple(SimpleNamespace(observation=f) for f in frames[::-1])
    pack=SimpleNamespace(root=Path("not-a-pack"),frames=frames)
    monkeypatch.setattr(spatial_zoom_capture,"read_spatial_rgb",lambda *a,**k:pytest.fail("Read bad frame order"))
    with pytest.raises(ValueError,match="fixed prediction order"):
        spatial_zoom_capture.capture_zoom_role(object(),pack,decisions)
