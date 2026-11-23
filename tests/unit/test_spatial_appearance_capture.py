import hashlib
from types import SimpleNamespace

import pytest

from hybrid_drive.experiments import spatial_appearance_capture as capture
from hybrid_drive.experiments.io import write_json


def test_new_appearance_set_is_checked_against_all_seven_previous_packs(monkeypatch):
    old = SimpleNamespace(
        packs={r: object() for r in capture.ROLES}, traces={r: object() for r in capture.ROLES},
        manifest_sha256="a"*64, manifest={"model_hash": "b"*64, "model_receipt": {"calibrated_policy": {}}},
    )
    tracking, new, trace = object(), object(), object()
    inspected = []
    monkeypatch.setattr(capture, "open_tracking_packs", lambda *args, **kwargs: (old, tracking))
    monkeypatch.setattr(capture, "reopen_tracking_capture", lambda *args, **kwargs: (trace, {}))
    monkeypatch.setattr(capture, "read_temporal_spatial_pack", lambda root, *, role: new)
    monkeypatch.setattr(
        capture, "verify_independent_temporal_final",
        lambda pack, roots, *, expected_role: inspected.append((pack, set(roots), expected_role)),
    )
    video = dict(development="dev", holdout="historical-1", stress="stress")
    earlier = dict(reference="train", previous_final_reused_as_dev="historical-2", evaluation="historical-3")
    scope = capture.open_appearance_inputs(
        video, earlier, "vc", "tracking-root", "tc", "new-root", video_capture_sha256="a"*64,
        tracking_capture_sha256="c"*64,
    )
    assert inspected == [(new, set(video.values()) | set(earlier.values()) | {"tracking-root"},
                          capture.APPEARANCE_ROLE)]
    assert scope.packs[capture.APPEARANCE_ROLE] is new
    assert scope.previous_traces[capture.TRACKING_ROLE] is trace
    assert scope.previous_capture_hashes[capture.TRACKING_ROLE] == "c"*64
    assert {"vc","tc","new-root"} <= set(scope.protected_roots)


@pytest.mark.parametrize("key,value",[
    ("model_hash","f"*64),("labels_opened",True),("observation_anchors",{}),("descriptor_protocol",{}),
])
def test_recomputed_header_digest_does_not_authorize_bad_provenance(tmp_path, monkeypatch, key, value):
    old_hashes = {role: "a"*64 for role in (*capture.ROLES,capture.TRACKING_ROLE)}
    scope = SimpleNamespace(model_hash="b"*64,previous_capture_hashes=old_hashes)
    manifest = {
        "schema": "hybrid-drive-spatial-appearance-capture-v1", "status":"captured", "revision":"0"*40,
        "model_hash":scope.model_hash, "observation_anchors":{**old_hashes,capture.APPEARANCE_ROLE:"c"*64},
        "holdout_trace_sha256":"c"*64, "roles": {},
        "descriptor_protocol":capture.descriptor_protocol(),"labels_opened":False,
    }
    manifest[key]=value
    path=tmp_path/"appearance-capture.json"
    write_json(path,manifest)
    monkeypatch.setattr(capture,"read_spatial_trace",lambda *a,**k:pytest.fail("Read trace without provenance"))
    with pytest.raises(ValueError,match="bound source observations"):
        capture.reopen_appearance_capture(tmp_path,scope,expected_capture_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
