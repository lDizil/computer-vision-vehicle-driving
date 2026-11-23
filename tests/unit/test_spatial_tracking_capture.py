import hashlib
from types import SimpleNamespace

import pytest

from hybrid_drive.experiments import spatial_tracking_capture as capture
from hybrid_drive.experiments.io import write_json


def test_all_six_prior_packs_are_part_of_admission(monkeypatch):
    earlier = object()
    final = object()
    clips = dict(development="dev", holdout="previous-test", stress="stress")
    old = dict(reference="training", evaluation="accuracy-eval", previous_final_reused_as_dev="accuracy-dev")
    inspected = []

    monkeypatch.setattr(capture, "reopen_video_capture", lambda *args, **kwargs: earlier)
    monkeypatch.setattr(capture, "read_temporal_spatial_pack",
                        lambda root, *, role: final if root == "new" and role == capture.TRACKING_ROLE else None)
    monkeypatch.setattr(capture, "verify_independent_temporal_final",
                        lambda pack, roots, *, expected_role: inspected.append((pack, set(roots), expected_role)))
    assert capture.open_tracking_packs(clips, old, "bundle", "new", video_capture_sha256="0"*64) == (earlier, final)
    assert inspected == [(final, set(clips.values()) | set(old.values()), capture.TRACKING_ROLE)]


def _capture_manifest(tmp_path):
    previous = SimpleNamespace(manifest_sha256="a"*64, manifest={"model_hash": "b"*64})
    final = SimpleNamespace(manifest_hash="c"*64, frames=(object(),))
    manifest = {
        "schema": "hybrid-drive-vehicle-tracking-capture-v1", "status": "captured", "revision": "0"*40,
        "prior_video_capture_sha256": previous.manifest_sha256, "model_hash": previous.manifest["model_hash"],
        "pack_hash": final.manifest_hash, "trace_sha256": "d"*64, "frames": 1, "inference_wall_s": 0.1,
        "labels_opened": False,
    }
    path = tmp_path / "tracking-capture.json"
    return previous, final, manifest, path


def _publish(path, payload):
    write_json(path, payload)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_reopen_binds_same_model_and_exact_video_capture(tmp_path, monkeypatch):
    old, final, payload, path = _capture_manifest(tmp_path)
    calls = []
    trace = SimpleNamespace(role=capture.TRACKING_ROLE, inference_wall_s=0.1)

    def read_trace(source, pack, **kwargs):
        calls.append((source.name, pack, kwargs))
        return trace

    monkeypatch.setattr(capture, "read_spatial_trace", read_trace)
    digest = _publish(path, payload)
    assert capture.reopen_tracking_capture(tmp_path, old, final, expected_capture_sha256=digest) == (trace, payload)
    assert calls == [("tracking-observations.json", final,
                      dict(expected_sha256=payload["trace_sha256"], expected_model_hash=payload["model_hash"]))]
    calls.clear()
    with pytest.raises(ValueError, match="byte hash"):
        capture.reopen_tracking_capture(tmp_path, old, final, expected_capture_sha256="e"*64)
    assert not calls


@pytest.mark.parametrize("field,value", [
    ("model_hash", "e"*64), ("labels_opened", True), ("prior_video_capture_sha256", "f"*64), ("frames", 0),
])
def test_even_rehashed_inconsistent_receipt_never_opens_observations(tmp_path, monkeypatch, field, value):
    previous, final, manifest, file = _capture_manifest(tmp_path)
    monkeypatch.setattr(capture, "read_spatial_trace", lambda *args, **kwargs: pytest.fail("Trace opened too early"))
    manifest[field] = value
    with pytest.raises(ValueError, match="different frozen input"):
        capture.reopen_tracking_capture(tmp_path, previous, final, expected_capture_sha256=_publish(file, manifest))
