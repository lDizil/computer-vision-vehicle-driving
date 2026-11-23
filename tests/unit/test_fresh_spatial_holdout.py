import copy

import pytest

from hybrid_drive.experiments import fresh_spatial_holdout as holdout


def _recording(day, vehicle, slot):
    return {
        "recording_id": f"2021.06.{day:02d}.12.00.00_veh-{vehicle}",
        "log_token": f"log-{slot}",
        "samples": [
            {
                "frame_token": f"frame-{slot}-{i}",
                "jpeg": {"member": f"member-{slot}-{i}", "sha256": f"jpeg-{slot}-{i}"},
                "rgb": {"pixels_sha256": f"rgb-{slot}-{i}"},
            }
            for i in range(32)
        ],
    }


def _manifests():
    keys = ("dataset", "openscene_revision", "navsim_revision", "nuplan_revision", "rgb", "loc", "selection")
    protocol = dict.fromkeys(keys, "same")
    reference = {
        "manifest_hash": "original",
        "sources": {**protocol, "schema": "original"},
        "splits": {
            "train": {"recordings": [_recording(3, 1, 0)]},
            "dev": {"recordings": [_recording(4, 1, 1)]},
            "holdout": {"recordings": [_recording(5, 1, 2)]},
        },
    }
    fresh = {
        "manifest_hash": "fresh",
        "sources": {**protocol, "schema": holdout.EVALUATION_SOURCE_SCHEMA},
        "splits": {"holdout": {"recordings": [_recording(6, 1, 3), _recording(7, 1, 4)]}},
    }
    return {"reference": reference, "fresh": fresh}


@pytest.fixture
def manifests(monkeypatch):
    values = _manifests()
    monkeypatch.setattr(holdout, "read_spatial_manifest", values.__getitem__)
    return values


def test_admits_distinct_pinned_holdout_without_targets(manifests):
    assert holdout.read_fresh_spatial_holdout("reference", "fresh") is manifests["fresh"]


def test_rejects_reusing_old_holdout_even_if_training_never_saw_it(manifests):
    original = manifests["reference"]["splits"]["holdout"]["recordings"][0]
    manifests["fresh"]["splits"]["holdout"]["recordings"][0] = copy.deepcopy(original)
    with pytest.raises(ValueError, match="by recording"):
        holdout.read_fresh_spatial_holdout("reference", "fresh")


def test_rejects_renamed_recording_from_same_vehicle_and_date(manifests):
    manifests["fresh"]["splits"]["holdout"]["recordings"][0]["recording_id"] = "2021.06.05.13.00.00_veh-1"
    with pytest.raises(ValueError, match="date and vehicle"):
        holdout.read_fresh_spatial_holdout("reference", "fresh")


@pytest.mark.parametrize("kind,field", [("jpeg", "sha256"), ("rgb", "pixels_sha256")])
def test_rejects_duplicate_captured_content_under_new_identities(manifests, kind, field):
    earlier = manifests["reference"]["splits"]["dev"]["recordings"][0]["samples"][0]
    manifests["fresh"]["splits"]["holdout"]["recordings"][0]["samples"][0][kind][field] = earlier[kind][field]
    with pytest.raises(ValueError, match="digest"):
        holdout.read_fresh_spatial_holdout("reference", "fresh")


def test_rejects_different_target_definition(manifests):
    manifests["fresh"]["sources"]["loc"] = "different"
    with pytest.raises(ValueError, match="target protocol"):
        holdout.read_fresh_spatial_holdout("reference", "fresh")



def test_frozen_manifest_mismatch_stops_before_opening_targets(monkeypatch):
    from hybrid_drive.neural import spatial_data

    opened = []
    monkeypatch.setattr(spatial_data, "read_spatial_manifest", lambda _: {"manifest_hash": "changed"})
    monkeypatch.setattr(spatial_data, "read_spatial_targets", lambda *args: opened.append(args))
    with pytest.raises(ValueError, match="frozen manifest"):
        spatial_data.SpatialAlignmentDataset("fresh", "holdout", expected_manifest_hash="verified")
    assert not opened
