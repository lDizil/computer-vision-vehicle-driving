import copy

import pytest

from hybrid_drive.assets.spatial_manifest import POLICY_EVALUATION_SOURCE_SCHEMA
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
    policy = {
        "manifest_hash": "policy",
        "sources": {**protocol, "schema": POLICY_EVALUATION_SOURCE_SCHEMA},
        "splits": {"holdout": {"recordings": [_recording(8, 1, 5), _recording(9, 1, 6)]}},
    }
    return {"reference": reference, "fresh": fresh, "policy": policy}


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



def _read_policy():
    return holdout.read_fresh_spatial_holdout(
        "reference", "policy", expected_schema=POLICY_EVALUATION_SOURCE_SCHEMA,
        previously_seen_roots=("fresh",),
    )


def test_admits_policy_holdout_distinct_from_reference_and_previous_holdout(manifests):
    assert _read_policy() is manifests["policy"]


def test_cannot_exchange_the_calibration_and_policy_holdout_packs(manifests):
    with pytest.raises(ValueError, match="pinned holdout-only pack"):
        holdout.read_fresh_spatial_holdout("reference", "policy")
    with pytest.raises(ValueError, match="pinned holdout-only pack"):
        holdout.read_fresh_spatial_holdout(
            "reference", "fresh", expected_schema=POLICY_EVALUATION_SOURCE_SCHEMA,
        )


def test_rejects_reusing_the_same_manifest_as_a_seen_pack(manifests):
    with pytest.raises(ValueError, match="different from the previously seen pack"):
        holdout.read_fresh_spatial_holdout(
            "reference", "policy", expected_schema=POLICY_EVALUATION_SOURCE_SCHEMA,
            previously_seen_roots=("fresh", "policy"),
        )


def test_rejects_a_group_from_the_previous_holdout(manifests):
    recording = manifests["policy"]["splits"]["holdout"]["recordings"][0]
    recording["recording_id"] = "2021.06.06.13.00.00_veh-1"
    with pytest.raises(ValueError, match="previously seen pack by date and vehicle"):
        _read_policy()


@pytest.mark.parametrize(("kind", "field"), [("jpeg", "sha256"), ("rgb", "pixels_sha256")])
def test_rejects_reused_image_content_from_the_previous_holdout(manifests, kind, field):
    earlier = manifests["fresh"]["splits"]["holdout"]["recordings"][0]["samples"][0]
    manifests["policy"]["splits"]["holdout"]["recordings"][0]["samples"][0][kind][field] = earlier[kind][field]
    with pytest.raises(ValueError, match="previously seen pack by .* digest"):
        _read_policy()


def test_rejects_a_seen_pack_using_a_different_camera_protocol(manifests):
    manifests["fresh"]["sources"]["rgb"] = "different"
    with pytest.raises(ValueError, match="target protocol of the previously seen pack"):
        _read_policy()


def test_frozen_manifest_mismatch_stops_before_opening_targets(monkeypatch):
    from hybrid_drive.neural import spatial_data

    opened = []
    monkeypatch.setattr(spatial_data, "read_spatial_manifest", lambda _: {"manifest_hash": "changed"})
    monkeypatch.setattr(spatial_data, "read_spatial_targets", lambda *args: opened.append(args))
    with pytest.raises(ValueError, match="frozen manifest"):
        spatial_data.SpatialAlignmentDataset("fresh", "holdout", expected_manifest_hash="verified")
    assert not opened
