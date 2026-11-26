from dataclasses import replace

import pytest

from hybrid_drive.experiments.object_vru_geometry import GroundHit, GroundObject
from hybrid_drive.experiments.vru_chain_rescue import EARLY, rescue_chains, track_chains
from hybrid_drive.experiments.vru_observation_fusion import profile_vru_observations


FRONT = (1.,) + (0.,) * 35
DISTINCT = (0., 1.) + (0.,) * 34


def seen(key, y, *, camera="front", appearance=FRONT, kind="person"):
    xy = (10., y)
    hit = GroundHit(f"hit-{key}", kind, camera, xy, (0., 0., 1.5), .8, .2, appearance)
    return GroundObject(key, kind, xy, .8, (hit,))


def truth_for(now, previous, old):
    frames = [profile_vru_observations(*pair) for pair in (now, previous, old)]
    sources = {
        name: track_chains(name, *(getattr(frame, key) for frame in frames), .5, .5)
        for name, key in ((EARLY, "combined"), ("ssdlite", "ssdlite"), ("faster_rcnn", "faster_rcnn"))
    }
    return frames[0], sources


def _bridge_window(*, extra=(), old_camera="side", prior_y=-.40, appearance=FRONT):
    return truth_for(
        ((seen("n", 0., appearance=appearance),), (seen("n", .02), *extra)),
        ((seen("p", prior_y, appearance=appearance),), (seen("p", -.40),)),
        ((seen("o", -.80, camera=old_camera, appearance=appearance),), (seen("o", -.82),)),
    )


def _select(profiles, sources):
    return rescue_chains(profiles, sources[EARLY], sources["ssdlite"], sources["faster_rcnn"])


def test_same_snapshot_rescue_retains_the_primary_center_and_the_full_native_chain():
    profiles, sources = _bridge_window()
    base = sources[EARLY]["ssd:n"]
    assert base.confirmation.status == "no_older_link"
    assert sources["faster_rcnn"]["rcnn:n"].confirmation.status == "confirmed"
    exact, bridge = _select(profiles, sources)
    assert exact == sources[EARLY]
    assert set(bridge) == set(exact) == {"ssd:n"}
    repaired = bridge["ssd:n"]
    assert repaired == replace(sources["faster_rcnn"]["rcnn:n"], rescue="same_snapshot")
    assert repaired.initial.previous_object_id == "rcnn:p" and repaired.confirmation.older_object_id == "rcnn:o"


def test_exact_native_rescue_restores_an_older_link_without_changing_the_recent_link():
    profiles, sources = truth_for(
        ((), (seen("n", 0.),)), ((), (seen("p", -.40),)),
        ((seen("block", -.80, camera="side"),), (seen("o", -.80),)),
    )
    assert sources[EARLY]["rcnn:n"].confirmation.status == "no_older_link"
    exact, bridge = _select(profiles, sources)
    assert exact == bridge
    assert exact["rcnn:n"].rescue == "native"
    assert exact["rcnn:n"].initial == sources[EARLY]["rcnn:n"].initial
    assert exact["rcnn:n"].confirmation.status == "confirmed"


@pytest.mark.parametrize("options", [
    {"extra": (seen("other", -.90, kind="bicycle"),)},
    {"prior_y": .40},
    {"appearance": DISTINCT},
])
def test_bridge_abstains_on_nearby_competitor_conflicting_velocity_or_appearance(options):
    profiles, sources = _bridge_window(**options)
    exact, bridge = _select(profiles, sources)
    assert exact == bridge == sources[EARLY]


def test_already_confirmed_early_chain_is_not_replaced():
    profiles, sources = _bridge_window(old_camera="front")
    assert sources[EARLY]["ssd:n"].confirmation.status == "confirmed"
    exact, bridge = _select(profiles, sources)
    assert exact == bridge == sources[EARLY]


def test_rescue_cannot_steal_an_old_detection_from_a_confirmed_mixed_chain():
    profiles, sources = _bridge_window()
    donor = sources["faster_rcnn"]["rcnn:n"]
    # A separate early chain actually owns the same namespaced older observation.
    sources[EARLY]["rcnn:another"] = replace(donor, origin_arm=EARLY, origin_object_id="rcnn:another")
    exact, bridge = _select(profiles, sources)
    assert bridge == exact == sources[EARLY]
