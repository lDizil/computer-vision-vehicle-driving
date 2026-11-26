"""Offline VRU chain repair without changing the current detector centers."""

from collections import Counter
from dataclasses import dataclass, replace
from math import dist, fsum

from hybrid_drive.experiments.object_vru_association import ObjectMotion, associate_objects
from hybrid_drive.experiments.triple_vru_confirmation import Confirmation, confirm_objects, frozen_two_frame_velocity
from hybrid_drive.experiments.vru_observation_fusion import MIN_CROSS_DETECTOR_DISTANCE_M, ObservationProfiles


MAX_BRIDGE_DISTANCE_M = .75
MAX_BRIDGE_APPEARANCE_L1 = .60
MAX_VELOCITY_DELTA_MPS = .60
EARLY = "ssd_primary"
SINGLES = ("ssdlite", "faster_rcnn")
RESCUES = ("native_rescue", "bridge_rescue")
ARMS = (*SINGLES, EARLY, *RESCUES)


@dataclass(frozen=True)
class Chain:
    origin_arm: str
    origin_object_id: str
    geometry: ObjectMotion
    initial: ObjectMotion
    older: ObjectMotion | None
    confirmation: Confirmation
    rescue: str | None = None


def track_chains(arm, current, previous, older, dt, older_dt):
    geometry = associate_objects(current, previous, dt, mode="geometry")
    initial = associate_objects(current, previous, dt, mode="appearance")
    confirmed = confirm_objects(current, previous, older, initial, dt, older_dt)
    back = {} if older is None else dict(zip(
        (obj.object_id for obj in previous), associate_objects(previous, older, older_dt, mode="appearance"),
        strict=True,
    ))
    return {obj.object_id: Chain(arm, obj.object_id, geom, hop, back.get(hop.previous_object_id), decision)
            for obj, geom, hop, decision in zip(current, geometry, initial, confirmed, strict=True)}


def rescue_chains(current: ObservationProfiles, early, ssd, faster):
    """Keep confirmed early chains; repair only missing older links.

    First use the exact native object with the same recent link. A second
    arm can borrow a suppressed Faster R-CNN chain when both current camera
    observations and the measured recent motion agree. A borrowed chain keeps
    its own identities: it is never reconstructed from mixed-frame positions.
    """
    native = ssd | faster
    direct = {}
    for obj in current.combined:
        key = obj.object_id
        base, source = early[key], native[key]
        if (_can_rescue(base, source) and base.initial.previous_object_id == source.initial.previous_object_id):
            direct[key] = replace(source, rescue="native")
    exact = _add_without_conflicts(early, direct)

    bridges = {}
    for obj, donor in _unique_suppressed_pairs(current):
        base, source = exact[obj.object_id], faster[donor.object_id]
        if _can_rescue(base, source) and _agree_on_camera(obj, donor, base, source):
            bridges[obj.object_id] = replace(source, rescue="same_snapshot")
    return exact, _add_without_conflicts(exact, bridges)


def _can_rescue(base, source):
    if base.confirmation.status != "no_older_link" or source.confirmation.status != "confirmed":
        return False
    recent = pair_velocity(base)
    trend = source.confirmation.trend_velocity_xy_mps
    return (fsum(a * b for a, b in zip(recent, trend, strict=True)) > 0
            and dist(recent, trend) <= MAX_VELOCITY_DELTA_MPS)


def pair_velocity(chain):
    return frozen_two_frame_velocity(chain.initial, radial_gain=0., tangential_gain=1.)


def _add_without_conflicts(base, proposals):
    occupied = {part for chain in base.values() if chain.confirmation.status == "confirmed" for part in _links(chain)}
    counts = Counter(part for chain in proposals.values() for part in _links(chain))
    return base | {key: chain for key, chain in proposals.items()
                   if all(part not in occupied and counts[part] == 1 for part in _links(chain))}


def _links(chain):
    # IDs are detector-namespaced and local to a snapshot: time is part of the key.
    return (("previous", chain.initial.previous_object_id), ("older", chain.confirmation.older_object_id))


def _unique_suppressed_pairs(profiles):
    kept = {obj.object_id for obj in profiles.combined}
    by_primary = {}
    by_secondary = Counter()
    for a in profiles.ssdlite:
        near = tuple(b for b in profiles.faster_rcnn if dist(a.xy, b.xy) < MIN_CROSS_DETECTOR_DISTANCE_M)
        by_primary[a.object_id] = near
        by_secondary.update(b.object_id for b in near)
    for a in profiles.ssdlite:
        near = by_primary[a.object_id]
        if len(near) != 1:
            continue
        b = near[0]
        if (b.object_id not in kept and by_secondary[b.object_id] == 1 and a.kind == b.kind
                and dist(a.xy, b.xy) <= MAX_BRIDGE_DISTANCE_M):
            yield a, b


def _agree_on_camera(a, b, base, source):
    primary, donor = ({hit.camera_name: hit for hit in obj.hits} for obj in (a, b))
    shared = primary.keys() & donor.keys()
    if not shared.intersection(base.initial.common_cameras, source.confirmation.common_cameras):
        return False
    return all(
        dist(primary[name].xy, donor[name].xy) <= MAX_BRIDGE_DISTANCE_M
        and fsum(abs(p - q) for p, q in zip(primary[name].appearance, donor[name].appearance, strict=True))
        <= MAX_BRIDGE_APPEARANCE_L1 for name in shared
    )
