"""Require a common-camera track and consistent motion over two past intervals."""

from dataclasses import dataclass
from math import fsum, isfinite

from hybrid_drive.experiments.object_vru_association import associate_objects


MIN_STEP_M = .10
MIN_BASELINE_M = .30
MIN_RATE_AGREEMENT_MPS = .8
RELATIVE_RATE_ALLOWANCE = .25
MAX_CAMERA_RATE_DIFFERENCE_MPS = 1.
MAX_APPEARANCE_L1 = 1.10


@dataclass(frozen=True)
class Confirmation:
    status: str
    older_object_id: str | None = None
    common_cameras: tuple[str, ...] = ()
    trend_velocity_xy_mps: tuple[float, float] = (0., 0.)


def confirm_objects(current, middle, oldest, initial, latest_dt, older_dt):
    """Both historical sets must already be expressed in current ego axes.

    ``None`` means the third observation does not exist. An available empty
    observation deliberately means it exists but contains no eligible detections.
    No abstention encodes a stationary classification.
    """
    _check_age(latest_dt)
    pairs = tuple(zip(current, initial, strict=True))
    if oldest is None:
        if older_dt is not None:
            raise ValueError("An absent older bundle cannot have an interval")
        return tuple(Confirmation("no_older_observation" if motion.status == "estimated" else "no_initial_link")
                     for _, motion in pairs)
    _check_age(older_dt)
    previous = {m.object_id: m for m in middle}
    earlier = {o.object_id: o for o in oldest}
    back = dict(zip(previous, associate_objects(middle, oldest, older_dt, mode="appearance"), strict=True))
    if len(previous) != len(middle) or len(earlier) != len(oldest):
        raise ValueError("Duplicate object identity within one observation")
    results = []
    for now, hop in pairs:
        if hop.status != "estimated":
            results.append(Confirmation("no_initial_link"))
            continue
        matched = previous.get(hop.previous_object_id)
        if matched is None or matched.kind != now.kind:
            raise ValueError("The initial link must refer to the exact middle observation")
        prior_hop = back[matched.object_id]
        if prior_hop.status != "estimated":
            results.append(Confirmation("no_older_link"))
            continue
        old = earlier[prior_hop.previous_object_id]
        results.append(_measure(now, matched, old, hop, latest_dt, older_dt))
    return tuple(results)


def _measure(current, mid, old, first, latest_dt, older_dt):
    now, middle, earlier = ({hit.camera_name: hit for hit in o.hits} for o in (current, mid, old))
    common = sorted(set(now) & set(middle) & set(earlier))
    if not common:
        return Confirmation("no_common_camera")
    nx, ny = first.radial_unit_xy
    tangent = (-ny, nx)
    current_rate = _dot(first.velocity_xy_mps, tangent)
    accepted_rates, weights = [], []
    stamps = (-latest_dt - older_dt, -latest_dt, 0.)
    for name in common:
        a, b, c = earlier[name], middle[name], now[name]
        if fsum(abs(x - y) for x, y in zip(a.appearance, c.appearance, strict=True)) > MAX_APPEARANCE_L1:
            return Confirmation("inconsistent_appearance")
        positions = tuple(_dot(point.xy, tangent) for point in (a, b, c))
        oldest_step, latest_step = positions[1] - positions[0], positions[2] - positions[1]
        if min(abs(oldest_step), abs(latest_step)) < MIN_STEP_M or abs(positions[2] - positions[0]) < MIN_BASELINE_M:
            return Confirmation("below_motion_floor")
        first_rate, last_rate = oldest_step / older_dt, latest_step / latest_dt
        if not _agree(first_rate, last_rate) or not _agree(last_rate, current_rate):
            return Confirmation("inconsistent_motion")
        accepted_rates.append(_trend(stamps, positions))
        weights.append(min(a.confidence, b.confidence, c.confidence) /
                       max(.15, a.uncertainty_m + b.uncertainty_m + c.uncertainty_m)**2)
    if any(left * right <= 0 or abs(left - right) > MAX_CAMERA_RATE_DIFFERENCE_MPS
           for i, left in enumerate(accepted_rates) for right in accepted_rates[:i]):
        return Confirmation("inconsistent_cameras")
    denom = fsum(weights)
    if denom <= 0:
        return Confirmation("no_reliable_camera_weight")
    rate = fsum(x * w for x, w in zip(accepted_rates, weights, strict=True)) / denom
    return Confirmation("confirmed", old.object_id, tuple(common), tuple(a * rate for a in tangent))


def frozen_two_frame_velocity(hop, *, radial_gain, tangential_gain):
    if hop.status != "estimated":
        return 0., 0.
    radial, velocity = hop.radial_unit_xy, hop.velocity_xy_mps
    parallel = _dot(velocity, radial)
    return tuple(tangential_gain * value + (radial_gain - tangential_gain) * parallel * axis
                 for value, axis in zip(velocity, radial, strict=True))


def _check_age(value):
    if type(value) not in (int, float) or not isfinite(value) or not .4 <= value <= .6:
        raise ValueError("Expected an observed interval between 0.4 and 0.6 seconds")


def _agree(left, right):
    return left * right > 0 and abs(left - right) <= MIN_RATE_AGREEMENT_MPS + RELATIVE_RATE_ALLOWANCE * max(abs(left), abs(right))


def _dot(v, w):
    return fsum(a * b for a, b in zip(v, w, strict=True))


def _trend(times, positions):
    mean_t = fsum(times) / len(times)
    mean_pos = fsum(positions) / len(positions)
    return fsum((t-mean_t) * (v-mean_pos) for t, v in zip(times, positions, strict=True)) / fsum((t-mean_t)**2 for t in times)
