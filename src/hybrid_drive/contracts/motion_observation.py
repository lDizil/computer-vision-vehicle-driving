"""Associated OpenScene body-motion samples, not synchronized state guarantees."""

from dataclasses import dataclass, field
import math
from numbers import Integral
import re
import sys

from ._validation import finite_real, tuple_snapshot


OBSERVATION_METHOD = "openscene-body-motion-four-associated-frames-v1"
HEADING_RATE_METHOD = "body-qdot-pinned-pyquaternion-heading-v1"
POSE_RATE_METHOD = "past-shortest-heading-arc-over-associated-frame-time-v1"


@dataclass(frozen=True)
class MotionSample:
    """The frame time identifies a LiDAR association, not the sensor capture time.

    Heading rate is a qdot/body-gyro projection model using the pinned NAVSIM
    heading convention. It is not a synchronized measured yaw rate or a bound.
    """

    frame_token: str
    metadata_scene_token: str
    source_list_index: int
    metadata_frame_idx: int
    frame_timestamp_us: int
    global_translation_m: tuple[float, float, float]
    quaternion_wxyz: tuple[float, float, float, float]
    body_velocity_mps: tuple[float, float, float]
    body_acceleration_mps2: tuple[float, float, float]
    body_angular_rate_radps: tuple[float, float, float]
    pose_capture_timestamp_us: None = None
    motion_capture_timestamp_us: None = None
    heading_rad: float | None = field(init=False)
    associated_planar_heading_rate_radps: float | None = field(init=False)
    heading_rate_failure: str | None = field(init=False)
    heading_rate_uncertainty_radps: None = field(init=False, default=None)

    def __post_init__(self):
        for name in ("frame_token", "metadata_scene_token"):
            _hex(getattr(self, name), 16, name)
        for name in ("source_list_index", "metadata_frame_idx", "frame_timestamp_us"):
            object.__setattr__(self, name, _integer(getattr(self, name), name))
        for name, size in (("global_translation_m", 3), ("quaternion_wxyz", 4),
                           ("body_velocity_mps", 3), ("body_acceleration_mps2", 3),
                           ("body_angular_rate_radps", 3)):
            values = tuple_snapshot(getattr(self, name), name)
            if len(values) != size:
                raise ValueError(name + " has an invalid shape")
            object.__setattr__(self, name, tuple(finite_real(value, name) for value in values))
        if self.pose_capture_timestamp_us is not None or self.motion_capture_timestamp_us is not None:
            raise ValueError("OpenScene does not retain the linked pose/motion capture timestamp")
        heading, rate, failure = _heading_from_body_rate(self.quaternion_wxyz, self.body_angular_rate_radps)
        object.__setattr__(self, "heading_rad", heading)
        object.__setattr__(self, "associated_planar_heading_rate_radps", rate)
        object.__setattr__(self, "heading_rate_failure", failure)


def _heading_from_body_rate(quaternion, angular_rate):
    scale = max(abs(value) for value in quaternion)
    if scale == 0:
        raise ValueError("quaternion_wxyz must define an orientation")
    scaled = tuple(value / scale for value in quaternion)
    norm = math.hypot(*scaled)
    w, x, y, z = (value / norm for value in scaled)
    # This is the pinned NAVSIM Quaternion.yaw_pitch_roll[0], not ZYX yaw.
    numerator, denominator = 2 * (w * z - x * y), 1 - 2 * (y * y + z * z)
    radius = math.hypot(numerator, denominator)
    # atan2 loses its angle at the singularity; this is a floating-point guard,
    # not sensor uncertainty or a physical admission threshold.
    if radius <= 32 * sys.float_info.epsilon:
        return None, None, "heading_singularity"
    heading = math.atan2(numerator, denominator)
    # Algebraically differentiating N/D under qdot=.5*q*(0,omega) leaves
    # coefficient +1 for omega_z. Avoid halving subnormal angular rates.
    coupling = 2 * (w * y + x * z) / radius
    coefficients = (-(denominator / radius) * coupling, (numerator / radius) * coupling, 1.0)
    scale = max(abs(value) for value in angular_rate)
    if scale == 0:
        return heading, 0.0, None
    normalized = tuple(value / scale for value in angular_rate)
    terms = tuple(coefficient * value for coefficient, value in zip(coefficients, normalized))
    if (any(original != 0 and value == 0 for original, value in zip(angular_rate, normalized))
            or any(coefficient != 0 and value != 0 and term == 0
                   for coefficient, value, term in zip(coefficients, normalized, terms))):
        return heading, None, "numeric_range"
    scaled_rate = math.fsum(terms)
    rate = scaled_rate * scale
    if not math.isfinite(rate) or (scaled_rate != 0 and rate == 0):
        return heading, None, "numeric_range"
    return heading, rate, None


@dataclass(frozen=True)
class MotionObservation:
    base_bundle_hash: str
    source_revision: str
    navsim_revision: str
    nuplan_revision: str
    metadata_sha256: str
    source_manifest_hash: str
    samples: tuple[MotionSample, ...]
    past_interval_mean_heading_rates_radps: tuple[float | None, ...] = field(init=False)

    def __post_init__(self):
        for name in ("base_bundle_hash", "metadata_sha256", "source_manifest_hash"):
            _hex(getattr(self, name), 64, name)
        for name in ("source_revision", "navsim_revision", "nuplan_revision"):
            _hex(getattr(self, name), 40, name)
        samples = tuple_snapshot(self.samples, "samples")
        if len(samples) != 4 or any(not isinstance(value, MotionSample) for value in samples):
            raise ValueError("Exactly four past/current MotionSample values are required")
        if len({sample.frame_token for sample in samples}) != 4:
            raise ValueError("Motion frame tokens must be unique")
        rates = []
        for previous, current in zip(samples, samples[1:]):
            if (current.metadata_scene_token != previous.metadata_scene_token
                    or current.source_list_index != previous.source_list_index + 1
                    or current.metadata_frame_idx != previous.metadata_frame_idx + 1
                    or current.frame_timestamp_us <= previous.frame_timestamp_us):
                raise ValueError("Motion samples must be consecutive ordered frames of one source chunk")
            rate = None
            if previous.heading_rad is not None and current.heading_rad is not None:
                delta = math.remainder(current.heading_rad - previous.heading_rad, 2 * math.pi)
                if abs(delta) != math.pi:
                    rate = delta / ((current.frame_timestamp_us - previous.frame_timestamp_us) / 1e6)
            rates.append(rate)
        object.__setattr__(self, "samples", samples)
        object.__setattr__(self, "past_interval_mean_heading_rates_radps", tuple(rates))

    @property
    def current(self) -> MotionSample:
        return self.samples[-1]


def _integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or not 0 <= value < 2**63:
        raise ValueError(name + " must be a nonnegative bounded integer")
    return int(value)


def _hex(value, length, name):
    if not isinstance(value, str) or not re.fullmatch("[0-9a-f]{" + str(length) + "}", value):
        raise ValueError(name + " must be a lowercase hex identifier")
