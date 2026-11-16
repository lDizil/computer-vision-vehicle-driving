from dataclasses import dataclass, field
from enum import Enum
import math
from typing import ClassVar

import numpy as np
from numpy.typing import ArrayLike, NDArray

from hybrid_drive.contracts._validation import finite_real
from hybrid_drive.contracts.motion import EgoMotion


MODEL_ASSUMPTIONS = (
    "frozen_snapshot_coassociation_not_measurement_synchronization",
    "constant_path_curvature",
    "constant_body_heading_curvature",
    "prescribed_variable_sideslip_not_vehicle_dynamics",
)


class PlanarMotionFailure(str, Enum):
    UNSUPPORTED_MOTION = "unsupported_motion"
    NUMERIC_RANGE = "numeric_range"


class PlanarMotionError(ValueError):
    def __init__(self, reason: PlanarMotionFailure):
        super().__init__(reason.value)
        self.reason = reason


@dataclass(frozen=True)
class PlanarMotionBounds:
    """Conditional continuous upper bounds, not measured peaks or vehicle safety."""

    speed_mps: float
    acceleration_mps2: float
    jerk_mps3: float
    heading_rate_radps: float
    heading_acceleration_radps2: float
    heading_jerk_radps3: float
    sideslip_rate_radps: float
    assumptions: tuple[str, ...] = MODEL_ASSUMPTIONS


@dataclass(frozen=True)
class PlanarMotionHypothesis:
    """Point-mass path and independently supplied body-heading extrapolation.

    Position follows constant path curvature kappa; body heading is rho*s,
    where rho is the associated heading rate divided by initial speed. Thus
    beta(s)=beta0+(kappa-rho)*s, not a no-slip or fixed-sideslip vehicle model.
    Inputs are co-associated under a frozen-snapshot assumption; this class
    neither measures their capture times nor establishes synchronization.
    """

    motion: EgoMotion
    heading_rate_radps: float
    speed_mps: float = field(init=False)
    sideslip_rad: float = field(init=False)
    tangential_acceleration_mps2: float = field(init=False)
    normal_acceleration_mps2: float = field(init=False)
    path_curvature_per_m: float = field(init=False)
    heading_curvature_per_m: float = field(init=False)
    _tangent: tuple[float, float] = field(init=False, repr=False)

    model_id: ClassVar[str] = "constant-path-heading-curvature-v1"
    assumptions: ClassVar[tuple[str, ...]] = MODEL_ASSUMPTIONS

    @classmethod
    def from_motion(cls, motion: EgoMotion, heading_rate_radps: float) -> "PlanarMotionHypothesis":
        return cls(motion, heading_rate_radps)

    def __post_init__(self):
        if not isinstance(self.motion, EgoMotion):
            raise ValueError("motion must be EgoMotion")
        rate = finite_real(self.heading_rate_radps, "heading_rate_radps")
        motion = self.motion
        if motion.vx_mps <= 0:
            raise PlanarMotionError(PlanarMotionFailure.UNSUPPORTED_MOTION)
        speed = math.hypot(motion.vx_mps, motion.vy_mps)
        _finite(speed, math.hypot(motion.ax_mps2, motion.ay_mps2))
        tx, ty = motion.vx_mps / speed, motion.vy_mps / speed
        if tx == 0 or (motion.vy_mps != 0 and ty == 0):
            raise PlanarMotionError(PlanarMotionFailure.NUMERIC_RANGE)
        try:
            tangential = math.fsum((_product(tx, motion.ax_mps2), _product(ty, motion.ay_mps2)))
            normal = math.fsum((_product(-ty, motion.ax_mps2), _product(tx, motion.ay_mps2)))
        except OverflowError as exc:
            raise PlanarMotionError(PlanarMotionFailure.NUMERIC_RANGE) from exc
        curvature = _ratio(_ratio(normal, speed), speed)
        heading_curvature = _ratio(rate, speed)
        slip = math.atan2(motion.vy_mps, motion.vx_mps)
        _finite(tangential, normal, slip, curvature - heading_curvature)
        if not -math.pi / 2 < slip < math.pi / 2:
            raise PlanarMotionError(PlanarMotionFailure.NUMERIC_RANGE)
        values = {
            "heading_rate_radps": rate,
            "speed_mps": speed,
            "sideslip_rad": slip,
            "tangential_acceleration_mps2": tangential,
            "normal_acceleration_mps2": normal,
            "path_curvature_per_m": curvature,
            "heading_curvature_per_m": heading_curvature,
            "_tangent": (tx, ty),
        }
        for name, value in values.items():
            object.__setattr__(self, name, value)

    def poses_at_distances(self, distances_m: ArrayLike) -> NDArray[np.float64]:
        """Map nonnegative arc lengths to x/y and unwrapped body heading."""
        distances = np.asarray(distances_m)
        if distances.ndim != 1 or not distances.size or distances.dtype.kind not in "iuf":
            raise ValueError("distances_m must be a nonempty real vector")
        with np.errstate(over="ignore"):
            distances = distances.astype(np.float64)
        if not np.isfinite(distances).all() or np.any(distances < 0):
            raise ValueError("distances_m must be finite and nonnegative")
        poses = np.empty((len(distances), 3), dtype=np.float64)
        tx, ty = self._tangent
        for index, distance in enumerate(distances):
            distance = float(distance)
            angle = _product(self.path_curvature_per_m, distance)
            if abs(angle) <= 1:
                along = _product(distance, _sinc(angle))
                # This ordering retains subnormal angles before halving them.
                normal = _product(_product(distance, angle), 0.5 * _sinc(angle / 2) ** 2)
            else:
                along = math.sin(angle) / self.path_curvature_per_m
                normal = 2 * math.sin(angle / 2) ** 2 / self.path_curvature_per_m
            try:
                x = math.fsum((_product(along, tx), _product(-normal, ty)))
                y = math.fsum((_product(along, ty), _product(normal, tx)))
            except OverflowError as exc:
                raise PlanarMotionError(PlanarMotionFailure.NUMERIC_RANGE) from exc
            heading = _product(self.heading_curvature_per_m, distance)
            _finite(x, y, heading)
            poses[index] = x, y, heading
        return poses

    def continuous_bounds(
        self, peak_speed_mps: float, max_abs_tangential_acceleration_mps2: float,
        max_abs_tangential_jerk_mps3: float,
    ) -> PlanarMotionBounds:
        """Assume 0<=v<=V, |a_t|<=B, |j_t|<=q over the complete profile.

        Frenet derivatives are a=(a_t,kappa*v²) and
        j=(j_t-kappa²*v³,3*kappa*v*a_t). Bounds include every phase,
        not only output samples. The caller owns scalar-profile certification.
        Outward rounding guards the bounds of the stored model coefficients.
        """
        speed = finite_real(peak_speed_mps, "peak_speed_mps")
        acceleration = finite_real(max_abs_tangential_acceleration_mps2, "max_abs_tangential_acceleration_mps2")
        jerk = finite_real(max_abs_tangential_jerk_mps3, "max_abs_tangential_jerk_mps3")
        if speed < self.speed_mps or acceleration < abs(self.tangential_acceleration_mps2) or jerk < 0:
            raise ValueError("Profile bounds must include the initial state and nonnegative jerk bound")
        omega = _upper_product(abs(self.path_curvature_per_m), speed)
        lateral = _upper_product(omega, speed)
        tangent_jerk = _upper_sum(jerk, _upper_product(lateral, omega))
        normal_jerk = _upper_product(3, omega, acceleration)
        rho = abs(self.heading_curvature_per_m)
        slip_lower, slip_upper = _difference_interval(self.path_curvature_per_m, self.heading_curvature_per_m)
        slip_curvature = max(abs(slip_lower), abs(slip_upper))
        return PlanarMotionBounds(
            speed, _upper_hypot(acceleration, lateral), _upper_hypot(tangent_jerk, normal_jerk),
            _upper_product(rho, speed), _upper_product(rho, acceleration), _upper_product(rho, jerk),
            _upper_product(slip_curvature, speed),
        )

    def sideslip_range(self, stop_distance_m: float) -> tuple[float, float]:
        """Conservative unwrapped beta enclosure, without a calibrated slip cap."""
        distance = finite_real(stop_distance_m, "stop_distance_m")
        if distance < 0:
            raise ValueError("stop_distance_m must be nonnegative")
        start = self.sideslip_rad
        if distance == 0 or self.path_curvature_per_m == self.heading_curvature_per_m:
            return start, start
        lower, upper = _difference_interval(self.path_curvature_per_m, self.heading_curvature_per_m)
        if distance != 1:
            lower = math.nextafter(lower * distance, -math.inf) if lower else 0.0
            upper = math.nextafter(upper * distance, math.inf) if upper else 0.0
            _finite(lower, upper)
        if start != 0:
            lower = math.nextafter(start + lower, -math.inf) if lower else start
            upper = math.nextafter(start + upper, math.inf) if upper else start
        _finite(lower, upper)
        return min(start, lower), max(start, upper)


def _finite(*values):
    if not all(math.isfinite(value) for value in values):
        raise PlanarMotionError(PlanarMotionFailure.NUMERIC_RANGE)


def _ratio(numerator, denominator):
    value = numerator / denominator
    _finite(value)
    if numerator != 0 and value == 0:
        raise PlanarMotionError(PlanarMotionFailure.NUMERIC_RANGE)
    return value


def _product(left, right):
    value = left * right
    _finite(value)
    if left != 0 and right != 0 and value == 0:
        raise PlanarMotionError(PlanarMotionFailure.NUMERIC_RANGE)
    return value


def _sinc(angle):
    return math.sin(angle) / angle if angle else 1.0


def _difference_interval(left, right):
    difference = left - right
    _finite(difference)
    if left == right or left == 0 or right == 0:
        return difference, difference
    lower = math.nextafter(difference, -math.inf)
    upper = math.nextafter(difference, math.inf)
    _finite(lower, upper)
    return lower, upper


def _upper_product(*values):
    if 0 in values:
        return 0.0
    factors = sorted(value for value in values if value != 1)
    if not factors:
        return 1.0
    result = factors[0]
    for value in factors[1:]:
        result = math.nextafter(result * value, math.inf)
        _finite(result)
    return result


def _upper_sum(left, right):
    if left == 0 or right == 0:
        return left + right
    value = math.nextafter(left + right, math.inf)
    _finite(value)
    return value


def _upper_hypot(left, right):
    if left == 0 or right == 0:
        return left + right
    value = math.nextafter(math.hypot(left, right), math.inf)
    _finite(value)
    return value
