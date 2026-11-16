from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class BrakingProfile:
    down_s: float
    hold_s: float
    up_s: float
    down_distance_m: float
    stop_distance_m: float
    plateau_start_speed_mps: float
    peak_deceleration_mps2: float
    peak_speed_mps: float

    @property
    def stop_time_s(self) -> float:
        return self.down_s + self.hold_s + self.up_s


def build_profile(v: float, a: float, cap: float, jerk: float, minimum_speed: float) -> BrakingProfile:
    equivalent_speed = v + minimum_speed
    saturation_speed = cap * (cap / jerk)
    if equivalent_speed >= saturation_speed:
        peak = cap
        hold = (equivalent_speed - saturation_speed) / peak
        down = (a + peak) / jerk
    else:
        peak = -a if a < 0 and v == minimum_speed else math.sqrt(jerk) * math.sqrt(equivalent_speed)
        hold = 0.0
        down = (v - minimum_speed) / (peak - a) if a < 0 else (a + peak) / jerk
    if not math.isfinite(peak) or peak <= 0:
        raise FloatingPointError("Unrepresentable braking peak")
    up = peak / jerk
    release_speed = peak * up / 2
    plateau_speed = release_speed + peak * hold
    down_distance = down * (plateau_speed + down * (peak / 3 - a / 6))
    release_distance = release_speed * up / 3
    stop_distance = down_distance + hold * (release_speed + peak * hold / 2) + release_distance
    peak_speed = v + minimum_speed if a > 0 else v
    values = (equivalent_speed, down, hold, up, release_speed, plateau_speed, down_distance, stop_distance, peak_speed)
    if not all(math.isfinite(value) and value >= 0 for value in values):
        raise FloatingPointError("Unrepresentable stopping profile")
    release_start = down + hold
    stop_time = release_start + up
    if up <= 0 or not math.isfinite(stop_time) or stop_time <= release_start or stop_distance <= 0:
        raise FloatingPointError("Unrepresentable stopping endpoint")
    if hold > 0 and release_start <= down:
        raise FloatingPointError("Unrepresentable plateau duration")
    return BrakingProfile(down, hold, up, down_distance, stop_distance, plateau_speed, peak, peak_speed)


def sample_distances(profile: BrakingProfile, v: float, a: float, jerk: float, times) -> np.ndarray:
    distances = np.full(len(times), profile.stop_distance_m)
    down = times < profile.down_s
    t = times[down]
    distances[down] = (v + (a / 2 - jerk * t / 6) * t) * t
    release_start = profile.down_s + profile.hold_s
    hold = (times >= profile.down_s) & (times < release_start)
    t = times[hold] - profile.down_s
    distances[hold] = profile.down_distance_m + (profile.plateau_start_speed_mps - profile.peak_deceleration_mps2 * t / 2) * t
    up = (times >= release_start) & (times < profile.stop_time_s)
    ratio = (profile.stop_time_s - times[up]) / profile.up_s
    release_distance = profile.peak_deceleration_mps2 * profile.up_s / 2 * profile.up_s / 3
    # Evaluate backward from rest, rather than clipping a rounded negative velocity.
    distances[up] = profile.stop_distance_m - release_distance * ratio**3
    return distances
