from dataclasses import dataclass

from ._validation import finite_real


@dataclass(frozen=True)
class EgoMotion:
    """Measured motion at t_ref in the trajectory's fixed ego frame."""

    vx_mps: float
    vy_mps: float
    ax_mps2: float
    ay_mps2: float

    def __post_init__(self):
        for name in ("vx_mps", "vy_mps", "ax_mps2", "ay_mps2"):
            object.__setattr__(self, name, finite_real(getattr(self, name), name))
