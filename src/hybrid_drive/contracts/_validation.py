import math
from numbers import Real

import numpy as np


def finite_real(value: float, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be a finite real number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    return result


def identifier(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonblank string")
    return value


def tuple_snapshot(value, name: str) -> tuple:
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a sequence, not a string")
    try:
        return tuple(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a sequence") from exc
