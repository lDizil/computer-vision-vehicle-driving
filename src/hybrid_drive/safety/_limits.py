import math


def exceeds_limit(value: float, bound: float) -> bool:
    return value > bound and not math.isclose(value, bound, rel_tol=1e-9, abs_tol=1e-9)


def same_interval(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-12, abs_tol=0)
