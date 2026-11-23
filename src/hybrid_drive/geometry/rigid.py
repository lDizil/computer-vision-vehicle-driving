"""Checked rigid transforms, using column vectors and w,x,y,z quaternions."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def rigid_transform(value: ArrayLike, name: str) -> NDArray[np.float64]:
    transform = matrix(value, (4, 4), name)
    if not np.array_equal(transform[3], [0, 0, 0, 1]):
        raise ValueError(f"{name} must have homogeneous bottom row [0, 0, 0, 1]")
    rotation = transform[:3, :3]
    with np.errstate(over="ignore", invalid="ignore"):
        orthonormal = np.allclose(rotation @ rotation.T, np.eye(3), rtol=0, atol=1e-9)
        determinant = np.linalg.det(rotation)
    if not orthonormal or not np.isclose(determinant, 1, rtol=0, atol=1e-9):
        raise ValueError(f"{name} rotation must be proper orthonormal (determinant +1)")
    return transform


def pose_from_quaternion(translation: ArrayLike, rotation_wxyz: ArrayLike) -> NDArray[np.float64]:
    quaternion = matrix(rotation_wxyz, (4,), "rotation_wxyz")
    with np.errstate(over="ignore", invalid="ignore"):
        norm = np.linalg.norm(quaternion)
    if not np.isclose(norm, 1, rtol=0, atol=1e-5):
        raise ValueError("rotation_wxyz must be a unit quaternion")
    w, x, y, z = quaternion / norm
    pose = np.eye(4)
    pose[:3, :3] = (
        (1 - 2 * (y*y + z*z), 2 * (x*y - w*z), 2 * (x*z + w*y)),
        (2 * (x*y + w*z), 1 - 2 * (x*x + z*z), 2 * (y*z - w*x)),
        (2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x*x + y*y)),
    )
    pose[:3, 3] = matrix(translation, (3,), "translation")
    return pose


def invert_pose(value: ArrayLike) -> NDArray[np.float64]:
    pose = rigid_transform(value, "pose")
    inverse = np.eye(4)
    inverse[:3, :3] = pose[:3, :3].T
    with np.errstate(over="ignore", invalid="ignore"):
        inverse[:3, 3] = -inverse[:3, :3] @ pose[:3, 3]
    if not np.isfinite(inverse).all():
        raise ValueError("inverse pose must be finite")
    return inverse


def matrix(value: ArrayLike, shape: tuple[int, ...], name: str) -> NDArray[np.float64]:
    result = numeric_array(value, name)
    if result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    return result


def numeric_array(value: ArrayLike, name: str) -> NDArray[np.float64]:
    try:
        array = np.asarray(value)
        # NumPy otherwise silently coerces mixed Python booleans to integers.
        contains_bool = any(isinstance(item, (bool, np.bool_)) for item in np.asarray(value, dtype=object).flat)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain finite real numeric values") from exc
    if array.dtype.kind not in "fiu" or contains_bool:
        raise ValueError(f"{name} must contain finite real numeric values, not booleans")
    with np.errstate(over="ignore", invalid="ignore"):
        result = array.astype(np.float64, copy=True)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain finite real numeric values")
    return result
