"""Causal Depth Pro 2 W4 Core ML adapter for pinned rectified camera frames."""

from __future__ import annotations

from hashlib import sha256
import math
import os
from pathlib import Path
from select import select
from struct import Struct
import subprocess
from threading import Lock
from time import monotonic
from typing import Hashable

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from hybrid_drive.assets.files import check_components, verify_file
from hybrid_drive.experiments.bounded_io import canonical_json, decode_json, read_bounded, require_hex, require_keys
from hybrid_drive.geometry.camera import CameraCalibration


_WIDTH, _HEIGHT = 640, 360
_RIG_WIDTH, _RIG_HEIGHT = 768, 432
_CAMERA_SLOTS = 6
_DEPTH_BYTES = _WIDTH * _HEIGHT * 4
_REQUEST = Struct("<4sBBH")
_REPLY = Struct("<4sI")
PINNED_MANIFESTS_SHA256 = frozenset({
    "d0b1efcaa538ed6bff8bc25a3f4a3e6b7a87fd33bb31436ae73d6fca20eb3bf6",
    "516b3bb979e42d6a894f38ec4ac170f91b99c2a75d08e48b6d7dcd951704274b",
})
PROFILE_KEYS = {"worker_sha256", "checkpoint_manifest_sha256", "compiled_bundle_sha256"}


def checkpoint_profile(worker, compiled, manifest):
    """Bind the executable and compiled bundle, verifying the pinned weight bytes."""
    raw = read_bounded(manifest, 16384)
    manifest_sha = sha256(raw).hexdigest()
    if manifest_sha not in PINNED_MANIFESTS_SHA256:
        raise ValueError("Expected a pinned Depth Pro 2 checkpoint manifest")
    spec = decode_json(raw)
    compiled = Path(compiled)
    check_components(compiled)
    if compiled.name != Path(spec["directory"]).with_suffix(".mlmodelc").name or not compiled.is_dir():
        raise ValueError("Expected the compiled Depth Pro 2 bundle for this manifest")
    weights = spec["files"]["Data/com.apple.CoreML/weights/weight.bin"]
    weight_name = "weights/weight.bin"
    if verify_file(compiled / weight_name, **weights) is None:
        raise ValueError("Compiled Depth Pro 2 bundle is missing pinned weights")
    fingerprints = {}
    for i, path in enumerate(compiled.rglob("*")):
        if i >= 64:
            raise ValueError("Unexpected Depth Pro 2 bundle layout")
        check_components(path)
        if not path.is_file():
            continue
        name = path.relative_to(compiled).as_posix()
        fingerprints[name] = weights["sha256"] if name == weight_name else sha256(read_bounded(path, 16 * 1024**2)).hexdigest()
    if "model.mil" not in fingerprints or weight_name not in fingerprints:
        raise ValueError("Expected a complete compiled Depth Pro 2 graph and weights")
    return {"worker_sha256": sha256(read_bounded(worker, 16 * 1024**2)).hexdigest(),
            "checkpoint_manifest_sha256": manifest_sha,
            "compiled_bundle_sha256": sha256(canonical_json(fingerprints)).hexdigest()}


def validate_depthpro_profile(profile):
    require_keys(profile, PROFILE_KEYS, "Depth Pro 2 execution profile")
    for name, digest in profile.items():
        require_hex(digest, 64, name)
    if profile["checkpoint_manifest_sha256"] not in PINNED_MANIFESTS_SHA256:
        raise ValueError("Expected a pinned Depth Pro 2 checkpoint manifest")
    return profile


class DepthPro2Error(RuntimeError):
    """A terminated worker cannot be reused, because its clip history is unknown."""


class DepthPro2:
    """Reuse one local compiled model and isolate its state by camera.

    Pass the same key for successive observations from one camera, in capture
    order. Start each clip with `reset=True` per camera or `reset_streams()`.
    A prediction returns optical-axis Z in metres in the input raster; invalid
    depth is NaN. The worker sees only the supplied current frame and its state.
    """

    def __init__(
        self,
        worker_path: str | Path,
        compiled_model_path: str | Path,
        *,
        timeout_s: float = 120.0,
    ) -> None:
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("A positive worker timeout is required")
        worker, model = Path(worker_path).absolute(), Path(compiled_model_path).absolute()
        if not worker.is_file() or model.suffix != ".mlmodelc" or not model.is_dir():
            raise ValueError("Depth Pro 2 requires a local worker and a compiled .mlmodelc package")

        self._timeout_s = timeout_s
        self._slots: dict[Hashable, tuple[int, str]] = {}
        self._lock = Lock()
        self._process: subprocess.Popen[bytes] | None = subprocess.Popen(
            [worker, model], stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0,
        )
        assert self._process.stdin is not None and self._process.stdout is not None
        self._input_fd, self._output_fd = self._process.stdin.fileno(), self._process.stdout.fileno()
        try:
            os.set_blocking(self._input_fd, False)
            os.set_blocking(self._output_fd, False)
        except OSError:
            self._close_worker()
            raise

    def __enter__(self) -> DepthPro2:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def close(self) -> None:
        with self._lock:
            self._close_worker()

    def reset_streams(self) -> None:
        """Each reused slot will send a reset on its next frame."""
        with self._lock:
            self._slots.clear()

    def predict(
        self, rgb: NDArray[np.uint8], calibration: CameraCalibration, *, stream_key: Hashable, reset: bool = False,
    ) -> NDArray[np.float32]:
        if (
            not isinstance(rgb, np.ndarray)
            or (rgb.shape, rgb.dtype) != ((_RIG_HEIGHT, _RIG_WIDTH, 3), np.dtype(np.uint8))
            or not isinstance(calibration, CameraCalibration)
            or (calibration.width, calibration.height) != (_RIG_WIDTH, _RIG_HEIGHT)
        ):
            raise ValueError("Depth Pro 2 expects a calibrated uint8 768x432 RGB frame")
        focal_px = float(calibration.resized(_WIDTH, _HEIGHT).intrinsic[0, 0])
        pixels = Image.fromarray(rgb).resize((_WIDTH, _HEIGHT), Image.Resampling.BILINEAR).tobytes()

        with self._lock:
            if self._process is None:
                raise DepthPro2Error("Depth Pro 2 worker is closed")
            stream = self._slots.get(stream_key)
            if stream is None:
                if len(self._slots) == _CAMERA_SLOTS:
                    raise ValueError("Depth Pro 2 accepts six live camera streams; reset before a new clip")
                slot = len(self._slots)
                reset = True
            else:
                slot, camera_name = stream
                if camera_name != calibration.name and not reset:
                    raise ValueError("Reset a temporal stream before changing its camera")
            try:
                raw = self._predict_inverse(_REQUEST.pack(b"DPQ2", slot, int(bool(reset)), 0) + pixels)
            except BaseException:
                self._close_worker()
                raise
            self._slots[stream_key] = (slot, calibration.name)
        return _to_optical_depth(raw, focal_px)

    def _predict_inverse(self, message: bytes) -> bytes:
        deadline = monotonic() + self._timeout_s
        remaining = memoryview(message)
        while remaining:
            timeout = max(0., deadline - monotonic())
            if not select([], [self._input_fd], [], timeout)[1]:
                raise DepthPro2Error("Timed out sending frame to Depth Pro 2 worker")
            try:
                count = os.write(self._input_fd, remaining)
            except BlockingIOError:
                continue
            if count == 0:
                raise DepthPro2Error("Depth Pro 2 worker stopped reading")
            remaining = remaining[count:]

        magic, size = _REPLY.unpack(self._read(_REPLY.size, deadline))
        if magic == b"DPE2" and 1 <= size <= 4096:
            message = self._read(size, deadline).decode("utf-8", errors="replace")
            raise DepthPro2Error(f"Depth Pro 2 worker error: {message}")
        if magic != b"DPR2" or size != _DEPTH_BYTES:
            raise DepthPro2Error("Invalid reply header from Depth Pro 2 worker")
        return self._read(size, deadline)

    def _read(self, size: int, deadline: float) -> bytes:
        result = bytearray()
        while len(result) < size:
            timeout = max(0., deadline - monotonic())
            if not select([self._output_fd], [], [], timeout)[0]:
                raise DepthPro2Error("Timed out reading reply from Depth Pro 2 worker")
            try:
                piece = os.read(self._output_fd, size - len(result))
            except BlockingIOError:
                continue
            if not piece:
                raise DepthPro2Error("Depth Pro 2 worker closed its reply early")
            result.extend(piece)
        return bytes(result)

    def _close_worker(self) -> None:
        process, self._process = self._process, None
        if process is None:
            return
        assert process.stdin is not None and process.stdout is not None
        process.stdin.close()
        process.stdout.close()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def _to_optical_depth(buffer: bytes, focal_px: float) -> NDArray[np.float32]:
    inverse = np.frombuffer(buffer, dtype="<f4").reshape(_HEIGHT, _WIDTH)
    depth = np.full(inverse.shape, np.nan, dtype=np.float32)
    # Apple returns f_x / (width * Z), not 1/Z. Near sky/masked pixels
    # remain missing when mapped back; interpolation could mix backgrounds.
    with np.errstate(over="ignore", divide="ignore"):
        np.divide(focal_px / _WIDTH, inverse, out=depth, where=(inverse > 0) & np.isfinite(inverse))
    depth[~np.isfinite(depth)] = np.nan
    rows = ((2 * np.arange(_RIG_HEIGHT) + 1) * _HEIGHT) // (2 * _RIG_HEIGHT)
    cols = ((2 * np.arange(_RIG_WIDTH) + 1) * _WIDTH) // (2 * _RIG_WIDTH)
    return depth[rows[:, None], cols]
