"""Exact-size, SHA256-verified opaque downloads from pinned HTTP sources."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import re
import time
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from .files import AtomicFile, PreparationError, verify_file


MIB = 1024 * 1024
MAX_TRANSFER_BYTES = 4 * 1024 * MIB
CHUNK_SIZE = 64 * 1024


@dataclass(frozen=True)
class AssetSpec:
    url: str
    revision: str
    size_bytes: int
    sha256: str
    etag: str

    @classmethod
    def from_dict(cls, value):
        return cls(**{key: value[key] for key in ("url", "revision", "size_bytes", "sha256", "etag")})

    def validate(self):
        parsed = urlsplit(self.url)
        if parsed.scheme != "https" or parsed.hostname != "huggingface.co" or parsed.username or parsed.password:
            raise PreparationError("Only pinned public Hugging Face HTTPS sources are allowed")
        if parsed.query or parsed.fragment or not re.fullmatch(r"[0-9a-f]{40}", self.revision):
            raise PreparationError("Expected an immutable source revision")
        if f"/resolve/{self.revision}/" not in parsed.path:
            raise PreparationError("Source URL does not contain the pinned revision")
        if not re.fullmatch(r"[0-9a-f]{64}", self.sha256) or not re.fullmatch(r"[0-9a-f]{40,64}", self.etag):
            raise PreparationError("Invalid content digest or ETag")
        if not 0 < self.size_bytes <= MAX_TRANSFER_BYTES:
            raise PreparationError("Asset exceeds download byte limit")


class TransferBudget:
    def __init__(self, rate_limit_mib=4, max_bytes=MAX_TRANSFER_BYTES, timeout_seconds=3600):
        if not math.isfinite(rate_limit_mib) or not 0 < rate_limit_mib <= 128:
            raise PreparationError("Rate limit must be finite and in (0, 128] MiB/s")
        if not 0 < max_bytes <= MAX_TRANSFER_BYTES or not 0 < timeout_seconds <= 7200:
            raise PreparationError("Invalid transfer budget or deadline")
        self.rate = rate_limit_mib * MIB
        self.max_bytes = max_bytes
        self.timeout = timeout_seconds
        self.started = time.monotonic()
        self.received = 0
        self.next_progress = 128 * MIB

    def check(self):
        if time.monotonic() - self.started > self.timeout:
            raise PreparationError("Download deadline exceeded")

    def consume(self, count):
        self.check()
        self.received += count
        if self.received > self.max_bytes:
            raise PreparationError("Download byte budget exceeded")
        delay = self.started + self.received / self.rate - time.monotonic()
        if delay > 0:
            if time.monotonic() + delay - self.started > self.timeout:
                raise PreparationError("Rate limit would exceed download deadline")
            time.sleep(delay)
        if self.received >= self.next_progress:
            print(f"Transferred {self.received} bytes of <= {self.max_bytes}", flush=True)
            self.next_progress += 128 * MIB


def _check_response(response, spec):
    if response.status != 200:
        raise PreparationError("Expected a complete HTTP 200 response")
    if response.headers.get("Content-Encoding", "identity").lower() != "identity":
        raise PreparationError("Unexpected HTTP content encoding")
    if response.headers.get("Content-Length") != str(spec.size_bytes):
        raise PreparationError("Remote size differs from the pinned source")
    if response.headers.get("ETag", "").strip('"') != spec.etag:
        raise PreparationError("Remote ETag differs from the pinned source")
    revision = response.headers.get("X-Repo-Commit")
    if revision is not None and revision != spec.revision:
        raise PreparationError("Remote revision differs from the pinned source")


def pin_remote_asset(spec, *, opener=urlopen):
    spec.validate()
    request = Request(spec.url, headers={"Accept-Encoding": "identity"}, method="HEAD")
    with opener(request, timeout=60) as response:
        _check_response(response, spec)


class RemoteReader:
    def __init__(self, response, spec, budget):
        self.response = response
        self.spec = spec
        self.budget = budget
        self.received = 0
        self.digest = hashlib.sha256()

    def read(self, size=CHUNK_SIZE):
        self.budget.check()
        if size < 0:
            size = CHUNK_SIZE
        size = min(size, CHUNK_SIZE, self.spec.size_bytes - self.received)
        if size <= 0:
            return b""
        block = self.response.read(size)
        if not block:
            raise PreparationError("Truncated HTTP payload")
        if len(block) > size:
            raise PreparationError("HTTP read exceeds its byte budget")
        self.received += len(block)
        self.digest.update(block)
        self.budget.consume(len(block))
        return block

    def finish(self):
        if self.received != self.spec.size_bytes:
            raise PreparationError("Truncated HTTP payload")
        if self.response.read(1):
            raise PreparationError("HTTP payload exceeds pinned size")
        if self.digest.hexdigest() != self.spec.sha256:
            raise PreparationError("Downloaded SHA256 differs from pinned source")


def open_remote(spec, budget, *, opener=urlopen):
    pin_remote_asset(spec, opener=opener)
    budget.check()
    response = opener(Request(spec.url, headers={"Accept-Encoding": "identity", "If-Match": f'"{spec.etag}"'}), timeout=60)
    try:
        _check_response(response, spec)
    except BaseException:
        response.close()
        raise
    return response


def download_verified_asset(spec, destination, *, rate_limit_mib=4, budget=None, opener=urlopen):
    """Download opaque bytes, or reuse only an exact size+SHA256 match."""
    if not isinstance(spec, AssetSpec):
        spec = AssetSpec.from_dict(spec)
    spec.validate()
    record = verify_file(destination, size_bytes=spec.size_bytes, sha256=spec.sha256)
    if record is not None:
        return {**record, "bytes_transferred": 0, "reused": True}
    budget = budget or TransferBudget(rate_limit_mib)
    with open_remote(spec, budget, opener=opener) as response:
        reader = RemoteReader(response, spec, budget)
        with AtomicFile(destination) as temporary:
            while reader.received < spec.size_bytes:
                temporary.out.write(reader.read())
            reader.finish()
            temporary.commit()
    return {"size_bytes": spec.size_bytes, "sha256": spec.sha256, "bytes_transferred": reader.received, "reused": False}
