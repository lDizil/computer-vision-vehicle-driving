#!/usr/bin/env python3
"""Download one real NAVSIM log and the vector maps needed by nuPlan v1.2."""

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import sys
import tarfile
import time
from urllib.request import Request, urlopen
import zlib


CHUNK_SIZE = 64 * 1024
MIB = 1024 * 1024
MAX_DOWNLOAD_BYTES = 900 * MIB
MAX_OUTPUT_BYTES = 1300 * MIB
MAX_MAP_BYTES = 512 * MIB
MAX_LOG_BYTES = 16 * MIB
MAX_PREFIX_BYTES = 8 * MIB
MAX_TAR_BYTES = 64 * MIB
SOURCES_PATH = Path(__file__).with_name("navsim-sample-sources.json")


class PreparationError(ValueError):
    pass


def prepare(root, sources, rate_limit_mib=4, opener=urlopen):
    root = Path(root).resolve()
    validate_sources(root, sources)
    limiter = RateLimiter(rate_limit_mib)
    log = sources["log"]
    maps = sources["maps"]
    download_bytes = log["prefix_size_bytes"] + sum(f["compressed_size_bytes"] for f in maps["files"])
    print(f"Sample download ceiling: {download_bytes} bytes; limit: {rate_limit_mib:g} MiB/s", flush=True)
    root.mkdir(parents=True, exist_ok=True)
    records = []

    target = destination(root, log["path"])
    record = reuse_file(target, log)
    if record is None:
        print(f"Downloading {log['path']} from a bounded archive prefix", flush=True)
        prefix = b"".join(read_range(log, 0, log["prefix_size_bytes"], limiter, opener))
        if hashlib.sha256(prefix).hexdigest() != log["prefix_sha256"]:
            raise PreparationError("Log archive prefix SHA256 mismatch")
        record = write_verified(target, log, lambda out: extract_log(prefix, log, out))
    records.append({"path": log["path"], **record})

    for spec in maps["files"]:
        target = destination(root, spec["path"])
        record = reuse_file(target, spec)
        if record is None:
            print(f"Downloading {spec['path']} ({spec['compressed_size_bytes']} compressed bytes)", flush=True)
            chunks = read_range(maps, spec["offset"], spec["compressed_size_bytes"], limiter, opener)
            record = write_verified(target, spec, lambda out: inflate_member(chunks, spec["size_bytes"], out))
        records.append({"path": spec["path"], **record})

    result = {
        "sources": sources,
        "full_archive_hashes_verified": False,
        "pickle_deserialized": False,
        "files": records,
    }
    manifest = root / "sample-manifest.json"
    partial = manifest.with_name(manifest.name + ".partial")
    with partial.open("x", encoding="utf-8") as out:
        json.dump(result, out, indent=2)
        out.write("\n")
    os.replace(partial, manifest)
    print(f"Verified {len(records)} files; manifest: {manifest}", flush=True)
    return result


def validate_sources(root, sources):
    log = sources["log"]
    files = sources["maps"]["files"]
    if len(files) != 5 or sum(f["path"].endswith("/map.gpkg") for f in files) != 4:
        raise PreparationError("Expected four vector maps and one map metadata file")
    if sum(f["path"] == "maps/nuplan-maps-v1.0.json" for f in files) != 1:
        raise PreparationError("Unexpected map metadata path")
    if not 0 < log["size_bytes"] <= MAX_LOG_BYTES or not 0 < log["prefix_size_bytes"] <= MAX_PREFIX_BYTES:
        raise PreparationError("Log or compressed prefix exceeds its size limit")
    safe_relative(log["member"])
    paths = [log["path"]] + [f["path"] for f in files]
    if len(paths) != len(set(paths)):
        raise PreparationError("Duplicate destination path")
    for path in paths:
        destination(root, path)
    download_size = log["prefix_size_bytes"]
    output_size = log["size_bytes"]
    check_range(log, 0, log["prefix_size_bytes"])
    for spec in files:
        safe_relative(spec["member"])
        if not 0 < spec["size_bytes"] <= MAX_MAP_BYTES:
            raise PreparationError("Map member exceeds its uncompressed size limit")
        check_range(sources["maps"], spec["offset"], spec["compressed_size_bytes"])
        download_size += spec["compressed_size_bytes"]
        output_size += spec["size_bytes"]
    if download_size > MAX_DOWNLOAD_BYTES or output_size > MAX_OUTPUT_BYTES:
        raise PreparationError("Sample exceeds the total byte budget")


def safe_relative(name):
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts or "\\" in name or name == ".":
        raise PreparationError(f"Unsafe relative path: {name}")
    return path


def destination(root, name):
    path = root.joinpath(*safe_relative(name).parts)
    if path.is_symlink() or not path.resolve().is_relative_to(root):
        raise PreparationError(f"Destination escapes root or is a symlink: {name}")
    return path


def check_range(source, start, length):
    if start < 0 or length <= 0 or length > MAX_DOWNLOAD_BYTES or start + length > source["archive_size_bytes"]:
        raise PreparationError("HTTP range exceeds the archive or byte budget")


class RateLimiter:
    def __init__(self, mib_per_second):
        if not math.isfinite(mib_per_second) or mib_per_second <= 0:
            raise PreparationError("Rate limit must be a positive finite number")
        self.bytes_per_second = mib_per_second * MIB
        self.started = time.monotonic()
        self.received = 0

    def wait(self, byte_count):
        self.received += byte_count
        delay = self.started + self.received / self.bytes_per_second - time.monotonic()
        if delay > 0:
            time.sleep(delay)


def read_range(source, start, length, limiter, opener):
    check_range(source, start, length)
    end = start + length - 1
    headers = {"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"}
    if source.get("etag"):
        headers["If-Match"] = '"' + source["etag"] + '"'
    request = Request(source["url"], headers=headers)
    with opener(request, timeout=60) as response:
        expected_range = f"bytes {start}-{end}/{source['archive_size_bytes']}"
        if response.status != 206 or response.headers.get("Content-Range") != expected_range:
            raise PreparationError("Server did not honor the exact HTTP Range request")
        if response.headers.get("Content-Encoding", "identity") != "identity":
            raise PreparationError("Unexpected HTTP content encoding")
        if source.get("etag") and response.headers.get("ETag", "").strip('"') != source["etag"]:
            raise PreparationError("Archive ETag does not match the pinned source")
        content_length = response.headers.get("Content-Length")
        if content_length is not None and int(content_length) != length:
            raise PreparationError("Unexpected HTTP Content-Length")
        remaining = length
        while remaining:
            requested = min(CHUNK_SIZE, remaining)
            block = response.read(requested)
            if not block:
                raise PreparationError("Truncated HTTP range payload")
            if len(block) > requested:
                raise PreparationError("HTTP range read exceeds its byte budget")
            remaining -= len(block)
            limiter.wait(len(block))
            yield block
        if response.read(1):
            raise PreparationError("HTTP range payload exceeds its declared length")


def reuse_file(path, spec):
    if not path.exists():
        return None
    try:
        record = verify_file(path, spec)
    except PreparationError as exc:
        raise PreparationError(f"Existing file is corrupt; remove it explicitly before retrying: {path}: {exc}") from exc
    print(f"Reusing verified {path.name}", flush=True)
    return record


def verify_file(path, spec):
    if not path.is_file() or path.stat().st_size != spec["size_bytes"]:
        raise PreparationError("File size mismatch")
    digest = hashlib.sha256()
    crc = 0
    with path.open("rb") as source:
        while True:
            block = source.read(CHUNK_SIZE)
            if not block:
                break
            digest.update(block)
            crc = zlib.crc32(block, crc)
    sha256 = digest.hexdigest()
    crc32 = f"{crc & 0xffffffff:08x}"
    if spec.get("sha256") and sha256 != spec["sha256"]:
        raise PreparationError("File SHA256 mismatch")
    if spec.get("crc32") and crc32 != spec["crc32"]:
        raise PreparationError("ZIP member CRC32 mismatch")
    if not spec.get("sha256") and not spec.get("crc32"):
        raise PreparationError("File has no expected checksum")
    checks = []
    if spec.get("sha256"):
        checks.append("sha256")
    if spec.get("crc32"):
        checks.append("crc32")
    return {"size_bytes": spec["size_bytes"], "sha256": sha256, "crc32": crc32,
            "verified_against": "+".join(checks)}


def write_verified(path, spec, writer):
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    if partial.exists() or partial.is_symlink():
        raise PreparationError(f"Partial file already exists; remove it explicitly: {partial}")
    out = partial.open("xb")
    try:
        with out:
            writer(out)
        record = verify_file(partial, spec)
        os.replace(partial, path)
        return record
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def extract_log(prefix, spec, out):
    # The prefix deliberately ends before the gzip trailer; only the exact complete tar member is used.
    raw = zlib.decompressobj(16 + zlib.MAX_WBITS).decompress(prefix, MAX_TAR_BYTES + 1)
    if len(raw) > MAX_TAR_BYTES:
        raise PreparationError("Tar prefix exceeds its uncompressed size limit")
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as archive:
        for member in archive:
            safe_relative(member.name)
            if member.name != spec["member"]:
                continue
            if not member.isfile() or member.size != spec["size_bytes"]:
                raise PreparationError("Selected tar member is not a regular file of the expected size")
            if member.offset_data + member.size > len(raw):
                raise PreparationError("Selected tar member is truncated")
            with archive.extractfile(member) as source:
                while True:
                    block = source.read(CHUNK_SIZE)
                    if not block:
                        break
                    out.write(block)
            return
    raise PreparationError("Selected log was not found in the bounded tar prefix")


def inflate_member(chunks, expected_size, out):
    decoder = zlib.decompressobj(-15)
    written = 0
    for block in chunks:
        while block:
            data = decoder.decompress(block, min(CHUNK_SIZE, expected_size - written + 1))
            block = decoder.unconsumed_tail
            if decoder.unused_data:
                raise PreparationError("ZIP member contains trailing compressed data")
            if written + len(data) > expected_size:
                raise PreparationError("ZIP member exceeds its uncompressed size limit")
            out.write(data)
            written += len(data)
    if not decoder.eof or written != expected_size:
        raise PreparationError("Truncated ZIP deflate stream or uncompressed size mismatch")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--rate-limit-mib", type=float, default=4)
    args = parser.parse_args()
    try:
        with SOURCES_PATH.open(encoding="utf-8") as source:
            sources = json.load(source)
        prepare(args.root, sources, args.rate_limit_mib)
    except (OSError, ValueError, tarfile.TarError, zlib.error) as exc:
        print(f"Preparation failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
