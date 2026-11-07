"""Read only three allowlisted JPEGs from a bounded gzip tar stream."""
from __future__ import annotations

import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
import warnings
import zlib

from .fetch import AssetSpec, CHUNK_SIZE, RemoteReader, open_remote
from .files import AtomicFile, PreparationError, destination, safe_relative, verify_file


MAX_MEMBER_BYTES = 16 * 1024 * 1024
MAX_TAR_BYTES = 8 * 1024 * 1024 * 1024
MAX_MEMBERS = 100000
MAX_JPEG_PIXELS = 16 * 1024 * 1024


def validate_jpeg(path):
    try:
        from PIL import Image
    except ImportError as exc:
        raise PreparationError("Pillow is required for real JPEG decoding; use --jpeg-validator-image if running on the host") from exc
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as image:
                if image.format != "JPEG" or image.mode != "RGB" or not 0 < image.width * image.height <= MAX_JPEG_PIXELS:
                    raise PreparationError("Expected a bounded RGB JPEG")
                size = list(image.size)
                image.verify()
            with Image.open(path) as image:
                image.load()
        return {"width": size[0], "height": size[1], "format": "JPEG", "mode": "RGB"}
    except PreparationError:
        raise
    except Exception as exc:
        raise PreparationError("JPEG integrity or decoding failed") from exc


def docker_jpeg_validator(image):
    # A pre-existing project runtime decodes data only, with no host mounts or network.
    if not image or image.startswith("-") or len(image) > 128:
        raise PreparationError("Invalid JPEG validator image")
    program = """import io,json,sys,warnings
from PIL import Image
warnings.simplefilter('error', Image.DecompressionBombWarning)
b=sys.stdin.buffer.read(16777217)
if not 0 < len(b) <= 16777216: raise ValueError('JPEG byte limit')
with Image.open(io.BytesIO(b)) as im:
 if im.format != 'JPEG' or im.mode != 'RGB' or not 0 < im.width*im.height <= 16777216: raise ValueError('Expected bounded RGB JPEG')
 width,height=im.size
 im.verify()
with Image.open(io.BytesIO(b)) as im: im.load()
print(json.dumps(dict(width=width,height=height,format='JPEG',mode='RGB')))
"""

    def validate(path):
        if Path(path).stat().st_size > MAX_MEMBER_BYTES:
            raise PreparationError("JPEG byte limit")
        try:
            result = subprocess.run(
                ["docker", "run", "--rm", "-i", "--network", "none", "--cpus", "1", "--memory", "256m",
                 "--read-only", "--cap-drop", "ALL", "--security-opt", "no-new-privileges", image, "python", "-c", program],
                input=Path(path).read_bytes(), capture_output=True, timeout=60, check=True,
            )
            record = json.loads(result.stdout)
            if set(record) != {"width", "height", "format", "mode"}:
                raise ValueError("Unexpected JPEG validation output")
            return record
        except (subprocess.SubprocessError, ValueError, OSError) as exc:
            raise PreparationError("Container JPEG integrity or decoding failed") from exc

    return validate


class BoundedTarReader:
    def __init__(self, stream):
        self.stream = stream
        self.received = 0

    def read_exact(self, size):
        if size < 0 or self.received + size > MAX_TAR_BYTES:
            raise PreparationError("Uncompressed tar byte budget exceeded")
        result = bytearray()
        while len(result) < size:
            block = self.stream.read(min(CHUNK_SIZE, size - len(result)))
            if not block:
                raise PreparationError("Truncated tar member or header")
            result.extend(block)
            self.received += len(block)
        return bytes(result)

    def skip(self, size):
        while size:
            block_size = min(size, CHUNK_SIZE)
            self.read_exact(block_size)
            size -= block_size


def _validate_entries(entries):
    if len(entries) != 3 or [entry["camera"] for entry in entries] != ["CAM_L0", "CAM_F0", "CAM_R0"]:
        raise PreparationError("Expected exactly CAM_L0, CAM_F0, CAM_R0 in model input order")
    names = set()
    paths = set()
    for entry in entries:
        safe_relative(entry["member"])
        safe_relative(entry["path"])
        if not entry["path"].endswith(".jpg") or entry["member"] in names or entry["path"] in paths:
            raise PreparationError("Duplicate or invalid camera allowlist")
        names.add(entry["member"])
        paths.add(entry["path"])


def reuse_cameras(root, source, previous, validator=validate_jpeg):
    entries = source["files"]
    _validate_entries(entries)
    if previous is not None:
        if previous.get("source") != source or not isinstance(previous.get("files"), list):
            raise PreparationError("Existing camera manifest does not match pinned source")
        records = previous["files"]
        if (len(records) != 3 or not all(isinstance(record, dict) for record in records)
                or [record.get("path") for record in records] != [entry["path"] for entry in entries]):
            raise PreparationError("Invalid existing camera manifest")
    else:
        records = [None] * len(entries)
    found = []
    for entry, record in zip(entries, records):
        path = destination(root, entry["path"])
        if path.exists() and record is None:
            raise PreparationError("Existing camera lacks a verified provenance manifest")
        if record is not None:
            if any(record.get(key) != value for key, value in entry.items()):
                raise PreparationError("Existing camera identity differs from allowlist")
            checked = verify_file(path, size_bytes=record.get("size_bytes"), sha256=record.get("sha256"))
            if checked is None:
                raise PreparationError("Manifest camera file is missing")
            decoded = validator(path)
            if any(record.get(key) != value for key, value in decoded.items()):
                raise PreparationError("Existing camera decoding differs from manifest")
            found.append(record)
    return found if len(found) == 3 else None


def stream_cameras(root, source, budget, *, opener=None, validator=validate_jpeg):
    entries = source["files"]
    _validate_entries(entries)
    for entry in entries:
        if destination(root, entry["path"]).exists():
            raise PreparationError("Refusing to overwrite an existing camera")
    spec = AssetSpec.from_dict(source)
    kwargs = {} if opener is None else {"opener": opener}
    staged = {}
    records = {}
    seen = set()
    try:
        with open_remote(spec, budget, **kwargs) as response:
            remote = RemoteReader(response, spec, budget)
            with gzip.GzipFile(fileobj=remote, mode="rb") as stream:
                tar = BoundedTarReader(stream)
                allowlist = {entry["member"]: entry for entry in entries}
                while len(records) < len(entries):
                    header = tar.read_exact(512)
                    if header == b"\0" * 512:
                        raise PreparationError("Camera archive ended before all allowlisted files were found")
                    member = tarfile.TarInfo.frombuf(header, "utf-8", "strict")
                    safe_relative(member.name)
                    if member.name in seen:
                        raise PreparationError("Duplicate tar member")
                    seen.add(member.name)
                    if len(seen) > MAX_MEMBERS:
                        raise PreparationError("Tar member count exceeds budget")
                    if not (member.isreg() or member.isdir()):
                        raise PreparationError("Tar links and extended/special entries are forbidden")
                    if member.size < 0 or member.size > MAX_MEMBER_BYTES or (member.isdir() and member.size):
                        raise PreparationError("Tar member exceeds its size limit")
                    entry = allowlist.get(member.name)
                    if entry is not None:
                        if not member.isreg() or member.size == 0:
                            raise PreparationError("Allowlisted camera is not a nonempty regular file")
                        temporary = AtomicFile(destination(root, entry["path"]))
                        staged[member.name] = temporary
                        digest = hashlib.sha256()
                        remaining = member.size
                        while remaining:
                            block = tar.read_exact(min(remaining, CHUNK_SIZE))
                            temporary.out.write(block)
                            digest.update(block)
                            remaining -= len(block)
                        temporary.out.flush()
                        decoded = validator(temporary.partial)
                        records[member.name] = {**entry, "size_bytes": member.size, "sha256": digest.hexdigest(), **decoded}
                        print(f"Decoded {entry['camera']}: {member.size} bytes, {decoded['width']}x{decoded['height']}", flush=True)
                    else:
                        tar.skip(member.size)
                    tar.skip((-member.size) % 512)
            # Intentionally stop at the last required member, without storing the archive.
            transferred = remote.received
            prefix_hash = remote.digest.hexdigest()
            scanned_bytes = tar.received
        for temporary in staged.values():
            temporary.commit()
        return {
            "source": source, "files": [records[entry["member"]] for entry in entries],
            "bytes_transferred": transferred, "compressed_prefix_sha256": prefix_hash,
            "uncompressed_bytes_scanned": scanned_bytes, "members_scanned": len(seen),
            "full_archive_sha256_verified": False, "stopped_after_allowlist": True,
        }
    except (tarfile.TarError, EOFError, gzip.BadGzipFile, zlib.error, UnicodeError, OSError) as exc:
        raise PreparationError("Invalid or truncated camera archive") from exc
    finally:
        for temporary in staged.values():
            temporary.close()
