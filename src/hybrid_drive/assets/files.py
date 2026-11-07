"""Filesystem guards and content verification; never deserialize model files."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import tempfile


class PreparationError(ValueError):
    pass


def safe_relative(name):
    if not isinstance(name, str) or not name or len(name) > 512 or "\\" in name or "\x00" in name:
        raise PreparationError("Invalid asset path")
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in ("", ".", "..") for part in name.split("/")):
        raise PreparationError(f"Unsafe asset path: {name}")
    return path


def checked_root(root):
    root = Path(os.path.abspath(root))
    check_components(root)
    root.mkdir(parents=True, exist_ok=True)
    check_components(root)
    return root


def check_components(path):
    for component in [path, *path.parents]:
        if component.is_symlink():
            raise PreparationError(f"Symlink in asset path: {component}")
    if path.exists() and not (path.is_dir() or path.is_file()):
        raise PreparationError(f"Non-regular asset path: {path}")


def destination(root, name):
    path = Path(root).joinpath(*safe_relative(name).parts)
    check_components(path)
    return path


def verify_file(path, *, size_bytes, sha256):
    check_components(path)
    if not path.exists():
        return None
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size != size_bytes:
            raise PreparationError(f"Existing file is corrupt or not regular: {path}")
        digest = hashlib.sha256()
        with os.fdopen(descriptor, "rb", closefd=False) as source:
            for block in iter(lambda: source.read(64 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != sha256:
            raise PreparationError(f"Existing file is corrupt: {path}")
        return {"size_bytes": info.st_size, "sha256": sha256}
    finally:
        os.close(descriptor)


class AtomicFile:
    def __init__(self, target):
        self.target = Path(target)
        check_components(self.target)
        self.target.parent.mkdir(parents=True, exist_ok=True)
        check_components(self.target)
        descriptor, name = tempfile.mkstemp(prefix=".ltf-", suffix=".partial", dir=self.target.parent)
        self.partial = Path(name)
        self.out = os.fdopen(descriptor, "wb")
        self.committed = False

    def commit(self, *, replace=False):
        self.out.flush()
        os.fsync(self.out.fileno())
        self.out.close()
        check_components(self.target)
        if replace:
            os.replace(self.partial, self.target)
        else:
            # Hard-link publication is atomic and cannot overwrite an existing file.
            os.link(self.partial, self.target, follow_symlinks=False)
            self.partial.unlink()
        self.committed = True

    def close(self):
        if not self.out.closed:
            self.out.close()
        if not self.committed:
            self.partial.unlink(missing_ok=True)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def read_manifest(path):
    check_components(path)
    if not path.exists():
        return None
    if not path.is_file() or path.stat().st_size > 128 * 1024:
        raise PreparationError("Invalid existing manifest")
    try:
        with path.open(encoding="utf-8") as source:
            result = json.load(source)
    except (ValueError, OSError) as exc:
        raise PreparationError("Invalid existing manifest") from exc
    if not isinstance(result, dict):
        raise PreparationError("Invalid existing manifest")
    return result


def write_manifest(path, payload):
    with AtomicFile(path) as temporary:
        temporary.out.write((json.dumps(payload, indent=2) + "\n").encode())
        temporary.commit(replace=True)
