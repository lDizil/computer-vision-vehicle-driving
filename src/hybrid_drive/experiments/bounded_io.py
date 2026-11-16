"""Bounded strict JSON and regular-file reads for portable experiment artifacts."""

import json
import os
import re
import stat

from hybrid_drive.assets.files import check_components


def read_bounded(path, maximum):
    check_components(path)
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as source:
        info = os.fstat(source.fileno())
        if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= maximum:
            raise ValueError("Artifact file exceeds byte budget or is not regular")
        data = source.read(maximum + 1)
    if not 0 < len(data) <= maximum:
        raise ValueError("Artifact file exceeds byte budget")
    return data


def decode_json(data):
    try:
        return json.loads(data, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    except (UnicodeError, RecursionError) as exc:
        raise ValueError("Invalid bounded JSON") from exc


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError("Nonfinite JSON")


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def require_keys(value, expected, name):
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(name + " has missing or unknown keys")


def require_hex(value, length, name):
    if not isinstance(value, str) or not re.fullmatch("[0-9a-f]{" + str(length) + "}", value):
        raise ValueError(name + " must be a lowercase hex digest")
