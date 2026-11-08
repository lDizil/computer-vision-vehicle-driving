import copy
import hashlib
import io
import json
from pathlib import Path

import pytest

from hybrid_drive.assets.files import PreparationError
from hybrid_drive.assets.slow import prepare_slow, verify_slow


REVISION = "a" * 40
NAMES = ("SmolVLM-256M-Instruct-Q8_0.gguf", "mmproj-SmolVLM-256M-Instruct-Q8_0.gguf")


class FakeResponse(io.BytesIO):
    def __init__(self, body, spec):
        super().__init__(body)
        self.status = 200
        self.headers = {"Content-Length": str(spec["size_bytes"]), "ETag": '"' + spec["etag"] + '"',
                        "X-Repo-Commit": spec["revision"]}


class FakeHTTP:
    def __init__(self, sources, bodies):
        self.specs = {spec["url"]: spec for spec in sources["files"]}
        self.bodies = dict(zip(self.specs, bodies))
        self.requests = []

    def __call__(self, request, timeout):
        self.requests.append(request)
        spec = self.specs[request.full_url]
        body = b"" if request.get_method() == "HEAD" else self.bodies[request.full_url]
        return FakeResponse(body, spec)


@pytest.fixture
def assets(tmp_path):
    bodies = (b"GGUF fixture text model", b"GGUF fixture vision projector")
    sources = {"schema": "hybrid-drive-smolvlm-q8-v1", "model_version": "fixture-q8-v1", "files": [
        {"path": name, "url": f"https://huggingface.co/example/fixture/resolve/{REVISION}/{name}",
         "revision": REVISION, "size_bytes": len(body), "sha256": hashlib.sha256(body).hexdigest(), "etag": "b" * 64}
        for name, body in zip(NAMES, bodies)
    ]}
    return tmp_path.resolve(), sources, bodies, FakeHTTP(sources, bodies)


def forbid(*args, **kwargs):
    raise AssertionError("Read-only verification must not download or write")


def forbid_repairs(monkeypatch):
    monkeypatch.setattr("hybrid_drive.assets.slow.download_verified_asset", forbid)
    monkeypatch.setattr("hybrid_drive.assets.slow.write_manifest", forbid)
    monkeypatch.setattr(Path, "mkdir", forbid)


def test_two_asset_manifest_reuses_verified_content_without_any_http(assets):
    root, sources, bodies, http = assets
    prepared = prepare_slow(root, sources, rate_limit_mib=128, opener=http)
    assert prepared["bytes_transferred_this_run"] == sum(map(len, bodies))
    assert [request.get_method() for request in http.requests] == ["HEAD", "GET", "HEAD", "GET"]
    expected = {"sources": sources, "files": [
        {key: spec[key] for key in ("path", "sha256", "size_bytes")} for spec in sources["files"]
    ]}
    assert json.loads((root / "smolvlm-manifest.json").read_text()) == expected
    for name, body in zip(NAMES, bodies):
        assert (root / name).read_bytes() == body
    http.requests.clear()
    reused = prepare_slow(root, sources, opener=http)
    assert reused == {**expected, "bytes_transferred_this_run": 0}
    assert http.requests == []
    assert verify_slow(root, sources) == expected


def test_inference_verification_is_readonly_even_for_nonwritable_asset_tree(assets, monkeypatch):
    root, sources, _, http = assets
    prepared = prepare_slow(root, sources, rate_limit_mib=128, opener=http)
    paths = [root / name for name in (*NAMES, "smolvlm-manifest.json")]
    before = {path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in paths}
    for path in paths:
        path.chmod(0o444)
    root.chmod(0o555)
    try:
        with monkeypatch.context() as guard:
            forbid_repairs(guard)
            verified = verify_slow(root, sources)
        assert verified == {key: prepared[key] for key in ("sources", "files")}
        assert {path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in paths} == before
    finally:
        root.chmod(0o755)
        for path in paths:
            path.chmod(0o644)


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("damage", ["missing", "size", "sha256"])
def test_inference_rejects_missing_or_corrupt_files_without_repairs(assets, monkeypatch, name, damage):
    root, sources, _, http = assets
    prepare_slow(root, sources, rate_limit_mib=128, opener=http)
    target = root / name
    if damage == "missing":
        target.unlink()
    elif damage == "size":
        target.write_bytes(b"truncated")
    else:
        target.write_bytes(b"x" * target.stat().st_size)
    before = {path.name: path.read_bytes() for path in root.iterdir()}
    forbid_repairs(monkeypatch)
    with pytest.raises(PreparationError):
        verify_slow(root, sources)
    assert {path.name: path.read_bytes() for path in root.iterdir()} == before


@pytest.mark.parametrize("damage", ["missing", "invalid-json", "sources", "digest", "extra-file"])
def test_manifest_is_a_required_exact_pin_not_a_hint_to_redownload(assets, monkeypatch, damage):
    root, sources, _, http = assets
    prepare_slow(root, sources, rate_limit_mib=128, opener=http)
    path = root / "smolvlm-manifest.json"
    manifest = json.loads(path.read_text())
    if damage == "missing":
        path.unlink()
    elif damage == "invalid-json":
        path.write_text("not JSON")
    else:
        if damage == "sources":
            manifest["sources"]["model_version"] = "untrusted-model"
        elif damage == "digest":
            manifest["files"][0]["sha256"] = "c" * 64
        else:
            manifest["files"].append(copy.deepcopy(manifest["files"][0]))
        path.write_text(json.dumps(manifest))
    forbid_repairs(monkeypatch)
    with pytest.raises(PreparationError):
        verify_slow(root, sources)


def test_prepare_will_not_replace_corrupt_existing_model_or_changed_manifest(assets):
    root, sources, _, http = assets
    prepare_slow(root, sources, rate_limit_mib=128, opener=http)
    target = root / NAMES[0]
    target.write_bytes(b"corrupt")
    http.requests.clear()
    with pytest.raises(PreparationError, match="corrupt"):
        prepare_slow(root, sources, opener=http)
    assert target.read_bytes() == b"corrupt"
    assert http.requests == []
    changed = copy.deepcopy(sources)
    changed["model_version"] = "another-version"
    with pytest.raises(PreparationError, match="differs"):
        prepare_slow(root, changed, opener=http)
    assert http.requests == []


def test_partial_preparation_never_publishes_manifest_and_can_reuse_first_file(assets):
    root, sources, bodies, http = assets
    second_url = sources["files"][1]["url"]
    http.bodies[second_url] = b"x" * len(bodies[1])
    with pytest.raises(PreparationError, match="SHA256"):
        prepare_slow(root, sources, rate_limit_mib=128, opener=http)
    assert (root / NAMES[0]).read_bytes() == bodies[0]
    assert not (root / NAMES[1]).exists()
    assert not (root / "smolvlm-manifest.json").exists()
    assert list(root.glob("*.partial")) == []
    http.requests.clear()
    http.bodies[second_url] = bodies[1]
    completed = prepare_slow(root, sources, rate_limit_mib=128, opener=http)
    assert completed["bytes_transferred_this_run"] == len(bodies[1])
    assert [request.full_url for request in http.requests] == [second_url, second_url]
    verify_slow(root, sources)


def test_aggregate_transfer_budget_is_enforced_before_filesystem_or_http(assets):
    root, sources, _, http = assets
    for spec in sources["files"]:
        spec["size_bytes"] = 200 * 1024**2
    target = root / "not-created"
    with pytest.raises(PreparationError, match="320 MiB"):
        prepare_slow(target, sources, opener=http)
    assert not target.exists()
    assert http.requests == []
