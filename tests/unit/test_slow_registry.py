import copy
import hashlib
import io
import json
from pathlib import Path

import pytest

from hybrid_drive.assets import slow
from hybrid_drive.assets.files import PreparationError
from hybrid_drive.assets.slow_registry import resolve_model_key


SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
MIB = 1024**2


def sources_for(key):
    return json.loads((SCRIPTS / resolve_model_key(key).sources_filename).read_text())


@pytest.fixture
def qwen_assets(tmp_path):
    sources = sources_for("qwen3vl-2b")
    bodies = (b"GGUF Qwen text fixture", b"GGUF Qwen projector fixture")
    for spec, body in zip(sources["files"], bodies):
        spec["size_bytes"] = len(body)
        spec["sha256"] = hashlib.sha256(body).hexdigest()
    requests = []

    def opener(request, timeout):
        requests.append(request)
        index = next(index for index, spec in enumerate(sources["files"]) if spec["url"] == request.full_url)
        spec = sources["files"][index]
        response = io.BytesIO(b"" if request.get_method() == "HEAD" else bodies[index])
        response.status = 200
        response.headers = {"Content-Length": str(spec["size_bytes"]), "ETag": spec["etag"],
                            "X-Repo-Commit": spec["revision"]}
        return response

    return tmp_path, sources, bodies, requests, opener


def test_qwen_preparation_reuses_verified_pair_without_overwriting_existing_contents(qwen_assets):
    root, sources, bodies, requests, opener = qwen_assets
    unrelated = root / "keep.txt"
    unrelated.write_bytes(b"unrelated existing content")
    prepared = slow.prepare_slow(root, sources, rate_limit_mib=128, opener=opener)
    assert prepared["bytes_transferred_this_run"] == sum(map(len, bodies))
    assert [request.get_method() for request in requests] == ["HEAD", "GET", "HEAD", "GET"]
    profile = resolve_model_key("qwen3vl-2b")
    manifest_path = root / profile.manifest_filename
    before = {path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in root.iterdir()}
    requests.clear()
    reused = slow.prepare_slow(root, sources, opener=opener)
    assert reused["bytes_transferred_this_run"] == 0
    assert requests == []
    assert slow.verify_slow(root, sources) == json.loads(manifest_path.read_text())
    assert {path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in root.iterdir()} == before
    assert unrelated.read_bytes() == b"unrelated existing content"
    assert not (root / "smolvlm-manifest.json").exists()


def test_existing_corrupt_projector_fails_before_downloading_missing_text_model(qwen_assets):
    root, sources, _, requests, opener = qwen_assets
    projector = root / sources["files"][1]["path"]
    projector.write_bytes(b"existing corrupt content must not be replaced")
    with pytest.raises(PreparationError, match="corrupt"):
        slow.prepare_slow(root, sources, opener=opener)
    assert projector.read_bytes() == b"existing corrupt content must not be replaced"
    assert not (root / sources["files"][0]["path"]).exists()
    assert requests == []


def test_existing_manifest_records_are_not_repaired_or_overwritten(qwen_assets):
    root, sources, _, requests, opener = qwen_assets
    slow.prepare_slow(root, sources, rate_limit_mib=128, opener=opener)
    manifest_path = root / resolve_model_key("qwen3vl-2b").manifest_filename
    manifest = json.loads(manifest_path.read_text())
    manifest["files"][1]["sha256"] = "c" * 64
    original = json.dumps(manifest).encode()
    manifest_path.write_bytes(original)
    requests.clear()
    with pytest.raises(PreparationError, match="manifest differs"):
        slow.prepare_slow(root, sources, opener=opener)
    assert manifest_path.read_bytes() == original
    assert requests == []


@pytest.mark.parametrize("key,cap", [("256m", 320), ("500m", 600), ("qwen3vl-2b", 1536)])
def test_model_budget_is_checked_before_any_write_or_network(tmp_path, monkeypatch, key, cap):
    sources = sources_for(key)
    for spec in sources["files"]:
        spec["size_bytes"] = cap * MIB // 2
    assert slow._validate(sources) == cap * MIB
    sources["files"][1]["size_bytes"] += 1
    monkeypatch.setattr(slow, "download_verified_asset", lambda *args, **kwargs: pytest.fail("unexpected download"))
    target = tmp_path / key
    with pytest.raises(PreparationError, match=f"{cap} MiB"):
        slow.prepare_slow(target, sources)
    assert not target.exists()


@pytest.mark.parametrize("damage", ["smol-schema", "smol-projector", "extra-file", "reordered", "missing-pin"])
def test_qwen_rejects_substitution_or_unbounded_sources_before_filesystem(tmp_path, damage):
    sources = sources_for("qwen3vl-2b")
    if damage == "smol-schema":
        sources["schema"] = resolve_model_key("500m").schema
    elif damage == "smol-projector":
        sources["files"][1]["path"] = resolve_model_key("500m").projector_filename
    elif damage == "extra-file":
        sources["files"].append(copy.deepcopy(sources["files"][0]))
    elif damage == "reordered":
        sources["files"].reverse()
    else:
        sources["files"][0].pop("sha256")
    target = tmp_path / damage
    with pytest.raises(PreparationError):
        slow.prepare_slow(target, sources)
    assert not target.exists()


def test_unknown_model_key_is_not_silently_mapped_to_smol():
    with pytest.raises(ValueError, match="Unknown Slow model"):
        resolve_model_key("qwen")
