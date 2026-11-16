import copy
import hashlib
import io
import tarfile

from PIL import Image
import pytest

from hybrid_drive.assets import temporal


def test_selection_is_twenty_successive_full_windows_not_model_outputs():
    sources = temporal.load_temporal_sources()
    cases = sources["cases"]
    assert [case["source_list_index"] for case in cases] == list(range(27,47))
    assert cases[-1]["frame_timestamp_us"]-cases[0]["frame_timestamp_us"] == 9_500_066
    assert len({camera["member"] for case in cases for camera in case["cameras"]}) == 60
    assert [frame["source_list_index"] for frame in sources["selection"]["identity_window"]] == list(range(24,57))


@pytest.mark.parametrize("corruption", ["index", "timestamp", "chunk", "duplicate", "camera", "traversal", "history"])
def test_selection_corruption_rejected(corruption):
    sources = copy.deepcopy(temporal.load_temporal_sources())
    if corruption == "index": sources["cases"][0]["source_list_index"] = True
    elif corruption == "timestamp": sources["selection"]["identity_window"][5]["timestamp"] += 100_000
    elif corruption == "chunk": sources["selection"]["identity_window"][5]["scene_token"] = "0"*16
    elif corruption == "duplicate": sources["cases"].pop()
    elif corruption == "camera": sources["cases"][0]["cameras"].reverse()
    elif corruption == "traversal": sources["cases"][0]["cameras"][0]["path"] = "../escape.jpg"
    else: sources["selection"]["num_history_frames"] = 1
    with pytest.raises(ValueError): temporal.validate_temporal_sources(sources)


def archive_fixture(tmp_path,monkeypatch,*,missing=False,link=False):
    sources = copy.deepcopy(temporal.load_temporal_sources())
    buffer = io.BytesIO()
    Image.new("RGB",(8,6),(20,40,60)).save(buffer,format="JPEG")
    jpeg = buffer.getvalue()
    path = tmp_path/"fixture.tgz"
    entries = [camera for case in sources["cases"] for camera in case["cameras"]]
    with tarfile.open(path,"w:gz",format=tarfile.USTAR_FORMAT) as archive:
        for entry in entries[:-1] if missing else entries:
            member = tarfile.TarInfo(entry["member"])
            member.size = len(jpeg)
            archive.addfile(member,io.BytesIO(jpeg))
        if link:
            member = tarfile.TarInfo("unsafe-link")
            member.type = tarfile.SYMTYPE
            member.linkname = entries[0]["member"]
            archive.addfile(member)
    sources["camera_archive"] |= {"size_bytes":path.stat().st_size,"sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
    monkeypatch.setattr(temporal,"_PINNED_ARCHIVE",sources["camera_archive"])
    return sources,path


def test_cached_verified_extraction_retains_sixty_inputs_and_reuse_never_rescans(tmp_path,monkeypatch):
    sources,archive = archive_fixture(tmp_path,monkeypatch)
    root = tmp_path/"prepared"
    manifest = temporal.prepare_temporal_sample(root,sources,archive)
    assert len(manifest["files"]) == 60
    assert temporal.verify_temporal_sample(root,sources) == manifest
    monkeypatch.setattr(temporal,"_verified_archive",lambda *args:pytest.fail("Repeated archive scan"))
    assert temporal.prepare_temporal_sample(root,sources,archive) == manifest
    (root/manifest["files"][0]["path"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError): temporal.verify_temporal_sample(root,sources)


@pytest.mark.parametrize("failure", ["hash", "missing", "link"])
def test_preparation_never_publishes_partial_or_unverified_input(tmp_path,monkeypatch,failure):
    sources,archive = archive_fixture(tmp_path,monkeypatch,missing=failure=="missing",link=failure=="link")
    if failure == "hash":
        data = bytearray(archive.read_bytes());data[-1] ^= 1;archive.write_bytes(data)
    root = tmp_path/"prepared"
    with pytest.raises(ValueError): temporal.prepare_temporal_sample(root,sources,archive)
    assert not root.exists()
    assert not list(tmp_path.glob(".temporal-*"))
