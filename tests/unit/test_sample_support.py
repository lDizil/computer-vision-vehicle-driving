import shutil

import pytest

from scripts.sample_support import writable_map_workspace


def test_map_locks_can_be_created_without_writing_verified_readonly_sources(tmp_path):
    root = tmp_path / "data"
    files = {"maps/city/version/map.gpkg": b"verified-vector-map", "maps/metadata.json": b"{}"}
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        path.chmod(0o444)
    maps = root / "maps"
    maps.chmod(0o555)
    workspace = None
    try:
        workspace = writable_map_workspace(root, [{"path": name} for name in files])
        (workspace / ".maplocks").mkdir()
        (workspace / ".maplocks" / "native.lock").touch()
        for name, content in files.items():
            link = workspace / name.removeprefix("maps/")
            assert link.is_symlink() and link.read_bytes() == content
            assert (root / name).read_bytes() == content
        assert not (maps / ".maplocks").exists()
    finally:
        maps.chmod(0o755)
        if workspace is not None:
            shutil.rmtree(workspace)


def test_map_workspace_rejects_a_path_escaping_the_prepared_map_root(tmp_path):
    with pytest.raises(ValueError):
        writable_map_workspace(tmp_path, [{"path": "maps/../../outside"}])
