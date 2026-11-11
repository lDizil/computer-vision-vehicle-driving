import importlib.util
from pathlib import Path

import pytest


SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


@pytest.fixture
def qualification_cli():
    spec = importlib.util.spec_from_file_location("qualification_cli", SCRIPTS / "run_slow_qualification.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    return script


@pytest.mark.parametrize("status,exit_code", [("completed", 0), ("failed", 1)])
def test_cli_exit_code_tracks_execution_not_semantic_quality(qualification_cli, status, exit_code, tmp_path, monkeypatch):
    script = qualification_cli
    monkeypatch.setattr(script, "run_qualification", lambda *args, **kwargs: (tmp_path, {"status": status}))
    monkeypatch.setattr("sys.argv", [str(script.__file__), "--assets-root", str(tmp_path),
                                   "--output-root", str(tmp_path), "--revision", "a" * 40])
    assert script.main() == exit_code
