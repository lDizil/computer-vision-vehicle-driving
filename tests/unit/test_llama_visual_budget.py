from contextlib import contextmanager
from dataclasses import replace
import io
import json
from types import SimpleNamespace
from urllib.error import HTTPError

import pytest

from hybrid_drive.qualification.cases import build_suite
from hybrid_drive.slow import llama


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    revision = "d" * 40
    (tmp_path / "SOURCE_REVISION").write_text(revision)
    sources = {"model_version": "fixture-q8", "runtime_revision": revision, "files": [
        {"path": "model.gguf"}, {"path": "projector.gguf"},
    ]}
    config = llama.LlamaServerFactory(str(tmp_path), sources, server_path=str(tmp_path / "llama-server"))
    state = SimpleNamespace(config=config, launches=[], probes=[], requests=[], http_error=None)

    @contextmanager
    def reservation():
        yield SimpleNamespace(bind=lambda address: None, getsockname=lambda: ("127.0.0.1", 12345))

    def launch(command, **kwargs):
        state.launches.append(command)
        return SimpleNamespace(poll=lambda: 0, wait=lambda timeout: 0)

    def devices(command, **kwargs):
        state.probes.append(command)
        return SimpleNamespace(stdout="  MTL0: Apple fixture GPU\n", stderr="")

    def open_http(request, *, timeout):
        state.requests.append(request)
        if state.http_error is not None:
            raise state.http_error
        return io.BytesIO(b'{"choices":[{"finish_reason":"stop","message":{"content":"{}"}}]}')

    monkeypatch.setattr(llama, "verify_slow", lambda *args: None)
    monkeypatch.setattr(llama.sys, "platform", "darwin")
    monkeypatch.setattr(llama.subprocess, "run", devices)
    monkeypatch.setattr(llama.subprocess, "Popen", launch)
    monkeypatch.setattr(llama.socket, "socket", reservation)
    monkeypatch.setattr(llama, "build_opener", lambda *args: SimpleNamespace(open=open_http))
    monkeypatch.setattr(llama.LlamaVisionBackend, "_wait_ready", lambda self: None)
    return state


@pytest.fixture
def context():
    return build_suite().cases[0].to_context("fixture-q8", request_id="budget-request",
                                           submitted_at_monotonic_s=100, deadline_monotonic_s=101)


@pytest.mark.parametrize("changes,message", [
    ({"image_min_tokens": 0}, "Image minimum"),
    ({"image_min_tokens": 1025}, "Image minimum"),
    ({"image_min_tokens": True}, "Image minimum"),
    ({"image_min_tokens": 1.5}, "Image minimum"),
    ({"image_max_tokens": 0}, "Image maximum"),
    ({"image_max_tokens": 1025}, "Image maximum"),
    ({"image_max_tokens": True}, "Image maximum"),
    ({"image_max_tokens": 1.5}, "Image maximum"),
    ({"image_max_tokens": None}, "Image maximum"),
    ({"image_min_tokens": 65}, "must not exceed"),
    ({"context_size": 1023}, "Context size"),
    ({"context_size": 16385}, "Context size"),
    ({"context_size": True}, "Context size"),
    ({"context_size": 4096.0}, "Context size"),
    ({"cache_prompt": 0}, "cache_prompt"),
    ({"cache_prompt": "false"}, "cache_prompt"),
])
def test_invalid_visual_budget_fails_before_device_probe_or_launch(runtime, changes, message):
    with pytest.raises(ValueError, match=message):
        replace(runtime.config, execution_profile="metal", **changes)()
    assert runtime.probes == [] and runtime.launches == []


def test_default_visual_budget_preserves_the_cpu_command_and_explicit_cache(runtime, context):
    backend = runtime.config()
    try:
        assert runtime.launches == [[
            runtime.config.server_path, "-m", runtime.config.asset_root + "/model.gguf",
            "--mmproj", runtime.config.asset_root + "/projector.gguf",
            "--host", "127.0.0.1", "--port", "12345",
            "--threads", "1", "--threads-batch", "1", "--parallel", "1", "--ctx-size", "4096",
            "--image-max-tokens", "64", "--offline", "--no-webui", "--log-disable",
            "--gpu-layers", "0", "--no-mmproj-offload",
        ]]
        assert runtime.probes == []
        assert backend.generate(context) == "{}"
        assert json.loads(runtime.requests[-1].data)["cache_prompt"] is True
    finally:
        backend.close()


@pytest.mark.parametrize("minimum,maximum,context_size", [(1, 1, 1024), (1024, 1024, 8192), (None, 1024, 16384)])
def test_visual_budget_bounds_reach_the_server_command(runtime, minimum, maximum, context_size):
    backend = replace(runtime.config, image_min_tokens=minimum, image_max_tokens=maximum,
                      context_size=context_size)()
    try:
        command = runtime.launches[0]
        assert command[command.index("--image-max-tokens") + 1] == str(maximum)
        assert command[command.index("--ctx-size") + 1] == str(context_size)
        if minimum is None:
            assert "--image-min-tokens" not in command
        else:
            assert command[command.index("--image-min-tokens") + 1] == str(minimum)
    finally:
        backend.close()


def test_cache_can_be_disabled_in_the_serialized_inference_request(runtime, context):
    backend = replace(runtime.config, cache_prompt=False)()
    try:
        assert backend.generate(context) == "{}"
        request = runtime.requests[-1]
        assert request.full_url.endswith("/v1/chat/completions")
        assert json.loads(request.data)["cache_prompt"] is False
    finally:
        backend.close()


def test_actual_context_overflow_remains_the_server_error(runtime, context):
    error = HTTPError("http://127.0.0.1:12345/v1/chat/completions", 400,
                      "request exceeds context size", {}, io.BytesIO(b"actual server token budget error"))
    runtime.http_error = error
    backend = replace(runtime.config, image_min_tokens=1024, image_max_tokens=1024, context_size=1024)()
    try:
        with pytest.raises(HTTPError) as caught:
            backend.generate(context)
        assert caught.value is error
        assert caught.value.read() == b"actual server token budget error"
    finally:
        backend.close()
