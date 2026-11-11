from dataclasses import replace
import base64
import io
import json
import subprocess
from types import SimpleNamespace
from urllib.error import URLError

import numpy as np
from PIL import Image
import pytest

from hybrid_drive import Pose2D, Trajectory
from hybrid_drive.assets.files import PreparationError
from hybrid_drive.contracts.motion import EgoMotion
from hybrid_drive.contracts.planning import Candidate, CandidateSet
from hybrid_drive.contracts.slow import SlowRequest
from hybrid_drive.slow.context import CAMERA_NAMES, CameraFrame, SlowContext
from hybrid_drive.slow import llama
from hybrid_drive.slow.protocol import MAX_RESPONSE_BYTES, parse_response


class FakeResponse(io.BytesIO):
    def __init__(self, body):
        super().__init__(body)
        self.read_sizes = []

    def read(self, size=-1):
        assert size > 0, "Local HTTP must not perform an unbounded read"
        self.read_sizes.append(size)
        return super().read(size)


class FakeHTTP:
    def __init__(self):
        self.calls = []
        self.responses = []
        self.health = b'{"status":"ok"}'
        self.reply = {"choices": [{"finish_reason": "stop", "message": {"content": "{}"}}]}
        self.raw_reply = None
        self.health_error = None

    def open(self, request, timeout):
        self.calls.append((request, timeout))
        if request.full_url.endswith("/health"):
            if self.health_error is not None:
                raise self.health_error
            body = self.health
        else:
            body = self.raw_reply if self.raw_reply is not None else json.dumps(self.reply).encode("utf-8")
        response = FakeResponse(body)
        self.responses.append(response)
        return response


class FakeProcess:
    def __init__(self):
        self.returncode = None
        self.stubborn = False
        self.calls = []

    def poll(self):
        return self.returncode

    def terminate(self):
        self.calls.append("terminate")
        if not self.stubborn:
            self.returncode = 0

    def kill(self):
        self.calls.append("kill")
        self.returncode = -9

    def wait(self, timeout):
        self.calls.append(("wait", timeout))
        if self.returncode is None:
            raise subprocess.TimeoutExpired("fixture-server", timeout)
        return self.returncode


class FakeSocket:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def bind(self, address):
        assert address == ("127.0.0.1", 0)

    def getsockname(self):
        return "127.0.0.1", 12345


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    revision = "d" * 40
    marker = tmp_path / "SOURCE_REVISION"
    marker.write_text(revision + "\n")
    sources = {"model_version": "fixture-q8", "runtime_revision": revision, "files": [
        {"path": "SmolVLM-256M-Instruct-Q8_0.gguf"}, {"path": "mmproj-SmolVLM-256M-Instruct-Q8_0.gguf"},
    ]}
    config = llama.LlamaServerFactory(str(tmp_path), sources, server_path=str(tmp_path / "llama-server"),
                                      max_tokens=123, seed=77, startup_timeout_s=0.1, request_timeout_s=0.5)
    state = SimpleNamespace(config=config, marker=marker, process=FakeProcess(), http=FakeHTTP(),
                            launches=[], handlers=[])

    def popen(command, **kwargs):
        state.launches.append((command, kwargs))
        return state.process

    def opener(*handlers):
        state.handlers.extend(handlers)
        return state.http

    monkeypatch.setattr(llama, "verify_slow", lambda root, pins: None)
    monkeypatch.setattr(llama.subprocess, "Popen", popen)
    monkeypatch.setattr(llama.socket, "socket", FakeSocket)
    monkeypatch.setattr(llama, "build_opener", opener)
    return state


@pytest.fixture
def context():
    plan = CandidateSet("plan", "snapshot", 10, Pose2D(4, 5, 0.2), (
        Candidate("left", Trajectory([[1, 0.1, 0.1], [2, 0.2, 0.2]], 0.25)),
        Candidate("straight", Trajectory([[1, 0, 0], [2, 0, 0]], 0.25)),
    ))
    cameras = tuple(CameraFrame(name, (np.arange(3 * 5 * 3, dtype=np.uint8) + i * 50).reshape(3, 5, 3))
                    for i, name in enumerate(CAMERA_NAMES))
    return SlowContext(SlowRequest("request", plan, 100, 101, 10.5, "fixture-q8"), EgoMotion(4, 0, 0, 0), cameras)


def test_generation_sends_ordered_real_rgb_full_plan_and_strict_deterministic_schema(runtime, context, monkeypatch):
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.invalid:9999")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.invalid:9999")
    raw = json.dumps({"context_hash": context.content_hash, "status": "VETO",
                      "vetoed_candidate_ids": ["left"], "explanation": "Visible left obstruction."})
    runtime.http.reply["choices"][0]["message"]["content"] = raw
    backend = runtime.config()
    try:
        assert backend.generate(context) == raw
        parse_response(raw, context).validate_for(context.request)
        assert [handler.proxies for handler in runtime.handlers] == [{}]
        command, options = runtime.launches[0]
        assert command[command.index("--host") + 1] == "127.0.0.1"
        assert command[command.index("--gpu-layers") + 1] == "0"
        assert "--offline" in command and "--no-mmproj-offload" in command
        assert options == {"stdin": subprocess.DEVNULL, "stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
        request, timeout = runtime.http.calls[-1]
        assert request.full_url == "http://127.0.0.1:12345/v1/chat/completions"
        assert request.get_method() == "POST" and timeout == 0.5
        payload = json.loads(request.data)
        assert (payload["temperature"], payload["seed"], payload["max_tokens"], payload["stream"]) == (0, 77, 123, False)
        assert len(payload["messages"]) == 1 and payload["messages"][0]["role"] == "user"
        content = payload["messages"][0]["content"]
        assert len(content) == 7
        for i, camera in enumerate(context.cameras):
            assert content[i * 2] == {"type": "text", "text": camera.name}
            encoded = content[i * 2 + 1]["image_url"]["url"]
            assert encoded.startswith("data:image/png;base64,")
            with Image.open(io.BytesIO(base64.b64decode(encoded.split(",", 1)[1]))) as decoded:
                assert decoded.mode == "RGB"
                np.testing.assert_array_equal(np.asarray(decoded), camera.rgb)
        metadata = json.loads(content[-1]["text"].split("Context JSON:\n", 1)[1])
        assert metadata["context_hash"] == context.content_hash
        candidates = metadata["request"]["plan"]["candidates"]
        assert [(candidate["candidate_id"], candidate["times_s"], candidate["poses"]) for candidate in candidates] == [
            ("left", [0.25, 0.5], [[1.0, 0.1, 0.1], [2.0, 0.2, 0.2]]),
            ("straight", [0.25, 0.5], [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        ]
        response_format = payload["response_format"]
        assert response_format["type"] == "json_object"
        schema = response_format["schema"]
        assert set(schema) == {"oneOf"} and len(schema["oneOf"]) == 3
        for branch, status in zip(schema["oneOf"], ("UNKNOWN", "NO_VETO", "VETO")):
            assert branch["type"] == "object" and branch["additionalProperties"] is False
            assert set(branch["required"]) == {"context_hash", "status", "vetoed_candidate_ids", "explanation"}
            assert branch["properties"]["context_hash"]["const"] == context.content_hash
            assert branch["properties"]["status"] == {"type": "string", "const": status}
            assert branch["properties"]["vetoed_candidate_ids"] == {
                "type": "array", "items": {"type": "string", "enum": ["left", "straight"]},
                "minItems": 1 if status == "VETO" else 0, "maxItems": 2 if status == "VETO" else 0,
            }
            assert branch["properties"]["explanation"]["maxLength"] == 512
    finally:
        backend.close()


@pytest.mark.parametrize("finish_reason", ["length", "content_filter", None])
def test_truncated_or_incomplete_generation_is_not_returned_as_json(runtime, context, finish_reason):
    runtime.http.reply["choices"][0]["finish_reason"] = finish_reason
    backend = runtime.config()
    try:
        with pytest.raises(ValueError, match="truncated|completed"):
            backend.generate(context)
    finally:
        backend.close()


@pytest.mark.parametrize("raw,valid", [
    ("x" * MAX_RESPONSE_BYTES, True), ("x" * (MAX_RESPONSE_BYTES + 1), False),
    ("界" * (MAX_RESPONSE_BYTES // 3), True), ("界" * (MAX_RESPONSE_BYTES // 3 + 1), False),
])
def test_generated_content_limit_is_utf8_bytes_not_characters(runtime, context, raw, valid):
    runtime.http.reply["choices"][0]["message"]["content"] = raw
    backend = runtime.config()
    try:
        if valid:
            assert backend.generate(context) == raw
        else:
            with pytest.raises(ValueError, match="oversized"):
                backend.generate(context)
    finally:
        backend.close()


def test_http_envelope_reads_only_cap_plus_one_and_refuses_oversized_body(runtime, context):
    runtime.http.raw_reply = b" " * (llama.MAX_HTTP_BYTES + 100)
    backend = runtime.config()
    try:
        with pytest.raises(ValueError, match="response exceeds byte budget"):
            backend.generate(context)
        assert runtime.http.responses[-1].read_sizes == [llama.MAX_HTTP_BYTES + 1]
        assert runtime.http.responses[-1].closed
        runtime.http.health = b"{}" + b" " * (llama.MAX_HTTP_BYTES - 2)
        assert backend._json("/health", timeout=0.1) == {}
        assert runtime.http.responses[-1].read_sizes == [llama.MAX_HTTP_BYTES + 1]
    finally:
        backend.close()


def test_oversized_local_request_is_rejected_before_any_http(runtime):
    backend = runtime.config()
    runtime.http.calls.clear()
    try:
        with pytest.raises(ValueError, match="input exceeds IPC budget"):
            backend._json("/v1/chat/completions", {"text": "x" * (4 * 1024**2)}, timeout=0.1)
        assert runtime.http.calls == []
    finally:
        backend.close()


def test_wrong_model_context_is_rejected_before_inference_http(runtime, context):
    backend = runtime.config()
    runtime.http.calls.clear()
    other = replace(context, request=replace(context.request, model_version="another-model"))
    try:
        with pytest.raises(ValueError, match="another model"):
            backend.generate(other)
        assert runtime.http.calls == []
    finally:
        backend.close()


def test_startup_timeout_terminates_server_with_bounded_waits(runtime, monkeypatch):
    now = [0.0]
    monkeypatch.setattr(llama.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(llama.time, "sleep", lambda seconds: now.__setitem__(0, now[0] + seconds))
    runtime.http.health_error = URLError("fixture not ready")
    with pytest.raises(TimeoutError, match="startup deadline"):
        replace(runtime.config, startup_timeout_s=0.04)()
    assert len(runtime.http.calls) == 2
    assert all(0 < timeout <= 0.2 for _, timeout in runtime.http.calls)
    assert runtime.process.calls == ["terminate", ("wait", 0.15), ("wait", 0.15)]
    assert runtime.process.returncode is not None


def test_server_exit_during_startup_is_detected_without_health_retry(runtime):
    runtime.process.returncode = 7
    with pytest.raises(RuntimeError, match="exited during startup"):
        runtime.config()
    assert runtime.http.calls == []
    assert runtime.process.calls == [("wait", 0.15)]


def test_close_escalates_stubborn_process_to_kill_and_is_idempotent(runtime):
    backend = runtime.config()
    runtime.process.stubborn = True
    backend.close()
    assert runtime.process.calls == ["terminate", ("wait", 0.15), "kill", ("wait", 0.15)]
    assert runtime.process.returncode == -9
    backend.close()
    assert runtime.process.calls == ["terminate", ("wait", 0.15), "kill", ("wait", 0.15)]


@pytest.mark.parametrize("marker", [None, "unrelated-revision"])
def test_unpinned_runtime_marker_prevents_server_launch(runtime, marker):
    if marker is None:
        runtime.marker.unlink()
    else:
        runtime.marker.write_text(marker)
    with pytest.raises((ValueError, RuntimeError)):
        runtime.config()
    assert runtime.launches == []
    assert runtime.http.calls == []


def test_unverified_model_assets_prevent_even_local_server_launch(runtime, monkeypatch):
    def reject(*args):
        raise PreparationError("fixture corrupt model")

    monkeypatch.setattr(llama, "verify_slow", reject)
    with pytest.raises(PreparationError, match="corrupt model"):
        runtime.config()
    assert runtime.launches == []
    assert runtime.http.calls == []
