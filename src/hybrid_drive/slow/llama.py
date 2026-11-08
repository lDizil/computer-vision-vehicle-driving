"""Local-only llama.cpp backend, owned and terminated by the Slow worker group."""

from dataclasses import dataclass
import base64
import io
import json
import math
from pathlib import Path
import socket
import subprocess
import time
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

from hybrid_drive.assets.slow import verify_slow
from .protocol import MAX_RESPONSE_BYTES, make_prompt


MAX_HTTP_BYTES = 256 * 1024


@dataclass(frozen=True)
class LlamaServerFactory:
    asset_root: str
    sources: dict
    server_path: str = "/opt/llama/llama-server"
    max_tokens: int = 256
    seed: int = 0
    startup_timeout_s: float = 30
    request_timeout_s: float = 60

    @property
    def model_version(self):
        return self.sources["model_version"]

    def __call__(self):
        return LlamaVisionBackend(self)


class LlamaVisionBackend:
    def __init__(self, config: LlamaServerFactory):
        verify_slow(config.asset_root, config.sources)
        marker = Path(config.server_path).with_name("SOURCE_REVISION")
        try:
            runtime_revision = marker.read_text().strip()
        except OSError as exc:
            raise ValueError("Local model runtime provenance is missing") from exc
        if runtime_revision != config.sources["runtime_revision"]:
            raise ValueError("Local model runtime differs from the pinned revision")
        if isinstance(config.max_tokens, bool) or not isinstance(config.max_tokens, int) or not 1 <= config.max_tokens <= 512:
            raise ValueError("Generation must be bounded to 1..512 tokens")
        for value in (config.startup_timeout_s, config.request_timeout_s):
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or not 0 < value <= 120:
                raise ValueError("HTTP/startup timeouts must be in (0, 120] seconds")
        if isinstance(config.seed, bool) or not isinstance(config.seed, int) or not 0 <= config.seed < 2**32:
            raise ValueError("Seed must be an integer in [0, 2**32)")
        self.config = config
        self._process = None
        self._opener = build_opener(ProxyHandler({}))
        # Bind only loopback. The enclosing inference container has --network none.
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]
        self._url = f"http://127.0.0.1:{port}"
        root = Path(config.asset_root)
        command = [
            config.server_path, "-m", str(root / config.sources["files"][0]["path"]),
            "--mmproj", str(root / config.sources["files"][1]["path"]),
            "--host", "127.0.0.1", "--port", str(port), "--threads", "1", "--threads-batch", "1",
            "--gpu-layers", "0", "--no-mmproj-offload", "--parallel", "1", "--ctx-size", "4096",
            "--image-max-tokens", "64", "--offline", "--no-webui", "--log-disable",
        ]
        # Inherit the worker's group: killing a client alone would leave inference running.
        self._process = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            self._wait_ready()
        except BaseException:
            self.close()
            raise

    def _wait_ready(self):
        deadline = time.monotonic() + self.config.startup_timeout_s
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                raise RuntimeError("Local model server exited during startup")
            try:
                self._json("/health", timeout=min(0.2, max(0.01, deadline - time.monotonic())))
                return
            except (URLError, HTTPError, TimeoutError, OSError):
                time.sleep(0.02)
        raise TimeoutError("Local model server startup deadline exceeded")

    def _json(self, path, payload=None, *, timeout):
        data = None if payload is None else json.dumps(payload, allow_nan=False).encode()
        if data is not None and len(data) > 4 * 1024**2:
            raise ValueError("Local inference input exceeds IPC budget")
        request = Request(self._url + path, data=data, headers={"Content-Type": "application/json"})
        with self._opener.open(request, timeout=timeout) as response:
            raw = response.read(MAX_HTTP_BYTES + 1)
        if len(raw) > MAX_HTTP_BYTES:
            raise ValueError("Local server response exceeds byte budget")
        return json.loads(raw)

    def generate(self, context):
        from PIL import Image

        if context.request.model_version != self.config.model_version:
            raise ValueError("Slow request targets another model version")
        content = []
        for camera in context.cameras:
            content.append({"type": "text", "text": camera.name})
            image = io.BytesIO()
            Image.fromarray(camera.rgb, mode="RGB").save(image, format="PNG")
            url = "data:image/png;base64," + base64.b64encode(image.getvalue()).decode("ascii")
            content.append({"type": "image_url", "image_url": {"url": url}})
        content.append({"type": "text", "text": make_prompt(context)})
        payload = {
            "messages": [{"role": "user", "content": content}], "temperature": 0,
            "seed": self.config.seed, "max_tokens": self.config.max_tokens, "stream": False,
            "response_format": {"type": "json_object", "schema": response_schema(context)},
        }
        reply = self._json("/v1/chat/completions", payload, timeout=self.config.request_timeout_s)
        choices = reply.get("choices")
        if not isinstance(choices, list) or len(choices) != 1 or choices[0].get("finish_reason") != "stop":
            raise ValueError("Local model response was truncated or not a completed choice")
        raw = choices[0].get("message", {}).get("content")
        if not isinstance(raw, str) or len(raw.encode("utf-8")) > MAX_RESPONSE_BYTES:
            raise ValueError("Local model returned invalid or oversized content")
        return raw

    def close(self):
        process = self._process
        self._process = None
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=0.15)
            except subprocess.TimeoutExpired:
                process.kill()
        try:
            process.wait(timeout=0.15)
        except subprocess.TimeoutExpired:
            pass  # The enclosing worker still kills the entire group on hard timeout.


def response_schema(context):
    return {
        "type": "object", "properties": {
            "context_hash": {"type": "string", "const": context.content_hash},
            "status": {"type": "string", "enum": ["UNKNOWN", "NO_VETO", "VETO"]},
            "vetoed_candidate_ids": {"type": "array", "items": {
                "type": "string", "enum": [value.candidate_id for value in context.request.plan.candidates],
            }, "maxItems": len(context.request.plan.candidates)},
            "explanation": {"type": "string", "maxLength": 512},
        }, "required": ["context_hash", "status", "vetoed_candidate_ids", "explanation"],
        "additionalProperties": False,
    }
