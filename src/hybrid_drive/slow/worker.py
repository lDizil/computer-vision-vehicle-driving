"""Bounded, single-owner subprocess inference; poll diagnostics are not a history."""

from collections import Counter, deque
from dataclasses import dataclass
import ctypes
import multiprocessing as mp
import os
import pickle
import signal
import struct
import sys
import time

from hybrid_drive.contracts._validation import finite_real, identifier
from hybrid_drive.contracts.slow import SlowResult

from .context import SlowContext
from .protocol import make_prompt, parse_response


_OUTPUT_LIMIT = 8192
_NOTICE = struct.Struct("!Q64scI")
_TASK = struct.Struct("!QI")
_INPUT_LIMIT = 4 * 1024 * 1024
_STOP_GRACE_S = 0.025
_CLOSE_BUDGET_S = 0.5


@dataclass(frozen=True)
class SlowCompletion:
    context: SlowContext
    result: SlowResult | None
    received_at_monotonic_s: float
    outcome: str
    raw_response: str | None = None


class AsyncSlowWorker:
    """One active request plus one latest-wins pending request.

    submit/poll never wait for inference or process termination. Call poll on
    fast ticks to enforce deadlines. At most two diagnostic completions are
    retained; dropped_completion_count exposes overwritten audit events. A
    consumer needing full history must poll between submissions. Single-thread
    ownership is required. close has a fixed wall-clock cleanup budget.
    """

    def __init__(self, factory, model_version: str, *, clock=time.monotonic):
        identifier(model_version, "model_version")
        if not callable(factory) or not callable(clock):
            raise ValueError("factory and clock must be callable")
        if getattr(factory, "model_version", model_version) != model_version:
            raise ValueError("Factory model_version does not match worker")
        self.model_version = model_version
        self._factory = factory
        self._clock = clock
        self._mp = mp.get_context("spawn")
        self._process = None
        self._input = self._task_reader = self._input_buffer = None
        self._reader = self._writer = self._buffer = None
        self._active = self._pending = None
        self._token = 0
        self._stopping = False
        self._stop_started = 0.0
        self._events = deque(maxlen=2)
        self.dropped_completion_count = 0
        self.outcome_counts = Counter()
        self.closed = False

    @property
    def queue_depth(self) -> int:
        return int(self._active is not None) + int(self._pending is not None)

    @property
    def active_request_id(self) -> str | None:
        return None if self._active is None else self._active.request.request_id

    @property
    def pending_request_id(self) -> str | None:
        return None if self._pending is None else self._pending.request.request_id

    def now_monotonic_s(self) -> float:
        now = finite_real(self._clock(), "clock")
        if now < 0:
            raise ValueError("clock must be nonnegative")
        return now

    def submit(self, context: SlowContext) -> str:
        if self.closed:
            raise RuntimeError("Worker is closed")
        if not isinstance(context, SlowContext):
            raise ValueError("context must be SlowContext")
        make_prompt(context)
        request = context.request
        if request.model_version != self.model_version:
            raise ValueError("Request model_version does not match worker")
        if request.request_id in (self.active_request_id, self.pending_request_id):
            raise ValueError("Duplicate active or pending request_id")
        now = self.now_monotonic_s()
        if now < request.submitted_at_monotonic_s:
            raise ValueError("Request submission time is in the future")
        self._service(now)
        if now > request.deadline_monotonic_s:
            self._emit(context, None, now, "timeout")
            return "expired"
        if self._active is None and self._pending is None and not self._stopping:
            self._dispatch(context, now)
            return "started"
        disposition = "queued"
        if self._pending is not None:
            self._emit(self._pending, None, now, "superseded")
            disposition = "replaced_pending"
        self._pending = context
        return disposition

    def poll(self) -> tuple[SlowCompletion, ...]:
        if not self.closed:
            self._service(self.now_monotonic_s())
        events = tuple(self._events)
        self._events.clear()
        return events

    def _service(self, now):
        if self._active is not None and now > self._active.request.deadline_monotonic_s:
            self._emit(self._active, None, now, "timeout")
            self._active = None
            self._begin_stop()
        if self._stopping:
            self._reap()
        elif self._process is not None:
            self._read_completion(now)
            if self._process is not None and not self._process.is_alive():
                if self._active is not None:
                    self._emit(self._active, None, now, "process_exit")
                    self._active = None
                self._begin_stop()
                self._reap()
        if self._pending is not None and now > self._pending.request.deadline_monotonic_s:
            self._emit(self._pending, None, now, "timeout")
            self._pending = None
        if self._active is None and self._pending is not None and not self._stopping:
            context, self._pending = self._pending, None
            self._dispatch(context, now)

    def _dispatch(self, context, now):
        self._active = context
        self._token += 1
        try:
            if self._process is None:
                self._task_reader, self._input = self._mp.Pipe(duplex=False)
                self._input_buffer = self._mp.RawArray(ctypes.c_ubyte, _INPUT_LIMIT)
                self._reader, self._writer = self._mp.Pipe(duplex=False)
                self._buffer = self._mp.RawArray(ctypes.c_ubyte, _OUTPUT_LIMIT)
                self._process = self._mp.Process(
                    target=_child_main,
                    args=(self._factory, self.model_version, self._task_reader,
                          self._input_buffer, self._writer, self._buffer),
                    daemon=True,
                )
                self._process.start()
                self._writer.close()
                self._task_reader.close()
            payload = pickle.dumps(context, protocol=pickle.HIGHEST_PROTOCOL)
            if len(payload) > _INPUT_LIMIT:
                raise ValueError("Serialized context exceeds input bound")
            memoryview(self._input_buffer).cast("B")[:len(payload)] = payload
            self._input.send_bytes(_TASK.pack(self._token, len(payload)))
        except Exception:
            self._emit(context, None, now, "backend_error")
            self._active = None
            self._begin_stop()

    def _read_completion(self, now):
        if self._active is None or not self._reader.poll(0):
            return
        raw = None
        try:
            notice = self._reader.recv_bytes(_NOTICE.size)
            token, digest, code, length = _NOTICE.unpack(notice)
            if token != self._token or digest.decode("ascii") != self._active.content_hash:
                raise ValueError("Mismatched worker envelope")
            outcome, result = "backend_error", None
            if code == b"C":
                if length > _OUTPUT_LIMIT:
                    raise ValueError("Oversized worker output")
                raw = bytes(memoryview(self._buffer).cast("B")[:length]).decode("utf-8")
                result = parse_response(raw, self._active)
                result.validate_for(self._active.request)
                outcome = "completed"
            elif code == b"I":
                outcome = "invalid_output"
            elif code != b"B":
                raise ValueError("Unknown worker outcome")
        except (EOFError, OSError):
            return
        except (ValueError, UnicodeError, struct.error):
            outcome, result = "invalid_output", None
        self._emit(self._active, result, now, outcome, raw)
        self._active = None
        if outcome == "backend_error":
            self._begin_stop()

    def _emit(self, context, result, now, outcome, raw=None):
        if len(self._events) == self._events.maxlen:
            self.dropped_completion_count += 1
        self._events.append(SlowCompletion(context, result, now, outcome, raw))
        self.outcome_counts[outcome] += 1

    def _begin_stop(self):
        if self._process is None:
            self._dispose()
            return
        if not self._stopping:
            self._stopping = True
            self._stop_started = time.monotonic()
            self._signal(signal.SIGTERM)

    def _signal(self, sig):
        process = self._process
        if process is None or process.pid is None:
            return
        if os.name == "posix":
            try:
                # A descendant can remain in the group after its leader exits.
                os.killpg(process.pid, sig)
                return
            except ProcessLookupError:
                pass
        if process.is_alive():
            process.kill() if sig == signal.SIGKILL else process.terminate()

    def _reap(self):
        process = self._process
        if process is None or process.pid is None:
            self._dispose()
            return
        if time.monotonic() - self._stop_started >= _STOP_GRACE_S:
            self._signal(signal.SIGKILL)
        if process.is_alive():
            return
        process.join(timeout=0)
        self._signal(signal.SIGKILL)
        if _group_has_live_members(process.pid):
            return
        self._dispose()

    def _dispose(self):
        for endpoint in (self._reader, self._writer, self._input, self._task_reader):
            if endpoint is not None:
                endpoint.close()
        if self._process is not None:
            if self._process.pid is not None and self._process.is_alive():
                return
            self._process.close()
        self._process = None
        self._input = self._task_reader = self._input_buffer = None
        self._reader = self._writer = self._buffer = None
        self._stopping = False

    def close(self) -> None:
        self.closed = True
        self._active = self._pending = None
        if self._process is None:
            self._dispose()
            return
        if not self._stopping:
            try:
                self._input.send_bytes(_TASK.pack(0, 0))
            except (ValueError, OSError):
                pass
            self._stopping = True
            self._stop_started = time.monotonic()
        until = time.monotonic() + _CLOSE_BUDGET_S
        while self._process is not None and time.monotonic() < until:
            self._reap()
            if self._process is not None:
                time.sleep(0.002)
        if self._process is not None:
            self._signal(signal.SIGKILL)
            raise RuntimeError("Worker termination not confirmed within cleanup budget")

    def __enter__(self):
        if self.closed:
            raise RuntimeError("Worker is closed")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def _group_has_live_members(pgid):
    if os.name != "posix":
        return False
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    if not sys.platform.startswith("linux"):
        # SIGKILL was sent to the whole group, and the leader is reaped.
        return False
    # Orphan zombies cannot run or fork; waiting for PID 1 to reap them would
    # prevent recovery in containers whose init does not promptly reap children.
    with os.scandir("/proc") as processes:
        for entry in processes:
            if not entry.name.isdigit():
                continue
            try:
                with open(entry.path + "/stat") as handle:
                    fields = handle.read(4096).rsplit(")", 1)[1].split()
                if int(fields[2]) == pgid and fields[0] != "Z":
                    return True
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                continue
    return False


def _child_main(factory, model_version, inputs, input_buffer, writer, buffer):
    if os.name == "posix":
        os.setsid()
    backend = None
    try:
        while True:
            token, length = _TASK.unpack(inputs.recv_bytes(_TASK.size))
            if token == 0:
                return
            if length > _INPUT_LIMIT:
                return
            context = pickle.loads(bytes(memoryview(input_buffer).cast("B")[:length]))
            code, output = b"B", b""
            try:
                if backend is None:
                    backend = factory()
                if getattr(backend, "model_version", model_version) != model_version:
                    raise ValueError("Backend model version mismatch")
                raw = backend.generate(context)
                if not isinstance(raw, str) or len(raw) > _OUTPUT_LIMIT:
                    code = b"I"
                else:
                    output = raw.encode("utf-8")
                    code = b"C" if len(output) <= _OUTPUT_LIMIT else b"I"
            except UnicodeError:
                code, output = b"I", b""
            except Exception:
                # Neither exception messages nor model traceback cross IPC.
                code, output = b"B", b""
            if code != b"C":
                output = b""
            memoryview(buffer).cast("B")[:len(output)] = output
            writer.send_bytes(_NOTICE.pack(token, context.content_hash.encode("ascii"), code, len(output)))
    finally:
        if backend is not None and callable(getattr(backend, "close", None)):
            try:
                backend.close()
            except Exception:
                pass
        inputs.close()
        writer.close()
