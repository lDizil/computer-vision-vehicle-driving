from dataclasses import dataclass
import json
import math
import multiprocessing as mp
import os
import signal
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pytest

from hybrid_drive.contracts.motion import EgoMotion
from hybrid_drive.contracts.planning import Candidate, CandidateSet
from hybrid_drive.contracts.slow import SlowRequest
from hybrid_drive.contracts.trajectory import Pose2D, Trajectory
from hybrid_drive.slow.context import CAMERA_NAMES, CameraFrame, SlowContext
from hybrid_drive.slow.worker import AsyncSlowWorker


class Clock:
    def __init__(self, value=10):
        self.value = value

    def __call__(self):
        return self.value


def context(request_id="request", *, submitted=10, deadline=20, model_version="v1"):
    plan = CandidateSet("plan-" + request_id, "snapshot", 100, Pose2D(0, 0, 0),
                        (Candidate("first", Trajectory([[x, 0, 0] for x in (1, 2, 3)], 0.5)),))
    request = SlowRequest(request_id, plan, submitted, deadline, 102, model_version)
    cameras = tuple(CameraFrame(name, np.zeros((2, 3, 3), dtype=np.uint8)) for name in CAMERA_NAMES)
    return SlowContext(request, EgoMotion(2, 0, 0, 0), cameras)


@dataclass
class FakeFactory:
    kind: str = "ok"
    marker: str | None = None
    model_version: str = "v1"

    def __call__(self):
        if self.kind == "factory_error":
            raise RuntimeError("private exception message")
        return FakeBackend(self.kind, self.marker)


@dataclass
class RecoveringFactory:
    marker: str
    failure: str
    model_version: str = "v1"

    def __call__(self):
        marker = Path(self.marker)
        calls = int(marker.read_text()) + 1 if marker.exists() else 1
        marker.write_text(str(calls))
        if calls == 1 and self.failure == "factory":
            raise RuntimeError("first initialization fails")
        return FakeBackend("error" if calls == 1 else "ok", None)


class FakeBackend:
    model_version = "v1"

    def __init__(self, kind, marker):
        self.kind = kind
        self.marker = marker
        self.calls = 0

    def generate(self, supplied):
        self.calls += 1
        kind = self.kind if supplied.request.request_id != "recover" else "ok"
        if kind == "ignore":
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
        if kind == "grandchild":
            child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
            Path(self.marker).write_text(str(child.pid))
        elif self.marker:
            Path(self.marker).write_text(str(os.getpid()))
        if kind in ("hang", "grandchild", "ignore"):
            while True:
                time.sleep(1)
        if kind == "exit":
            os._exit(17)
        if kind == "error":
            raise RuntimeError("private exception message")
        if kind == "oversized":
            return "X" * 8193
        if kind == "multibyte":
            return "é" * 8192
        if kind == "unicode":
            return "\ud800"
        if kind == "invalid":
            return "not JSON"
        return json.dumps(dict(context_hash=supplied.content_hash, status="NO_VETO",
                               vetoed_candidate_ids=[], explanation=str(self.calls)))

    def close(self):
        if self.marker and self.kind == "close_marker":
            Path(self.marker).write_text("closed")


def wait_for_completion(worker, *, timeout=8):
    until = time.monotonic() + timeout
    while time.monotonic() < until:
        events = worker.poll()
        if events:
            return events
        time.sleep(0.005)
    pytest.fail("No worker completion within test budget")


def wait_for_file(path, timeout=8):
    until = time.monotonic() + timeout
    while time.monotonic() < until:
        if path.exists():
            return path.read_text()
        time.sleep(0.005)
    pytest.fail("Backend did not start within test budget")


def wait_for_reap(worker):
    until = time.monotonic() + 3
    while worker._process is not None and time.monotonic() < until:
        worker.poll()
        time.sleep(0.005)
    assert worker._process is None


def test_real_spawn_persistent_model_and_trusted_envelope():
    clock = Clock()
    with AsyncSlowWorker(FakeFactory(), "v1", clock=clock) as worker:
        first = context()
        assert worker.submit(first) == "started"
        pid = worker._process.pid
        events = wait_for_completion(worker)
        assert len(events) == 1
        completion = events[0]
        assert completion.context is first
        assert completion.result.request_id == first.request.request_id
        assert completion.result.model_version == "v1"
        assert completion.result.explanation == "1"
        assert completion.received_at_monotonic_s == 10
        assert completion.outcome == "completed"
        assert completion.raw_response is not None
        assert worker.queue_depth == 0
        assert worker.submit(context("next")) == "started"
        assert wait_for_completion(worker)[0].result.explanation == "2"
        assert worker._process.pid == pid
    assert worker.closed
    assert worker._process is None


@pytest.mark.parametrize("kind,outcome", [
    ("invalid", "invalid_output"), ("oversized", "invalid_output"),
    ("multibyte", "invalid_output"), ("unicode", "invalid_output"),
    ("error", "backend_error"), ("factory_error", "backend_error"),
    ("exit", "process_exit"),
])
def test_child_failures_are_explicit_bounded_and_do_not_leak_exception_text(kind, outcome):
    with AsyncSlowWorker(FakeFactory(kind), "v1", clock=Clock()) as worker:
        worker.submit(context())
        completion = wait_for_completion(worker)[0]
        assert completion.outcome == outcome
        assert completion.result is None
        assert completion.raw_response == ("not JSON" if kind == "invalid" else None)
        assert "private exception message" not in repr(completion)


def test_latest_pending_wins_and_diagnostics_cannot_grow_without_polling(tmp_path):
    marker = tmp_path / "started"
    with AsyncSlowWorker(FakeFactory("hang", str(marker)), "v1", clock=Clock()) as worker:
        worker.submit(context("active"))
        wait_for_file(marker)
        assert worker.submit(context("pending-0")) == "queued"
        for i in range(1, 100):
            assert worker.submit(context(f"pending-{i}")) == "replaced_pending"
            assert worker.queue_depth == 2
        assert worker.active_request_id == "active"
        assert worker.pending_request_id == "pending-99"
        events = worker.poll()
        assert len(events) == 2
        assert [item.context.request.request_id for item in events] == ["pending-97", "pending-98"]
        assert all(item.outcome == "superseded" for item in events)
        assert worker.dropped_completion_count == 97
        assert worker.outcome_counts["superseded"] == 99


def test_timeout_is_inclusive_kills_real_process_and_can_restart(tmp_path):
    clock = Clock()
    marker = tmp_path / "started"
    with AsyncSlowWorker(FakeFactory("ignore", str(marker)), "v1", clock=clock) as worker:
        worker.submit(context(deadline=12))
        old_pid = int(wait_for_file(marker))
        worker.submit(context("recover", deadline=15))
        clock.value = 12
        assert worker.poll() == ()
        assert worker.active_request_id == "request"
        clock.value = math.nextafter(12, math.inf)
        before = time.monotonic()
        assert worker.poll()[0].outcome == "timeout"
        assert time.monotonic() - before < 0.25
        events = wait_for_completion(worker)
        assert events[0].context.request.request_id == "recover"
        assert events[0].outcome == "completed"
        assert worker._process.pid != old_pid
        assert not any(child.pid == old_pid for child in mp.active_children())


def test_pending_and_active_expiration_produce_at_most_two_events(tmp_path):
    clock = Clock()
    marker = tmp_path / "started"
    with AsyncSlowWorker(FakeFactory("hang", str(marker)), "v1", clock=clock) as worker:
        worker.submit(context(deadline=12))
        wait_for_file(marker)
        worker.submit(context("pending", deadline=11))
        clock.value = 13
        events = worker.poll()
        assert len(events) == 2
        assert [item.outcome for item in events] == ["timeout", "timeout"]
        assert worker.queue_depth == 0
        wait_for_reap(worker)


def test_deadline_includes_spawn_and_already_expired_does_not_start():
    clock = Clock(12)
    with AsyncSlowWorker(FakeFactory(), "v1", clock=clock) as worker:
        assert worker.submit(context(deadline=11)) == "expired"
        assert worker._process is None
        assert worker.poll()[0].outcome == "timeout"
        worker.submit(context(deadline=13))
        clock.value = 14
        assert worker.poll()[0].outcome == "timeout"
        wait_for_reap(worker)


@pytest.mark.parametrize("which", ["active", "pending"])
def test_duplicate_in_flight_ids_are_rejected(which, tmp_path):
    marker = tmp_path / "started"
    with AsyncSlowWorker(FakeFactory("hang", str(marker)), "v1", clock=Clock()) as worker:
        worker.submit(context("active"))
        wait_for_file(marker)
        worker.submit(context("pending"))
        with pytest.raises(ValueError, match="Duplicate"):
            worker.submit(context(which))


def test_model_identity_and_submission_time_are_checked_before_spawn():
    with pytest.raises(ValueError, match="Factory"):
        AsyncSlowWorker(FakeFactory(model_version="other"), "v1")
    with AsyncSlowWorker(FakeFactory(), "v1", clock=Clock()) as worker:
        with pytest.raises(ValueError, match="model_version"):
            worker.submit(context(model_version="other"))
        with pytest.raises(ValueError, match="future"):
            worker.submit(context(submitted=11))
        assert worker._process is None
    with pytest.raises(RuntimeError, match="closed"):
        worker.submit(context())
    worker.close()


def test_close_calls_backend_cleanup_and_is_bounded(tmp_path):
    marker = tmp_path / "closed"
    worker = AsyncSlowWorker(FakeFactory("close_marker", str(marker)), "v1", clock=Clock())
    worker.submit(context())
    wait_for_completion(worker)
    before = time.monotonic()
    worker.close()
    assert time.monotonic() - before < 1
    assert marker.read_text() == "closed"
    assert worker._process is None


def _pid_is_live(pid):
    if sys.platform.startswith("linux"):
        try:
            state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
            return state != "Z"
        except FileNotFoundError:
            return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


@pytest.mark.skipif(os.name != "posix", reason="Process-group lifecycle is POSIX-specific")
@pytest.mark.parametrize("shutdown", ["timeout", "close"])
def test_shutdown_kills_inherited_grandchild_process_group(shutdown, tmp_path):
    marker = tmp_path / "grandchild"
    clock = Clock()
    worker = AsyncSlowWorker(FakeFactory("grandchild", str(marker)), "v1", clock=clock)
    try:
        worker.submit(context(deadline=12))
        grandchild = int(wait_for_file(marker))
        assert _pid_is_live(grandchild)
        if shutdown == "timeout":
            clock.value = 13
            assert worker.poll()[0].outcome == "timeout"
            wait_for_reap(worker)
        else:
            worker.close()
        until = time.monotonic() + 2
        while _pid_is_live(grandchild) and time.monotonic() < until:
            time.sleep(0.005)
        assert not _pid_is_live(grandchild)
    finally:
        worker.close()


def test_unpicklable_factory_fails_explicitly_without_leaking_a_child():
    worker = AsyncSlowWorker(lambda: None, "v1", clock=Clock())
    try:
        worker.submit(context())
        assert worker.poll()[0].outcome == "backend_error"
        assert worker._process is None
    finally:
        worker.close()


def test_max_camera_payload_is_bounded_and_crosses_real_spawn():
    supplied = context()
    cameras = tuple(CameraFrame(name, np.zeros((480, 640, 3), dtype=np.uint8)) for name in CAMERA_NAMES)
    supplied = SlowContext(supplied.request, supplied.motion, cameras)
    with AsyncSlowWorker(FakeFactory(), "v1", clock=Clock()) as worker:
        worker.submit(supplied)
        reply = wait_for_completion(worker)[0]
        assert reply.outcome == "completed"
        assert reply.context is supplied


@pytest.mark.parametrize("failure", ["generate", "factory"])
def test_backend_error_restarts_process_and_factory_before_serving_next_request(failure, tmp_path):
    marker = tmp_path / "factory-calls"
    factory = RecoveringFactory(str(marker), failure)
    with AsyncSlowWorker(factory, "v1", clock=Clock()) as worker:
        worker.submit(context("first"))
        failed_pid = worker._process.pid
        failed = wait_for_completion(worker)[0]
        assert failed.outcome == "backend_error"
        assert failed.result is None
        assert marker.read_text() == "1"
        assert worker.queue_depth == 0
        # This ID must not use FakeBackend's special test-only recovery path.
        worker.submit(context("next"))
        recovered = wait_for_completion(worker)[0]
        assert recovered.context.request.request_id == "next"
        assert recovered.outcome == "completed"
        assert recovered.result.explanation == "1"
        assert marker.read_text() == "2"
        assert worker._process.pid != failed_pid
        assert not any(child.pid == failed_pid for child in mp.active_children())
