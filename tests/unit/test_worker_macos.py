from dataclasses import dataclass
import errno
import json
import signal
import subprocess
import time
from types import SimpleNamespace

import pytest

from hybrid_drive.slow import worker as runtime


class FakeProcess:
    pid = 12345

    def __init__(self, *, alive=False):
        self.alive = alive
        self.signals = []
        self.closed = False

    def is_alive(self):
        return self.alive

    def kill(self):
        self.signals.append(signal.SIGKILL)

    def terminate(self):
        self.signals.append(signal.SIGTERM)

    def join(self, *, timeout):
        assert timeout == 0

    def close(self):
        self.closed = True


def denied_signal(pgid, sig):
    raise PermissionError(errno.EPERM, "Operation not permitted")


def worker_with_process(monkeypatch, *, platform="darwin", alive=False):
    monkeypatch.setattr(runtime.sys, "platform", platform)
    monkeypatch.setattr(runtime.os, "killpg", denied_signal)
    worker = runtime.AsyncSlowWorker(lambda: None, "test")
    worker._process = FakeProcess(alive=alive)
    worker._stopping = True
    worker._stop_started = time.monotonic()
    return worker


@pytest.mark.parametrize("output", ["42 Ss\n", "12345 Z\n42 R+\n"])
def test_darwin_eperm_only_confirms_cleanup_after_dead_leader_and_no_live_group(monkeypatch, output):
    worker = worker_with_process(monkeypatch)
    probes = []

    def snapshot(command, **kwargs):
        probes.append((command, kwargs))
        return SimpleNamespace(stdout=output)

    monkeypatch.setattr(runtime.subprocess, "run", snapshot)
    worker._signal(signal.SIGKILL)
    assert worker._process is not None
    assert worker._stopping
    assert probes == []
    worker._reap()
    assert worker._process is None
    assert worker._signal_error is None
    assert probes[0][0] == ["/bin/ps", "-axo", "pgid=,stat="]
    assert probes[0][1] == dict(stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                               text=True, check=True, timeout=0.15)


def test_darwin_eperm_with_live_group_cannot_confirm_cleanup_and_preserves_cause(monkeypatch):
    worker = worker_with_process(monkeypatch)
    process = worker._process
    monkeypatch.setattr(runtime.subprocess, "run", lambda *args, **kwargs:
                        SimpleNamespace(stdout="12345 Z\n12345 S+\n"))
    monkeypatch.setattr(runtime, "_CLOSE_BUDGET_S", 0.005)
    worker._signal(signal.SIGKILL)
    worker._reap()
    assert worker._process is process
    with pytest.raises(RuntimeError, match="not confirmed") as failure:
        worker.close()
    assert isinstance(failure.value.__cause__, PermissionError)
    assert failure.value.__cause__.errno == errno.EPERM
    assert not process.closed


def test_darwin_eperm_with_live_leader_tries_individual_signals_but_does_not_confirm_cleanup(monkeypatch):
    worker = worker_with_process(monkeypatch, alive=True)
    process = worker._process
    monkeypatch.setattr(runtime.subprocess, "run", lambda *args, **kwargs:
                        pytest.fail("A live leader cannot confirm cleanup through ps"))
    monkeypatch.setattr(runtime, "_CLOSE_BUDGET_S", 0.005)
    worker._signal(signal.SIGTERM)
    worker._signal(signal.SIGKILL)
    assert process.signals == [signal.SIGTERM, signal.SIGKILL]
    assert worker._stopping
    with pytest.raises(RuntimeError, match="not confirmed") as failure:
        worker.close()
    assert isinstance(failure.value.__cause__, PermissionError)
    assert worker._process is process
    assert not process.closed


def test_linux_eperm_is_unchanged_even_with_dead_leader(monkeypatch):
    worker = worker_with_process(monkeypatch, platform="linux")
    monkeypatch.setattr(runtime.subprocess, "run", lambda *args, **kwargs:
                        pytest.fail("Linux lifecycle must not invoke ps"))
    with pytest.raises(PermissionError):
        worker._signal(signal.SIGKILL)


@pytest.mark.parametrize("result", [
    subprocess.TimeoutExpired("ps", 0.15), subprocess.CalledProcessError(1, "ps"),
    PermissionError(errno.EPERM, "ps denied"), "", "\n", "bad row\n",
    "12345 unexpected-state\n", "12345 S extra-column\n",
])
def test_darwin_probe_failure_or_unparseable_output_fails_closed(monkeypatch, result):
    worker = worker_with_process(monkeypatch)

    def snapshot(*args, **kwargs):
        if isinstance(result, Exception):
            raise result
        return SimpleNamespace(stdout=result)

    monkeypatch.setattr(runtime.subprocess, "run", snapshot)
    assert runtime._group_has_live_members(FakeProcess.pid)
    monkeypatch.setattr(runtime, "_CLOSE_BUDGET_S", 0.005)
    with pytest.raises(RuntimeError, match="not confirmed") as failure:
        worker.close()
    assert isinstance(failure.value.__cause__, PermissionError)
    assert worker._process is not None


@pytest.mark.parametrize("output,live", [
    ("12345 S+\n42 R\n", True), ("42 Ss\n12345 Z+\n", False),
    ("42 Ss\n", False),
])
def test_darwin_membership_uses_actual_non_zombie_processes_without_killpg_probe(monkeypatch, output, live):
    monkeypatch.setattr(runtime.sys, "platform", "darwin")
    monkeypatch.setattr(runtime.os, "killpg", lambda *args:
                        pytest.fail("Darwin membership must be checked independently of killpg"))
    monkeypatch.setattr(runtime.subprocess, "run", lambda *args, **kwargs:
                        SimpleNamespace(stdout=output))
    assert runtime._group_has_live_members(FakeProcess.pid) is live


@dataclass
class TruncatedFactory:
    oversized: bool

    def __call__(self):
        return TruncatedBackend(self.oversized)


class TruncatedBackend:
    def __init__(self, oversized):
        self.oversized = oversized

    def generate(self, supplied):
        raw = json.dumps(dict(context_hash=supplied.content_hash, status="VETO",
                              vetoed_candidate_ids=["first"], explanation="partial generation"))
        if self.oversized:
            raw = "X" * 8193
        raise runtime.InvalidSlowOutput("provider truncated completion", raw_response=raw)


@pytest.mark.parametrize("oversized", [False, True])
def test_truncated_completion_is_not_applied_and_only_bounded_raw_crosses_spawn(oversized):
    from test_slow_worker import Clock, context, wait_for_completion

    supplied = context()
    with runtime.AsyncSlowWorker(TruncatedFactory(oversized), "v1", clock=Clock()) as worker:
        worker.submit(supplied)
        completion = wait_for_completion(worker)[0]
        assert completion.outcome == "invalid_output"
        assert completion.result is None
        if oversized:
            assert completion.raw_response is None
        else:
            assert json.loads(completion.raw_response) == dict(
                context_hash=supplied.content_hash, status="VETO",
                vetoed_candidate_ids=["first"], explanation="partial generation",
            )
            assert len(completion.raw_response.encode("utf-8")) <= 8192
        assert "provider truncated completion" not in repr(completion)


def test_darwin_leader_exit_during_individual_signal_still_requires_group_proof(monkeypatch):
    worker = worker_with_process(monkeypatch, alive=True)
    process = worker._process

    def exit_during_kill():
        process.alive = False
        raise ProcessLookupError(errno.ESRCH, "leader exited")

    monkeypatch.setattr(process, "kill", exit_during_kill)
    monkeypatch.setattr(runtime.subprocess, "run", lambda *args, **kwargs:
                        SimpleNamespace(stdout="42 Ss\n"))
    worker._signal(signal.SIGKILL)
    assert worker._process is process
    worker._reap()
    assert worker._process is None
    assert process.closed


def test_close_limits_darwin_ps_to_remaining_overall_budget(monkeypatch):
    worker = worker_with_process(monkeypatch)
    process = worker._process
    clock = SimpleNamespace(now=100.0)
    worker._stop_started = clock.now
    timeouts = []

    def join_near_deadline(*, timeout):
        assert timeout == 0
        clock.now = 100.49

    def expired_probe(*args, **kwargs):
        timeouts.append(kwargs["timeout"])
        clock.now += kwargs["timeout"]
        raise subprocess.TimeoutExpired("ps", kwargs["timeout"])

    monkeypatch.setattr(runtime.time, "monotonic", lambda: clock.now)
    monkeypatch.setattr(runtime.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(process, "join", join_near_deadline)
    monkeypatch.setattr(runtime.subprocess, "run", expired_probe)
    with pytest.raises(RuntimeError, match="not confirmed") as failure:
        worker.close()
    assert timeouts == [pytest.approx(0.01)]
    assert clock.now == pytest.approx(100.5)
    assert isinstance(failure.value.__cause__, PermissionError)
    assert worker._process is process
    assert worker._stopping
    assert not process.closed


@pytest.mark.parametrize("remaining", [0.0, -0.01])
def test_exhausted_darwin_cleanup_budget_cannot_probe_or_release_pending_work(monkeypatch, remaining):
    worker = worker_with_process(monkeypatch)
    process = worker._process
    pending = object()
    worker._pending = pending
    worker._stop_started = 99.0
    monkeypatch.setattr(runtime.time, "monotonic", lambda: 100.0)
    monkeypatch.setattr(runtime.subprocess, "run", lambda *args, **kwargs:
                        pytest.fail("An exhausted cleanup budget cannot start ps"))
    worker._reap(deadline_monotonic_s=100.0 + remaining)
    assert worker._process is process
    assert worker._pending is pending
    assert worker._stopping
    assert isinstance(worker._signal_error, PermissionError)
    assert not process.closed
