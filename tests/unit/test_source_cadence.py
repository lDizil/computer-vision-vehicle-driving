from dataclasses import dataclass
import json
from types import SimpleNamespace

import numpy as np
import pytest

from hybrid_drive.contracts.motion import EgoMotion
from hybrid_drive.contracts.planning import Candidate, CandidateSet
from hybrid_drive.contracts.slow import RiskStatus, SlowRequest, SlowResult
from hybrid_drive.contracts.trajectory import Pose2D, Trajectory
from hybrid_drive.experiments import source_cadence as replay
from hybrid_drive.experiments.protocol import content_hash
from hybrid_drive.assets.temporal import load_temporal_sources
from hybrid_drive.slow.context import CAMERA_NAMES, CameraFrame, SlowContext
from hybrid_drive.slow.worker import AsyncSlowWorker, SlowCompletion


class Clock:
    def __init__(self):self.value = 100.
    def __call__(self):return self.value
    def sleep(self,seconds):self.value += seconds


@dataclass
class Factory:
    duration: float = 12
    mismatch: bool = False
    veto: bool = False
    outcome: str = "completed"
    drop_count: int = 0
    omit_discard_event: bool = False
    submit_wall_s: float = 0
    cleanup_error: bool = False
    protocol_name: str = replay.PROTOCOL_NAME
    model_version: str = "fixture"
    sources: object = None
    def parse_output(self,*args):raise AssertionError("Fixture never invokes model/parser")
    def __call__(self):raise AssertionError("No real child/model in deterministic replay fixture")


class Worker:
    def __init__(self,factory,model_version,*,clock,response_parser):
        self.factory,self.clock = factory,clock
        self.active,self.pending,self.events = None,None,[]
        self.started = []
        self.dropped_completion_count = factory.drop_count
    @property
    def active_request_id(self):return None if self.active is None else self.active.request.request_id
    @property
    def pending_request_id(self):return None if self.pending is None else self.pending.request.request_id
    @property
    def queue_depth(self):return int(self.active is not None)+int(self.pending is not None)
    def dispatch(self,context):
        self.active,self.ready = context,self.clock()+self.factory.duration
        self.started.append(context.request.request_id)
    def service(self):
        if self.active is not None and self.clock() >= self.ready:
            context,self.active = self.active,None
            request = context.request
            result = SlowResult(request.request_id, "wrong" if self.factory.mismatch else request.plan.snapshot_id,
                                request.plan.plan_id,request.plan.candidates_hash,request.model_version,
                                RiskStatus.VETO if self.factory.veto else RiskStatus.NO_VETO,
                                (request.plan.candidates[0].candidate_id,) if self.factory.veto else (),"<script>untrusted</script>")
            self.events.append(SlowCompletion(context,result if self.factory.outcome == "completed" else None,
                                             self.clock(),self.factory.outcome,"<script>raw</script>"))
        if self.active is None and self.pending is not None:
            context,self.pending = self.pending,None
            self.dispatch(context)
    def submit(self,context):
        self.clock.sleep(self.factory.submit_wall_s)
        self.service()
        if self.active is None and self.pending is None:
            self.dispatch(context);return "started"
        outcome = "queued"
        if self.pending is not None:
            self.events.append(SlowCompletion(self.pending,None,self.clock(),"superseded"))
            outcome = "replaced_pending"
        self.pending = context
        return outcome
    def poll(self):
        self.service()
        events,self.events = tuple(self.events),[]
        return events
    def discard_pending(self,request_id):
        if self.pending_request_id != request_id:return False
        if not self.factory.omit_discard_event:self.events.append(SlowCompletion(self.pending,None,self.clock(),"discarded_pending"))
        self.pending = None
        return True
    def close(self):
        if self.factory.cleanup_error:raise RuntimeError("fixture cleanup error")


def fixture_inputs(monkeypatch):
    sources = load_temporal_sources()
    path = Trajectory([[1,0,0],[2,0,0],[3,0,0]],.5)
    images = tuple(CameraFrame(name,np.zeros((2,3,3),np.uint8)) for name in CAMERA_NAMES)
    bundles = tuple(SimpleNamespace(motion=EgoMotion(2,0,0,0),views=images,trajectory=path,index=i) for i in range(20))
    rows = [{k:case[k] for k in ("case_id","scene_token","frame_timestamp_us")} |
            {"bundle_hash":f"{i:064x}","fast_cpu_s":.123,"fast_wall_s":.456} for i,case in enumerate(sources["cases"])]
    manifest = {"cases":rows,"temporal_fast_hash":"a"*64,"source_revision":"b"*40,"sources_hash":"c"*64}
    monkeypatch.setattr(replay,"load_temporal_fast",lambda root:manifest)
    monkeypatch.setattr(replay,"load_temporal_bundles",lambda *a:bundles)
    monkeypatch.setattr(replay,"render_plan_overlay",lambda image,path:image)
    def plan(bundle):
        i = bundle.index
        return CandidateSet(f"p{i}",f"s{i}",rows[i]["frame_timestamp_us"]/1e6,Pose2D(0,0,0),(Candidate("fast",path),))
    monkeypatch.setattr(replay,"plan_for_bundle",plan)
    return manifest,bundles,plan,images


def run_fixture(tmp_path,monkeypatch,factory):
    inputs = fixture_inputs(monkeypatch)
    clock = Clock()
    workers = []
    def worker(*a,**kw):
        value = Worker(*a,**kw);workers.append(value);return value
    root,result = replay.run_source_cadence(tmp_path/"frozen",factory,tmp_path/"replay",revision="d"*40,
                                           clock=clock,sleep=clock.sleep,worker_factory=worker)
    return root,result,workers[0],inputs


def test_slow_finish_after_stream_retains_twenty_terminal_outcomes_and_original_clocks(tmp_path,monkeypatch):
    root,result,worker,_ = run_fixture(tmp_path,monkeypatch,Factory(veto=True))
    frames = json.loads((root/"frames.json").read_text())
    completions = json.loads((root/"completions.json").read_text())
    assert result["status"] == "completed"
    assert result["delivered_frames"] == 20 and result["maximum_queue_depth"] == 2
    assert result["source_span_s"] == 9.500066
    assert result["terminal_audit_complete"] and result["dropped_completion_count"] == 0
    assert result["frames_hash"] == content_hash(frames) and result["completions_hash"] == content_hash(completions)
    assert result["outcome_counts"] == {"superseded":18,"discarded_pending":1,"completed":1}
    assert result["native_completed_count"] == 1 and result["normal_queue_termination_count"] == 19
    assert result["slow_technical_failure_count"] == 0 and result["measurement_status"] == "completed"
    assert len(worker.started) == 1
    assert result["finished_at_monotonic_s"] > result["stream_end_monotonic_s"]
    completed = next(row for row in completions if row["outcome"] == "completed")
    assert completed["source_window_expired"] and not completed["matches_current_plan"]
    assert not completed["applicable_at_observation"] and not completed["used_for_fast_decision"]
    assert completed["arbitration_disposition"] == "plan_mismatch"
    assert all(row["fast_cpu_s"] == .123 and not row["slow_used"] for row in frames)
    assert all(row["delivery_to_decision_s"] >= row["fast_decision_wall_s"] for row in frames)
    assert all(row["decision_status"] == "candidate" for row in frames)
    assert completed["status"] == "VETO" and completed["vetoed_candidate_ids"] == ["fast"]
    assert frames[-1]["delivered_at_monotonic_s"]-frames[0]["delivered_at_monotonic_s"] == pytest.approx(9.500066)
    report = (root/"report.html").read_text()
    assert "&lt;script&gt;raw&lt;/script&gt;" in report and "<script>raw" not in report


@pytest.mark.parametrize("duration,expect_applicable,expect_used", [(0,True,True),(.2,True,False),(.6,False,False)])
def test_completion_admission_does_not_revise_already_returned_fast_decisions(tmp_path,monkeypatch,duration,expect_applicable,expect_used):
    root,result,_,_ = run_fixture(tmp_path,monkeypatch,Factory(duration=duration))
    completed = [row for row in json.loads((root/"completions.json").read_text()) if row["outcome"] == "completed"]
    assert result["status"] == "completed"
    assert any(row["applicable_at_observation"] for row in completed) is expect_applicable
    assert any(row["used_for_fast_decision"] for row in completed) is expect_used


def test_half_open_source_interval_and_result_mismatch_are_explicit(tmp_path,monkeypatch):
    sources = load_temporal_sources()
    first_interval = (sources["cases"][1]["frame_timestamp_us"]-sources["cases"][0]["frame_timestamp_us"])/1e6
    root,_,_,_ = run_fixture(tmp_path,monkeypatch,Factory(duration=first_interval))
    completed = [row for row in json.loads((root/"completions.json").read_text()) if row["outcome"] == "completed"]
    assert completed[0]["source_window_expired"] and not completed[0]["applicable_at_observation"]
    root,result,_,_ = run_fixture(tmp_path,monkeypatch,Factory(duration=0,mismatch=True))
    completed = json.loads((root/"completions.json").read_text())
    assert result["fast_decisions_using_slow"] == 0
    assert all(row["arbitration_disposition"] == "response_mismatch" for row in completed)


def test_cleanup_failure_and_output_into_frozen_input_are_not_success(tmp_path,monkeypatch):
    _,result,_,_ = run_fixture(tmp_path,monkeypatch,Factory(cleanup_error=True))
    assert result["status"] == "failed" and not result["cleanup_confirmed"]
    with pytest.raises(ValueError,match="outside"):
        replay.run_source_cadence(tmp_path/"frozen",Factory(),tmp_path/"frozen"/"new",revision="d"*40)


def test_delivery_to_decision_includes_submission_overhead_not_frozen_fast_inference(tmp_path,monkeypatch):
    root,_,_,_ = run_fixture(tmp_path,monkeypatch,Factory(submit_wall_s=.02))
    frames = json.loads((root/"frames.json").read_text())
    assert all(row["delivery_to_decision_s"] >= .019 for row in frames)
    assert all(row["fast_decision_wall_s"] == 0 for row in frames)
    assert all(row["fast_cpu_s"] == .123 and row["fast_wall_s"] == .456 for row in frames)


@pytest.mark.parametrize("factory", [Factory(drop_count=1),Factory(omit_discard_event=True)])
def test_missing_or_overflowed_terminal_audit_fails_pipeline(tmp_path,monkeypatch,factory):
    _,result,_,_ = run_fixture(tmp_path,monkeypatch,factory)
    assert result["status"] == result["measurement_status"] == "failed"
    assert "audit" in result["error"]


@pytest.mark.parametrize("outcome", ["backend_error","timeout"])
def test_technical_slow_failure_is_not_confused_with_normal_queue_policy(tmp_path,monkeypatch,outcome):
    _,result,_,_ = run_fixture(tmp_path,monkeypatch,Factory(outcome=outcome))
    assert result["measurement_status"] == "completed" and result["status"] == "completed_with_slow_failures"
    assert result["slow_status"] == "technical_failures_observed"
    assert result["slow_technical_failure_count"] == 1 and result["slow_technical_outcomes"] == {outcome:1}
    assert result["native_completed_count"] == 0 and result["normal_queue_termination_count"] == 19


def test_discard_pending_never_services_dispatches_or_cancels_active(monkeypatch):
    clock = Clock()
    worker = AsyncSlowWorker(Factory(),"fixture",clock=clock)
    path = Trajectory([[1,0,0],[2,0,0],[3,0,0]],.5)
    plan = CandidateSet("plan","snapshot",0,Pose2D(0,0,0),(Candidate("fast",path),))
    images = tuple(CameraFrame(name,np.zeros((2,3,3),np.uint8)) for name in CAMERA_NAMES)
    def context(name):return SlowContext(SlowRequest(name,plan,100,160,.5,"fixture"),EgoMotion(2,0,0,0),images)
    active,pending = context("active"),context("pending")
    worker._active,worker._pending = active,pending
    monkeypatch.setattr(worker,"_service",lambda *a:pytest.fail("discard serviced work"))
    monkeypatch.setattr(worker,"_dispatch",lambda *a:pytest.fail("discard dispatched work"))
    assert not worker.discard_pending("active") and worker.pending_request_id == "pending"
    assert worker.discard_pending("pending")
    assert worker.active_request_id == "active" and worker.pending_request_id is None
    event = worker._events.pop()
    assert event.context is pending and event.outcome == "discarded_pending"
    worker.close()
