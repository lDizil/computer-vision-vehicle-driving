from dataclasses import dataclass, replace
import json

import pytest

from hybrid_drive.contracts.slow import RiskStatus, SlowResult
from hybrid_drive.qualification.cases import build_suite
from hybrid_drive.qualification.prompt import projected_prompt
from hybrid_drive.qualification.runner import run_qualification, wait_for_completion
from hybrid_drive.slow.worker import SlowCompletion
from hybrid_drive.slow.llama import LlamaServerFactory


@dataclass(frozen=True)
class Factory:
    model_version: str
    seed: int = 0
    sources: dict = None
    execution_profile: str = "cpu"
    threads: int = 1
    max_tokens: int = 512
    decision_first: bool = True
    image_min_tokens: int | None = None
    image_max_tokens: int = 64
    context_size: int = 4096
    cache_prompt: bool = True
    prompts_for = LlamaServerFactory.prompts_for
    parse_output = LlamaServerFactory.parse_output
    expected_output_fields = LlamaServerFactory.expected_output_fields
    protocol_name = LlamaServerFactory.protocol_name
    prompt_builder = staticmethod(projected_prompt)


class Worker:
    instances = []
    outcome = "completed"
    fail_close = False
    hash_first = False

    def __init__(self, factory, version, *, response_parser):
        self.context = None
        self.closed = False
        self.queue_depth = 0
        self.instances.append(self)

    def submit(self, context):
        self.context = context
        self.queue_depth = 1
        return "started"

    def poll(self):
        if not self.queue_depth:
            return ()
        self.queue_depth = 0
        request = self.context.request
        result = None
        if self.outcome == "completed":
            result = SlowResult(request.request_id, request.plan.snapshot_id, request.plan.plan_id,
                                request.plan.candidates_hash, request.model_version, RiskStatus.UNKNOWN,
                                explanation="Not enough visible evidence")
        payload = {"status": "UNKNOWN", "vetoed_candidate_ids": [],
                   "explanation": "Not enough visible evidence", "context_hash": self.context.content_hash}
        if self.hash_first:
            payload = {"context_hash": payload.pop("context_hash"), **payload}
        raw = json.dumps(payload)
        return (SlowCompletion(self.context, result, request.submitted_at_monotonic_s,
                               self.outcome, raw),)

    def close(self):
        self.closed = True
        if self.fail_close:
            raise RuntimeError("cannot confirm model termination")


@pytest.fixture
def environment(monkeypatch):
    from hybrid_drive.qualification import runner

    Worker.instances = []
    monkeypatch.setattr(Worker, "outcome", "completed")
    monkeypatch.setattr(Worker, "fail_close", False)
    monkeypatch.setattr(Worker, "hash_first", False)
    monkeypatch.setattr(runner, "AsyncSlowWorker", Worker)
    sources = {"runtime_revision": "a" * 40}
    return build_suite().select(("q00", "q08")), {"small": Factory("small-v1", sources=sources),
                                               "large": Factory("large-v1", sources=sources)}


def test_real_report_and_matched_matrix_preserve_negative_unknown_result(environment, tmp_path):
    suite, factories = environment
    directory, result = run_qualification(suite, factories, tmp_path, revision="a" * 40, seeds=(0, 1))
    rows = json.loads((directory / "records.json").read_text())
    assert len(rows) == 8
    assert result["status"] == "completed"
    assert result["summary"]["usage"] == "advisory_only"
    assert result["summary"]["models"]["small"]["rates"]["candidate_coverage"] == 0
    assert result["summary"]["models"]["small"]["diagnostic_gate"]["status"] == "fail"
    assert result["summary"]["paired_comparisons"][0]["matched_attempts"] == 4
    assert all(worker.closed for worker in Worker.instances)
    assert len(list((directory / "prompts").glob("*.txt"))) == 8
    html = (directory / "report.html").read_text()
    assert "Техническое выполнение: completed" in html
    provenance = json.loads((directory / "manifest.json").read_text())["provenance"]
    assert provenance["commit_reference_supplied"] is True
    assert "reproducible_commit" not in provenance
    for row in rows:
        assert list(json.loads(row["raw_response"])) == [
            "status", "vetoed_candidate_ids", "explanation", "context_hash"]
        assert row["input_hash"] == next(case.input_hash for case in suite.cases if case.case_id == row["case_id"])
        case = next(case for case in suite.cases if case.case_id == row["case_id"])
        restored = case.to_context(factories[row["model_id"]].model_version, request_id=row["request_id"],
                                   submitted_at_monotonic_s=row["submitted_at_monotonic_s"],
                                   deadline_monotonic_s=row["deadline_monotonic_s"])
        assert restored.content_hash == row["context_hash"] == json.loads(row["raw_response"])["context_hash"]


@pytest.mark.parametrize("outcome,status", [("invalid_output", "completed"), ("timeout", "failed"),
                                            ("backend_error", "failed"), ("process_exit", "failed")])
def test_diagnostic_format_failures_and_infrastructure_errors_stay_in_denominators(
        environment, tmp_path, monkeypatch, outcome, status):
    suite, factories = environment
    monkeypatch.setattr(Worker, "outcome", outcome)
    directory, result = run_qualification(suite, factories, tmp_path, revision="a" * 40)
    assert result["status"] == status
    assert result["summary"]["models"]["small"]["outcomes"][outcome] == 2
    counts = result["summary"]["models"]["small"]["confusion"]
    assert counts["failed_positive"] == counts["total_positive"] == 1
    assert counts["failed_negative"] == counts["total_negative"] == 1
    assert len(json.loads((directory / "records.json").read_text())) == 4


def test_unconfirmed_cleanup_stops_before_starting_another_model(environment, tmp_path, monkeypatch):
    suite, factories = environment
    monkeypatch.setattr(Worker, "fail_close", True)
    with pytest.raises(RuntimeError, match="termination"):
        run_qualification(suite, factories, tmp_path, revision="a" * 40)
    assert len(Worker.instances) == 1
    directory, = tmp_path.iterdir()
    assert len(json.loads((directory / "records.json").read_text())) == 2
    assert json.loads((directory / "failure.json").read_text())["matrix_complete"] is False
    assert not (directory / "result.json").exists()


def test_wait_ignores_foreign_completion_and_never_relabels_it_as_current(environment):
    suite, _ = environment
    context = suite.cases[0].to_context("v1", request_id="current", submitted_at_monotonic_s=0,
                                       deadline_monotonic_s=1)
    foreign = replace(context, request=replace(context.request, request_id="old"))

    class Deliveries:
        queue_depth = 1

        def poll(self):
            if self.queue_depth == 1:
                self.queue_depth = 2
                return (SlowCompletion(foreign, None, 0.1, "invalid_output"),)
            self.queue_depth = 0
            return (SlowCompletion(context, None, 0.2, "invalid_output"),)

    assert wait_for_completion(Deliveries(), context).context is context


def test_cleanup_failure_after_last_model_retains_complete_report_but_never_success(
        environment, tmp_path, monkeypatch):
    suite, factories = environment
    monkeypatch.setattr(Worker, "fail_close", True)
    directory, result = run_qualification(suite, {"small": factories["small"]}, tmp_path, revision="a" * 40)
    assert result["status"] == "failed"
    assert result["execution_error"] == "RuntimeError"
    assert json.loads((directory / "failure.json").read_text())["matrix_complete"] is True
    html = (directory / "report.html").read_text()
    assert "Техническое выполнение: failed" in html
    assert "RuntimeError" in html
    assert "Техническое выполнение: completed" not in html


def test_prompt_construction_error_is_explicit_incomplete_run_not_fabricated_model_record(
        environment, tmp_path):
    suite, factories = environment

    @dataclass(frozen=True)
    class BrokenPromptFactory(Factory):
        @staticmethod
        def prompt_builder(context):
            raise ValueError("Broken input construction")

    broken = BrokenPromptFactory("broken-v1", sources=factories["small"].sources)
    with pytest.raises(ValueError, match="input construction"):
        run_qualification(suite, {"broken": broken}, tmp_path, revision="a" * 40)
    directory, = tmp_path.iterdir()
    assert json.loads((directory / "records.json").read_text()) == []
    assert json.loads((directory / "failure.json").read_text())["matrix_complete"] is False
    assert Worker.instances[0].closed


@pytest.mark.parametrize("raw", ["x" * 8193, "я" * 8192, "\ud800"])
def test_rejected_provider_raw_is_bounded_even_before_ipc(raw):
    from hybrid_drive.slow.protocol import InvalidSlowOutput

    assert InvalidSlowOutput("truncated", raw).raw_response is None


@pytest.mark.parametrize("field,value", [("threads", 6), ("execution_profile", "metal"),
                                         ("max_tokens", 256), ("decision_first", False)])
def test_model_comparison_rejects_confounding_execution_profiles(environment, tmp_path, field, value):
    suite, factories = environment
    factories["large"] = replace(factories["large"], **{field: value})
    with pytest.raises(ValueError, match="same execution/generation profile"):
        run_qualification(suite, factories, tmp_path, revision="a" * 40)
    assert not list(tmp_path.iterdir())


def test_decision_first_checks_real_raw_order_and_marks_protocol_failure(environment, tmp_path, monkeypatch):
    suite, factories = environment
    monkeypatch.setattr(Worker, "hash_first", True)
    directory, result = run_qualification(suite, factories, tmp_path, revision="a" * 40)
    assert result["status"] == "failed"
    records = json.loads((directory / "records.json").read_text())
    for row in records:
        assert row["outcome"] == "backend_error"
        assert row["error_type"] == "DecisionOrderError"
        assert row["status"] is None and row["vetoed_candidate_ids"] == []
        assert next(iter(json.loads(row["raw_response"]))) == "context_hash"
    assert all(value["outcomes"]["backend_error"] == 2 for value in result["summary"]["models"].values())
    assert "Техническое выполнение: failed" in (directory / "report.html").read_text()


def test_default_hash_first_consumers_keep_their_original_order(environment, tmp_path, monkeypatch):
    suite, factories = environment
    factories = {model: replace(factory, decision_first=False) for model, factory in factories.items()}
    monkeypatch.setattr(Worker, "hash_first", True)
    directory, result = run_qualification(suite, factories, tmp_path, revision="working-tree")
    assert result["status"] == "completed"
    records = json.loads((directory / "records.json").read_text())
    assert all(row["outcome"] == "completed" and row["status"] == "UNKNOWN" for row in records)
    provenance = json.loads((directory / "manifest.json").read_text())["provenance"]
    assert provenance["commit_reference_supplied"] is False


def test_prompt_program_hash_includes_each_declared_conditional_stage(environment, tmp_path, monkeypatch):
    import hashlib

    suite, factories = environment
    monkeypatch.setattr(Factory, "prompts_for", lambda self, context: ("visibility question", projected_prompt(context)))
    directory, result = run_qualification(suite, factories, tmp_path, revision="a" * 40,
                                          prompt_version="two-stage-test")
    rows = json.loads((directory / "records.json").read_text())
    context = suite.cases[0].to_context("small-v1", request_id="irrelevant", submitted_at_monotonic_s=0,
                                       deadline_monotonic_s=1)
    program = json.dumps(("visibility question", projected_prompt(context)), ensure_ascii=False,
                         separators=(",", ":"))
    assert rows[0]["prompt_sha256"] == hashlib.sha256(program.encode()).hexdigest()
    assert all(row["declared_prompt_stages"] == 2 for row in rows)
    assert json.loads((directory / "manifest.json").read_text())["provenance"]["prompt_version"] == "two-stage-test"
    assert result["status"] == "completed"
