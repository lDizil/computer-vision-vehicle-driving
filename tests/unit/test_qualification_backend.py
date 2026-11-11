from dataclasses import replace
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hybrid_drive import Pose2D, Trajectory
from hybrid_drive.assets import slow as slow_assets
from hybrid_drive.assets.files import PreparationError
from hybrid_drive.contracts.motion import EgoMotion
from hybrid_drive.contracts.planning import Candidate, CandidateSet
from hybrid_drive.contracts.slow import RiskStatus, SlowRequest
from hybrid_drive.qualification.cases import build_suite
from hybrid_drive.qualification.prompt import projected_prompt
from hybrid_drive.slow import llama
from hybrid_drive.slow.context import CAMERA_NAMES, CameraFrame, SlowContext
from hybrid_drive.slow.protocol import MAX_PROMPT_BYTES, make_prompt, parse_response


MIB = 1024**2
SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def pinned_sources(variant, sizes):
    revision = "a" * 40
    names = (f"SmolVLM-{variant}-Instruct-Q8_0.gguf", f"mmproj-SmolVLM-{variant}-Instruct-Q8_0.gguf")
    schema = "hybrid-drive-smolvlm-q8-v1" if variant == "256M" else "hybrid-drive-smolvlm-500m-q8-v1"
    return {"schema": schema, "files": [
        {"path": name, "url": f"https://huggingface.co/example/fixture/resolve/{revision}/{name}",
         "revision": revision, "size_bytes": size, "sha256": "b" * 64, "etag": "c" * 64}
        for name, size in zip(names, sizes)
    ]}


@pytest.mark.parametrize("variant,cap", [("256M", 320), ("500M", 600)])
def test_each_model_variant_has_its_own_aggregate_transfer_cap(tmp_path, monkeypatch, variant, cap):
    sources = pinned_sources(variant, (cap * MIB // 2, cap * MIB // 2))
    budgets = []

    def download(spec, target, *, budget, **kwargs):
        budgets.append(budget)
        return {"sha256": spec.sha256, "size_bytes": spec.size_bytes}

    monkeypatch.setattr(slow_assets, "download_verified_asset", download)
    result = slow_assets.prepare_slow(tmp_path / "accepted", sources)
    assert len(budgets) == 2 and budgets[0] is budgets[1]
    assert budgets[0].max_bytes == cap * MIB
    assert sum(record["size_bytes"] for record in result["files"]) == cap * MIB
    sources["files"][1]["size_bytes"] += 1
    with pytest.raises(PreparationError, match=f"{cap} MiB"):
        slow_assets.prepare_slow(tmp_path / "rejected", sources)
    assert len(budgets) == 2
    assert not (tmp_path / "rejected").exists()


def test_500m_cannot_reuse_the_256m_projector_or_schema_before_any_download(tmp_path, monkeypatch):
    downloads = []
    monkeypatch.setattr(slow_assets, "download_verified_asset", lambda *args, **kwargs: downloads.append(args))
    for damage in ("projector", "schema"):
        sources = pinned_sources("500M", (400 * MIB, 110 * MIB))
        if damage == "projector":
            sources["files"][1]["path"] = "mmproj-SmolVLM-256M-Instruct-Q8_0.gguf"
        else:
            sources["schema"] = "hybrid-drive-smolvlm-q8-v1"
        with pytest.raises(PreparationError, match="Unexpected Slow model files"):
            slow_assets.prepare_slow(tmp_path / damage, sources)
        assert not (tmp_path / damage).exists()
    assert downloads == []


@pytest.mark.parametrize("arguments,variant", [([], "256M"), (["--variant", "500m"], "500M")])
def test_preparation_variant_flag_selects_separate_pinned_model_and_projector(tmp_path, monkeypatch, capsys,
                                                                          arguments, variant):
    spec = importlib.util.spec_from_file_location("qualification_prepare_slow", SCRIPTS / "prepare_slow_sample.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    calls = []

    def prepare(root, sources, **kwargs):
        calls.append((root, sources))
        return {"prepared": variant}

    monkeypatch.setattr(script, "prepare_slow", prepare)
    monkeypatch.setattr("sys.argv", [str(script.__file__), "--root", str(tmp_path), *arguments])
    script.main()
    assert json.loads(capsys.readouterr().out) == {"prepared": variant}
    root, sources = calls[0]
    assert root == tmp_path and len(calls) == 1
    assert [entry["path"] for entry in sources["files"]] == [
        f"SmolVLM-{variant}-Instruct-Q8_0.gguf", f"mmproj-SmolVLM-{variant}-Instruct-Q8_0.gguf",
    ]
    assert all(f"SmolVLM-{variant}-Instruct-GGUF/resolve/" in entry["url"] for entry in sources["files"])
    assert slow_assets._validate(sources) == (320 if variant == "256M" else 600) * MIB


@pytest.fixture
def context():
    plan = CandidateSet("opaque-plan", "opaque-snapshot", 10, Pose2D(4, 5, 0.2), (
        Candidate("c-a9", Trajectory([[1, .4, .2], [3, 1.2, .5]], .25)),
        Candidate("c-b4", Trajectory([[1, -.4, -.2], [3, -1.2, -.5]], .25)),
    ))
    cameras = tuple(CameraFrame(name, np.full((2, 3, 3), index, dtype=np.uint8))
                    for index, name in enumerate(CAMERA_NAMES))
    request = SlowRequest("opaque-request", plan, 100, 101, 10.5, "fixture-q8")
    return SlowContext(request, EgoMotion(4, .1, -.2, .3), cameras)


class Reservation:
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
    (tmp_path / "SOURCE_REVISION").write_text(revision)
    sources = {"model_version": "fixture-q8", "runtime_revision": revision, "files": [
        {"path": "model.gguf"}, {"path": "projector.gguf"},
    ]}
    config = llama.LlamaServerFactory(str(tmp_path), sources, server_path=str(tmp_path / "llama-server"))
    state = SimpleNamespace(config=config, launches=[], queries=[], probes=[], reply="{}")

    def popen(command, **kwargs):
        state.launches.append(command)
        return SimpleNamespace(poll=lambda: 0, wait=lambda timeout: 0)

    def query(backend, path, payload=None, *, timeout):
        state.queries.append((path, payload, timeout))
        return {"choices": [{"finish_reason": "stop", "message": {"content": state.reply}}]}

    def devices(command, **kwargs):
        state.probes.append((command, kwargs))
        return SimpleNamespace(stdout="Available devices:\n  MTL0: Apple fixture GPU\n", stderr="")

    monkeypatch.setattr(llama, "verify_slow", lambda *args: None)
    monkeypatch.setattr(llama.subprocess, "Popen", popen)
    monkeypatch.setattr(llama.subprocess, "run", devices)
    monkeypatch.setattr(llama.socket, "socket", Reservation)
    monkeypatch.setattr(llama.LlamaVisionBackend, "_wait_ready", lambda self: None)
    monkeypatch.setattr(llama.LlamaVisionBackend, "_json", query)
    return state


def test_default_backend_stays_single_thread_cpu_and_uses_conservative_prompt(runtime, context):
    backend = runtime.config()
    try:
        assert backend.generate(context) == "{}"
        command = runtime.launches[0]
        assert command[command.index("--threads") + 1] == "1"
        assert command[command.index("--threads-batch") + 1] == "1"
        assert command[command.index("--gpu-layers") + 1] == "0"
        assert "--no-mmproj-offload" in command and "--mmproj-offload" not in command
        assert "--device" not in command and runtime.probes == []
        payload = runtime.queries[0][1]
        assert payload["messages"][0]["content"][-1]["text"] == make_prompt(context)
        assert "No calibration or pixel-to-trajectory projection is supplied" in make_prompt(context)
    finally:
        backend.close()


def test_metal_requires_native_macos_without_cpu_fallback(runtime, monkeypatch):
    monkeypatch.setattr(llama.sys, "platform", "linux")
    with pytest.raises(ValueError, match="native macOS"):
        replace(runtime.config, execution_profile="metal", threads=6)()
    assert runtime.probes == [] and runtime.launches == []


@pytest.mark.parametrize("stdout,stderr", [("", ""), ("CPU: host\n", "Metal support compiled in\n")])
def test_missing_metal_device_never_launches_cpu_fallback(runtime, monkeypatch, stdout, stderr):
    monkeypatch.setattr(llama.sys, "platform", "darwin")
    monkeypatch.setattr(llama.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout=stdout, stderr=stderr))
    with pytest.raises(ValueError, match="no silent CPU fallback"):
        replace(runtime.config, execution_profile="metal")()
    assert runtime.launches == []


def test_native_mtl0_device_is_parsed_and_both_thread_counts_and_offloads_are_explicit(runtime, monkeypatch):
    monkeypatch.setattr(llama.sys, "platform", "darwin")
    backend = replace(runtime.config, execution_profile="metal", threads=6)()
    try:
        assert runtime.probes == [([runtime.config.server_path, "--list-devices"], {
            "capture_output": True, "text": True, "timeout": 10, "check": True,
        })]
        command = runtime.launches[0]
        assert command[command.index("--device") + 1] == "MTL0"
        assert command[command.index("--gpu-layers") + 1] == "999"
        assert command[command.index("--threads") + 1] == "6"
        assert command[command.index("--threads-batch") + 1] == "6"
        assert "--mmproj-offload" in command and "--no-mmproj-offload" not in command
    finally:
        backend.close()


@pytest.mark.parametrize("threads", [0, 13, True, 1.5])
def test_invalid_thread_bounds_fail_before_device_probe_or_server_launch(runtime, threads):
    with pytest.raises(ValueError, match="CPU threads"):
        replace(runtime.config, threads=threads)()
    assert runtime.probes == [] and runtime.launches == []


def test_prompt_builder_injection_receives_actual_context_and_supplies_the_http_text(runtime, context):
    received = []

    def builder(value):
        received.append(value)
        return "projected input controlled by explicit DI"

    backend = replace(runtime.config, prompt_builder=builder)()
    try:
        backend.generate(context)
        assert received == [context]
        assert runtime.queries[0][0] == "/v1/chat/completions"
        assert runtime.queries[0][1]["messages"][0]["content"][-1] == {
            "type": "text", "text": "projected input controlled by explicit DI",
        }
    finally:
        backend.close()


@pytest.mark.parametrize("text,valid", [
    ("x" * MAX_PROMPT_BYTES, True), ("x" * (MAX_PROMPT_BYTES + 1), False),
    ("界" * (MAX_PROMPT_BYTES // 3), True), ("界" * (MAX_PROMPT_BYTES // 3 + 1), False),
    (None, False), (b"not-text", False), ("\ud800", False),
])
def test_injected_prompt_is_utf8_bounded_before_any_generation_http(runtime, context, text, valid):
    backend = replace(runtime.config, prompt_builder=lambda value: text)()
    try:
        if valid:
            assert backend.generate(context) == "{}"
            assert len(runtime.queries) == 1
        else:
            with pytest.raises(ValueError):
                backend.generate(context)
            assert runtime.queries == []
    finally:
        backend.close()


@pytest.mark.parametrize("status,ids,valid", [
    ("UNKNOWN", [], True), ("UNKNOWN", ["c-a9"], False),
    ("NO_VETO", [], True), ("NO_VETO", ["c-b4"], False),
    ("VETO", [], False), ("VETO", ["c-a9"], True), ("VETO", ["c-b4"], True),
    ("VETO", ["c-a9", "c-b4"], True), ("VETO", ["foreign-candidate"], False),
    ("VETO", ["c-a9", "c-b4", "c-a9"], False),
])
def test_schema_status_and_veto_cardinality_constraints_match_the_independent_parser(context, status, ids, valid):
    matching = []
    for branch in llama.response_schema(context)["oneOf"]:
        properties = branch["properties"]
        vetoes = properties["vetoed_candidate_ids"]
        if (properties["status"]["const"] == status
                and vetoes["minItems"] <= len(ids) <= vetoes["maxItems"]
                and all(candidate_id in vetoes["items"]["enum"] for candidate_id in ids)):
            matching.append(branch)
    assert len(matching) == (1 if valid else 0)
    raw = json.dumps({"context_hash": context.content_hash, "status": status,
                      "vetoed_candidate_ids": ids, "explanation": "Visible scene evidence."})
    if valid:
        result = parse_response(raw, context)
        assert result.status is RiskStatus(status)
        assert result.vetoed_candidate_ids == tuple(ids)
    else:
        with pytest.raises(ValueError):
            parse_response(raw, context)


@pytest.mark.parametrize("status", ["UNKNOWN", "NO_VETO"])
def test_inconsistent_model_output_is_not_rewritten_to_a_successful_status(runtime, context, status):
    raw = json.dumps({"context_hash": context.content_hash, "status": status,
                      "vetoed_candidate_ids": ["c-a9"], "explanation": "Ambiguous model output."})
    runtime.reply = raw
    backend = runtime.config()
    try:
        actual = backend.generate(context)
        assert actual == raw
        with pytest.raises(ValueError, match="Only VETO"):
            parse_response(actual, context)
    finally:
        backend.close()


def test_projected_prompt_binds_actual_candidate_order_to_colors_and_full_geometry(context):
    text = projected_prompt(context)
    payload = json.loads(text.rsplit("\n", 1)[1])
    assert set(payload) == {"axes", "candidates", "motion"}
    assert context.content_hash not in text
    assert payload["candidates"] == [
        {"id": "c-a9", "overlay_color": "cyan", "times_s": [.25, .5],
         "poses_xy_heading": [[1, .4, .2], [3, 1.2, .5]]},
        {"id": "c-b4", "overlay_color": "magenta", "times_s": [.25, .5],
         "poses_xy_heading": [[1, -.4, -.2], [3, -1.2, -.5]]},
    ]
    assert payload["motion"] == {"vx_mps": 4, "vy_mps": .1, "ax_mps2": -.2, "ay_mps2": .3}
    assert "first candidate CYAN" in text and "second MAGENTA" in text
    assert "path colours are NOT risk labels" in text
    assert "UNKNOWN if views lack scene evidence" in text


@pytest.mark.parametrize("changes", [
    {"model_version": "another-model"},
    {"request_id": "another-request"},
    {"submitted_at_monotonic_s": 200, "deadline_monotonic_s": 205},
    {"model_version": "another-model", "request_id": "another-request",
     "submitted_at_monotonic_s": 200, "deadline_monotonic_s": 205},
])
def test_same_blank_case_prompt_bytes_ignore_model_request_id_and_clocks(changes):
    case = next(case for case in build_suite().cases if case.case_id == "q08")
    context = case.to_context("fixture-q8", request_id="opaque-request",
                              submitted_at_monotonic_s=100, deadline_monotonic_s=101)
    changed = replace(context, request=replace(context.request, **changes))
    assert context.content_hash != changed.content_hash
    original = projected_prompt(context).encode("utf-8")
    updated = projected_prompt(changed).encode("utf-8")
    assert original == updated
    assert context.content_hash.encode("ascii") not in original
    assert changed.content_hash.encode("ascii") not in updated


def test_response_schema_places_request_hash_after_all_risk_decision_fields(context):
    schema = json.loads(json.dumps(llama.response_schema(context, decision_first=True)))
    assert len(schema["oneOf"]) == 3
    for branch in schema["oneOf"]:
        properties = branch["properties"]
        assert list(properties) == ["status", "vetoed_candidate_ids", "explanation", "context_hash"]


def test_default_conservative_schema_keeps_its_original_field_order(context):
    for branch in llama.response_schema(context)["oneOf"]:
        properties = branch["properties"]
        assert list(properties) == ["context_hash", "status", "vetoed_candidate_ids", "explanation"]
        assert properties["context_hash"] == {"type": "string", "const": context.content_hash}


def test_projected_prompt_still_changes_when_actual_candidate_geometry_changes(context):
    first, second = context.request.plan.candidates
    poses = first.trajectory.poses.copy()
    poses[0, 1] += .125
    candidate = Candidate(first.candidate_id, Trajectory(poses, first.trajectory.interval_s))
    plan = replace(context.request.plan, candidates=(candidate, second))
    changed = replace(context, request=replace(context.request, plan=plan))
    assert projected_prompt(context).encode("utf-8") != projected_prompt(changed).encode("utf-8")


def test_projected_prompt_cannot_observe_case_id_title_or_expected_labels():
    case = build_suite().cases[0]
    changed_labels = replace(case, case_id="CASE_ID_MUST_NOT_LEAK", title="TITLE_MUST_NOT_LEAK",
                             expected_status=RiskStatus.NO_VETO, expected_veto_ids=())
    kwargs = {"request_id": "same-request", "submitted_at_monotonic_s": 100,
              "deadline_monotonic_s": 101}
    original = projected_prompt(case.to_context("fixture-q8", **kwargs))
    relabelled = projected_prompt(changed_labels.to_context("fixture-q8", **kwargs))
    assert original == relabelled
    for hidden in (case.case_id, case.title, changed_labels.case_id, changed_labels.title,
                   "expected_status", "expected_veto_ids", "label_source", "hand_checked_geometry"):
        assert hidden not in original


@pytest.mark.parametrize("count", [1, 3])
def test_projected_prompt_rejects_an_unmatched_candidate_color_legend(context, count):
    candidates = context.request.plan.candidates[:count]
    if count == 3:
        candidates += (Candidate("c-extra", context.request.plan.candidates[0].trajectory),)
    plan = replace(context.request.plan, candidates=candidates)
    changed = replace(context, request=replace(context.request, plan=plan))
    with pytest.raises(ValueError, match="exactly two candidates"):
        projected_prompt(changed)
