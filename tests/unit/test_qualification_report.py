import base64
from copy import deepcopy
from dataclasses import replace
from html.parser import HTMLParser
from io import BytesIO

from PIL import Image
import pytest

from hybrid_drive.qualification.cases import build_suite
from hybrid_drive.qualification.metrics import summarize
from hybrid_drive.qualification.report import write_html


class Document(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags = []
        self.images = []
        self.external = []
        self.events = []

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)
        for name, value in attrs:
            if name.lower().startswith("on"):
                self.events.append((name, value))
            if name in ("src", "href", "action") and not value.startswith(("data:", "#")):
                self.external.append(value)
            if tag == "img" and name == "src":
                self.images.append(value)


@pytest.fixture
def report_data():
    suite = build_suite().select(["q00", "q01", "q08"])
    return suite, report_records(suite)


def report_records(suite, models=("model",), seeds=(7,)):
    return [dict(model_id=model, case_id=case.case_id, seed=seed, outcome="completed",
                    status=case.expected_status.value, vetoed_candidate_ids=list(case.expected_veto_ids),
                    explanation="Geometry is visible.", latency_s=index + 0.5, raw_response="{}",
                    input_hash=case.input_hash, prompt_sha256=f"{index:064x}", request_id=f"{model}-{index}-{seed}")
            for model in models for index, case in enumerate(suite.cases) for seed in seeds]


def test_standalone_report_embeds_exact_model_inputs_and_raw_views_as_png(tmp_path, report_data):
    suite, records = report_data
    summary = summarize(suite, records, expected_models=["model"], seeds=[7])
    path = tmp_path / "nested" / "qualification.html"
    write_html(path, suite, records, summary, provenance={"runtime": "CPU", "threads": 6})
    html = path.read_text()
    document = Document()
    document.feed(html)
    assert not document.external and not document.events
    assert "script" not in document.tags and "iframe" not in document.tags
    assert len(document.images) == 6 * len(suite.cases)
    for case_index, case in enumerate(suite.cases):
        for frame_index, frame in enumerate((*case.cameras, *case.raw_cameras)):
            uri = document.images[6 * case_index + frame_index]
            assert uri.startswith("data:image/png;base64,")
            image = Image.open(BytesIO(base64.b64decode(uri.split(",", 1)[1])))
            assert image.format == "PNG" and image.size == (frame.width, frame.height)
            assert image.tobytes() == frame.rgb.tobytes()
    assert "SMOKE — подмножество" in html and "not_road_qualified" in html
    assert "p99 не доказывает" in html and "Исходный ответ" in html
    assert "UNKNOWN-контроли" in html and "Контрфактические пары" in html
    assert "Техническое выполнение: not_supplied" in html and "Успешное завершение не подтверждено." in html


def test_all_untrusted_model_titles_responses_and_provenance_are_text_not_markup(tmp_path, report_data):
    suite, records = report_data
    attack = '</pre><script src="https://evil.invalid/a"></script><img src=x onerror="attack()">&'
    suite = replace(suite, cases=(replace(suite.cases[0], title=attack), *suite.cases[1:]))
    for record in records:
        record.update(model_id=attack, explanation=attack, raw_response=attack, request_id=attack)
    summary = summarize(suite, records, expected_models=[attack], seeds=[7])
    path = tmp_path / "qualification.html"
    write_html(path, suite, records, summary, provenance={attack: {"source": attack}})
    html = path.read_text()
    document = Document()
    document.feed(html)
    assert "script" not in document.tags and not document.events and not document.external
    assert attack not in html and "&lt;script" in html and "&quot;https://evil.invalid/a&quot;" in html
    assert "&amp;" in html


def test_failure_raw_response_and_status_are_preserved_but_never_shown_as_a_quality_success(tmp_path, report_data):
    suite, records = report_data
    records[0].update(outcome="invalid_output", raw_response="not-json\nfull response", status="VETO")
    summary = summarize(suite, records, expected_models=["model"], seeds=[7])
    path = tmp_path / "qualification.html"
    write_html(path, suite, records, summary, provenance={})
    html = path.read_text()
    assert "not-json\nfull response" in html and "invalid_output" in html
    assert "При отказе поля статуса и veto IDs не участвуют" in html
    assert "FAIL" in html and "Отрицательное качество — нормальный результат" in html


def test_report_cannot_present_an_altered_or_other_run_summary(tmp_path, report_data):
    suite, records = report_data
    summary = summarize(suite, records, expected_models=["model"], seeds=[7])
    summary = deepcopy(summary)
    summary["models"]["model"]["confusion"]["tp"] += 1
    with pytest.raises(ValueError, match="does not match"):
        write_html(tmp_path / "qualification.html", suite, records, summary, provenance={})


def test_report_refuses_paired_model_results_when_actual_prompt_hash_changes(tmp_path, report_data):
    suite, records = report_data
    records += [dict(record, model_id="other") for record in records]
    summary = summarize(suite, records, expected_models=["model", "other"], seeds=[7])
    records[-1]["prompt_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="identical prompt_sha256"):
        write_html(tmp_path / "qualification.html", suite, records, summary, provenance={})


def test_completed_matrix_does_not_hide_execution_cleanup_failure_or_allow_error_markup(tmp_path):
    suite = build_suite()
    records = report_records(suite, models=("small", "large"), seeds=(0, 1, 2))
    summary = summarize(suite, records, expected_models=["small", "large"], seeds=[0, 1, 2])
    assert len(records) == 60 and all(row["outcome"] == "completed" for row in records)
    assert all(value["diagnostic_gate"]["status"] == "pass" for value in summary["models"].values())
    attack = '</pre><script src="https://evil.invalid/error"></script><img src=x onerror="attack()">'
    execution = {"status": "failed", "execution_error": {"kind": "cleanup_failure", "details": attack}}
    path = tmp_path / "qualification.html"
    write_html(path, suite, records, summary, provenance={}, execution=execution)
    html = path.read_text()
    document = Document()
    document.feed(html)
    assert html.index("Техническое выполнение: failed") < html.index("<h2>Качество и отказы")
    assert "cleanup_failure" in html and "execution-failed" in html and "PASS" in html
    assert "not_road_qualified" in html and "Только advisory" in html
    assert attack not in html and "&lt;script" in html and "https://evil.invalid/error" in html
    assert "script" not in document.tags and not document.events and not document.external


def test_completed_execution_does_not_turn_smoke_quality_failure_into_road_qualification(tmp_path, report_data):
    suite, records = report_data
    summary = summarize(suite, records, expected_models=["model"], seeds=[7])
    path = tmp_path / "qualification.html"
    write_html(path, suite, records, summary, provenance={}, execution={"status": "completed", "execution_error": None})
    html = path.read_text()
    assert "Техническое выполнение: completed" in html and "FAIL" in html
    assert "not_road_qualified" in html and "Только advisory" in html
