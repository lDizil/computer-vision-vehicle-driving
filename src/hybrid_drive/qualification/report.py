from __future__ import annotations

import base64
from html import escape
from io import BytesIO
import json
from pathlib import Path

from hybrid_drive.qualification.metrics import summarize


def write_html(path, suite, records, summary, *, provenance: dict, execution: dict | None = None) -> None:
    records = tuple(records)
    verified = summarize(suite, records, expected_models=summary["expected_models"], seeds=summary["seeds"])
    if verified != summary:
        raise ValueError("HTML summary does not match the supplied records and suite")
    parts = [_HEADER, '<h1>Контролируемая проверка Slow-моделей</h1>', _execution_banner(execution),
             '<p class="warning">Только advisory: <code>not_road_qualified</code>. '
             'Даже идеальный результат не разрешает автоматическое вето. '
             'Это аналитически размеченные схемы, не дорожный benchmark и не независимый holdout.</p>',
             f'<p>Версия набора: {_text(suite.version)}. Охват: '
             f'{"SMOKE — подмножество" if summary["scope"] == "smoke_subset" else "полный контролируемый набор"}; '
             f'сцен: {summary["case_count"]}; seeds: {_text(summary["seeds"])}.</p>',
             f'<p>Роль набора: {_text(suite.manifest().get("suite_role", "controlled_diagnostic"))}.</p>',
              '<p>Для многошагового протокола raw response — trace с исходными ответами каждого этапа; '
              'объяснение итогового решения может быть детерминированным отображением этих категорий.</p>',
              '<p>Кандидаты на входных изображениях: первый — cyan (0, 200, 220), '
             'второй — magenta (220, 0, 180). Эталон показан отдельно и не добавлен к входу модели.</p>',
             '<h2>Качество и отказы</h2>', _quality_table(summary),
             '<h2>Измеренное время</h2>', _latency_table(summary),
             '<p>Секунды реальных попыток, включая отказы. Квантили вычислены из отдельных длительностей, '
             'не усреднены по прогонам. На малой выборке p99 не доказывает устойчивый хвост задержки.</p>',
             '<h2>Контрфактические пары</h2>', _pair_table(summary),
             '<p>Зачёт пары требует обоих точных эталонных ответов. Изменение объяснения или текста ответа не считается успехом.</p>',
             '<h2>Парное сравнение моделей</h2>', _comparison_table(summary),
             '<p>Сравниваем только одни и те же полные входы и seed. Это диагностическое сравнение, '
             'а не доказательство улучшения вождения; конфигурации моделей приведены в происхождении.</p>',
             '<h2>Сцены, входы и исходные ответы</h2>']
    for index, case in enumerate(suite.cases):
        parts.append(_case_section(index, case, records))
    parts.extend(['<h2>Определения и фиксированные критерии</h2>', _definitions(),
                  '<p>Полный диагностический PASS: все известные ответы точны, recall=100%, false veto=0%, '
                  'coverage=100%, все UNKNOWN-контроли корректны и все пары зачтены. '
                  'SMOKE не проходит полный gate. Отрицательное качество — нормальный результат измерения, не ошибка запуска.</p>',
                  _pre(summary["diagnostic_criteria"]), '<h2>Происхождение</h2>', _pre(provenance),
                  '<h2>Манифест набора</h2>', _pre(suite.manifest()), '</main></body></html>'])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")


def _execution_banner(execution):
    if execution is None:
        return ('<section class="execution"><h2>Техническое выполнение: not_supplied — статус не предоставлен.</h2>'
                '<p>Успешное завершение не подтверждено.</p></section>')
    if not isinstance(execution, dict) or execution.get("status") not in ("completed", "failed"):
        raise ValueError("execution must declare completed or failed status")
    if execution["status"] == "failed":
        title = "failed — выполнение завершилось ошибкой."
        style = "execution execution-failed"
    else:
        title = "completed — выполнение завершено."
        style = "execution"
    return (f'<section class="{style}"><h2>Техническое выполнение: {title}</h2>'
            '<p>Статус выполнения независим от качества ответов и дорожной квалификации.</p>'
            + _pre(execution) + '</section>')


def _quality_table(summary):
    rows = []
    for model, value in summary["models"].items():
        cm, rates, controls = value["confusion"], value["rates"], value["unknown_controls"]
        rows.append([model, f'{value["outcomes"]["completed"]}/{value["attempts"]}',
                     ", ".join(f"{key}={cm[key]}" for key in ("tp", "fp", "tn", "fn")),
                     f'{cm["abstained_positive"]}/{cm["abstained_negative"]}',
                     f'{cm["failed_positive"]}/{cm["failed_negative"]}',
                     f'{cm["total_positive"]}/{cm["total_negative"]}',
                     _percent(rates["risk_recall_all_attempts"]), _percent(rates["false_veto_rate"]),
                     _percent(rates["candidate_coverage"]),
                     f'{controls["correct"]}/{controls["attempts"]}', value["diagnostic_gate"]["status"].upper()])
    result = _table(["Модель", "Завершено / попыток", "Решения кандидатов", "UNKNOWN риск / clear",
                     "Отказ риск / clear", "Всего риск / clear", "Recall, %", "False veto, %", "Coverage, %",
                     "UNKNOWN-контроли", "Диагн. gate"], rows)
    for model, value in summary["models"].items():
        result += f'<h3>{_text(model)}: исходы и UNKNOWN-контроли</h3>'
        result += _pre({"outcomes": value["outcomes"], "unknown_controls": value["unknown_controls"],
                        "gate_checks": value["diagnostic_gate"]["checks"]})
    return result


def _latency_table(summary):
    return _table(["Модель", "Попыток", "Среднее, с", "p50, с", "p95, с", "p99, с"],
                  [[model, value["latency_s"]["sample_count"],
                    *(_number(value["latency_s"][key]) for key in ("mean", "p50", "p95", "p99"))]
                   for model, value in summary["models"].items()])


def _pair_table(summary):
    rows = []
    for model, value in summary["models"].items():
        for result in value["counterfactuals"]["results"]:
            rows.append([model, result["pair_id"], result["kind"], result["seed"],
                         result["first_case_id"], _yes(result["first_correct"]),
                         result["second_case_id"], _yes(result["second_correct"]), _yes(result["both_correct"])])
    return _table(["Модель", "Пара", "Изменение", "Seed", "Сцена 1", "Верно", "Сцена 2", "Верно", "Обе верны"], rows)


def _comparison_table(summary):
    return _table(["Модель A", "Модель B", "Пар попыток", "Обе верны", "Только A", "Только B", "Обе неверны",
                   "Среднее B−A, с"],
                  [[row["left_model"], row["right_model"], row["matched_attempts"], row["both_correct"],
                    row["left_only_correct"], row["right_only_correct"], row["neither_correct"],
                    _number(row["mean_latency_delta_s_right_minus_left"])] for row in summary["paired_comparisons"]])


def _case_section(index, case, records):
    parts = [f'<section id="case-{index}"><h3>{_text(case.case_id)} — {_text(case.title)}</h3>',
             f'<p>Эталон: {_text(case.expected_status.value)}, veto IDs: {_text(list(case.expected_veto_ids))}. '
             f'Полный вход SHA-256: <code>{_text(case.input_hash)}</code>.</p>', '<h4>Точный вход модели</h4>',
             _pictures(case.cameras), '<h4>Исходные виды до проекции кандидатов</h4>', _pictures(case.raw_cameras)]
    for row in records:
        if row["case_id"] != case.case_id:
            continue
        parts.extend([f'<h4>{_text(row["model_id"])} / seed {_text(row["seed"])}</h4>',
                      _table(["Исход", "Статус", "Veto IDs", "Время, с", "Request ID"],
                             [[row["outcome"], row["status"], row["vetoed_candidate_ids"],
                               _number(row["latency_s"]), row["request_id"]]]),
                      '<p>При отказе поля статуса и veto IDs не участвуют в оценке.</p>' if row["outcome"] != "completed" else "",
                      '<p>Объяснение:</p>', _pre(row["explanation"]), '<p>Исходный ответ:</p>', _pre(row["raw_response"])])
    parts.append('</section>')
    return "\n".join(parts)


def _pictures(frames):
    from PIL import Image

    parts = ['<div class="views">']
    for frame in frames:
        data = BytesIO()
        Image.fromarray(frame.rgb).save(data, format="PNG")
        encoded = base64.b64encode(data.getvalue()).decode("ascii")
        parts.append(f'<figure><img src="data:image/png;base64,{encoded}" '
                     f'alt="{_text(frame.name)}" width="{frame.width}" height="{frame.height}">'
                     f'<figcaption>{_text(frame.name)}</figcaption></figure>')
    return "\n".join(parts) + '</div>'


def _definitions():
    return ('<ul><li>TP/FP/TN/FN: только явно принятые VETO/NO_VETO решения по отдельным кандидатам известной сцены.</li>'
            '<li>Recall = TP / все рискованные попытки кандидатов, включая UNKNOWN и отказы.</li>'
            '<li>False veto = FP / все clear-попытки кандидатов, включая UNKNOWN и отказы.</li>'
            '<li>Coverage = явно принятые решения / все попытки кандидатов известных сцен.</li>'
            '<li>UNKNOWN-контроли не входят в эти знаменатели: правильный ответ для них — только завершённый UNKNOWN.</li></ul>')


def _table(headers, rows):
    headings = ''.join(f'<th scope="col">{_text(value)}</th>' for value in headers)
    body = ''.join('<tr>' + ''.join(f'<td>{_text(value)}</td>' for value in row) + '</tr>' for row in rows)
    if not body:
        body = f'<tr><td colspan="{len(headers)}">Нет элементов в этом охвате.</td></tr>'
    return f'<div class="table"><table><thead><tr>{headings}</tr></thead><tbody>{body}</tbody></table></div>'


def _pre(value):
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
    return '<pre>' + _text(value) + '</pre>'


def _text(value):
    return escape(str(value), quote=True)


def _number(value):
    return "—" if value is None else f"{value:.3f}"


def _percent(value):
    return "—" if value is None else f"{100 * value:.1f}"


def _yes(value):
    return "да" if value else "нет"


_HEADER = '''<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src data:; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'">
<title>Контролируемая проверка Slow-моделей</title>
<style>body{font:15px system-ui,sans-serif;color:#18202c;background:#f4f6f9;margin:0}main{max-width:1450px;margin:auto;padding:24px}
h1,h2,h3{line-height:1.25}h2{margin-top:36px;border-top:1px solid #ccd2dc;padding-top:20px}.warning{background:#fff0ce;padding:16px;border-left:5px solid #a56a00}
.execution h2{margin-top:0;border-top:0;padding-top:0}.execution-failed{border-left:5px solid #b42318;background:#fff0ee}
.table{overflow-x:auto}table{border-collapse:collapse;width:100%;background:white}th,td{border:1px solid #d4dae3;padding:8px;text-align:left;vertical-align:top}
th{background:#e7edf5}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#fff;padding:12px;border:1px solid #d4dae3}code{overflow-wrap:anywhere}
section{margin:22px 0;padding:16px;border:1px solid #ccd2dc;background:#fff}.views{display:flex;flex-wrap:wrap;gap:12px}figure{margin:0;flex:1 1 280px}
img{max-width:100%;height:auto;display:block}figcaption{padding:5px 0}p,li{line-height:1.5}</style></head><body><main>'''
