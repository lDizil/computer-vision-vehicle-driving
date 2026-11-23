"""Local evidence for the frozen geometry comparison on new clips."""

from html import escape

from hybrid_drive.assets.files import AtomicFile
from .temporal_spatial_gallery import write_temporal_spatial_gallery
from .temporal_spatial_metrics import TemporalTarget


_MODE_LABELS = {
    "baseline": "Исходные рамки",
    "zoom_full": "Полная поправка",
    "zoom_half": "Половина поправки",
    "gate": "Выбор поправки",
    "gate_supported": "Выбор с проверкой диапазона",
}
_ROLE_LABELS = {
    "tracking_holdout": "Dev · прежняя выборка трекинга",
    "appearance_holdout": "Dev · прежняя выборка внешнего вида",
    "holdout": "новые записи",
}
_OUTCOMES = {
    "baseline_retained_on_dev": "На dev ни одна поправка не прошла пороги. Сохраняем исходные рамки.",
    "preselected_passed": "Режим, выбранный на dev, подтвердился на новых записях.",
    "no_replacement": "Режим, выбранный на dev, не подтвердился. Сохраняем исходные рамки; замену по holdout не выбираем.",
}
_REASONS = {
    "insufficient_matched_gain": "Недостаточный прирост совпадений",
    "role_regression": "Суммарная потеря в наборе",
    "recording_drop": "Превышена допустимая потеря в записи",
    "missing_metrics": "Нет метрик",
    "invalid_metrics": "Некорректные метрики",
    "different_frozen_inputs_or_labels": "Другие кадры, прогнозы или разметка",
    "different_fixed_counts": "Изменилось число рамок или целей",
    "invalid_change_counts": "Некорректный подсчёт изменений",
}


def write_guard_gallery(path, pack, targets, forecasts, *, selected):
    projected = {
        sample: TemporalTarget(tuple(track is not None for track in row.track_ids), row.boxes)
        for sample, row in targets.items()
    }
    write_temporal_spatial_gallery(
        path, pack, projected, forecasts, selected_name=selected, baseline_name="baseline", mode_labels=_MODE_LABELS,
    )


def write_guard_report(root, result):
    status = result["status"]
    body = ""
    if status != "passed":
        title = "Запуск прерван" if status == "interrupted" else "Запуск не завершён"
        body = f'<section class="warning"><h2>{title}</h2><p>{escape(str(result.get("error", status)))}</p></section>'
    if "matrix" in result:
        body += _comparison(result["matrix"], gallery_ready=result.get("gallery") == "guard.html")
    page = f"""<!doctype html><html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'">
<title>MV — осторожное уточнение рамок</title>
<style>
*{{box-sizing:border-box}}body{{margin:0;background:#edf4f8;color:#16354a;font:16px/1.5 system-ui,sans-serif}}
main{{max-width:1120px;margin:auto;padding:24px 16px 40px}}section,header,footer{{background:#fff;border-radius:15px;
border:1px solid #cfdee8;padding:18px 22px;margin:14px 0}}header{{background:#183951;color:#f5fbff}}
h1{{font-size:clamp(1.7rem,4vw,2.6rem);line-height:1.15;margin:.2em 0 .4em}}
h2{{margin:.15em 0 .5em}}h3{{margin:1.15em 0 .45em}}p{{max-width:90ch;overflow-wrap:anywhere}}
a{{color:#135e68}}.button{{display:inline-block;padding:8px 12px;border-radius:8px;background:#bcf2de;color:#124633;
font-weight:700}}.warning{{border-left:5px solid #b4552d}}.verdict{{font-weight:650}}.muted{{color:#476277}}
.scroll{{overflow:auto}}table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}
th,td{{padding:8px 10px;text-align:left;vertical-align:top;border-bottom:1px solid #d8e6ed}}
thead th{{background:#e9f3f8}}tr.selected{{background:#e4f6ed}}@media (max-width:640px){{section,header,footer{{padding:14px}}}}
</style></head><body><main>
<header><h1>Когда стоит уточнять рамку?</h1><p>Офлайн-сравнение OpenScene · передняя камера CAM_F0.</p></header>
{body}
<footer><h2>Границы опыта</h2>
<p>На каждом кадре сопоставляем рамки и проекции разметки один-к-одному при IoU ≥ 0,5. «Восстановили» и
«потеряли» — цели, которые совпали только после поправки или только до неё. Их разница даёт чистый прирост.
Малая цель занимает менее 1% кадра.</p>
<p>Одинаковые кадры, оценки присутствия и число показанных рамок. Меняем лишь координаты видимых рамок.
Кроп использует те же пиксели 224×224. Правила трекинга и рабочий пайплайн не меняются; управления машиной нет.</p>
<a href="result.json">Метрики, параметры и контрольные хеши (JSON)</a></footer></main></body></html>"""
    with AtomicFile(root / "report.html") as output:
        output.out.write(page.encode("utf-8"))
        output.commit()


def _comparison(matrix, *, gallery_ready):
    selected = matrix["selection"]["name"]
    scores = matrix["holdout"]
    baseline, chosen = scores["baseline"], scores[selected]
    changes = chosen["changes"]
    diagnostic = matrix["holdout_diagnostic"]
    verdict = _OUTCOMES[diagnostic["outcome"]]
    reason = f'<p class="muted">{escape(_reason_text(diagnostic["reasons"]))}</p>' if diagnostic["reasons"] else ""
    gallery = '<p><a class="button" href="guard.html">Сравнить кадры и рамки</a></p>' if gallery_ready else ""
    dev = "".join(
        f'<h3>{escape(_ROLE_LABELS.get(role, role))}</h3>{_geometry(stats, selected)}'
        for role, stats in matrix["development"].items()
    )
    oracle = matrix["oracle"]
    return f"""
<section><h2>Новые записи · {baseline['frames']} кадров</h2>
<p class="verdict">{escape(verdict)}</p>{reason}
<p>Выбор на dev: <strong>{escape(_MODE_LABELS[selected])}</strong>. На новых кадрах число совпавших целей:
<strong>{baseline['detections_matched']} → {chosen['detections_matched']}</strong> из {baseline['labeled_boxes']}.
Восстановили {changes['gained_targets']}, потеряли {changes['lost_targets']}.
Каждый режим показывает ровно {baseline['output_boxes']} рамок за весь набор.</p>
{gallery}{_geometry(scores, selected)}
<h3>Выбранный режим по записям</h3>{_recordings(baseline, chosen)}
<h3>Действия двух осторожных режимов</h3>{_actions(scores, selected)}</section>
<section><h2>До открытия новых меток · dev</h2>{_selection(matrix['selection'])}{dev}</section>
<section><h2>Дополнительная диагностика на новых записях</h2>
<p><strong>Oracle: {oracle['oracle_detections_matched']} совпадений</strong> против
{oracle['baseline_detections_matched']} исходных. Это верхняя граница (+{oracle['recoverable_matches']})
для выбора между исходной, половинной и полной поправкой: oracle уже знает разметку. Как режим его не используем.</p>
<p>IDF1 показывает согласованность номеров объектов при повторном запуске тех же двух правил трекинга.
Это вторичная проверка: по ней режим не выбираем. Предел IDF1 учитывает геометрические совпадения, а не качество ID.</p>
{_tracking(scores, selected)}</section>"""


def _geometry(scores, selected):
    rows = []
    for name, label in _MODE_LABELS.items():
        stats = scores[name]
        change = stats["changes"]
        small = stats["projected_area"]["small_lt_0p01"]
        values = (
            label, stats["detections_matched"], f"+{change['gained_targets']}", f"−{change['lost_targets']}",
            change["changed_present_slots"], _percent(stats["recall"]), _percent(stats["mean_matched_iou"]),
            f"{small['detections_matched']} / {small['labeled_boxes']}",
        )
        rows.append((name == selected, values))
    return _table(
        ("Режим", "Совпало", "Восстановили", "Потеряли", "Изменили рамок", "Полнота", "Средний IoU", "Малые цели"),
        rows,
    )


def _recordings(baseline, chosen):
    rows = []
    for name, before in baseline["per_recording"].items():
        after = chosen["per_recording"][name]
        changes = chosen["changes"]["per_recording"][name]
        rows.append((False, (
            name, before["frames"], before["labeled_boxes"],
            f"{before['detections_matched']} → {after['detections_matched']}",
            f"+{changes['gained_targets']}", f"−{changes['lost_targets']}",
        )))
    return _table(("Запись", "Кадры", "Цели", "Совпало", "Восстановили", "Потеряли"), rows)


def _actions(scores, selected):
    rows = (
        (name == selected, (_MODE_LABELS[name], *(scores[name]["guard_actions"][key] for key in ("keep", "half", "full"))))
        for name in ("gate", "gate_supported")
    )
    return _table(("Режим", "Исходная рамка", "Половина поправки", "Полная поправка"), rows)


def _selection(selection):
    summary = selection["summary"]
    rule = (
        f"<p>Для допуска: минимум +{summary['minimum_matched_gain']} совпадения суммарно, "
        f"без ухудшения по набору и с потерей не более {summary['maximum_recording_net_drop']} на отдельной записи. "
        "При равном приросте предпочитаем менять меньше рамок.</p>"
    )
    rows = []
    for name, stats in selection["candidates"].items():
        gain = stats["matched_gain"]
        moved = stats["changed_present_slots"]
        rows.append((name == selection["name"], (
            _MODE_LABELS[name], "—" if gain is None else f"{gain:+d}", "—" if moved is None else moved,
            "Да" if stats["eligible"] else "Нет", _reason_text(stats["reasons"]) or "—",
        )))
    return rule + _table(("Режим", "Δ Совпало", "Изменили рамок", "Допуск на dev", "Причины отказа"), rows)


def _tracking(scores, selected):
    rows = []
    for name, label in _MODE_LABELS.items():
        stats = scores[name]
        rows.append((name == selected, (
            label, _percent(stats["idf1_ceiling"]),
            *(_percent(stats["tracking"][key]["fixed_match_idf1"]) for key in ("zone", "motion_strict")),
        )))
    return _table(("Режим", "Предел IDF1", "IDF1 по зонам", "IDF1 строгого движения"), rows)


def _reason_text(reasons):
    messages = []
    for reason in reasons:
        code, *scope = reason.split(":", 2)
        parts = [_REASONS.get(code, code), *(_ROLE_LABELS.get(value, value) for value in scope)]
        messages.append(": ".join(parts))
    return "; ".join(messages)


def _table(headers, rows):
    head = "".join(f'<th scope="col">{escape(column)}</th>' for column in headers)
    lines = []
    for selected, (name, *values) in rows:
        first = '<tr class="selected">' if selected else "<tr>"
        rest = "".join(f"<td>{escape(str(value))}</td>" for value in values)
        lines.append(f'{first}<th scope="row">{escape(str(name))}</th>{rest}</tr>')
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{"".join(lines)}</tbody></table></div>'


def _percent(value):
    return "—" if value is None else f"{value:.1%}".replace(".", ",")
