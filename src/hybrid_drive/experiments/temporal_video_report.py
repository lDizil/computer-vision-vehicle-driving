"""Aggregate offline video results without revealing per-frame predictions."""

from html import escape

from hybrid_drive.assets.files import AtomicFile


_METRICS = (
    ("Режим", None),
    ("Ложные «есть»", "false_present"), ("Ошибочные «нет»", "false_absent"),
    ("Полнота", "recall"), ("Не уверен", "uncertain"), ("Смены на стабильной цели", "flicker_edges"),
    ("Появления замечено", "delay_detected"), ("Медиана задержки", "delay_median_us"),
    ("IoU рамок", "macro_positive_iou"), ("Показано с IoU ≥ 0,5", "localized_iou_half"),
    ("Дрожание на общих парах", "paired_jitter"),
)


def write_temporal_video_report(root, result):
    matrix = result.get("matrix")
    details = _report_sections(matrix) if matrix else _failure(result)
    gallery = (
        '<a class="button" href="video.html">Смотреть клипы рядом</a>'
        if result.get("gallery") else ""
    )
    status = escape(result["status"])
    page = f"""<!doctype html><html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MV — стабильность зрения на видео</title>
<style>body{{font:16px/1.55 system-ui;color:#173044;background:#edf5f8;max-width:1250px;margin:30px auto;padding:0 18px}}
header,section{{padding:18px;border:1px solid #d5e4ec;border-radius:13px;margin:14px 0;background:#fff}}
header{{color:#f5fbff;background:linear-gradient(112deg,#142d47,#1b5b66)}}h1,h2{{line-height:1.2}}
.button{{display:inline-block;background:#bcf6e5;color:#174035;border-radius:8px;padding:9px 12px;font-weight:bold}}
.scroll{{overflow:auto}}table{{min-width:920px;width:100%;border-collapse:collapse}}th,td{{padding:8px 10px;text-align:left;
border-bottom:1px solid #d9e5ee}}th{{background:#eaf3f8}}tr.chosen{{background:#e9f9f4}}
.note{{padding:10px 12px;border-left:4px solid #ae7b31;background:#fff4db}}a{{color:#16587e}}</style></head><body>
<header><h1>Стабильность камеры во времени</h1><p>Последовательные кадры, один пространственный источник.</p>{gallery}</header>
<p>Статус запуска: {status} · <a href="result.json">Все измерения и доказательства</a></p>
{details}
</body></html>"""
    with AtomicFile(root / "report.html") as file:
        file.out.write(page.encode())
        file.commit()


def _report_sections(matrix):
    chosen = matrix["selection"]["name"]
    evaluation = matrix["holdout"][chosen]
    b = matrix["holdout"]["framewise"]
    counts = matrix["experiment_counts"]
    source = matrix["source_frames"]
    lead = (
        f'На dev выбрали <b>{escape(chosen)}</b>. На новых клипах: ложные «есть» {b["false_present"]} → '
        f'{evaluation["false_present"]}; смены на стабильной цели {b["flicker_edges"]} → '
        f'{evaluation["flicker_edges"]}; полнота {_pct(b["recall"])} → {_pct(evaluation["recall"])}.'
    )
    diagnostic = matrix["holdout_diagnostic"]["outcome"]
    message = {
        "framewise_selected_on_dev": "На dev ни один вариант не прошёл все условия улучшения.",
        "preselected_recipe_passed": "Выбранный вариант прошёл те же ограничения на новых записях.",
        "keep_framewise": "Вариант не прошёл ограничения на holdout; оставляем покадровый режим, не подбирая другой на тесте.",
    }[diagnostic]
    fault_names = matrix["faults"]
    examples = matrix["fault_invariants"]
    fault_rows = "".join(
        f'<tr><td>{escape(name)}</td><td>{arm["framewise"]["replay"]["accepted"]}</td>'
        f'<td>{arm["framewise"]["replay"]["dropped"]}</td><td>{arm["framewise"]["replay"]["withheld_stale"]}</td>'
        f'<td>{sum(arm["framewise"]["replay"]["rejected"].values())}</td></tr>'
        for name, arm in fault_names.items()
    )
    return f"""<section><h2>Итог на новых записях</h2><p>{lead}</p><p>{escape(message)}</p>
<p>{sum(source.values())} исходных кадров. {counts["clean"]} чистых прогонов и {counts["stress"]} стресс-прогонов.
Каждый режим получает одни и те же исходные оценки и рамки. Выбор делается на dev, не на финальных клипах.</p>
{_table(matrix["holdout"], chosen)}</section>
<section><h2>Отдельный dev</h2>{_table(matrix["development"], chosen)}</section>
<section><h2>Что происходит, когда кадры приходят не так</h2>
<p>Инварианты подтверждены для {len(examples)} режимов: поздние кадры дают эффект пропуска; допустимая задержка доставки
не меняет решения; после отклонённой вставки исходный прогон совпадает побитово по значениям решений и рамок.</p>
<div class="scroll"><table><thead><tr><th>Сценарий</th><th>Принято</th><th>Пропущено</th>
<th>Устарело</th><th>Отклонённых входов</th></tr></thead><tbody>{fault_rows}</tbody></table></div></section>
<section><h2>Как читать метрики</h2>
<p>Мерцание — любые смены «есть / нет / не уверен» при неизменной разметке внутри клипа. Задержка измеряется от
перехода эталона «нет → есть» до первого «есть»; уже присутствующие в начале клипа не считаются появлением.
Дрожание — изменение ошибки центра относительно движения эталона, в процентах размера изображения, на общих
парах, где оба режима показали рамки; это не измерение скорости машины и не трекинг ID.</p>
<p class="note">Источник даёт связанные с метаданными временные отметки примерно через 0,5 с. Это офлайн
проектные клипы по одному ракурсу, не проверка частоты контроллера и не оценка безопасности вождения.</p></section>"""


def _table(arms, selected):
    header = "".join(f"<th>{escape(title)}</th>" for title, _ in _METRICS)
    rows = []
    for arm, metrics in arms.items():
        cells = [arm]
        for _, key in _METRICS[1:]:
            value = metrics[key]
            if key == "recall":
                value = _pct(value)
            elif key == "delay_median_us":
                value = "—" if value is None else f"{value / 1000:.1f} мс"
            elif key == "delay_detected":
                value = f"{value} / {metrics['delay_appearances']}"
            elif key == "macro_positive_iou":
                value = "—" if value is None else f"{value:.3f}"
            elif key == "paired_jitter":
                p = value
                value = (
                    f"{_pct(p['baseline_mean'])} → {_pct(p['candidate_mean'])} ({p['count']})"
                    if p["count"] else "—"
                )
            cells.append(value)
        klass = ' class="chosen"' if arm == selected else ""
        rows.append(f"<tr{klass}>" + "".join(f"<td>{escape(str(v))}</td>" for v in cells) + "</tr>")
    return f'<div class="scroll"><table><thead><tr>{header}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


def _pct(value):
    return "—" if value is None else f"{value:.2%}"


def _failure(result):
    return f'<section><h2>Незавершённый запуск</h2><p>{escape(result.get("error", "Unknown error"))}</p></section>'
