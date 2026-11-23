"""Local tracking results and per-clip visual labels (never original source IDs)."""

from dataclasses import replace
from html import escape

from hybrid_drive.assets.files import AtomicFile
from .temporal_spatial_gallery import write_temporal_spatial_gallery
from .temporal_spatial_metrics import TemporalTarget


_LABELS = {
    "zone": "ID по зоне", "iou": "Пересечение рамок", "motion": "С учётом движения",
    "iou_gap": "Пересечение + 1 неопределённый кадр",
    "motion_gap": "Движение + 1 неопределённый кадр",
    "motion_strict": "Движение с отбрасыванием неоднозначных связей",
}


def write_tracking_gallery(path, pack, targets, predictions, *, selected):
    baseline = predictions["zone"]
    truth, public_truth = {}, {}
    clip_labels, previous = {}, None
    for record in baseline:
        obs = record.decision.observation
        key = (obs.recording_id, obs.clip_id)
        if key != previous:
            clip_labels = {}
            previous = key
        true = targets[obs.sample_id]
        for identifier in true.track_ids:
            if identifier is not None and identifier not in clip_labels:
                clip_labels[identifier] = f"G{len(clip_labels) + 1}"
        truth[obs.sample_id] = tuple(None if token is None else clip_labels[token] for token in true.track_ids)
        public_truth[obs.sample_id] = TemporalTarget(tuple(token is not None for token in true.track_ids), true.boxes)

    overlays = {
        name: tuple(replace(row.decision, reasons=row.reasons) for row in frames)
        for name, frames in predictions.items()
    }
    display_ids = {
        mode: tuple(tuple(None if i is None else f"P{i}" for i in row.track_ids) for row in frames)
        for mode, frames in predictions.items()
    }
    write_temporal_spatial_gallery(
        path, pack, public_truth, overlays, selected_name=selected, baseline_name="zone",
        mode_labels=_LABELS, truth_ids=truth, predicted_ids=display_ids,
    )


def write_tracking_report(root, result):
    if result.get("status") == "passed":
        outcome = result["matrix"]["holdout_diagnostic"]
        status = {
            "baseline_retained_on_dev": "В dev ни один новый вариант не прошёл условия. Оставили отсчёт по зонам.",
            "preselected_passed": "Предвыбранный вариант подтвердил свойства на новых клипах.",
            "no_replacement": "Предвыбранный вариант не подтвердился на новых клипах. По финальному тесту другой вариант не выбирали.",
        }[outcome["outcome"]]
        body = f"""<section><h2>Итог</h2><p>{escape(status)}</p><p><a href="tracking.html">Открыть видеоплеер с ID</a></p>
<p>Ассоциация не меняет координаты и оценки присутствия: меняются исключительно номера на уже показанных рамках.</p></section>
<section><h2>Новые записи</h2>{_table(result['matrix']['holdout'], outcome['evaluated_recipe'])}</section>
<section><h2>Dev, три старых видеонабора</h2>{_table(result['matrix']['development'], outcome['evaluated_recipe'])}</section>"""
    else:
        body = f"<section><h2>Прогон не завершён</h2><p>{escape(result.get('error', result['status']))}</p></section>"
    page = f"""<!doctype html><html lang="ru"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>MV — ID машины между кадрами</title>
<style>body{{max-width:1120px;margin:24px auto;background:#edf5fa;color:#14334b;font:16px/1.55 system-ui;padding:0 16px}}
section,header{{background:#fff;border:1px solid #cfdee8;border-radius:14px;padding:17px;margin:13px 0}}
header{{background:#173b59;color:#effaff}}h1{{margin:.2em 0}}a{{color:#175075}}.scroll{{overflow:auto}}
table{{width:100%;min-width:730px;border-collapse:collapse}}th,td{{padding:8px;border-bottom:1px solid #d6e2eb;text-align:left}}
th{{background:#e5f3fa}}tr.chosen{{background:#e4faf0}}</style>
<header><h1>Та же машина или новый трек?</h1><p>Локальное сравнение на последовательных фронтальных кадрах.</p></header>
{body}
<section><h2>Как читать</h2><p>Золотой ID G — эталон, бирюзовый P — прогноз. Номера существуют только внутри клипа.
IDF1 считаем по одинаковым покадровым IoU-сопоставлениям; это диагностика, не MOTChallenge.
На кадр и четверть приходится одна подходящая аннотация, а не все автомобили. Ни история, ни метки не создают рамки.
Ни абсолютная скорость, ни безопасность вождения не измеряются.</p><a href="result.json">Все счётчики и хеши</a></section></html>"""
    with AtomicFile(root / "report.html") as target:
        target.out.write(page.encode())
        target.commit()


def _table(arms, chosen):
    rows = []
    for name, metrics in arms.items():
        score = metrics["fixed_match_idf1"]
        cells = [
            _LABELS.get(name, name), "—" if score is None else f"{score:.1%}",
            metrics["detections_matched"], metrics["wrong_links"], metrics["adjacent_switches"],
            f"{metrics['cross_zone_correct']} / {metrics['cross_zone_links']}",
        ]
        klass = ' class="chosen"' if name == chosen else ""
        rows.append(f"<tr{klass}>" + "".join(f"<td>{escape(str(value))}</td>" for value in cells) + "</tr>")
    return (
        '<div class="scroll"><table><thead><tr><th>Режим</th><th>IDF1</th><th>Matched боксы</th>'
        '<th>Ошибочные связи</th><th>Смены ID</th><th>Через зону правильно</th></tr></thead>'
        '<tbody>' + "".join(rows) + "</tbody></table></div>"
    )
