"""Self-contained local views of an identity comparison with appearance."""

from html import escape

from hybrid_drive.assets.files import AtomicFile
from .spatial_tracking_report import write_tracking_gallery


_LABELS = {
    "zone": "ID по зоне", "motion_strict": "Движение без внешнего вида",
    "rgb": "Цвет рамки", "rgb_shuffled": "Цвет перемешан",
    "dino": "DINO внутри рамки", "dino_shuffled": "DINO перемешан",
    "contrast": "DINO минус фон", "contrast_shuffled": "DINO минус фон, перемешан",
}


def write_appearance_gallery(path, pack, targets, predictions, *, selected):
    write_tracking_gallery(path, pack, targets, predictions, selected=selected, mode_labels=_LABELS)


def write_appearance_report(root, result):
    if result.get("status") == "passed":
        matrix = result["matrix"]
        check = matrix["holdout_diagnostic"]
        status = {
            "baseline_retained_on_dev": "Ни один визуальный вариант не прошёл все dev-условия.",
            "preselected_passed": "Заранее выбранный вариант прошёл условия на новых кадрах.",
            "no_replacement": "Заранее выбранный вариант не подтвердился. Другой по финалу не подбираем.",
        }[check["outcome"]]
        body = f"""<section><h2>Что получилось</h2><p>{escape(status)}</p>
<p>Координаты рамок и решения о присутствии одинаковы. Отличаются только номера треков.</p>
<p><a class="button" href="appearance.html">Открыть 144 кадра с локальными ID</a></p></section>
<section><h2>Две новые записи</h2>{_tracking(matrix["holdout"])}{_signal(matrix["holdout"])}</section>
<section><h2>Dev: четыре знакомых набора</h2>{_tracking(matrix["development"])}{_signal(matrix["development"])}</section>"""
    else:
        body = f'<section><h2>Запуск не завершён</h2><p>{escape(result.get("error", result["status"]))}</p></section>'
    page = f"""<!doctype html><html lang="ru"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>MV — внешность машины и ID</title>
<style>body{{max-width:1130px;margin:25px auto;padding:0 16px;background:#edf6fa;color:#18374e;font:16px/1.55 system-ui}}
section,header{{background:#fff;border:1px solid #cfe3ed;border-radius:14px;padding:18px;margin:14px 0}}
header{{background:#173753;color:#f1fbff}}h1{{margin:.25em 0}}a{{color:#145777}}.button{{display:inline-block;
background:#c2f8e9;color:#143d35;padding:8px 12px;border-radius:9px;font-weight:700}}.scroll{{overflow:auto}}
table{{min-width:720px;width:100%;border-collapse:collapse}}th,td{{text-align:left;padding:8px;border-bottom:1px solid #d6e4ec}}
th{{background:#e8f4f9}}</style>
<header><h1>Внешний вид или только движение?</h1><p>Офлайн проверка на неизменных рамках.</p></header>
{body}
<section><h2>Как читать результат</h2>
<p>IDF1 сопоставляет номера треков после одинаковых геометрических совпадений; это диагностика, не MOTChallenge.
Проверка выбора визуального кандидата сравнивает кадры с эталонной идентичностью, но не управляет трекером.
Контроль переставляет признаки только внутри кадра; одиночные и некоторые случайно неизменившиеся позиции остаются
прежними, поэтому это не гарантированная нулевая модель. Веса и пороги не меняли; водительских решений нет.</p>
<a href="result.json">Числа, сравнения и хеши</a></section></html>"""
    with AtomicFile(root / "report.html") as target:
        target.out.write(page.encode())
        target.commit()


def _tracking(split):
    answers = []
    for name, m in split["tracking"].items():
        idf1 = m["fixed_match_idf1"]
        answer = [_LABELS.get(name, name), "—" if idf1 is None else f"{idf1:.1%}",
                  m["detections_matched"], m["wrong_links"], m["adjacent_switches"],
                  f"{m['cross_zone_correct']} / {m['cross_zone_links']}"]
        answers.append("<tr>" + "".join(f"<td>{escape(str(x))}</td>" for x in answer) + "</tr>")
    return ('<div class="scroll"><table><thead><tr><th>Режим</th><th>IDF1</th><th>Те же совпавшие рамки</th>'
            '<th>Неверные связи</th><th>Смены ID</th><th>Через зону правильно</th></tr></thead><tbody>'
            + "".join(answers) + "</tbody></table></div>")


def _signal(split):
    m = split["signal"]
    rows = []
    for name, p in m["profiles"].items():
        n = p["queries"]
        a = [ _LABELS.get(name, name), f"{p['actual_correct']} / {n}", f"{p['shuffled_correct']} / {n}",
              split["shuffle"][name]["changed_slots"]]
        rows.append("<tr>" + "".join(f"<td>{escape(str(x))}</td>" for x in a) + "</tr>")
    return (f"<p>Сравнимых соседних рамок с реальной альтернативой: {m['eligible_geometry']}.</p>"
            '<div class="scroll"><table><thead><tr><th>Признак</th><th>Правильный визуальный выбор</th>'
            '<th>После перестановки</th><th>Переставлено разных векторов</th></tr></thead><tbody>'
            + "".join(rows) + "</tbody></table></div>")
