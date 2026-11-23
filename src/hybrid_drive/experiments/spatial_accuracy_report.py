"""Local report for paired calibration and box refinement."""

from html import escape
import json

from hybrid_drive.assets.files import AtomicFile


def write_spatial_accuracy_report(root, manifest, result):
    rows = []
    for key, title in (
        ("previous", "Старый порог и рамки"),
        ("calibrated", "Новые пороги, старые рамки"),
        ("selected", "Новые пороги, выбранные рамки"),
    ):
        if key not in result.get("final", {}):
            continue
        mode = result["final"][key]
        decisions, geometry = mode["decisions"], mode["geometry"]
        quality = mode["localized_at_iou_half"]
        cells = [
            title, decisions["false_present"], decisions["true_present"], decisions["uncertain_count"],
            _percent(decisions["precision"]), _percent(decisions["positive_recall"]), _percent(decisions["coverage"]),
            _number(geometry["macro_positive_iou"]), quality["count"],
        ]
        rows.append("<tr>" + "".join(f"<td>{escape(str(value))}</td>" for value in cells) + "</tr>")
    status = escape(result["status"])
    gallery = '<p><a class="button" href="gallery.html">Открыть галерею решений</a></p>' if result.get("gallery") else ""
    if "selected_refinement_seed" in result:
        seed = result["selected_refinement_seed"]
        selected = ("Уточнение рамок не прошло критерии на dev: оставили исходную геометрию."
                    if seed is None else f"Уточнение рамок seed {seed} выбрано только по dev.")
    else:
        selected = "Выбор рамок не завершён."
    page = f"""<!doctype html><html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>MV · точность решений</title>
<style>body{{font:16px/1.55 system-ui;color:#153148;background:#edf4f8;max-width:1160px;margin:32px auto;padding:0 18px}}
h1,h2{{color:#153f61}}section,header{{padding:18px;margin:14px 0;border-radius:12px;background:white;border:1px solid #d5e5ef}}
header{{background:#122e44;color:#effbff}}header h1{{color:inherit}}.button{{background:#b9f3e7;color:#123e39;
padding:8px 14px;border-radius:9px;font-weight:700}}.scroll{{overflow:auto}}table{{min-width:810px;width:100%;border-collapse:collapse}}
th,td{{padding:9px;text-align:left;border-bottom:1px solid #d5e5ef}}th{{background:#edf5f8}}
pre{{white-space:pre-wrap;word-break:break-word}}</style></head><body>
<header><h1>«Есть», «нет» или «не уверен»</h1><p>Новые пороги и рамки на двух независимых записях.</p>{gallery}</header>
<section><h2>Сравнение на одинаковых кадрах</h2><p>{escape(selected)} Статус запуска: {status}.</p>
<div class="scroll"><table><thead><tr><th>Режим</th><th>Ложные «есть»</th><th>Верные «есть»</th>
<th>Не уверен</th><th>Точность «есть»</th><th>Полнота</th><th>Ответ без отказа</th>
<th>IoU по целям</th><th>Показано с IoU ≥ 0,5</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<p>«Не уверен» не считается правильным «нет». IoU мерит пересечение рамок на всех положительных целях независимо от порога.
Последняя колонка требует и уверенного «есть», и IoU ≥ 0,5.</p></section>
<section><h2>Условия</h2><p>Один фронтальный кадр, четыре зоны. Пороги и refiner выбраны на dev и сохранены до финальных меток.
Цели — проекции 3D-разметки. Метрики описывают офлайн-эксперимент, не безопасность вождения.</p>
<pre>{escape(json.dumps({"manifest": manifest, "result": result}, ensure_ascii=False, indent=2))}</pre></section>
</body></html>"""
    with AtomicFile(root / "report.html") as target:
        target.out.write(page.encode())
        target.commit()


def _number(value):
    return "—" if value is None else f"{value:.3f}"


def _percent(value):
    return "—" if value is None else f"{value:.1%}"
