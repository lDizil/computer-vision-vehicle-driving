"""Local report and two-pane comparison of held-out-from-fit crop refinements."""

from html import escape

from hybrid_drive.assets.files import AtomicFile
from .temporal_spatial_gallery import write_temporal_spatial_gallery
from .temporal_spatial_metrics import TemporalTarget


_LABELS = {
    "baseline": "Текущие рамки",
    "geometry_0": "Поправка без изображения · seed 0",
    "geometry_1": "Поправка без изображения · seed 1",
    "global_0": "Признаки всего кадра · seed 0",
    "global_1": "Признаки всего кадра · seed 1",
    "zoom_0": "DINO на крупном плане · seed 0",
    "zoom_1": "DINO на крупном плане · seed 1",
}
_SPLITS = {
    "tracking_holdout": "Клипы проверки трекинга (знакомые записи)",
    "appearance_holdout": "Клипы проверки внешнего вида (знакомые записи)",
}


def write_zoom_gallery(path, pack, targets, outputs, *, selected):
    boxes = {key: TemporalTarget(tuple(v is not None for v in row.track_ids), row.boxes) for key,row in targets.items()}
    write_temporal_spatial_gallery(
        path, pack, boxes, outputs, selected_name=selected, baseline_name="baseline", mode_labels=_LABELS,
    )


def write_zoom_report(root, result):
    if result.get("status") == "passed":
        m = result["matrix"]
        selected = m["selection"]["name"]
        primary = m["references"][m["primary_reference"]]
        summaries = "".join(
            f"<section><h2>{escape(_SPLITS.get(role, role))}</h2>{_table(stats, track=True)}</section>"
            for role, stats in m["references"].items()
        )
        lead = (
            f'<p>Выбор на dev: <b>{escape(_LABELS[selected])}</b>. На контрольных кадрах основного опыта '
            f'совпадения <b>{primary["baseline"]["detections_matched"]} → {primary[selected]["detections_matched"]}</b> '
            f'из {primary["baseline"]["labeled_boxes"]} размеченных целей.</p>'
        )
        body = f"""<section><h2>Что получилось</h2>{lead}
<p>Порог присутствия и число показанных боксов не менялись. Меняются только координаты исходных рамок.</p>
<a class="button" href="zoom.html">Сравнить рамки на тех же 144 кадрах</a></section>
<section><h2>Dev — выбор до открытия контрольных меток</h2>{_table(m["development"], track=False)}</section>{summaries}"""
    else:
        body = f'<section><h2>Запуск не завершён</h2><p>{escape(result.get("error", result["status"]))}</p></section>'
    page = f"""<!doctype html><html lang="ru"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>MV — рамка на крупном плане</title>
<style>body{{max-width:1100px;margin:26px auto;padding:0 16px;background:#eef5f8;color:#16364d;font:16px/1.55 system-ui}}
section,header{{background:#fff;border:1px solid #d1e4ed;padding:18px;border-radius:14px;margin:14px 0}}
header{{background:#193855;color:#f5fbff}}h1{{margin:.3em 0}}a{{color:#1a5a7e}}.button{{display:inline-block;
background:#bdf4e5;color:#143d34;padding:9px 12px;border-radius:8px;font-weight:700}}.scroll{{overflow:auto}}
table{{min-width:740px;width:100%;border-collapse:collapse}}th,td{{padding:8px;text-align:left;border-bottom:1px solid #d7e4ec}}
th{{background:#e8f3f9}}.note{{background:#fff3df;border-left:4px solid #edbb62;padding:10px}}</style>
<header><h1>Крупный план для уточнения рамок</h1><p>Те же исходные кадры, три вида признаков и два фиксированных seed.</p></header>
{body}
<section><h2>Что означают числа</h2>
<p>Совпадение — одно предсказание одной размеченной цели на кадре при IoU не менее 0,5. «Малые» — площадь эталона
меньше 1% кадра. Потолок IDF1 ограничен этими совпадениями; измеренный IDF1 переигран тем же трекером.
Кроп — переразметка уже имеющихся пикселей 224×224 для DINO, а не повышение исходного разрешения камеры.</p>
<p class="note">Контрольные записи исключены из этой подгонки и выбора, но уже использованы в прежних опытах.
Новым слепым тестом результат не считаем. Управление не меняли.</p><a href="result.json">Счётчики и доказательства</a></section></html>"""
    with AtomicFile(root / "report.html") as output:
        output.out.write(page.encode())
        output.commit()


def _table(scores, *, track):
    columns = ["Режим", "Совпало", "Показано", "Recall", "Малых совпало", "Потолок IDF1"]
    if track:
        columns += ["IDF1 зоны", "IDF1 строгого движения"]
    rows = []
    for name, stats in scores.items():
        area = stats["projected_area"]["small_lt_0p01"]
        row = [
            _LABELS.get(name, name), stats["detections_matched"], stats["output_boxes"],
            _pct(stats["recall"]), f"{area['detections_matched']} / {area['labeled_boxes']}",
            _pct(stats["idf1_ceiling"]),
        ]
        if track:
            row += [_pct(stats["tracking"][key]["fixed_match_idf1"]) for key in ("zone", "motion_strict")]
        rows.append("<tr>" + "".join(f"<td>{escape(str(v))}</td>" for v in row) + "</tr>")
    headers = "".join(f"<th>{escape(x)}</th>" for x in columns)
    return f'<div class="scroll"><table><thead><tr>{headers}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


def _pct(value):
    return "—" if value is None else f"{value:.1%}"
