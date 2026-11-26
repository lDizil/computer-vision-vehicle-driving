"""Render a local camera overlay report from the sealed, scored projection audit."""

import argparse
from base64 import b64encode
from hashlib import sha256
from html import escape
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from hybrid_drive.adapters.real_camera import _rectify_native_openscene_rgb
from hybrid_drive.experiments.bounded_io import decode_json, read_bounded
from hybrid_drive.experiments.vru_chain_capture import index_sha
from hybrid_drive.experiments.vru_fusion_holdout_data import load_fusion_inputs
from hybrid_drive.experiments.vru_projection_audit import SCORE_SCHEMA


CAUSE = {
    "matched": "Есть однозначный центр на земле",
    "ground_center_ambiguous": "Рядом есть центр, соответствие неоднозначно",
    "no_projected_target_envelope": "Нет 2D-контура из разметки",
    "unsupported_class": "Класс не поддерживается детекторами",
    "no_reliable_camera_box": "Нет 2D-бокса с достаточным совпадением",
    "camera_box_ambiguous": "2D-бокс нельзя отнести к одной цели",
    "camera_box_filtered": "2D-бокс есть, отсеян до проекции",
    "camera_projection_rejected": "2D-бокс есть, проекция отклонена",
    "camera_projection_far": "Проекция есть, но дальше 2 м от цели",
    "lost_after_camera_projection": "Близкий центр потерян после проекции",
}
STATES = {
    "score_floor": "ниже порога", "crop_edge": "на краю кропа", "deduplicated": "дубликат",
    "box_budget": "лимит боксов", "projected": "проекция принята", "projection_rejected": "проекция отклонена",
    "size": "маленький бокс", "edge": "край кадра", "invalid_ray": "луч не пересёк плоскость надёжно",
    "outside_roi": "вне области", "unstable": "чувствительно к сдвигу бокса",
}
MODEL = {"ssdlite": "SSDLite", "faster_rcnn": "Faster R-CNN"}
W, H = 768, 432


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("control-score", "control-inputs", "replay-score", "replay-inputs", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args(argv)
    control = _load(args.control_score, args.control_inputs)
    replay = _load(args.replay_score, args.replay_inputs)
    control_moving = control[0]["summary"]["moving"]
    missed = control_moving["unmatched_with_no_near_center"]
    with_box = control_moving["no_near_with_camera_witness"]["0.3"]
    with args.output.open("x", encoding="utf-8") as out:
        out.write(_HEAD + f"""
<header><div class="tag">MV · разбор ошибок компьютерного зрения</div>
<h1>Куда пропадают<br>детекции</h1>
<p>Два детектора → 2D-боксы → проекция на дорогу → объединение.</p>
<p class="focus"><b>{with_box} из {missed}</b> движущихся целей без близкого центра на контроле
имели 2D-бокс, который можно отнести к той же цели по контуру из разметки.</p></header>
{_section("01", "Отдельный контроль", control)}
{_section("02", "Повтор #54", replay)}
<footer>Зелёный контур — проекция 3D-разметки, а не ручная 2D-разметка. Отсутствие бокса не доказывает,
что объект был виден. Ниже внутреннего порога и NMS TorchVision трасса не видит боксы.
Поездки отобраны для диагностики; это не оценка частоты ошибок в реальном движении.</footer></main></body></html>""")


def _load(path, root):
    score = decode_json(read_bounded(path, 16 * 1024**2))
    if score["schema"] != SCORE_SCHEMA or score["input_sha256"] != index_sha(root):
        raise ValueError("Projection score and camera evidence do not share an index")
    return score, load_fusion_inputs(root), root


def _section(number, title, bundle):
    score, index, root = bundle
    summary = score["summary"]
    moving = summary["moving"]
    rows = "".join(f"<tr><td>{escape(CAUSE[name])}</td><td>{count}</td></tr>"
                   for name, count in sorted(moving["causes"].items(), key=lambda item: (-item[1], item[0])))
    plane = moving["diagnostic_plane_oracle"]
    return f"""<section><div class="tag">{number} · {summary['windows']} кадров · {summary['recordings']} поездки</div>
<h2>{escape(title)}</h2>
<div class="grid"><table><caption>{moving['targets']} размеченных движущихся целей</caption>
<thead><tr><th>Статус</th><th>Целей</th></tr></thead><tbody>{rows}</tbody></table>
<div class="panel"><h3>Контрольная проверка геометрии</h3><p>
Даже если в пересчёте луча подставить высоту низа из разметки, в радиус 2 м возвращаются
{plane['misses_moved_within_radius']} из {plane['unplaced_targets_with_owned_projected_hit']}
пропущенных целей с принятым 2D-боксом и его проекцией. Это диагностический предел для этой поправки,
а не новый прогноз.</p>
<p>Трасса дала {score['reference_current_objects_identical']} неизменных объектов обоих детекторов
при повторе кадров прошлого прогона.</p></div></div>
<h3>Те же кадры с наложением контуров</h3><div class="cards">
{''.join(_example(case, index, root) for case in _select(summary['cases']))}
</div></section>"""


def _select(cases):
    useful = [case for case in cases if case["category"] == "moving" and case["views"]]
    groups = ("no_reliable_camera_box", "camera_box_filtered", "camera_projection_rejected",
              "camera_projection_far", "ground_center_ambiguous", "matched")
    return [choice for cause in groups if (choice := next((case for case in useful if case["cause"] == cause), None)) is not None]


def _example(case, index, root):
    chosen = max(case["boxes"], key=lambda box: box["iou"], default=None)
    camera = chosen["camera_name"] if chosen else max(case["views"], key=lambda view: view["xyxy"][3]-view["xyxy"][1])["camera_name"]
    label = next(view["xyxy"] for view in case["views"] if view["camera_name"] == camera)
    view = next(row for row in index["frames"][case["sample_token"]]["views"] if row["camera"] == camera)
    raw = read_bounded(root / view["path"], 2 * 1024**2)
    if view["sha256"] != sha256(raw).hexdigest():
        raise ValueError("Report camera JPEG differs from the audited observation")
    with Image.open(BytesIO(raw), formats=("JPEG",)) as source:
        pixels, _ = _rectify_native_openscene_rgb(camera, np.asarray(source), view["intrinsic"], view["camera_to_ego_rotation"],
                                                   view["camera_to_ego_translation"], view["distortion"], W, H)
    data = BytesIO()
    Image.fromarray(pixels).save(data, format="JPEG", quality=87)
    jpg = b64encode(data.getvalue()).decode("ascii")
    if chosen:
        rect = chosen["box_xyxy"]
        status = STATES[chosen["state"]]
        if chosen["reason"]:
            status += ": " + STATES[chosen["reason"]]
        legend = f'<span class="det">{MODEL[chosen["detector"]]} · IoU {chosen["iou"]:.2f} · {escape(status)}</span>'
    else:
        rect, legend = label, "<span>Бокс модели не сопоставился</span>"
    crop = _crop(label, rect)
    overlay = _rect(label, "#46dfb5") + (_rect(rect, "#ffb576") if chosen else "")
    return f"""<article class="panel"><div class="tag">{escape(camera)} · {case['center_xy_m'][0]:.1f} м вперёд</div>
<h4>{escape(CAUSE[case['cause']])}</h4>
<svg viewBox="{crop}" role="img" aria-label="Увеличенный прямоугольник из кадра и границы бокса">
<image x="0" y="0" width="{W}" height="{H}" href="data:image/jpeg;base64,{jpg}"/>{overlay}</svg>
<div class="legend"><span class="gt">Контур из разметки</span> {legend}</div></article>"""


def _crop(a, b):
    x0, y0, x1, y1 = min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])
    width = max(104., (x1-x0) * 2.5, (y1-y0) * 3.2)
    height = width * .65
    width, height = min(width, W), min(height, H)
    left = max(0., min(W-width, (x0+x1-width)/2))
    top = max(0., min(H-height, (y0+y1-height)/2))
    return f"{left:.2f} {top:.2f} {width:.2f} {height:.2f}"


def _rect(box, color):
    x, y, right, bottom = box
    return (f'<rect x="{x:.2f}" y="{y:.2f}" width="{right-x:.2f}" height="{bottom-y:.2f}" '
            f'fill="none" stroke="{color}" stroke-width="1.2" vector-effect="non-scaling-stroke"/>')


_HEAD = """<!doctype html><html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>MV · аудит проекции</title>
<style>
:root{color-scheme:dark;--bg:#091624;--text:#eef8ff;--muted:#acc6d8;--a:#46dfb5;--orange:#ffb576;--edge:#284358;--card:#142638}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:16px/1.5 system-ui,sans-serif}
main{margin:auto;max-width:1120px;padding:30px clamp(14px,5vw,52px)}
header{padding:30px 0;background:radial-gradient(ellipse at 70% 30%,#1a3a51 0,transparent 64%)}
h1{font-size:clamp(38px,8vw,82px);letter-spacing:-.06em;line-height:.98;margin:.18em 0 .25em}
h2{font-size:clamp(28px,4vw,40px);line-height:1.2;margin:.2em 0 .4em}h3,h4{margin:.4em 0}
p,footer{color:var(--muted);max-width:78ch}.tag{color:var(--a);font-size:.8rem;letter-spacing:.1em;text-transform:uppercase}
.focus{border-left:3px solid var(--a);padding-left:12px}.focus b,.gt{color:var(--a)}.det{color:var(--orange)}
section{border-top:1px solid var(--edge);padding:24px 0}.grid,.cards{display:grid;gap:12px;grid-template-columns:repeat(auto-fit,minmax(min(100%,300px),1fr))}
.grid{margin-bottom:24px}.panel{background:var(--card);padding:14px;border:1px solid var(--edge);border-radius:12px}
table{border-collapse:collapse;font-variant-numeric:tabular-nums}caption{text-align:left;color:var(--muted)}
th,td{border-bottom:1px solid var(--edge);padding:6px;text-align:left}th:last-child,td:last-child{text-align:right}
svg{width:100%;display:block;background:var(--bg);border-radius:6px}.legend{display:grid;gap:3px;font-size:.84rem;margin:7px 0}
footer{border-top:1px solid var(--edge);padding-top:20px;font-size:.9rem}
</style></head><body><main>"""


if __name__ == "__main__":
    main()
