"""Render two scored VRU chain replays as one offline report with camera examples."""

import argparse
from base64 import b64encode
from hashlib import sha256
from html import escape
from io import BytesIO
from pathlib import Path

from PIL import Image

from hybrid_drive.experiments.bounded_io import decode_json, read_bounded
from hybrid_drive.experiments.vru_chain_capture import index_sha
from hybrid_drive.experiments.vru_chain_replay import SCORE_SCHEMA
from hybrid_drive.experiments.vru_fusion_holdout_data import load_fusion_inputs
from hybrid_drive.experiments.vru_chain_rescue import ARMS, EARLY, RESCUES


NAMES = {"ssdlite": "SSDLite", "faster_rcnn": "Faster R-CNN", EARLY: "Раннее объединение",
         "native_rescue": "+ Собственная цепочка", "bridge_rescue": "+ Согласованный донор"}
GATES = {"rescue_signal": "Порог пройден", "gain_not_established": "Улучшение не доказано",
         "insufficient_support": "Недостаточно совпадений"}
CHECKS = {"coverage_invariant": "Покрытие неизменно", "matched_moving_support": "≥12 найденных движущихся",
          "verified_moving_gain": "≥2 дополнительных проверенных цепочек", "verified_gain_spread": "≥2 поездок с приростом",
          "moving_error_reduction": "Ошибка смещения ниже на ≥0,03 м", "future_center_bound": "Будущий центр не хуже >0,05 м",
          "no_added_quiet_motion": "Нет новой скорости у стоящих", "no_added_wrong_links": "Нет новых связей с другой целью"}
CAUSES = {
    "current_missing_center": "Нет близкой детекции в текущем кадре",
    "current_ambiguous_or_nonmutual": "Неоднозначное сопоставление в текущем кадре",
    "previous_missing_center": "Нет близкой детекции на t−0,5 с",
    "older_missing_center": "Нет близкой детекции на t−1 с",
    "previous_ambiguous_or_nonmutual_center": "Неоднозначное сопоставление на t−0,5 с",
    "older_ambiguous_or_nonmutual_center": "Неоднозначное сопоставление на t−1 с",
    "older_no_reliable_history": "Трекер не нашёл надёжную связь с t−1 с",
    "older_ambiguous_history": "Несколько правдоподобных связей на t−1 с",
    "confirmation_confirmed": "Все три наблюдения сопоставлены",
    "confirmation_below_motion_floor": "Слишком малое смещение",
    "confirmation_inconsistent_motion": "Скорости на двух интервалах не согласованы",
}
PHASES = {"previous": "t−0,5 с", "older": "t−1 с", "origin": "детекция-источник", "confirmation": "подтверждение"}
STATES = {"missing_frame": "кадр недоступен", "missing_annotations": "нет разметки",
          "unlabeled_target": "цель не размечена на этом кадре", "unverified_link": "связь не сверяется с разметкой",
          "other_target": "связь с другой размеченной целью", "link_not_attempted": "связь не искали",
          "ambiguous_history": "несколько возможных связей", "nonmutual_history": "связь не взаимна",
          "no_reliable_history": "нет надёжной связи", "no_initial_link": "нет первой связи",
          "no_older_link": "нет второй связи", "no_older_observation": "нет третьего кадра",
          "no_common_camera": "нет общей камеры для трёх кадров", "inconsistent_appearance": "не совпадает внешность",
          "inconsistent_cameras": "камеры дают разные скорости"}
GROUPS = {"moving": "Движущийся объект", "quiet": "Стоящий объект", "intermediate": "Малое движение"}
PALETTE = ("#a3a0ff", "#8fa8bc", "#46dfb5")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("replay-score", "replay-inputs", "control-score", "control-inputs", "output"):
        parser.add_argument("--" + field, type=Path, required=True)
    args = parser.parse_args(argv)
    replay = _read(args.replay_score, args.replay_inputs)
    control = _read(args.control_score, args.control_inputs)
    report = _page(replay, control)
    with args.output.open("x", encoding="utf-8") as target:
        target.write(report)


def _read(score_path, input_root):
    result = decode_json(read_bounded(score_path, 16 * 1024**2))
    if result["schema"] != SCORE_SCHEMA or result["input_sha256"] != index_sha(input_root):
        raise ValueError("Report score and input camera cohort differ")
    return result, load_fusion_inputs(input_root), input_root


def _page(replay, control):
    control_summary = control[0]["summary"]
    n = control_summary["counts"]
    strongest = next((name for name in RESCUES if control_summary["comparisons"][name]["gate"]["status"] == "rescue_signal"), None)
    verdict = (NAMES[strongest] + ": порог пройден" if strongest else "Ни один из двух вариантов не прошёл порог")
    return _HEAD + f"""
<main>
<header><div class="eyebrow">MV · OFFLINE RESEARCH / VRU</div>
<h1>Три кадра.<br>Одна цепочка.</h1>
<p>Где теряется оценка скорости прохожих и велосипедистов после объединения двух детекторов.</p>
<div class="hero"><b>{escape(verdict)}</b><span>Независимый контроль · {n['windows']} окон · {n['recordings']} поездки</span></div>
</header>
<section><h2>Как читать результат</h2>
<p>Оба детектора видят одни и те же кадры. Для сравнения восстановления сохраняем текущий центр и все ранее подтверждённые связи.
Подтверждение трекером и пространственная сверка трёх наблюдений с одной разметкой — разные показатели.
Отсутствующая оценка для расчёта ошибки даёт нулевое смещение, но не означает, что объект стоит.</p>
<div class="flow"><span>Камеры t−1, t−0,5, t</span><span>Два детектора</span><span>Связи без будущих кадров</span><span>Сверка с разметкой</span></div>
</section>
{_section("01", "Новые поездки", "Контроль не пересекается по дню и машине с прогонами #50–54.", control)}
{_section("02", "Повтор #54", "Разбор известных результатов, не независимая проверка восстановления.", replay)}
<footer>OpenScene Mini · Ошибки считаются в системе машины на t · Только офлайн · Изображения и разметка остаются локально.</footer>
</main></body></html>"""


def _section(number, title, help_text, bundle):
    result, index, root = bundle
    summary = result["summary"]
    n = summary["counts"]
    lines = []
    for name in ARMS:
        arm = summary["arms"][name]
        moving, quiet = arm["moving"], arm["quiet"]
        cells = (NAMES[name], arm["all"]["matched"], moving["matched"], moving["confirmed"],
                 moving["verified_confirmed"], f"{quiet['nonzero_velocity']} / {quiet['matched']}",
                 _number(moving["displacement_mean_m"]), _number(moving["future_center_mean_m"]))
        lines.append(_row(cells, cls="baseline" if name == EARLY else ""))
    cols = ("Вариант", "Центры", "Движущиеся", "Подтв. скорости", "Сверены 3 кадра",
            "Скорость у стоящих", "Ошибка смещения, м", "Ошибка центра, м")
    gate = "".join(_gate(summary["comparisons"][name], name) for name in RESCUES)
    causes = _causes(summary["arms"][EARLY]["moving"]["causes"])
    dropout = _coverage_note(summary["cases"])
    cards = "".join(_example(case, index, root) for case in _select_cases(summary["cases"]))
    return f"""<section>
<div class="eyebrow">{number} / {n['windows']} окон · {n['annotated_tracks']} размеченных целей · {n['moving_tracks']} движущихся</div>
<h2>{escape(title)}</h2><p>{escape(help_text)}</p>
{_table(cols, "".join(lines))}
<p class="aside">Ошибки — примерно через 1 с, на целях, найденных соответствующим вариантом. У двух восстановлений состав совпадает с ранним объединением.
«Сверены 3 кадра» — совпадение с одной размеченной целью во всех трёх кадрах по консервативному сопоставлению, а не доказанный ID детекции.</p>
<div class="grid">{gate}</div><h3>Первый барьер для движущихся · раннее объединение</h3>{dropout}{causes}
<h3>Примеры на реальных кадрах</h3><p class="aside">Подобранные примеры, не частота событий. Схемы справа — проекции на дорогу, не рамки на фото.</p>
<div class="examples">{cards or '<p>Нет объектов для показа.</p>'}</div>
<p class="aside hash">Прогноз SHA256: {escape(result['prediction_sha256'])}</p></section>"""


def _gate(comparison, name):
    detail = comparison["gate"]
    checks = "".join(f'<li class="{("ok" if value else "bad")}">{("✓" if value else "—")} {escape(CHECKS[key])}</li>'
                     for key, value in detail["checks"].items())
    moving = comparison["moving"]
    return f"""<article class="panel"><b>{escape(NAMES[name])}</b><h3>{GATES[detail['status']]}</h3>
<p>Δ ошибки смещения: {_number(moving['displacement_change_m'], signed=True)} м.
Новых сверенных цепочек: {moving['verified_gained']}, в {moving['verified_gain_recordings']} поездках.</p>
<ul>{checks}</ul></article>"""


def _table(columns, rows):
    headers = "".join(f"<th>{escape(name)}</th>" for name in columns)
    return f'<div class="scroll"><table><thead><tr>{headers}</tr></thead><tbody>{rows}</tbody></table></div>'


def _row(cells, *, cls=""):
    return f'<tr class="{cls}">' + "".join(f"<td>{escape(str(value))}</td>" for value in cells) + "</tr>"


def _causes(causes):
    values = sorted(causes.items(), key=lambda item: (-item[1], item[0]))
    peak = max((count for _, count in values), default=1)
    rows = "".join(f"""<div class="bar"><span>{escape(_cause(key))}</span>
<meter min="0" max="{peak}" value="{count}">{count}</meter><b>{count}</b></div>""" for key, count in values)
    return f'<div class="bars">{rows}</div>'


def _cause(key):
    if key in CAUSES:
        return CAUSES[key]
    phase, _, state = key.partition("_")
    return f"{PHASES[phase]}: {STATES[state]}" if phase in PHASES and state in STATES else key


def _coverage_note(cases):
    moving = [case for case in cases if case["category"] == "moving"]
    unmatched = [case for case in moving if not case["arms"][EARLY]["matched"]]
    neither_near = sum(
        all(case["arms"][name]["cause"] == "current_missing_center" for name in ("ssdlite", "faster_rcnn"))
        for case in unmatched
    )
    return (f"<p>{len(unmatched)} из {len(moving)} движущихся целей не имеют однозначного текущего совпадения. "
            f"У {neither_near} из них ни один детектор не дал проецированного центра в радиусе 2 м.</p>")


def _select_cases(all_cases):
    changed = [case for case in all_cases if case["arms"]["bridge_rescue"].get("rescue")]
    changed.sort(key=lambda case: (
        not case["arms"]["bridge_rescue"]["verified"],
        case["arms"]["bridge_rescue"]["displacement_error_m"] - case["arms"][EARLY]["displacement_error_m"],
    ))
    losses = [case for case in all_cases if case["category"] == "moving" and case["arms"][EARLY]["matched"]
              and case["arms"][EARLY]["cause"] in ("previous_missing_center", "previous_ambiguous_or_nonmutual_center",
                                                  "older_missing_center", "older_ambiguous_or_nonmutual_center")
              and not case["arms"]["bridge_rescue"].get("rescue")]
    return (changed[:2] + losses[:1]) or [case for case in all_cases if case["arms"][EARLY]["matched"]][:1]


def _example(case, index, root):
    a, b = (case["arms"][name] for name in (EARLY, "bridge_rescue"))
    mode = "Связь восстановлена" if b["rescue"] else "Оценку скорости не меняем"
    camera = b["cameras"][0]
    anchor = next(row for row in index["anchors"] if row["sample_token"] == case["sample_token"])
    frames = "".join(_thumb(index, root, anchor[slot], camera, time)
                     for slot, time in (("older_token", "t−1"), ("previous_token", "t−0,5"), ("sample_token", "t")))
    ep = escape(case["sample_token"][:8])
    return f"""<article class="panel sp"><div><div class="eyebrow">{GROUPS[case['category']]} · {ep}</div>
<h3>{mode}</h3><p>{escape(_cause(a['cause']))} → {escape(_cause(b['cause']))}.</p>
<div class="frames">{frames}</div>
<p class="aside">Камера {escape(camera)} · ошибка смещения: {_number(a['displacement_error_m'])} → {_number(b['displacement_error_m'])} м.</p></div>
{_plot(case['truth'], a, b)}</article>"""


def _thumb(index, root, token, camera, label):
    if token is None:
        return f"<figure><div class='empty'>Кадра нет</div><figcaption>{label}</figcaption></figure>"
    view = next(view for view in index["frames"][token]["views"] if view["camera"] == camera)
    jpeg = read_bounded(root / view["path"], 2 * 1024**2)
    if sha256(jpeg).hexdigest() != view["sha256"]:
        raise ValueError("Report camera JPEG differs from the frozen index")
    with Image.open(BytesIO(jpeg), formats=("JPEG",)) as photo:
        photo.thumbnail((320, 180))
        preview = BytesIO()
        photo.save(preview, format="JPEG", quality=76, optimize=True)
    body = b64encode(preview.getvalue()).decode("ascii")
    return f'<figure><img src="data:image/jpeg;base64,{body}" alt="Камера {escape(camera)}, {label}"><figcaption>{label}</figcaption></figure>'


def _plot(truth, a, b):
    tracks = [
        [truth[f"{phase}_xy_m"] for phase in ("older", "previous", "current", "future")],
        _prediction_path(a), _prediction_path(b),
    ]
    points = [point for track in tracks for point in track if point is not None]
    front, lateral = zip(*points, strict=True)
    mid_x, mid_y = (min(front) + max(front))/2, (min(lateral) + max(lateral))/2
    span = max(max(front)-min(front), max(lateral)-min(lateral), 1.) * 1.25

    def coords(pos):
        x, y = pos
        return 150 - (y - mid_y) * 210 / span, 128 - (x - mid_x) * 210 / span

    lines = []
    for track, color in zip(tracks, PALETTE, strict=True):
        actual = [coords(point) for point in track if point is not None]
        glyph = " ".join(f"{x:.1f},{y:.1f}" for x, y in actual)
        lines.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.2" points="{glyph}"/>')
        for x, y in actual:
            lines.append(f'<circle fill="{color}" cx="{x:.1f}" cy="{y:.1f}" r="3.3"/>')
    legend = "".join(f'<span style="color:{color}">● {text}</span>'
                     for color, text in zip(PALETTE, ("Разметка", "До", "После"), strict=True))
    return f'''<div class="map"><svg viewBox="0 0 300 254" role="img" aria-label="Траектории сверху в координатах машины">
<path d="M45 22V232M97 22V232M150 22V232M203 22V232M255 22V232M45 22H255M45 75H255M45 128H255M45 181H255M45 232H255" stroke="#26405a" fill="none"/>
<text x="47" y="16">вперёд ↑</text><text x="203" y="248">вправо →</text>{''.join(lines)}</svg><div>{legend}</div></div>'''


def _prediction_path(row):
    return [row["history"][phase]["linked_xy_m"] for phase in ("older", "previous")] + [row["center_xy_m"], row["future_xy_m"]]


def _number(value, *, signed=False):
    return "—" if value is None else format(value, "+.3f" if signed else ".3f").replace(".", ",")


_HEAD = """<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MV — разбор трёхкадровых цепочек</title><style>
:root{color-scheme:dark;--bg:#091625;--tile:#122337;--text:#edf5fc;--muted:#a8bdd0;--a:#46dfb5;--edge:#28415a}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:16px/1.5 system-ui,sans-serif}
main{max-width:1160px;margin:auto;padding:36px clamp(16px,4vw,48px) 40px}
header{padding:32px 0 24px;background:radial-gradient(ellipse at 78% 30%,#16364f 0,transparent 63%)}
h1{margin:.13em 0 .18em;font-size:clamp(42px,9vw,90px);letter-spacing:-.065em;line-height:.95}
h2{font-size:clamp(26px,4vw,40px);letter-spacing:-.035em;margin:.18em 0 .25em}h3{font-size:20px;margin:12px 0}
p{margin:.3em 0 .85em;color:var(--muted);max-width:74ch}section{padding:32px 0;border-top:1px solid var(--edge)}
.eyebrow{font-size:.78rem;letter-spacing:.12em;text-transform:uppercase;color:var(--a);font-weight:700}
.hero,.flow{display:flex;flex-wrap:wrap;gap:12px;align-items:center;margin:22px 0 0}.hero span,.flow span{padding:7px 10px;border:1px solid var(--edge);border-radius:9px}
.hero b{color:var(--a);font-size:1.12rem}.hero span{color:var(--muted)}.aside,footer{font-size:.85rem;color:var(--muted)}.hash{overflow-wrap:anywhere}
.panel{padding:18px;border-radius:14px;border:1px solid var(--edge);background:var(--tile)}.grid,.examples{display:grid;gap:12px}.grid{grid-template-columns:repeat(auto-fit,minmax(min(100%,320px),1fr))}
.scroll{overflow-x:auto;margin:16px 0}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{padding:11px 8px;border-bottom:1px solid var(--edge);text-align:right}th{font-size:.8rem;color:var(--muted);vertical-align:bottom}th:first-child,td:first-child{text-align:left}
tr.baseline{background:#1a3144}.sp{display:grid;grid-template-columns:minmax(0,1.8fr) minmax(0,1fr);gap:12px}
.frames{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:4px}figure{margin:0}figure img,.empty{width:100%;aspect-ratio:16/9;border-radius:6px}
.empty{display:grid;place-items:center;background:var(--bg)}figcaption{font-size:.78rem;color:var(--muted)}.map{text-align:center}.map svg{width:100%;max-width:300px}
.map svg text{fill:var(--muted);font:10px system-ui}.map span{margin:0 4px;font-size:.8rem}
ul{list-style:none;margin:0;padding:0;font-size:.85rem}.ok{color:var(--a)}.bad{color:#f7b488}
.bar{display:grid;grid-template-columns:minmax(150px,2fr) minmax(50px,1fr) 24px;gap:10px;align-items:center;font-size:.88rem}
.bars{display:grid;gap:5px;margin-bottom:18px;max-width:760px}meter{width:100%;height:14px}
@media(max-width:680px){.sp{grid-template-columns:1fr}.map{justify-self:center;width:100%}.bar{gap:6px}main{padding-top:14px}}
</style></head><body>"""


if __name__ == "__main__":
    main()
