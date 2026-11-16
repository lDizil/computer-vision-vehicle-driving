"""Portable HTML evidence, not a scored model benchmark."""

import base64
from html import escape
import json


def write_real_report(path, bundle, manifest, records, result):
    def text(value):
        return escape(str(value))

    def image(name):
        encoded = base64.b64encode(path.with_name(name).read_bytes()).decode("ascii")
        return "data:image/png;base64," + encoded

    views = []
    for view in bundle.views:
        name = view.frame.name
        views.append(f'<article><h3>{text(name)}</h3><img alt="Rectified RGB {text(name)}" '
                     f'src="{image(name + "-rectified.png")}"><img alt="Fast path {text(name)}" '
                     f'src="{image(name + "-overlay.png")}"><p>Сверху — rectified RGB. Снизу — тот же кадр + Fast-путь.</p></article>')
    rows = "".join(f'<tr><td>{row["seed"]}</td><td>{text(row["outcome"])}</td>'
                   f'<td>{text(row["status"] or "—")}</td><td>{row["latency_s"]:.3f} с</td></tr>' for row in records)
    traces = "".join(f'<h3>Seed {row["seed"]}</h3><p>{text(row.get("explanation") or row.get("error_type") or "Нет результата")}</p>'
                     f'<pre>{text(row["raw_response"] or "Нет ответа модели")}</pre>' for row in records)
    provenance = json.dumps({**manifest, "result": result}, ensure_ascii=False, indent=2)
    body = f"""<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MV · Real path advisory</title><style>
:root{{color-scheme:dark;font:16px/1.5 system-ui;background:#101720;color:#dfe8f3}}body{{max-width:1440px;margin:auto;padding:32px}}
h1{{font-size:36px;line-height:1.2}}h2{{margin-top:36px}}.badge{{color:#45d6df}}.grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px}}
article{{padding:16px;border:1px solid #354455;border-radius:14px;background:#182330}}img{{width:100%;height:auto;display:block;margin-bottom:10px}}
p{{color:#b7c8db}}table{{border-collapse:collapse;width:100%}}td,th{{padding:12px;border-bottom:1px solid #354455;text-align:left}}
pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#182330;padding:16px;border-radius:12px;font-size:13px}}code{{overflow-wrap:anywhere}}
@media(max-width:900px){{.grid{{grid-template-columns:1fr}}body{{padding:16px}}}}
</style><header><span class="badge">MV / REAL FRAME / ADVISORY ONLY</span><h1>Настоящие камеры → Fast-путь → Slow</h1>
<p>Сцена <code>{text(bundle.scene_token)}</code> · один настоящий LTF-кандидат · {bundle.trajectory.num_poses} точек / {bundle.trajectory.horizon_s:g} с</p></header>
<section><h2>Что смотрим</h2><p>Бирюзовая линия — предложенный Fast-путь. Slow описывает только видимое пересечение с препятствием.
Результат не меняет траекторию и не поступает в Safety Veto. NO_VETO не означает безопасную дорогу.</p>
<div class="grid">{''.join(views)}</div></section>
<section><h2>Ответы Slow</h2><table><thead><tr><th>Seed</th><th>Выполнение</th><th>Advisory</th><th>Холодный запуск + ответ</th></tr></thead><tbody>{rows}</tbody></table>
<p>Это повторы одного кадра, не независимые дорожные сценарии. Разметки для оценки качества здесь нет.</p>{traces}</section>
<section><h2>Геометрия и время</h2><p>Исходную дисторсию исправляем один раз до resize; итоговые PNG уже rectified.
XY-путь поднят на плоскость z=0 текущей rear-axle системы. Высота дороги, roll/pitch и движение между экспозициями не восстановлены.
Время кадра известно; индивидуальное время съёмки камер неизвестно. Проекция — диагностическая, не точная граница безопасности.</p></section>
<section><h2>Проверяемость</h2><p>Bundle <code>{text(bundle.bundle_hash)}</code>. Модель получает только текущие RGB с линией;
future labels, reference trajectories и evaluator-метрики в её вход не попадают.</p><pre>{text(provenance)}</pre></section></html>"""
    path.write_text(body, encoding="utf-8")
