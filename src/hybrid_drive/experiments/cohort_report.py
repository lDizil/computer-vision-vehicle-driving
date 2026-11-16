"""Self-contained descriptive report of fixed real-scene offline experiments."""

import base64
from html import escape
import json
from pathlib import Path

from .bounded_io import read_bounded
from .offline_evidence import load_cohort_evidence_index
from .cohort_fast import load_fast_cohort


def write_cohort_report(run_dir, result, fast_run_root, evidence_root):
    run_dir, evidence_root = Path(run_dir), Path(evidence_root)
    cohort = load_fast_cohort(fast_run_root)
    try:
        index = load_cohort_evidence_index(evidence_root, cohort_hash=cohort["cohort_hash"])
    except (OSError, ValueError):
        index = None
    records = result["records"]

    def text(value):
        return escape(str(value))

    def number(value):
        return "—" if value is None else f"{value:.4f}"

    def image(path, caption):
        if not path.is_file():
            return f'<p>{text(caption)}: нет изображения</p>'
        data = base64.b64encode(read_bounded(path, 2 * 1024 * 1024)).decode("ascii")
        return f'<figure><img alt="{text(caption)}" src="data:image/png;base64,{data}"><figcaption>{text(caption)}</figcaption></figure>'

    summaries = []
    for mode, summary in result["summary"].items():
        metrics = summary["metrics_mean_valid_only"]
        summaries.append(f'<tr><td>{text(mode)}</td><td>{summary["valid"]}/{summary["attempted"]}</td>'
                         f'<td>{number(summary["score_mean_all_attempts_failure_zero"])}</td>'
                         f'<td>{number(metrics["no_at_fault_collisions"])}</td><td>{number(metrics["ego_progress"])}</td>'
                         f'<td>{number(summary["ade_m_valid_only"])}</td><td>{number(summary["fde_m_valid_only"])}</td></tr>')
    cards = []
    for i, case in enumerate(cohort["cases"]):
        evidence = None if index is None else index["cases"][i]
        rows = [row for row in records if row["case_id"] == case["case_id"]]
        table = "".join(f'<tr><td>{text(row["mode"])}</td><td>{text(row["status"])}</td>'
                        f'<td>{number((row["metrics"] or {}).get("score"))}</td><td>{text(row.get("slow_status", "—"))}</td>'
                        f'<td>{text(row.get("error") or row.get("fallback_reason") or "—")}</td></tr>' for row in rows)
        overlay = '<p>Нет native evidence.</p>'
        if evidence is not None and evidence["evidence_path"]:
            overlay = image(evidence_root / evidence["evidence_path"] / "CAM_F0.png", "Текущая FRONT-камера и frozen Fast-путь")
        plot = image(run_dir / case["case_id"] / "trajectory.png", "Выходы трёх режимов и recorded human trajectory — только evaluator/report")
        cards.append(f'<article><h2>{text(case["case_id"])} · source index {case["source_list_index"]}</h2>'
                     f'<p><code>{text(case["scene_token"])}</code></p><div class="images">{overlay}{plot}</div>'
                     f'<table><tr><th>Режим</th><th>Выбор</th><th>PDMS</th><th>Slow</th><th>Ошибка / fallback</th></tr>{table}</table></article>')
    audit = json.dumps({key: value for key, value in result.items() if key != "records"}, ensure_ascii=False, indent=2)
    csv_data = base64.b64encode((run_dir / "comparison.csv").read_bytes()).decode("ascii")
    html = f"""<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MV · Six real scenes</title><style>
:root{{color-scheme:dark;font:16px/1.5 system-ui;background:#101720;color:#e2ebf6}}body{{max-width:1400px;margin:auto;padding:32px}}
h1{{font-size:36px}}h2{{margin:0 0 12px}}p,figcaption{{color:#b5c7db}}a{{color:#40d0df}}code,pre{{overflow-wrap:anywhere}}
article{{border:1px solid #3a495a;background:#172330;border-radius:14px;padding:20px;margin:24px 0}}.images{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}
figure{{margin:0}}img{{width:100%;height:auto}}table{{width:100%;border-collapse:collapse;font-size:14px}}td,th{{text-align:left;padding:10px;border-bottom:1px solid #3a495a}}
pre{{white-space:pre-wrap;background:#172330;padding:20px;border-radius:12px;font-size:12px}}.table{{overflow:auto}}
@media(max-width:800px){{body{{padding:12px}}.images{{grid-template-columns:1fr}}}}
</style><header><p>MV / SIX FIXED FRAMES / OFFLINE ONLY</p><h1>Fast + проверки + Slow: реальный cohort</h1>
<p>Шесть непересекающихся окон одного recording. Кадры зафиксированы до моделей по temporal-valid правилу.
Это descriptive debug, не независимый validation split и не демонстрация безопасного управления.</p></header>
<section><h2>Итог по 18 запланированным записям</h2><p>PDMS — официальный NAVSIM v1.1. Для среднего PDMS ошибки получают явный штраф 0;
NC, progress и ADE/FDE усредняем только по valid результатам. NC — оценка No At-Fault Collisions, не количество измеренных аварий.</p>
<div class="table"><table><tr><th>Режим</th><th>Valid / planned</th><th>PDMS (все)</th><th>NC (valid)</th><th>Progress (valid)</th><th>ADE, м</th><th>FDE, м</th></tr>{''.join(summaries)}</table></div>
<p>Эффект Slow смотрим как combined − checks_only при неизменных обязательных проверках. Slow-only намеренно не исполняем.</p>
<a download="comparison.csv" href="data:text/csv;base64,{csv_data}">Скачать CSV</a></section>
<section>{''.join(cards)}</section><section><h2>Время и границы</h2><p>Linux воспроизводит frozen counterfactual decision в исходном native clock domain.
Это не свежий runtime-ответ: simulation time не продвигается. Автоматическое veto в реальный агент не включено.
Проекция XY→z=0 приблизительная; индивидуальное время камер неизвестно. Будущие reference labels появляются только после фиксации всех трёх выходов.</p>
<h2>Проверяемость</h2><pre>{text(audit)}</pre></section></html>"""
    (run_dir / "report.html").write_text(html, encoding="utf-8")
