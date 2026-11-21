"""Standalone research report with no source imagery or per-frame labels."""

from html import escape
import json


def write_spatial_report(root, manifest, result):
    def row(run, regime, metrics, recording="aggregate"):
        def display(key):
            value = metrics[key]
            return "—" if value is None else f"{value:.4f}"

        prefix = "macro_" if recording == "aggregate" else ""
        cells = [
            run["seed"],
            regime,
            recording,
            metrics["frames"],
            metrics["positives"],
            metrics["negatives"],
            display(prefix + "balanced_accuracy"),
            display(prefix + "roc_auc"),
            display(prefix + "positive_iou"),
        ]
        return "<tr>" + "".join(f"<td>{escape(str(cell))}</td>" for cell in cells) + "</tr>"

    summary, record_rows = [], []
    for run in result["runs"]:
        for key, label in (
            ("holdout", run["condition"]),
            ("holdout_shuffled_same_recording", "shuffled image"),
        ):
            if key not in run:
                continue
            metrics = run[key]
            summary.append(row(run, label, metrics))
            record_rows.extend(
                row(run, label, record, name) for name, record in metrics["per_recording"].items()
            )
    header = (
        "<thead><tr><th>Seed</th><th>Condition</th><th>Recording</th><th>Frames</th>"
        "<th>Positives</th><th>Negatives</th><th>Balanced accuracy</th><th>ROC AUC</th><th>Positive IoU</th></tr></thead>"
    )
    title = "OpenScene: spatial grounding on new recordings"
    html = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title>
<style>body{{font:16px system-ui;color:#192a44;background:#f4f7fc;max-width:1250px;margin:28px auto;padding:0 20px}}
section{{background:#fff;padding:18px;margin:16px 0;border:1px solid #dbe4f0;border-radius:10px;overflow:auto}}
h1,h2{{color:#1c4171}}table{{border-collapse:collapse;width:100%}}th,td{{padding:8px;text-align:left;border-bottom:1px solid #dce4ef}}
pre{{overflow-wrap:anywhere;white-space:pre-wrap}}.note{{background:#fff3db;padding:12px;border-left:4px solid #a96d15}}</style>
<h1>{title}</h1><p>Fixed four image queries per current front-camera frame.
Paired training with frozen DINOv2 and the same spatial head, data, initialization and optimizer.
The zero-feature control sees the same spatial coordinate prior but no pixels.</p>
<p class="note">Thresholds use dev recordings only. All weights and thresholds are serialized before final labels are opened.
Boxes are source cuboid projections, not manually annotated visible-pixel masks. Only two test recordings,
correlated frames and a single camera; this is not general detection quality, text/motion alignment or driving safety.</p>
<section><h2>Final holdout — recording-weighted metrics</h2>
<p>Macro presence first averages each class rate across recordings, then averages the two class rates. A recording with only negatives contributes to specificity. Macro IoU/AUC average defined per-recording scores. Positive box IoU includes every positive label, independent of presence prediction.
The shuffled diagnostic changes the image association within each recording, keeping the original threshold.</p>
<table>{header}<tbody>{''.join(summary)}</tbody></table></section>
<section><h2>Individual recordings</h2><table>{header}<tbody>{''.join(record_rows)}</tbody></table></section>
<section><h2>Provenance</h2><p>Status: {escape(result["status"])}</p>
<pre>{escape(json.dumps({"manifest": manifest,"result":result},indent=2))}</pre></section></html>"""
    (root / "report.html").write_text(html, encoding="utf-8")
