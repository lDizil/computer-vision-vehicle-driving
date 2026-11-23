"""Standalone aggregate report for the fixed/learned-prior comparison."""

from html import escape
import json


def write_proposal_report(root, manifest, result):
    rows = []
    per_recording = []
    for arm in result["runs"]:
        cases = (
            (arm["condition"], "holdout"),
            ("shuffled proposals (same frame)", "holdout_shuffled_rois_same_frame"),
        )
        for label, key in cases:
            if key in arm:
                rows.append(_row(arm["seed"], label, arm[key]))
                for name, decoder in arm[key]["decoder"]["per_recording"].items():
                    language = arm[key]["text"]["per_recording"][name]
                    per_recording.append(
                        _row(arm["seed"], label, {"decoder": decoder, "text": language}, name)
                    )
    tests = (
        "<tr><th>Seed</th><th>Condition</th><th>Recording</th><th>Frames</th><th>Positive queries</th><th>Decoder BA</th>"
        "<th>Decoder ROC AUC</th><th>Box IoU</th><th>Yes/no BA</th><th>Yes/no ROC AUC</th></tr>"
    )
    title = "Proposal alignment on the reused OpenScene reference"
    page = f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport"
content="width=device-width,initial-scale=1"><title>{title}</title>
<style>body{{font:16px system-ui;color:#192b43;background:#f5f8fc;max-width:1240px;margin:26px auto;padding:0 18px}}
h1,h2{{color:#1b4779}}section{{background:white;padding:18px;margin:16px 0;border:1px solid #d8e6f1;border-radius:10px;overflow:auto}}
table{{border-collapse:collapse;width:100%}}td,th{{padding:8px;text-align:left;border-bottom:1px solid #dce6ef}}
pre{{white-space:pre-wrap;overflow-wrap:anywhere}}.note{{background:#fff4dd;padding:12px;border-left:4px solid #af761b}}</style>
<h1>{title}</h1>
<p>Frozen DINO and Pythia. Both arms see the exact same whole-frame feature grid and same number of RoIs.
The treatment uses four ordered proposals from the seed-0 spatial checkpoint, with no confidence gating.
The control uses four fixed full-height quarters. Shuffling changes only RoIs, within each recording.</p>
<p class="note">The two reference recordings were used in the earlier spatial study, not a fresh test set.
Nothing here enables the runtime veto or evaluates driving safety. Token scores are yes/no comparisons after consumed LOC markers, not free-form answers.</p>
<section><h2>Reference holdout — recording-macro metrics</h2>
<p>All weights and both dev thresholds per run are fixed before opening the reference labels.
BA is balanced accuracy. Positive box IoU counts all positive labels, regardless of predicted presence.</p>
<table><thead>{tests}</thead><tbody>{''.join(rows)}</tbody></table></section>
<section><h2>Individual recordings</h2>
<table><thead>{tests}</thead><tbody>{''.join(per_recording)}</tbody></table></section>
<section><h2>Recorded run</h2><p>Status: {escape(result["status"])}</p>
<pre>{escape(json.dumps({"manifest": manifest, "result": result}, indent=2))}</pre></section></html>"""
    target = root / "report.html"
    temporary = root / "report.html.tmp"
    try:
        temporary.write_text(page, encoding="utf-8")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def _row(seed, label, metrics, recording=None):
    decoder, language = metrics["decoder"], metrics["text"]
    prefix = "" if recording is not None else "macro_"
    cells = [
        str(seed),
        label,
        recording or "macro",
        str(decoder["frames"]),
        str(decoder["positives"]),
        _value(decoder[prefix + "balanced_accuracy"]),
        _value(decoder[prefix + "roc_auc"]),
        _value(decoder[prefix + "positive_iou"]),
        _value(language[prefix + "balanced_accuracy"]),
        _value(language[prefix + "roc_auc"]),
    ]
    return "<tr>" + "".join(f"<td>{escape(cell)}</td>" for cell in cells) + "</tr>"


def _value(value):
    return "—" if value is None else f"{value:.4f}"
