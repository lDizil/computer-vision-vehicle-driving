"""Small self-contained summary; raw images and training annotations stay local."""

import html
import json
import math


def write_alignment_report(root, manifest, result, trace):
    def number(value):
        return "—" if value is None else f"{value:.4f}"

    rows = []
    for arm, measured in result["arms"].items():
        for key in ("holdout_before", "holdout_after", "train_after"):
            metrics = measured[key]
            cells = (
                arm,
                key,
                metrics["examples"],
                metrics["positives"],
                number(metrics["loss"]),
                number(metrics["presence_balanced_accuracy"]),
                number(metrics["forced_choice_yes_no_accuracy"]),
                number(metrics["positive_projected_box_iou"]),
                number(metrics["positive_center_l2_m"]),
            )
            rows.append("<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in cells) + "</tr>")
    header = (
        "Condition",
        "Partition / checkpoint",
        "Frames",
        "Positive labels",
        "LOC loss",
        "Presence balanced acc.",
        "Text yes/no choice acc.",
        "Positive 2D IoU",
        "Positive center L2, m",
    )
    html_table = "<table><thead><tr>" + "".join(f"<th>{cell}</th>" for cell in header) + "</tr></thead>"
    html_table += "<tbody>" + "".join(rows) + "</tbody></table>"
    error = f"<p>{html.escape(result['error'])}</p>" if result.get("error") else ""
    content = """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>OpenScene LOC alignment pilot</title>
<style>body{font:16px system-ui;background:#f4f7fc;color:#182a45;max-width:1240px;margin:30px auto;padding:0 20px}
section{background:white;border:1px solid #d7e2f2;border-radius:12px;padding:20px;margin:20px 0}
h1,h2{color:#193e72}.scroll{overflow:auto}table{border-collapse:collapse;width:100%}td,th{padding:12px 8px;
border-bottom:1px solid #dbe4f0;text-align:left;white-space:nowrap}pre{white-space:pre-wrap;overflow-wrap:anywhere}
.note{background:#fff3d9;border-left:4px solid #b67d13;padding:12px}</style>
<h1>OpenScene LOC alignment pilot</h1>
<p>Frozen pretrained DINOv2 and Pythia with a learnable prior tokenizer, query bridge and LOC decoder.
The two arms start from the same weights and train on the same examples and batch order. The control
replaces camera features with zeros, keeping identical geometry priors, prompt and supervision.</p>
<p class="note">Labels mean the nearest eligible source annotation in the specified camera region.
Projected 2D targets are not human visible-pixel boxes. A different recording is held out from optimization
and checkpoint selection. This pilot is not full Stage 1, MOT, VQA, driving quality or a safety assessment.</p>
<section><h2>Measured LOC metrics</h2>
<p>Regression uses positive examples, regardless of predicted validity. Text accuracy is forced yes/no
choice, not free-form generation. Loss terms are weighted by their supervised counts. A missing
denominator appears as —.</p><div class="scroll">"""
    content += html_table + "</div>" + _class_count_note(result) + error + "</section>"
    content += (
        "<section><h2>Optimizer trace</h2><p>Each point is one minibatch; losses vary with target mix.</p>"
    )
    recipe = manifest["recipe"]
    steps_per_arm = math.ceil(manifest["planned_examples"]["train"] / recipe["batch_size"]) * recipe["epochs"]
    content += _loss_chart(trace, steps_per_arm) + "</section>"
    content += "<section><h2>Provenance and checks</h2><pre>"
    content += html.escape(json.dumps({"manifest": manifest, "result": result}, indent=2))
    content += "</pre></section></html>"
    (root / "report.html").write_text(content, encoding="utf-8")


def _class_count_note(result):
    if not result["arms"]:
        return ""
    holdout = next(iter(result["arms"].values()))["holdout_after"]
    positive, negative = holdout["positives"], holdout["negatives"]
    note = f"<p>Holdout: {positive} positives and {negative} negatives from one different recording."
    if 0 < min(positive, negative) < 5:
        change = 50 / min(positive, negative)
        note += f" A single minority-class outcome changes balanced accuracy by {change:g} percentage points."
    return note + "</p>"


def _loss_chart(trace, steps_per_arm):
    if not trace:
        return "<p>No optimizer steps completed.</p>"
    upper = max(max(row["loss"] for row in trace), 1) * 1.05
    series = (
        ("image_features", "#245cbb", "Image features"),
        ("constant_zero_features", "#ae690d", "Zero features"),
    )
    svg = (
        '<svg viewBox="0 0 980 260" style="width:100%;max-width:980px;font:13px system-ui" '
        'role="img" aria-label="Optimizer losses for image and zero-feature arms">'
        '<path d="M70 42v165h865" fill="none" stroke="#7b8ca5"/>'
    )
    for i, (arm, color, label) in enumerate(series):
        losses = [row["loss"] for row in trace if row["arm"] == arm]
        if not losses:
            continue
        points = " ".join(
            f"{70 + 865 * n / max(1, steps_per_arm - 1):.2f},{207 - 165 * value / upper:.2f}"
            for n, value in enumerate(losses)
        )
        svg += f'<polyline points="{points}" stroke="{color}" stroke-width="2" fill="none"/>'
        svg += f'<text x="{85 + 160*i}" y="25" fill="{color}">{label}</text>'
    return (
        svg + f'<text x="5" y="50">{upper:.2f}</text><text x="43" y="208">0</text>'
        '<text x="380" y="236">Optimizer step within each condition</text></svg>'
    )
