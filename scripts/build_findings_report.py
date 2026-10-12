"""Render the findings report and export static scientific figures.

Consumes completed development outputs only. No API calls or model fitting.
"""
import ast
import hashlib
import html
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/banking77-findings"
FIG = OUT / "figures"
ANALYSIS = json.loads((OUT / "analysis.json").read_text(encoding="utf-8"))
TEAL, NAVY, RUST, GRAY = "#087f78", "#254768", "#b34d28", "#657485"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.labelcolor": "#253746", "text.color": "#253746",
    "axes.titleweight": "bold", "axes.titlepad": 14,
    "svg.fonttype": "none", "pdf.fonttype": 42})


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def accuracy():
    rows = [("EmbeddingGemma + LR", "embeddings", "embedding_lr_temperature", TEAL),
            ("EmbeddingGemma + MLP", "embeddings", "embedding_mlp_raw", TEAL),
            ("TF-IDF + LR", "tfidf", "tfidf_temperature", TEAL),
            ("Haiku 5.5", "haiku", "llm_same_raw", RUST),
            ("OpenAI Decisions", "openai", "decisions_confidence_raw", RUST)]
    fig, ax = plt.subplots(figsize=(9, 3.9), layout="constrained")
    for y, (label, run, method, color) in enumerate(rows):
        m = ANALYSIS["all_metrics"][run][method]
        value = m["accuracy"] * 100
        low, high = np.array(m["accuracy_ci95"]) * 100
        ax.errorbar(value, y, xerr=[[value-low], [high-value]], fmt="o", color=color,
                    markersize=8, capsize=4, linewidth=2)
        ax.text(100.9, y, f"{value:.1f}%", va="center", weight="bold", color=color)
    ax.set(yticks=range(5), yticklabels=[r[0] for r in rows], xlim=(65, 105), ylim=(4.6, -0.6),
           xlabel="Development accuracy (%)", title="Frozen embeddings led in the tested operating conditions")
    ax.set_xticks([65, 70, 75, 80, 85, 90, 95, 100])
    ax.axhspan(-.5, 2.5, color=TEAL, alpha=.05)
    ax.axhspan(2.5, 4.5, color=RUST, alpha=.04)
    ax.grid(axis="x", alpha=.15)
    save(fig, "classification_accuracy")


def confidence():
    rows = [("Original confidence", "llm_same_raw", RUST),
            ("Separate verifier", "llm_verifier_raw", NAVY),
            ("Learned correctness", "llm_correctness_raw", TEAL)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), layout="constrained")
    for ax, metric, title, limits in zip(axes, ["brier", "error_auroc"],
        ["Probability error decreased", "Error ranking improved"], [(0, .18), (.75, 1)]):
        for y, (_, method, color) in enumerate(rows):
            m = ANALYSIS["estimates"][method][metric]
            low, high = m["paired_bootstrap_ci95"]
            value = m["estimate"]
            ax.errorbar(value, y, xerr=[[value-low], [high-value]], fmt="o", color=color,
                        markersize=8, capsize=4, linewidth=2)
            ax.annotate(f"{value:.3f}", (value, y), xytext=(0, 12), textcoords="offset points",
                        ha="center", fontsize=10, weight="bold", color=color)
        ax.set(yticks=range(3), yticklabels=[r[0] for r in rows] if metric == "brier" else [],
               ylim=(2.6, -.7), xlim=limits, title=title,
               xlabel="Brier error (lower is better)" if metric == "brier" else "Error AUROC (higher is better)")
        ax.grid(axis="x", alpha=.15)
    save(fig, "haiku_confidence")


def risk():
    rows = [("Embedding LR calibrated", "embeddings", "embedding_lr_temperature", TEAL, "-"),
            ("Haiku original", "anthropic-batch-v2", "llm_same_raw", RUST, "--"),
            ("Haiku learned correctness", "anthropic-batch-v2", "llm_correctness_raw", NAVY, "-"),
            ("OpenAI native confidence", "openai-decisions-v2", "decisions_confidence_raw", GRAY, ":")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), layout="constrained")
    for label, run, method, color, style in rows:
        frame = pd.read_csv(ROOT / "runs" / run / "development" / f"{method}.risk_coverage.csv")
        for ax in axes:
            ax.step(frame.coverage, frame.risk, where="post", label=label, color=color,
                    linestyle=style, linewidth=1.7)
    for ax, top, title in zip(axes, [.30, .025], ["Full risk range", "Zoom on low observed error rates"]):
        ax.axhline(.01, color="#aa3636", linewidth=1, linestyle="--", zorder=0)
        ax.set(xlim=(0, 1), ylim=(0, top), xlabel="Fraction accepted automatically", title=title)
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0 if top == .30 else 1))
        ax.grid(alpha=.15)
    axes[0].set_ylabel("Observed error among accepted predictions")
    axes[0].legend(loc="upper left", frameon=True, fontsize=8)
    axes[1].scatter([.895], [3/895], color=TEAL, zorder=5, s=40)
    axes[1].annotate("895 accepted, 3 errors", (.895, 3/895), xytext=(.28, .013),
        arrowprops={"arrowstyle": "->", "color": TEAL}, fontsize=9, color=TEAL)
    axes[1].text(.03, .0105, "1% observed error reference", color="#aa3636", fontsize=8)
    save(fig, "risk_coverage")


def reliability():
    frame = pd.read_csv(ROOT / "runs/embeddings/development/embedding_lr_temperature.reliability.csv")
    intervals = np.array([ast.literal_eval(s) for s in frame.accuracy_ci95])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    ax = axes[0]
    ax.plot([0, 1], [0, 1], linestyle="--", color=GRAY, alpha=.7, linewidth=1)
    ax.errorbar(frame.confidence, frame.accuracy,
        yerr=[frame.accuracy - intervals[:, 0], intervals[:, 1] - frame.accuracy],
        fmt="o", color=TEAL, capsize=3, elinewidth=1.3, markersize=5)
    for i, row in frame.iterrows():
        # Alternating labels keep the sparse neighboring bins readable.
        offset = (8, -16) if i in [1, 3, 5] else (6, 6)
        if i == len(frame)-1:
            offset = (-58, 4)
        ax.annotate(f"n={int(row.n)}", (row.confidence, row.accuracy), xytext=offset,
                    textcoords="offset points", fontsize=8)
    ax.set(xlim=(0, 1.025), ylim=(0, 1.04), xlabel="Mean confidence in bin",
           ylabel="Observed fraction correct", title="Sparse bins have wide uncertainty")
    ax.grid(alpha=.15)
    ax = axes[1]
    counts = np.zeros(10, dtype=int)
    counts[frame.bin.to_numpy(dtype=int)] = frame.n.to_numpy()
    ax.bar((np.arange(10)+.5)/10, counts, width=.085, color=TEAL, alpha=.8)
    for i, n in enumerate(counts):
        if n:
            ax.text((i+.5)/10, n*1.12, str(n), ha="center", fontsize=9)
    ax.set(yscale="log", ylim=(1, 1600), xlim=(0, 1), xlabel="Confidence bin",
           ylabel="Examples per bin (log scale)", title="899 predictions lie in the final bin")
    ax.set_xticks(np.arange(0, 1.01, .2))
    ax.grid(axis="y", alpha=.15)
    save(fig, "reliability_counts")


CAPTIONS = {
    "classification_accuracy": "Figure 1. The same 1,000 development messages. Whiskers are marginal 95% Wilson accuracy intervals. Teal models used 5,992 labeled fitting examples; rust models used zero-shot intent-name prompts. This is not a comparison at equal task-specific training budgets.",
    "haiku_confidence": "Figure 2. The predictions are identical across all three methods. Whiskers are 95% bootstrap intervals conditional on fitted models, using 5,000 shared example resamples. Paired differences reported in the text, rather than overlap between marginal whiskers, assess improvement. Learned correctness also uses supervised evidence and additional calls.",
    "risk_coverage": "Figure 3. Descriptive development risk at each observed score threshold; tied scores enter together. The right panel expands the low-risk region and clips higher-risk curves. The dashed 1% line refers to observed error, whereas acceptance selection uses a Wilson upper endpoint. The highlighted threshold was selected on these data and requires independent validation.",
    "reliability_counts": "Figure 4. Temperature-scaled embedding LR. Points are equal-width bin averages with 95% Wilson intervals and example counts; empty bins are omitted. No connecting line or smoothing is used. The right panel uses a logarithmic count axis to show both sparse bins and the 899-example final bin.",
}
INSERT_BEFORE = {
    "Learning correctness materially improved Haiku confidence": "classification_accuracy",
    "Better error ranking did not guarantee low-risk automation": "haiku_confidence",
    "Calibration helped selectively": "risk_coverage",
    "What the findings justify next": "reliability_counts",
}


def inline(text):
    text = html.escape(text)
    text = re.sub(r"\[([^\]]+)\]\((https://[^\s)]+)\)", r'<a href="\2">\1</a>', text)
    return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)


def figure(name):
    svg = (FIG / f"{name}.svg").read_text(encoding="utf-8")
    svg = svg[svg.index("<svg"):]
    svg = svg.replace("<svg ", '<svg role="img" aria-label="' + html.escape(CAPTIONS[name], quote=True) + '" ', 1)
    return f'<figure>{svg}<figcaption>{html.escape(CAPTIONS[name])}</figcaption></figure>'


def render(markdown):
    pieces, navigation = [], []
    blocks = re.split(r"\n\s*\n", markdown.strip())
    for block in blocks:
        if block.startswith("# "):
            pieces.append(f'<header><p class="eyebrow">CONFIDENCE EXPERIMENT / RESEARCH REPORT</p><h1>{inline(block[2:])}</h1></header>')
        elif block.startswith("## "):
            title = block[3:]
            if title in INSERT_BEFORE:
                pieces.append(figure(INSERT_BEFORE[title]))
            anchor = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
            navigation.append(f'<a href="#{anchor}">{html.escape(title)}</a>')
            pieces.append(f'<h2 id="{anchor}">{inline(title)}</h2>')
        elif block.startswith("|"):
            rows = [[c.strip() for c in line.strip().strip("|").split("|")] for line in block.splitlines()]
            assert len(rows) > 2 and all(len(r) == len(rows[0]) for r in rows)
            align = ["right" if c.endswith(":") else "left" for c in rows[1]]
            heading = "".join(f'<th scope="col" style="text-align:{a}">{inline(c)}</th>' for a, c in zip(align, rows[0]))
            body = "".join("<tr>" + "".join(f'<td style="text-align:{a}">{inline(c)}</td>' for a, c in zip(align, row)) + "</tr>" for row in rows[2:])
            pieces.append(f'<div class="table-wrap"><table><thead><tr>{heading}</tr></thead><tbody>{body}</tbody></table></div>')
        else:
            pieces.append(f'<p>{inline(block)}</p>')
    return "\n".join(pieces), "\n".join(navigation)


STYLE = """
:root{color-scheme:light;--ink:#203644;--muted:#526572;--teal:#087f78;--paper:#fffefa}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:#edf1f1;color:var(--ink);font:17px/1.7 Georgia,'Times New Roman',serif}
.topbar{max-width:1060px;margin:auto;padding:20px 30px;display:flex;justify-content:space-between;align-items:center;font:13px/1.4 'Segoe UI',sans-serif;color:var(--muted)}
button{border:1px solid #bcc9cc;background:white;border-radius:5px;padding:8px 14px;cursor:pointer;color:var(--ink)}button:focus-visible,a:focus-visible{outline:3px solid #087f78;outline-offset:4px}
main{max-width:1060px;margin:0 auto 50px;background:var(--paper);padding:54px 90px 64px;border-top:6px solid var(--teal);box-shadow:0 8px 40px #2036440b}
header{margin-bottom:26px}.eyebrow{font:700 11px/1.5 'Segoe UI',sans-serif;letter-spacing:2px;color:var(--teal);margin:0 0 22px}
h1{font:700 clamp(34px,5vw,56px)/1.09 'Segoe UI',sans-serif;letter-spacing:-1.6px;max-width:750px;margin:0}
h2{font:650 25px/1.25 'Segoe UI',sans-serif;letter-spacing:-.4px;margin:52px 0 18px;scroll-margin-top:20px;border-top:1px solid #dbe3e1;padding-top:25px;break-after:avoid}
p{margin:0 0 20px;overflow-wrap:anywhere}header+p{font:12px/1.6 'Segoe UI',sans-serif;color:var(--muted);text-transform:uppercase;letter-spacing:1px;margin-bottom:32px}
strong{font-weight:700}a{color:#086e69;text-decoration-thickness:1px;text-underline-offset:3px}
nav{font:13px/1.6 'Segoe UI',sans-serif;border-top:1px solid #dbe3e1;margin-top:36px;padding-top:20px;display:grid;grid-template-columns:1fr 1fr;gap:7px 20px}nav a{text-decoration:none;color:var(--muted)}
.table-wrap{overflow-x:auto;margin:24px 0 25px}table{width:100%;border-collapse:collapse;font:13px/1.5 'Segoe UI',sans-serif;font-variant-numeric:tabular-nums}th{background:#e8f1ee;font-weight:650;padding:12px 10px;border-bottom:2px solid #aac8c0;vertical-align:bottom}td{padding:12px 10px;border-bottom:1px solid #dde5e2;vertical-align:top}tbody tr:nth-child(even){background:#f6f8f5}td:first-child{min-width:190px}
figure{margin:30px -22px 36px;padding:18px 15px 16px;background:white;border:1px solid #dbe4e0;break-inside:avoid}figure svg{display:block;width:100%;height:auto}figcaption{padding:12px 10px 0;font:12px/1.55 'Segoe UI',sans-serif;color:var(--muted)}
footer{margin-top:36px;padding-top:20px;border-top:1px solid #dbe3e1;font:12px/1.6 'Segoe UI',sans-serif;color:var(--muted)}
@media(max-width:700px){main{padding:30px 22px;box-shadow:none}body{font-size:16px}.topbar{padding:15px 22px}h1{letter-spacing:-1px}h2{font-size:23px}figure{margin:25px -12px}table{font-size:12px}nav{grid-template-columns:1fr}}
@media print{@page{size:A4;margin:16mm 15mm}body{background:white;font-size:10.5pt;line-height:1.5}.topbar,nav{display:none}main{padding:0;border:0;margin:0;max-width:none;box-shadow:none}h1{font-size:31pt}h2{font-size:16pt;margin-top:26px}p{margin-bottom:12px}figure{margin:18px 0;padding:10px}table{font-size:9pt}th,td{padding:7px}a{color:inherit}footer{font-size:9pt}}
"""


def main():
    # Completed extensions belong in the main report on every subsequent build.
    if (OUT / "sonnet-comparison.json").exists() and (ROOT / "reports/calibrator-comparison/analysis.json").exists():
        from consolidate_findings_report import main as consolidate
        return consolidate()
    for relative, expected in ANALYSIS["source_sha256"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected, relative
    accuracy()
    confidence()
    risk()
    reliability()
    markdown = (OUT / "report.md").read_text(encoding="utf-8")
    body, navigation = render(markdown)
    output = '<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Learning When to Trust a Classifier · BANKING77 findings</title><style>' + STYLE + '</style></head><body><div class="topbar"><span>BANKING77 · 9 October 2026</span><button onclick="window.print()">Print or save as PDF</button></div><main>' + body + '<nav aria-label="Report sections">' + navigation + '</nav><footer>Evidence package: <a href="analysis.json">analysis and source hashes</a> · <a href="report.md">Markdown source</a> · <a href="../../scripts/analyze_findings.py">analysis script</a> · <a href="../../scripts/build_findings_report.py">figure and report builder</a><br>All findings use development data. The official test set remains reserved for independent validation.</footer></main></body></html>'
    (OUT / "report.html").write_text(output, encoding="utf-8")
    (OUT / "figure_captions.json").write_text(json.dumps(CAPTIONS, indent=2), encoding="utf-8")
    from add_probability_comparisons import append_probability_comparisons
    append_probability_comparisons(OUT / "report.html")
    print(f"Verified {len(ANALYSIS['source_sha256'])} source hashes. Exported six figures as PNG, SVG and PDF.")
    print(f"Report: {OUT / 'report.html'} ({(OUT / 'report.html').stat().st_size:,} bytes; self-contained figures)")


if __name__ == "__main__":
    main()
