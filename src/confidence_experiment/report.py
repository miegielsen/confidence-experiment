import html
import io
import base64
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .io import write_json


def save_report(output, results, metadata, evaluation_role):
    output = Path(output)
    write_json(output / "metrics.json", results)
    write_json(output / "metadata.json", metadata)
    rows = []
    for method, result in results.items():
        rows.append({"method": method, **{k: result[k] for k in ("accuracy", "macro_f1", "brier", "binary_log_loss", "error_auroc", "error_average_precision", "aurc")},
                     "accepted_coverage": result["acceptance"]["coverage"], "accepted_risk": result["acceptance"]["risk"]})
    table = pd.DataFrame(rows)
    table.to_csv(output / "summary.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for method in results:
        reliability = pd.read_csv(output / f"{method}.reliability.csv")
        curve = pd.read_csv(output / f"{method}.risk_coverage.csv")
        intervals = reliability.accuracy_ci95.map(json.loads)
        # CSV round trips can put a Wilson endpoint a few ulps across 0 or 1.
        lower = [max(0.0, a - interval[0]) for a, interval in zip(reliability.accuracy, intervals)]
        upper = [max(0.0, interval[1] - a) for a, interval in zip(reliability.accuracy, intervals)]
        axes[0].errorbar(reliability.confidence, reliability.accuracy, yerr=[lower, upper],
                        fmt="o-", capsize=2, alpha=0.8, label=method)
        for row in reliability.itertuples():
            axes[0].annotate(str(row.n), (row.confidence, row.accuracy),
                             xytext=(3, 4), textcoords="offset points", fontsize=6)
        axes[1].plot(curve.coverage, curve.risk, label=method)
    axes[0].plot([0, 1], [0, 1], "--", color="gray")
    axes[0].set(xlabel="Mean predicted correctness", ylabel="Observed correctness", title="Reliability (95% Wilson intervals; labels = bin counts)", xlim=(0, 1), ylim=(0, 1))
    axes[1].set(xlabel="Accepted fraction", ylabel="Error rate of accepted predictions", title="Risk–coverage", xlim=(0, 1))
    axes[1].axhline(metadata["config"]["max_error"], color="gray", linestyle="--")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "plots.png", dpi=150)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=120)
    plt.close(fig)
    image = base64.b64encode(buffer.getvalue()).decode()
    notes = ("Development results. Thresholds are selected on these same development examples; acceptance results are optimistic. Official test results have not been inspected."
             if evaluation_role == "development" else "Official test evaluation with previously frozen models, calibration and development thresholds.")
    page = f"""<!doctype html><html lang="en"><meta charset="utf-8"><title>Confidence experiment</title>
<style>body{{font:16px system-ui;max-width:1350px;margin:40px auto;padding:0 20px;color:#18242e}}table{{border-collapse:collapse;font-size:13px}}td,th{{padding:9px;border-bottom:1px solid #ddd;text-align:right}}img{{max-width:100%}}pre{{white-space:pre-wrap}}</style>
<h1>BANKING77 confidence experiment</h1><p>{notes}</p>
<p>Confidence always means probability that the selected label is correct. Lower Brier, log loss and AURC are better; higher error AUROC/AP are better.</p>
{table.to_html(index=False, float_format=lambda x: f'{x:.4f}', border=0)}
<img alt="Reliability and risk coverage plots" src="data:image/png;base64,{image}">
<p>Wilson intervals describe binomial sampling uncertainty under an independent-example approximation. Development threshold search is adaptive; its intervals are not a guarantee of the target error rate. Test intervals are descriptive, without simultaneous multiple-method correction.</p>
<p>The CPU TF-IDF control is an additional baseline. It is not the frozen embedding approach from the experiment brief.</p>
<details><summary>Run metadata</summary><pre>{html.escape((output / 'metadata.json').read_text(encoding='utf-8'))}</pre></details></html>"""
    (output / "report.html").write_text(page, encoding="utf-8")
    return table
