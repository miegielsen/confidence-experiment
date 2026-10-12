"""Add probability diagnostics from completed development predictions only."""
import argparse
import hashlib
import html
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from confidence_experiment.io import write_json
from confidence_experiment.metrics import wilson

ROOT = Path(__file__).resolve().parents[1]
TEAL, RUST, GRAY = "#087f78", "#b34d28", "#657485"
PAIRS = [
    ("Embedding LR", "embeddings", "embedding_lr_raw", "Raw probabilities", "embedding_lr_temperature", "Temperature scaled"),
    ("Haiku 5.5", "anthropic-batch-v2", "llm_same_raw", "Original confidence", "llm_correctness_raw", "Learned correctness"),
    ("OpenAI Decisions", "openai-decisions-v2", "decisions_probability_raw", "Native selected-label probability", "decisions_probability_temperature", "Temperature transformed"),
]
CAPTIONS = {
    "probability_reliability": "Reliability comparisons on the same 1,000 development examples. Each row holds the classifier's selected labels fixed. Points show mean reported correctness probability versus observed fraction correct in ten shared equal-width bins; whiskers are nominal 95% Wilson intervals, without multiple-bin adjustment. The diagonal indicates perfect calibration. Count bars below each panel use a logarithmic axis and show every nonempty bin's sample size. Empty bins are omitted; no smoothing or connecting lines are used. OpenAI temperature inputs are log returned probabilities, not native logits. Haiku and Sonnet learned scores include extra inference calls and supervised TF-IDF evidence.",
    "probability_outcomes": "Scores split by observed outcome on the development set. Each curve is normalized within correct or incorrect cases, so its bin shares sum to 100%; the two outcome curves do not show the overall error prevalence. Legend counts retain that denominator. Strong separation helps triage errors, while reliability must be assessed in the other figure. Cross-classifier comparisons involve different mistakes and error rates; comparisons within Haiku or Sonnet hold the predictions fixed.",
}


def diagnostics(out, include_sonnet=False, include_opus=False):
    out = Path(out)
    directory = out / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    pairs = list(PAIRS)
    if include_sonnet:
        pairs.insert(2, ("Sonnet 5.5", "anthropic-sonnet-5-5", "llm_same_raw", "Original confidence", "llm_correctness_raw", "Learned correctness"))
    if include_opus:
        pairs.insert(3, ("Opus 5.5", "anthropic-opus-5-5", "llm_same_raw", "Original confidence", "llm_correctness_raw", "Learned correctness"))
    frames, hashes, bins, reference = {}, {}, {}, None
    for _, run, left, _, right, _ in pairs:
        for method in (left, right):
            path = ROOT / "runs" / run / "development" / f"{method}.predictions.csv"
            f = pd.read_csv(path).sort_values("id").reset_index(drop=True)
            assert len(f) == 1000 and f.id.is_unique
            assert np.array_equal(f.correct, f.truth == f.prediction)
            assert np.isfinite(f.confidence).all() and f.confidence.between(0, 1).all()
            if reference is None:
                reference = f[["id", "truth"]]
            else:
                assert f[["id", "truth"]].equals(reference)
            key = f"{run}:{method}"
            frames[key] = f
            hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
            index = np.minimum((f.confidence.to_numpy()*10).astype(int), 9)
            records = []
            for b in range(10):
                selected = f[index == b]
                if len(selected):
                    n, correct = len(selected), int(selected.correct.sum())
                    records.append({"bin": b, "n": n, "confidence": float(selected.confidence.mean()),
                        "accuracy": correct/n, "ci95": wilson(correct, n)})
            assert sum(r["n"] for r in records) == 1000
            saved_bins = pd.read_csv(path.with_name(f"{method}.reliability.csv"))
            assert dict(zip(saved_bins.bin, saved_bins.n)) == {r["bin"]: r["n"] for r in records}
            # Original float32 means can differ slightly from float64 means
            # after reading the same individual scores back from CSV.
            assert np.allclose(saved_bins.confidence, [r["confidence"] for r in records], atol=1e-6, rtol=0)
            assert np.allclose(saved_bins.accuracy, [r["accuracy"] for r in records], atol=1e-12, rtol=0)
            bins[key] = records
        assert frames[f"{run}:{left}"].prediction.equals(frames[f"{run}:{right}"].prediction)

    def save(fig, name):
        for ext in ("png", "svg", "pdf"):
            fig.savefig(directory / f"{name}.{ext}", bbox_inches="tight", dpi=180, facecolor="white")
        plt.close(fig)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
        "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none", "pdf.fonttype": 42})
    plot_pairs=[pair for pair in pairs if pair[1]!='anthropic-opus-5-5']
    fig = plt.figure(figsize=(11.6, 3.9*len(plot_pairs)), layout="constrained")
    outer = fig.add_gridspec(len(plot_pairs), 2)
    for row, (label, run, left, left_name, right, right_name) in enumerate(plot_pairs):
        for column, (method, variant) in enumerate(((left, left_name), (right, right_name))):
            grid = outer[row, column].subgridspec(2, 1, height_ratios=(3, 1), hspace=.03)
            ax, count_ax = fig.add_subplot(grid[0]), fig.add_subplot(grid[1])
            records = bins[f"{run}:{method}"]
            x, y = np.array([r["confidence"] for r in records]), np.array([r["accuracy"] for r in records])
            ci = np.array([r["ci95"] for r in records])
            ax.plot([0, 1], [0, 1], "--", color=GRAY, linewidth=1)
            ax.errorbar(x, y, yerr=np.maximum([y-ci[:, 0], ci[:, 1]-y], 0),
                fmt="o", markersize=4, capsize=3, color=TEAL, linewidth=1.3)
            ax.set(xlim=(0, 1.025), ylim=(0, 1.04), title=f"{label} · {variant}",
                ylabel="Observed fraction correct")
            ax.set_xticks(np.arange(0, 1.01, .2))
            ax.tick_params(axis="x", labelbottom=False)
            ax.yaxis.set_major_formatter(PercentFormatter(1))
            ax.grid(alpha=.13)
            for r in records:
                center = (r["bin"]+.5)/10
                count_ax.bar(center, r["n"], width=.08, color=TEAL, alpha=.65)
                count_ax.text(center, r["n"]*1.12, str(r["n"]), ha="center", fontsize=8)
            count_ax.set(yscale="log", ylim=(.8, 2000), xlim=(0, 1.025), ylabel="Bin n", xlabel="Reported correctness probability")
            count_ax.set_xticks(np.arange(0, 1.01, .2))
            count_ax.set_yticks([1, 100, 1000], labels=["1", "100", "1k"])
            count_ax.grid(axis="y", alpha=.13)
    save(fig, "probability_reliability")

    selected = [("Embedding LR calibrated", "embeddings:embedding_lr_temperature"),
        ("Haiku original", "anthropic-batch-v2:llm_same_raw"),
        ("Haiku learned", "anthropic-batch-v2:llm_correctness_raw"),
        ("OpenAI native probability", "openai-decisions-v2:decisions_probability_raw")]
    if include_sonnet:
        selected += [("Sonnet original", "anthropic-sonnet-5-5:llm_same_raw"),
                     ("Sonnet learned", "anthropic-sonnet-5-5:llm_correctness_raw")]
    if include_opus:
        selected += [("Opus original", "anthropic-opus-5-5:llm_same_raw"),
                     ("Opus learned", "anthropic-opus-5-5:llm_correctness_raw")]
    plot_selected=[pair for pair in selected if not pair[1].startswith('anthropic-opus-5-5:')]
    fig, axes = plt.subplots(len(plot_selected)//2, 2, figsize=(11.6, 3.5*(len(plot_selected)//2)), layout="constrained")
    edges = np.linspace(0, 1, 11)
    for ax, (label, key) in zip(axes.flat, plot_selected):
        frame = frames[key]
        for correct, color, outcome in ((True, TEAL, "Correct"), (False, RUST, "Incorrect")):
            scores = frame.loc[frame.correct == correct, "confidence"].to_numpy()
            if len(scores):
                # Match the experiment's bin assignment exactly, including
                # decimal scores such as 0.6 on floating-point boundaries.
                index = np.minimum((scores*10).astype(int), 9)
                shares = np.bincount(index, minlength=10)/len(scores)
                assert np.isclose(shares.sum(), 1)
                ax.stairs(shares, edges, color=color, linewidth=1.8, label=f"{outcome} (n={len(scores)})")
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Reported correctness probability",
            ylabel="Share within outcome group", title=label)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.legend(fontsize=8, loc="upper left", frameon=False)
        ax.grid(alpha=.13)
    save(fig, "probability_outcomes")
    if include_opus:
        from report_opus import export_opus_probabilities
        export_opus_probabilities(directory,frames,bins)

    high_confidence = []
    for label, key in selected:
        f = frames[key]
        mask = f.confidence >= .9
        n, errors = int(mask.sum()), int((~f.loc[mask, "correct"]).sum())
        high_confidence.append({"method": label, "n": n, "errors": errors, "risk": errors/n if n else None})
    data = {"role": "development", "n": 1000, "bins": 10, "binning": "equal width; score 1 in final bin",
        "intervals": "Nominal pointwise 95% Wilson; no multiple-bin adjustment", "reliability": bins,
        "high_confidence": high_confidence, "source_sha256": hashes, "captions": CAPTIONS,
        "sonnet_included": include_sonnet, "opus_included":include_opus, "official_test_evaluated": False}
    write_json(out / "probability-comparisons.json", data)
    return data


def section_markdown(data):
    rows = "\n".join(f"| {r['method']} | {r['n']} | {r['errors']} | {r['risk']:.2%} |" if r["n"] else
        f"| {r['method']} | 0 | — | — |" for r in data["high_confidence"])
    return f"""## Comparing the probability curves

Reliability plots answer whether a reported probability matches observed correctness. A point below the diagonal indicates overconfidence in that bin; a point above it indicates underconfidence. Each comparison holds the classifier's selected labels fixed. The histogram beneath each panel shows how much evidence supports its points, and nominal Wilson intervals show sampling uncertainty. The plots use shared equal-width bins without smoothing or connecting sparse estimates.

The second figure separates scores assigned to correct and incorrect predictions. Greater separation helps prioritize review, but does not by itself establish calibrated probabilities. Each outcome distribution is normalized separately; its legend reports the actual number of cases. Cross-classifier comparisons also change which examples are errors, so the within-classifier comparisons are the stronger method comparison.

The raw embedding scores are underconfident in several populated bins. Temperature scaling moves more predictions toward high confidence and reduces probability error. Haiku's learned scores assign low confidence to a much larger share of its mistakes while leaving all selected labels unchanged. The tested OpenAI temperature transformation makes several bins underconfident and worsens Brier error. Sparse individual points should be interpreted with their counts and intervals.

| Method | Scores at least 90% | Errors in that group | Observed error rate |
|---|---:|---:|---:|
{rows}

The 90% cutoff is a descriptive reference point shared across methods. It is separate from the threshold chosen for the 1% automation criterion and does not constitute a deployment guarantee. Probability quality, error ranking, and coverage should be assessed together. The OpenAI temperature comparison uses log returned probabilities as a surrogate for native logits; the learned correctness comparisons also add supervised evidence and extra calls.
"""


def append_probability_comparisons(report_path, include_sonnet=False):
    from build_findings_report import STYLE, render
    report_path = Path(report_path)
    data = diagnostics(report_path.parent, include_sonnet)
    markdown = section_markdown(data)
    (report_path.parent / "probability-section.md").write_text(markdown, encoding="utf-8")
    body, _ = render(markdown)
    for name in ("probability_reliability", "probability_outcomes"):
        svg = (report_path.parent / "figures" / f"{name}.svg").read_text(encoding="utf-8")
        svg = svg[svg.index("<svg"):].replace("<svg ", '<svg role="img" aria-label="' + html.escape(CAPTIONS[name], quote=True) + '" ', 1)
        body += f"<figure>{svg}<figcaption>{html.escape(CAPTIONS[name])}</figcaption></figure>"
    standalone_body = body.replace('<h2 id="comparing-the-probability-curves">', '<h1 id="comparing-the-probability-curves">', 1)
    standalone_body = standalone_body.replace('Comparing the probability curves</h2>', 'Comparing the probability curves</h1>', 1)
    standalone = '<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Probability curve comparisons</title><style>' + STYLE + '</style></head><body><main>' + standalone_body + '</main></body></html>'
    (report_path.parent / "probability-comparisons.html").write_text(standalone, encoding="utf-8")
    text = report_path.read_text(encoding="utf-8")
    begin, end = "<!-- probability comparisons begin -->", "<!-- probability comparisons end -->"
    section = begin + body + end
    if begin in text:
        pattern = re.escape(begin) + r".*?" + re.escape(end)
        text, count = re.subn(pattern, lambda match: section, text, flags=re.S)
        assert count == 1
    else:
        anchors = ("why-the-embedding-curves-looked-volatile", "implications-for-the-next-experiment")
        for anchor in anchors:
            marker = f'<h2 id="{anchor}">'
            if marker in text:
                text = text.replace(marker, section + marker, 1)
                break
        else:
            raise ValueError("No report insertion boundary found")
    report_path.write_text(text, encoding="utf-8")
    print(f"Added probability comparisons: {report_path}; Sonnet included={include_sonnet}", flush=True)
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default="reports/banking77-findings/report.html")
    parser.add_argument("--include-sonnet", action="store_true")
    args = parser.parse_args()
    append_probability_comparisons(ROOT / args.report, args.include_sonnet)
