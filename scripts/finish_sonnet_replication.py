"""Wait for the existing Sonnet worker, then build its paired findings report.

This process never submits requests or retries a failed experiment.
"""
from datetime import datetime, timezone
import html
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from confidence_experiment.io import read_json, write_json

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/anthropic-sonnet-5-5"
OUT = ROOT / "reports/sonnet-5-5-findings"


def now():
    return datetime.now(timezone.utc).isoformat()


def interval(row, scale=1, digits=4):
    low, high = row["ci95"]
    return f"{low*scale:.{digits}f} to {high*scale:.{digits}f}"


def figures(data, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    from build_findings_report import TEAL, NAVY, RUST, GRAY
    destination = out / "figures"
    destination.mkdir(parents=True, exist_ok=True)

    def save(fig, name):
        for ext in ("png", "svg", "pdf"):
            fig.savefig(destination / f"{name}.{ext}", bbox_inches="tight", dpi=180, facecolor="white")
        plt.close(fig)

    previous = read_json(ROOT / "reports/banking77-findings/analysis.json")
    rows = [("Embedding LR", previous["all_metrics"]["embeddings"]["embedding_lr_temperature"], TEAL),
            ("Embedding MLP", previous["all_metrics"]["embeddings"]["embedding_mlp_raw"], TEAL),
            ("TF-IDF LR", previous["all_metrics"]["tfidf"]["tfidf_temperature"], TEAL),
            ("Sonnet 5.5", data["metrics"]["sonnet"]["llm_same_raw"], NAVY),
            ("Haiku 5.5", data["metrics"]["haiku"]["llm_same_raw"], RUST),
            ("OpenAI Decisions", previous["all_metrics"]["openai"]["decisions_confidence_raw"], GRAY)]
    fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
    for i, (_, row, color) in enumerate(rows):
        value = row["accuracy"]
        low, high = row["accuracy_ci95"]
        ax.errorbar(value, i, xerr=[[value-low], [high-value]], fmt="o", color=color, capsize=4)
        ax.text(1.01, i, f"{value:.1%}", va="center", weight="bold", color=color)
    ax.set(yticks=range(len(rows)), yticklabels=[r[0] for r in rows], ylim=(5.6, -.6), xlim=(.65, 1.06),
           title="Classification on the same development messages", xlabel="Accuracy")
    ax.set_xticks(np.arange(.65, 1.01, .05))
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(axis="x", alpha=.15)
    save(fig, "classification")

    methods = [("Original", "llm_same_raw", RUST), ("Verifier", "llm_verifier_raw", NAVY),
               ("Learned", "llm_correctness_raw", TEAL)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), layout="constrained")
    for ax, metric, title in zip(axes, ("brier", "error_auroc"), ("Probability error", "Error ranking")):
        for i, (label, method, color) in enumerate(methods):
            row = data["estimates"][f"sonnet:{method}"][metric]
            value, (low, high) = row["estimate"], row["ci95"]
            ax.errorbar(value, i, xerr=[[value-low], [high-value]], fmt="o", color=color, capsize=4)
            ax.annotate(f"{value:.3f}", (value, i), xytext=(0, 11), textcoords="offset points", ha="center", color=color)
        ax.set(yticks=range(3), yticklabels=[r[0] for r in methods] if metric == "brier" else [],
               ylim=(2.5, -.6), xlabel="Brier error (lower is better)" if metric == "brier" else "Error AUROC (higher is better)", title=title)
        if metric == "brier":
            ax.set_xlim(left=0)
        else:
            ax.set_xlim(.5, 1.01)
        ax.grid(axis="x", alpha=.15)
    save(fig, "confidence")

    rows = [("Embedding LR calibrated", "embeddings", "embedding_lr_temperature", TEAL),
            ("Sonnet original", "anthropic-sonnet-5-5", "llm_same_raw", RUST),
            ("Sonnet learned", "anthropic-sonnet-5-5", "llm_correctness_raw", NAVY),
            ("Haiku learned", "anthropic-batch-v2", "llm_correctness_raw", GRAY)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for label, run, method, color in rows:
        frame = pd.read_csv(ROOT / "runs" / run / "development" / f"{method}.risk_coverage.csv")
        for ax in axes:
            ax.step(frame.coverage, frame.risk, where="post", color=color, label=label, linewidth=1.6)
    for ax, top in zip(axes, (.30, .025)):
        ax.axhline(.01, color=RUST, linestyle="--", linewidth=1)
        ax.set(xlim=(0, 1), ylim=(0, top), xlabel="Fraction accepted")
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0 if top == .30 else 1))
        ax.grid(alpha=.15)
    axes[0].set(title="Observed development risk", ylabel="Errors among accepted predictions")
    axes[1].set_title("Zoom on low observed error rates")
    axes[0].legend(loc="upper left", fontsize=8)
    save(fig, "risk_coverage")


def make_report(data, out=OUT):
    from build_findings_report import STYLE, render
    out.mkdir(parents=True, exist_ok=True)
    previous = read_json(ROOT / "reports/banking77-findings/analysis.json")
    m, c = data["metrics"], data["comparisons"]
    s, h = m["sonnet"]["llm_same_raw"], m["haiku"]["llm_same_raw"]
    learned = m["sonnet"]["llm_correctness_raw"]
    brier_gain = (s["brier"]-learned["brier"])/s["brier"]
    gain = c["sonnet_original_minus_learned_brier"]
    direction = "improved" if gain["estimate"] > 0 else "worsened"
    uncertainty = ("The paired interval supports an improvement on this sample." if gain["ci95"][0] > 0 else
        "The paired interval supports a deterioration on this sample." if gain["ci95"][1] < 0 else
        "The paired interval includes zero, so this sample does not establish a reliable change.")
    gap = c["embedding_minus_sonnet_accuracy"]
    verdict = ("The embedding classifier retains a clear accuracy lead within this sample." if gap["ci95"][0] > 0 else
        "Sonnet has a clear accuracy lead within this sample." if gap["ci95"][1] < 0 else
        "The accuracy difference between Sonnet and the embedding LR remains uncertain within this sample.")
    best_name, best = max(((name, row) for name, row in m["sonnet"].items() if name.startswith("llm_")),
                         key=lambda item: item[1]["acceptance"]["coverage"])
    accepted = best["acceptance"]
    automation = (f"The largest Sonnet acceptance set contained **{accepted['accepted']} predictions and {accepted['errors']} errors**, "
        f"or **{accepted['coverage']:.1%} development coverage**, using {best_name}." if accepted["accepted"] else
        "**No tested Sonnet threshold met the 1% error criterion under the Wilson rule.**")
    accuracy_rows = [("EmbeddingGemma plus LR", previous["all_metrics"]["embeddings"]["embedding_lr_temperature"]),
        ("EmbeddingGemma plus MLP", previous["all_metrics"]["embeddings"]["embedding_mlp_raw"]),
        ("TF-IDF plus LR", previous["all_metrics"]["tfidf"]["tfidf_temperature"]),
        ("Sonnet 5.5", s), ("Haiku 5.5", h),
        ("OpenAI Decisions", previous["all_metrics"]["openai"]["decisions_confidence_raw"])]
    classification = "\n".join(f"| {name} | {row['accuracy']:.1%} | {row['macro_f1']:.3f} | {row['errors']} |" for name, row in accuracy_rows)
    labels = {"llm_same_raw": "Original confidence", "llm_same_sigmoid": "Original plus sigmoid",
        "llm_verifier_raw": "Separate verifier", "llm_verifier_sigmoid": "Verifier plus sigmoid",
        "llm_correctness_raw": "Learned correctness", "llm_correctness_sigmoid": "Learned plus sigmoid"}
    confidence = "\n".join(f"| {label} | {m['sonnet'][method]['brier']:.4f} | {m['sonnet'][method]['binary_log_loss']:.4f} | {m['sonnet'][method]['error_auroc']:.3f} |" for method, label in labels.items())
    comparisons = "\n".join([
        f"| Sonnet minus Haiku accuracy | {c['sonnet_minus_haiku_accuracy']['estimate']*100:+.1f} pp | {interval(c['sonnet_minus_haiku_accuracy'], 100, 1)} pp |",
        f"| Embedding LR minus Sonnet accuracy | {gap['estimate']*100:+.1f} pp | {interval(gap, 100, 1)} pp |",
        f"| Sonnet original minus learned Brier error | {gain['estimate']:+.4f} | {interval(gain)} |",
        f"| Sonnet original minus verifier Brier error | {c['sonnet_original_minus_verifier_brier']['estimate']:+.4f} | {interval(c['sonnet_original_minus_verifier_brier'])} |",
        f"| Sonnet learned minus original error AUROC | {c['sonnet_learned_minus_original_auroc']['estimate']:+.4f} | {interval(c['sonnet_learned_minus_original_auroc'])} |"])
    acceptance_rows = [("Embedding LR calibrated", m["embedding"]["embedding_lr_temperature"]),
        (f"Sonnet best observed ({best_name})", best), ("Haiku learned correctness", m["haiku"]["llm_correctness_raw"])]
    acceptance_table = "\n".join(f"| {name} | {row['acceptance']['coverage']:.1%} | {row['acceptance']['accepted']} | {row['acceptance']['errors']} |" for name, row in acceptance_rows)
    markdown = f"""# Sonnet 5.5 and the Value of Learned Confidence

**BANKING77 replication · 1,000 shared development examples · Generated {now()[:10]}**

Sonnet 5.5 correctly classified **{1000-s['errors']} of 1,000 banking messages ({s['accuracy']:.1%})**, compared with **{h['accuracy']:.1%} for Haiku 5.5** and **97.1% for the frozen embedding LR**. {verdict}

Holding Sonnet's predictions fixed, learned correctness {direction} Brier error from **{s['brier']:.4f} to {learned['brier']:.4f}** ({abs(brier_gain):.1%} relative change). {uncertainty} {automation}

These are development findings. The official 3,080-example test set has not been evaluated, and acceptance thresholds were selected using the same development outcomes shown here.

## A matched replication

The replication uses the same seed, 5,992 labeled classifier-fitting examples, 2,000 correctness-training examples, 1,000 calibration examples, and 1,000 development messages as the earlier experiment. The exact development IDs and reference labels match. TF-IDF vocabulary, IDF values, learned class coefficients, and intercepts were checked against the Haiku run.

Sonnet replaced Haiku in all three calls: initial classification, blind second classification, and verification. Prompts, intent-name-only category definitions, structured schemas, 256-token output limit, learned confidence features, and acceptance rule were retained. No labeled demonstrations were added to the LLM prompts. The supervised classifiers used task-specific fitting labels, so cross-family accuracy comparisons are not controlled comparisons at equal labeled-data budgets.

Sonnet 5.5 rejects thinking=disabled. Its supported between_tools setting disables up-front thinking; these requests contain no tools. Haiku used disabled. This API-specific setting is recorded in cache keys and metadata. Model access and the exact three-call protocol were checked before full batch submission. [Anthropic thinking documentation](https://platform.claude.com/docs/en/build-with-claude/thinking)

## How Sonnet changed classification

| Classifier | Accuracy | Macro F1 | Errors out of 1,000 |
|---|---:|---:|---:|
{classification}

OpenAI's unsuccessful predictions include one refusal. These results cover the implemented zero-shot LLM configurations and supervised training conditions; they do not establish the best achievable performance of any provider.

## Confidence on Sonnet predictions

All Sonnet confidence variants share the same initial labels and therefore the same classification accuracy. The estimator changes the probability assigned to a selected label; it does not repair the underlying label.

| Confidence method | Brier error ↓ | Binary log loss ↓ | Error AUROC ↑ |
|---|---:|---:|---:|
{confidence}

Brier error is average squared error against the correct/incorrect outcome; lower is better. Error AUROC measures how well low confidence ranks mistakes; higher is better. Brier error also reflects discrimination and error prevalence, so differences between classifiers cannot be interpreted as pure calibration improvements. The within-Sonnet comparisons are the cleaner confidence-method experiment.

The learned estimator combines original and verifier confidence, blind label agreement, input length and an empty-input flag, supervised TF-IDF support for the chosen label, and the TF-IDF top-two margin. It is trained on the separate correctness partition; sigmoid transformations are fitted on the calibration partition. There are no embedding-neighbor features or feature ablations in this replication. A separate call to the same Sonnet model does not imply independent errors.

## What the paired comparisons establish

| Paired difference | Estimate | 95% bootstrap interval |
|---|---:|---:|
{comparisons}

Positive Brier differences in this table favor the second method because lower Brier error is better. Positive accuracy or AUROC differences favor the first method. Intervals use 5,000 resamples of shared example IDs, seed 20261009. They condition on the fitted models and calibrators, without retraining or adjustments for method selection and multiple comparisons. An interval spanning zero leaves the direction of improvement unresolved on this sample.

## Automation under the 1% rule

The acceptance rule chooses the largest confidence-threshold group whose upper endpoint of a 95% Wilson error interval is at most 1%. This is stricter than requiring an observed error rate below 1%.

| Candidate | Development coverage | Accepted | Errors among accepted |
|---|---:|---:|---:|
{acceptance_table}

{automation} The Sonnet row also selects among confidence methods on development outcomes. Its coverage is an exploratory operating point, not an independently validated guarantee. Freeze the model, confidence method, and threshold before official-test evaluation. Zero coverage means no threshold qualified under this rule; it does not imply that every prediction is incorrect or that ranking offers no review value.

## Implications for the next experiment

{verdict} The next classification comparison should preserve the labeled-data and prompt conditions explicitly. Richer intent descriptions and demonstrations would answer a different question and should be evaluated as a separate configuration.

For confidence, prioritize the paired within-classifier evidence above. If learned correctness helps, ablate TF-IDF support, blind agreement, and verification to determine which information justifies extra labels and calls. If its gain is small or uncertain, avoid assuming that the Haiku result transfers automatically to a stronger classifier.

Sparse reliability bins still need example counts and uncertainty intervals. More representative evaluation data improves the precision of calibration comparisons, particularly when a classifier makes few mistakes. Repeat the strongest approach on representative document data and evaluate unsupported inputs separately. Full encoder fine-tuning remains deferred.

## Cost and evidence

The Sonnet ledger records **${data['spending']['ledgers'][-1]['usd']:.2f}**, including its synchronous smoke check. Combined historical experiment spending is approximately **${data['spending']['combined_usd']:.2f}**. These are token-based estimates using configured prices, not billing invoices, and exclude local compute. Batch rates were $1/M input and $5/M output at setup. [Anthropic pricing](https://platform.claude.com/docs/en/about-claude/pricing)

The paired analysis records source hashes and verifies matching data and fixed labels within each confidence family. The complete Sonnet development report contains per-method predictions, reliability bins, risk–coverage curves, per-category results, and disagreements. All findings are from one training seed and one development split; annotation ambiguity, public-benchmark exposure, and distribution shift remain limitations.
"""
    figures(data, out)
    body, navigation = render(markdown)
    captions = {
        "classification": "The same 1,000 development messages; whiskers are marginal 95% Wilson accuracy intervals. The three supervised classifiers used fitting labels; the LLM prompts used intent names without demonstrations.",
        "confidence": "Sonnet labels are held fixed. Whiskers are conditional 95% bootstrap intervals. Paired differences in the table assess improvements; overlap between these marginal intervals is not the paired test.",
        "risk_coverage": "Development risk at observed confidence thresholds, preserving ties. The 1% line marks observed error, whereas acceptance selection uses a Wilson upper endpoint. The right panel clips higher-risk curves. Thresholds require independent validation.",
    }
    def embed(name):
        svg = (out / "figures" / f"{name}.svg").read_text(encoding="utf-8")
        svg = svg[svg.index("<svg"):].replace("<svg ", '<svg role="img" aria-label="' + html.escape(captions[name], quote=True) + '" ', 1)
        return f"<figure>{svg}<figcaption>{html.escape(captions[name])}</figcaption></figure>"
    for anchor, name in (("confidence-on-sonnet-predictions", "classification"),
                         ("what-the-paired-comparisons-establish", "confidence"),
                         ("implications-for-the-next-experiment", "risk_coverage")):
        marker = f'<h2 id="{anchor}">'
        assert body.count(marker) == 1
        body = body.replace(marker, embed(name) + marker, 1)
    result = '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Sonnet 5.5 findings</title><style>' + STYLE + '</style></head><body><main>' + body + '<nav aria-label="Report sections">' + navigation + '</nav><footer><a href="../banking77-findings/sonnet-comparison.json">Paired analysis and source hashes</a> · <a href="../../runs/anthropic-sonnet-5-5/development/report.html">Full Sonnet development report</a> · <a href="../banking77-findings/report.html">Original findings report</a></footer></main></body></html>'
    (out / "report.html").write_text(result, encoding="utf-8")
    if out == OUT:
        from add_probability_comparisons import append_probability_comparisons
        append_probability_comparisons(out / "report.html", include_sonnet=True)
    write_json(out / "summary.json", {"created_utc": now(), "role": "development", "sonnet_accuracy": s["accuracy"],
        "original_brier": s["brier"], "learned_brier": learned["brier"], "best_coverage": accepted["coverage"],
        "best_method": best_name, "report": str(out / "report.html")})
    # Make the added result discoverable from the existing local report.
    original = ROOT / "reports/banking77-findings/report.html"
    if out == OUT:
        text = original.read_text(encoding="utf-8")
        notice = '<aside id="sonnet-replication" style="margin:0 0 28px;padding:15px 18px;background:#e8f1ee;font:15px/1.5 system-ui,sans-serif"><strong>Sonnet 5.5 replication is complete.</strong> <a href="../sonnet-5-5-findings/report.html">Read the matched comparison and updated findings</a>.</aside>'
        if 'id="sonnet-replication"' not in text:
            assert text.count("<main>") == 1
            original.write_text(text.replace("<main>", "<main>"+notice, 1), encoding="utf-8")
    return out / "report.html"


def main():
    os.chdir(ROOT)
    if os.name == "nt":
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.c_void_p(-1), 0x4000)
    lock = RUN / ".comparison.lock"
    RUN.mkdir(parents=True, exist_ok=True)
    try:
        with lock.open("x") as handle:
            handle.write(str(os.getpid()))
    except FileExistsError:
        print("A Sonnet comparison finisher already owns this run.", flush=True)
        return
    status_path = RUN / "comparison_status.json"
    state = {"state": "waiting_for_results", "pid": os.getpid(), "started_utc": now(),
        "report": str(OUT / "report.html"), "official_test_evaluated": False}
    try:
        while True:
            supervisor = read_json(RUN / "background_status.json")
            if supervisor["state"] in {"failed", "cancelled", "canceled"}:
                raise RuntimeError(f"Sonnet worker {supervisor['state']}; no API requests retried.")
            if supervisor["state"] == "complete":
                break
            state.update(updated_utc=now())
            write_json(status_path, state)
            time.sleep(15)
        state.update(state="analyzing", updated_utc=now())
        write_json(status_path, state)
        subprocess.run([sys.executable, str(ROOT / "scripts/analyze_sonnet_comparison.py")], check=True)
        analysis = read_json(ROOT / "reports/banking77-findings/sonnet-comparison.json")
        report = make_report(analysis)
        state.update(state="complete", report=str(report), completed_utc=now())
        write_json(status_path, state)
        print(f"Sonnet findings report ready: {report}", flush=True)
    except Exception as error:
        state.update(state="failed", error=f"{type(error).__name__}: {error}", completed_utc=now())
        write_json(status_path, state)
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
