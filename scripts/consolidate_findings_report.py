"""Build the complete report from original, Sonnet and calibration evidence.

No API calls, model fitting, threshold search, or official test access.
The pre-consolidation prose is retained for reproducible editorial updates.
"""
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import re
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

import build_findings_report as builder
from add_probability_comparisons import diagnostics, section_markdown, CAPTIONS as PROB_CAPTIONS
from confidence_experiment.io import read_json, write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/banking77-findings"
CAL = ROOT / "reports/calibrator-comparison"
TITLE = "Learning When to Trust a Classifier"


def compose(original, sonnet, calibrators, probability):
    blocks = re.split(r"\n\s*\n", original.strip())
    replacements = {}

    def replace(prefix, text):
        matched = [b for b in blocks if b.startswith(prefix)]
        assert len(matched) == 1, prefix
        replacements[matched[0]] = text

    replace("A frozen embedding model", "A frozen embedding model with a linear classifier correctly classified **971 of 1,000 banking queries**. Sonnet 5.5 classified 830 correctly, Haiku 5.5 classified 757, and OpenAI Decisions classified 733. With task-specific labeled examples available, the supervised embedding approach remains the strongest candidate for further validation.")
    replace("The confidence experiment produced", "Learning correctness was useful for both Claude models. Keeping their predictions unchanged, a small learned estimator reduced Brier error by **59.0% for Haiku and 55.7% for Sonnet**, before final probability calibration. Verification alone showed no clear benefit. Temperature scaling, logistic calibration, and isotonic regression produced different outcomes across confidence sources; greater flexibility did not produce a consistent advantage.")
    replace("The operational results show", "The calibrated embedding classifier could accept **895 development predictions with three errors** under the experiment’s threshold-selection rule. Learned confidence improved LLM triage, but no tested Haiku or Sonnet threshold met the same stringent 1% error criterion. These are development findings; the official test set remains unevaluated.")
    replace("| Data role", "| Data role | Examples | Purpose |\n|---|---:|---|\n| Classifier fitting | 5,992 | Train the supervised classifiers |\n| Correctness training | 2,000 | Train separate Haiku and Sonnet correctness estimators |\n| Calibration | 1,000 | Fit temperature, logistic and isotonic transformations |\n| Development | 1,000 | Compare methods and choose acceptance thresholds |\n| Official test | 3,080 | Reserved for final evaluation |")
    replace("| Classifier |", "| Classifier | Accuracy | Macro F1 | Errors out of 1,000 |\n|---|---:|---:|---:|\n| EmbeddingGemma plus logistic regression | **97.1%** | **0.972** | **29** |\n| EmbeddingGemma plus a small MLP | 96.9% | 0.970 | 31 |\n| TF-IDF plus logistic regression | 87.1% | 0.873 | 129 |\n| Anthropic Sonnet 5.5 | 83.0% | 0.829 | 170 |\n| Anthropic Haiku 5.5 | 75.7% | 0.756 | 243 |\n| OpenAI Decisions using gpt-6-luna | 73.3% | 0.735 | 267 |")
    replace("The embedding classifier’s lead", "The embedding classifier led Sonnet by **14.1 percentage points**, with a paired 95% bootstrap interval of **11.9–16.3 points**. Its leads over Haiku and OpenAI Decisions were 21.4 points and 23.8 points, with intervals of 18.8–24.0 and 21.1–26.5 points. These gaps are large within this development sample, conditional on the fitted models, available task labels, and the particular prompting setup.")
    replace("| Confidence method |", "| Confidence method | Brier error ↓ | Binary log loss ↓ | Error AUROC ↑ |\n|---|---:|---:|---:|\n| Haiku’s original same-call confidence | 0.1454 | 0.4434 | 0.815 |\n| Separate Haiku verifier | 0.1388 | 0.4261 | 0.831 |\n| Learned correctness model | 0.0596 | **0.2093** | **0.961** |\n| Learned correctness plus logistic on score | **0.0578** | 0.2100 | **0.961** |")
    replace("The learned estimator used", "The Haiku and Sonnet correctness estimators used separate regularized logistic regressions trained on the same 2,000-example correctness partition. Features were original confidence, verifier confidence, agreement with a blind second classification, input length, an empty-input flag, and two signals from the shared supervised TF-IDF classifier: support for the LLM’s chosen label and the margin between its top two classes. All were available without the development example’s true label. The blind classifier and verifier were additional calls to the same Claude model; their errors need not be independent.")
    replace("| Candidate |", "| Candidate | Accepted | Errors among accepted | Observed error rate | Development coverage |\n|---|---:|---:|---:|---:|\n| Embedding LR with temperature scaling | **895** | **3** | **0.34%** | **89.5%** |\n| Embedding LR with logistic on log-odds | 892 | 3 | 0.34% | 89.2% |\n| Embedding LR with isotonic | 845 | 1 | 0.12% | 84.5% |\n| Embedding MLP with raw probabilities | 867 | 2 | 0.23% | 86.7% |\n| TF-IDF, all tested scores | 0 | — | — | 0% |\n| Haiku, all tested scores | 0 | — | — | 0% |\n| Sonnet, all tested scores | 0 | — | — | 0% |\n| OpenAI Decisions, all tested scores | 0 | — | — | 0% |")
    replace("Haiku shows why", "Both Claude models show why a strong AUROC is insufficient for this target. Their 500 highest raw learned-confidence predictions contained **eight errors for Haiku (1.6%) and six for Sonnet (1.2%)**. The scores moved many errors toward the review end of the queue, but no tested threshold qualified under the Wilson rule. Zero coverage means no threshold qualified, rather than every prediction being wrong or review ranking being useless.")
    replace("**Advance frozen", "**Advance frozen EmbeddingGemma plus logistic regression with temperature scaling as the main candidate.** It had the best observed classification and acceptance results. Neither the MLP nor a more flexible calibration map showed a convincing advantage. Freeze the complete candidate and its threshold before evaluating the untouched official test set.")
    replace("**Use learned correctness", "**Use learned correctness when retaining an LLM classifier has a business reason.** The gain replicated with Sonnet, but its higher classification accuracy still did not satisfy the stringent automation rule. Remove feature groups in turn: native confidence alone, supervised TF-IDF evidence, blind agreement, and verifier evidence. This will identify which signals justify additional calls and labeled-data requirements. Compare that system against using the supervised classifier directly.")
    replace("These are development results from", "These are development results from one split and one training seed. Each paired bootstrap uses 5,000 shared example resamples and percentile intervals, with seed 20261009. The intervals condition on fitted models and calibrators; they omit calibration-training and retraining variation, method selection, threshold search, and multiple-comparison adjustment. The later scalar calibration comparison adds exploratory development comparisons. Within each confidence family, selected labels stay fixed, making those comparisons stronger than causal comparisons across classifier families.")
    replace("Early Haiku requests", "Early Haiku requests encountered truncation with adaptive thinking and a 256-token output limit. Its comparison was restarted uniformly with thinking disabled. Sonnet used the same 256-token limit and between_tools thinking mode because its API rejected disabled; the protocol supplied no tools. This configuration difference constrains attribution of the Sonnet–Haiku gap solely to model identity. Historical and discarded calls remain in the ledger.")
    spending = sonnet["spending"]["combined_usd"]
    replace("Recorded API spending", f"Recorded API spending was approximately **${spending:.2f}**, including **$22.46 for the Sonnet replication** and $2.25 in earlier ledgers. These are token-based ledger estimates including exploratory and discarded calls; they exclude local compute and are not billing invoices. Logistic and isotonic comparisons reused cached outputs and incurred no additional API calls. Batch waiting and resumed work make recorded run durations unsuitable for a clean provider speed comparison.")
    replace("The accompanying analysis records", "The accompanying evidence records source SHA-256 hashes and verifies matching development IDs, frozen artifacts, and unchanged selected labels within each confidence family. It combines the original analysis, the paired Sonnet replication, and the 48 score variants from 11 confidence sources in the scalar calibration comparison. Scientific source outputs remain under runs/embeddings, runs/pilot, runs/anthropic-batch-v2, runs/anthropic-sonnet-5-5, runs/openai-decisions-v2, and reports/calibrator-comparison. The dataset revision is 57ec275d8078af65b7731c2a98be812d844a6d6b; the frozen embedding revision is 57c266a740f537b4dc058e1b0cda161fd15afa75. No official test predictions were generated for this report.")

    sonnet_methods = [("Original confidence", "llm_same_raw"),
                      ("Original plus logistic on score", "llm_same_sigmoid"),
                      ("Separate verifier", "llm_verifier_raw"),
                      ("Verifier plus logistic on score", "llm_verifier_sigmoid"),
                      ("Learned correctness", "llm_correctness_raw"),
                      ("Learned plus logistic on score", "llm_correctness_sigmoid")]
    srows = []
    for title, key in sonnet_methods:
        m = sonnet["metrics"]["sonnet"][key]
        srows.append(f"| {title} | {m['brier']:.4f} | {m['binary_log_loss']:.4f} | {m['error_auroc']:.3f} |")
    sonnet_section = """## Sonnet replicated the gain from learned correctness

Sonnet classified **83.0%** of the development messages correctly, an improvement of **7.3 percentage points over Haiku**, with a paired interval of **5.1–9.6 points**. It remained below the supervised TF-IDF and embedding classifiers in these operating conditions. The replication used the same splits, prompts, feature definitions, and supervised TF-IDF support, with a separate correctness estimator for Sonnet.

| Sonnet confidence method | Brier error ↓ | Binary log loss ↓ | Error AUROC ↑ |
|---|---:|---:|---:|
""" + "\n".join(srows) + """

Holding all 830 correct and 170 incorrect Sonnet predictions fixed, learned correctness reduced Brier error from **0.1201 to 0.0531**, a **55.7% reduction** before final calibration. The absolute reduction was 0.0669, with a paired interval of **0.0582–0.0758**. Error AUROC rose from **0.835 to 0.952**; the increase of 0.117 had an interval of 0.092–0.144. Logistic calibration on the learned score reduced Brier error further to **0.0505**.

Verification alone did not reproduce that gain: the verifier’s raw Brier error was 0.1249, slightly worse than Sonnet’s original 0.1201. The original-minus-verifier difference was −0.0049, with an interval of −0.0106 to +0.0006. Together with Haiku’s uncertain verifier improvement, this supports testing learned combinations of evidence rather than assuming a second same-model call is a reliable confidence measure.

The absolute learned-confidence Brier score was lower for Sonnet than for Haiku, but their different error prevalences and error sets prevent interpreting that difference as pure calibration superiority. The within-model reductions establish the replicated result.
"""
    names = {"embedding_lr": "Embedding LR", "embedding_mlp": "Embedding MLP", "tfidf": "TF-IDF LR",
             "haiku_same": "Haiku native", "haiku_verifier": "Haiku verifier", "haiku_correctness": "Haiku learned",
             "sonnet_same": "Sonnet native", "sonnet_verifier": "Sonnet verifier", "sonnet_correctness": "Sonnet learned",
             "decisions_confidence": "OpenAI confidence", "decisions_probability": "OpenAI probability"}
    if "opus_same" in calibrators["results"]:
        names.update({"opus_same":"Opus native", "opus_verifier":"Opus verifier", "opus_correctness":"Opus learned"})
    rows = []
    for key, label in names.items():
        methods = calibrators["results"][key]
        values = [f"{methods[m]['brier']:.5f}" if m in methods else "—"
                  for m in ("raw", "temperature", "logistic_score", "logistic_logit", "isotonic")]
        rows.append("| " + " | ".join([label, *values]) + " |")
    calibration_section = """## Temperature logistic and isotonic calibration

Temperature scaling divides the classifier’s class logits by one learned positive temperature before softmax. It changes the sharpness of the class distribution while preserving the selected class. For the embedding LR, it reduced Brier error from **0.03279 to 0.02081**, a **36.5% reduction**; binary log loss fell from 0.1326 to 0.0710. The original paired Brier improvement interval was 0.0082–0.0157. Simplicity helped here. [Guo and colleagues, 2017](https://proceedings.mlr.press/v70/guo17a.html)

The scalar comparison adds two logistic calibrators and isotonic regression. Logistic on score learns a slope and intercept for the original confidence; it is the sigmoid calibration already used in the LLM runs. Logistic on log-odds instead uses logit(p) as its feature. Both use regularization C=1. Isotonic learns a flexible non-decreasing mapping, with greater risk of overfitting when evidence is sparse. It can introduce tied scores and change ranking metrics. [Scikit-learn calibration documentation](https://scikit-learn.org/stable/modules/calibration.html)

All scalar mappings were fitted on the same **1,000 calibration examples** and evaluated on the separate **1,000 development examples**, estimating whether the unchanged selected label was correct. Temperature instead optimized multiclass log loss over 77 classes. An advantage in binary correctness Brier therefore does not establish better full class probabilities. Dashes indicate that the temperature transform was unavailable for that scalar confidence source.

| Confidence source | Raw | Temperature | Logistic score | Logistic log-odds | Isotonic |
|---|---:|---:|---:|---:|---:|
""" + "\n".join(rows) + """

Values are development Brier errors; lower is better. Temperature had the lowest embedding LR point estimate. Isotonic minus temperature was **+0.00177**, with a paired 95% interval of **−0.00004 to +0.00348**. This sample therefore leaves the difference uncertain. Only **39 calibration errors** informed the embedding LR mapping, and 29 informed the embedding MLP mapping; 1,000 calibration examples do not imply 1,000 informative error cases.

For Haiku native confidence, isotonic and logistic on log-odds were nearly tied: 0.14164 and 0.14168. Sonnet native confidence favored logistic on score by point estimate. For Sonnet learned correctness, isotonic’s Brier error of **0.05328** exceeded logistic-on-score’s **0.05052**, with a conditional paired difference interval of **+0.00051 to +0.00503**. Isotonic assigned probability exactly one to **three incorrect development predictions**, and binary log loss was **0.2241 versus 0.1886**. The log-loss calculation clips probabilities at 10⁻⁷, consistently with the original reports; an incorrect exact-one probability has infinite un-clipped log loss.

OpenAI’s selected-label probability improved from 0.14741 to **0.14400 with logistic on log-odds** and 0.14547 with isotonic. The tested temperature transform worsened it to 0.19034. That transform used log returned probabilities clipped at 10⁻¹² as a surrogate for native logits and optimized multiclass log loss. Native confidence and selected-label probability remained almost identical signals in this run. The result supports checking the transformation and target together.

All fitted logistic slopes were positive, so their strictly increasing maps preserved error ranking and threshold acceptance groups. Isotonic introduced ties: embedding LR coverage under the exploratory Wilson rule fell from **89.2% with raw or logistic scores to 84.5% with isotonic**, versus 89.5% with temperature. A calibrator can improve probability estimates yet affect usable acceptance thresholds through ties. The development-selected coverage figures still require independent validation.
"""
    new = [replacements.get(b, b) for b in blocks]
    # Replace the superseded calibration section as a single coherent section.
    start = new.index("## Calibration helped selectively")
    end = new.index("## Why the embedding curves looked volatile")
    new[start:end] = re.split(r"\n\s*\n", calibration_section.strip()) + re.split(r"\n\s*\n", probability.strip())
    location = new.index("## Better error ranking did not guarantee low-risk automation")
    new[location:location] = re.split(r"\n\s*\n", sonnet_section.strip())
    return "\n\n".join(new) + "\n"


def figures(sonnet, opus=None):
    # Use original metrics and Sonnet metrics directly for all six classifiers.
    data = builder.ANALYSIS["all_metrics"]
    rows = [("EmbeddingGemma + LR", data["embeddings"]["embedding_lr_temperature"], builder.TEAL),
            ("EmbeddingGemma + MLP", data["embeddings"]["embedding_mlp_raw"], builder.TEAL),
            ("TF-IDF + LR", data["tfidf"]["tfidf_temperature"], builder.TEAL),
            ("Sonnet 5.5", sonnet["metrics"]["sonnet"]["llm_same_raw"], builder.NAVY),
            ("Haiku 5.5", data["haiku"]["llm_same_raw"], builder.RUST),
            ("OpenAI Decisions", data["openai"]["decisions_confidence_raw"], builder.GRAY)]
    if opus:
        rows.insert(3,("Opus 5.5",opus["metrics"]["opus"]["llm_same_raw"],"#854397"))
    fig, ax = plt.subplots(figsize=(9, 4.2), layout="constrained")
    for i, (_, m, color) in enumerate(rows):
        p, (low, high) = m["accuracy"], m["accuracy_ci95"]
        ax.errorbar(p, i, xerr=[[p-low], [high-p]], fmt="o", color=color, capsize=4, markersize=7)
        ax.text(1.01, i, f"{p:.1%}", va="center", weight="bold", color=color)
    ax.set(yticks=range(len(rows)), yticklabels=[r[0] for r in rows], ylim=(len(rows)-.4, -.6), xlim=(.65, 1.06),
           title="Classification on the same development messages", xlabel="Development accuracy")
    ax.set_xticks(np.arange(.65, 1.01, .05))
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(axis="x", alpha=.15)
    builder.save(fig, "classification_accuracy")

    families=("haiku", "sonnet", "opus") if opus else ("haiku", "sonnet")
    fig, axes = plt.subplots(len(families), 2, figsize=(10.5, 3.5*len(families)), layout="constrained")
    methods = [("Original", "llm_same_raw", builder.RUST), ("Verifier", "llm_verifier_raw", builder.NAVY),
               ("Learned", "llm_correctness_raw", builder.TEAL)]
    for row, family in enumerate(families):
        for col, metric in enumerate(("brier", "error_auroc")):
            ax = axes[row, col]
            for y, (_, name, color) in enumerate(methods):
                m = (opus if family=="opus" else sonnet)["estimates"][f"{family}:{name}"][metric]
                p, (low, high) = m["estimate"], m["ci95"]
                ax.errorbar(p, y, xerr=[[p-low], [high-p]], fmt="o", color=color, capsize=4, markersize=7)
                ax.annotate(f"{p:.3f}", (p,y), xytext=(0,10), textcoords="offset points", ha="center", color=color)
            ax.set(yticks=range(3), yticklabels=[r[0] for r in methods] if col == 0 else [], ylim=(2.6,-.7),
                   xlim=(0,.18) if col == 0 else (.75,1), title=family.title() + (" probability error" if col == 0 else " error ranking"),
                   xlabel="Brier error (lower is better)" if col == 0 else "Error AUROC (higher is better)")
            ax.grid(axis="x", alpha=.15)
    builder.save(fig, "llm_confidence")

    candidates = [("Embedding LR temperature", "embeddings", "embedding_lr_temperature", builder.TEAL, "-"),
                  ("Haiku original", "anthropic-batch-v2", "llm_same_raw", builder.RUST, "--"),
                  ("Haiku learned", "anthropic-batch-v2", "llm_correctness_raw", builder.RUST, "-"),
                  ("Sonnet original", "anthropic-sonnet-5-5", "llm_same_raw", builder.NAVY, "--"),
                  ("Sonnet learned", "anthropic-sonnet-5-5", "llm_correctness_raw", builder.NAVY, "-"),
                  ("OpenAI native", "openai-decisions-v2", "decisions_confidence_raw", builder.GRAY, ":")]
    if opus:
        candidates += [("Opus original","anthropic-opus-5-5","llm_same_raw","#854397","--"),
                       ("Opus learned","anthropic-opus-5-5","llm_correctness_raw","#854397","-")]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), layout="constrained")
    for label, run, method, color, style in candidates:
        frame = pd.read_csv(ROOT / "runs" / run / "development" / f"{method}.risk_coverage.csv", float_precision="round_trip")
        for ax in axes:
            ax.step(frame.coverage, frame.risk, where="post", color=color, linestyle=style, label=label, linewidth=1.6)
    for ax, top, title in zip(axes, (.30,.025), ("Observed development risk", "Zoom on low observed risk")):
        ax.axhline(.01, color="#aa3636", linestyle="--", linewidth=1)
        ax.set(xlim=(0,1), ylim=(0,top), xlabel="Fraction accepted", title=title)
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0 if top == .30 else 1))
        ax.grid(alpha=.15)
    axes[0].set_ylabel("Errors among accepted predictions")
    axes[0].legend(loc="upper left", fontsize=7.5)
    axes[1].scatter([.895], [3/895], color=builder.TEAL)
    axes[1].annotate("895 accepted, 3 errors", (.895,3/895), xytext=(.25,.014),
                     arrowprops={"arrowstyle":"->", "color":builder.TEAL}, color=builder.TEAL, fontsize=9)
    builder.save(fig, "risk_coverage")
    builder.reliability()
    for source, name in (("mappings", "calibration_mappings"), ("reliability", "calibration_reliability")):
        for extension in ("svg", "png", "pdf"):
            shutil.copyfile(CAL / f"{source}.{extension}", OUT / "figures" / f"{name}.{extension}")
    if opus:
        for source,name in (("opus_mappings","opus_calibration_mappings"),("opus_reliability","opus_calibration_reliability")):
            for extension in ("svg","png","pdf"):
                shutil.copyfile(CAL/f"{source}.{extension}",OUT/"figures"/f"{name}.{extension}")


def main():
    original = read_json(OUT / "analysis.json")
    sonnet = read_json(OUT / "sonnet-comparison.json")
    calibration = read_json(CAL / "analysis.json")
    assert calibration['script_sha256']==hashlib.sha256((ROOT/'scripts/compare_calibrators.py').read_bytes()).hexdigest()
    cost_path = OUT / "cost-frontiers.json"
    cost = read_json(cost_path) if cost_path.exists() else None
    opus_path=OUT/"opus-comparison.json"
    opus=read_json(opus_path) if opus_path.exists() else None
    sources = {}
    for evidence in (original, sonnet, calibration, *([cost] if cost else []), *([opus] if opus else [])):
        for relative, expected in evidence["source_sha256"].items():
            assert relative not in sources or sources[relative] == expected, relative
            assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected, relative
            sources[relative] = expected
    for key in calibration["comparisons"]:
        assert calibration["comparisons"][key]["isotonic_fit"]["logistic_logit_slope"] > 0
    models = __import__("joblib").load(CAL / "calibrators.joblib")
    assert all(value["logistic_score"].model.coef_[0,0] > 0 for value in models.values())
    prob = diagnostics(OUT, include_sonnet=True, include_opus=bool(opus))
    probability = section_markdown(prob).replace("Haiku's learned scores assign", "Haiku's and Sonnet's learned scores assign")
    if opus:
        probability=probability.replace("Haiku's and Sonnet's learned scores assign low confidence to a much larger share of its mistakes while leaving all selected labels unchanged.","The Claude panels compare each model's original and learned confidence while keeping selected labels unchanged. Their score distributions and error counts show whether learned evidence separates mistakes from correct predictions.")
    (OUT / "probability-section.md").write_text(probability, encoding="utf-8")
    baseline = OUT / "report.before-sonnet-and-calibrators.md"
    if not baseline.exists():
        shutil.copyfile(OUT / "report.md", baseline)
    markdown = compose(baseline.read_text(encoding="utf-8"), sonnet, calibration, probability)
    if opus:
        from report_opus import integrate_opus
        markdown=integrate_opus(markdown,opus,calibration)
    if cost:
        cost_section = (OUT / "cost-section.md").read_text(encoding="utf-8")
        markdown = markdown.replace("## What the findings justify next", cost_section.strip() + "\n\n## What the findings justify next", 1)
        opening = "Cost changes the choice of confidence system. At observed batch fees, learned Haiku was cheaper and had lower Brier error than Sonnet native confidence, while Sonnet retained higher classification accuracy. Separate Pareto frontiers for accuracy, Brier error and error ranking make those trade-offs visible. Local compute, labeling and review costs remain explicit unpriced inputs rather than being treated as free.\n\n"
        if opus:
            opening="Cost and confidence quality change the deployment choice. The cost comparison includes all three Claude models, OpenAI Decisions and local classifiers, counting every call required by each pipeline. Separate Pareto frontiers distinguish classification accuracy, Brier error and error ranking. Local compute, labeling and review costs remain unpriced inputs in the sensitivity analysis.\n\n"
        markdown = markdown.replace("## The experiment in brief", opening + "## The experiment in brief", 1)
        markdown = markdown.replace("Freeze the complete candidate and its threshold before evaluating the untouched official test set.", "Freeze the complete candidate and its threshold before evaluating the untouched official test set. Price local inference, setup labels and human review before claiming total economic dominance.")
    (OUT / "report.md").write_text(markdown, encoding="utf-8")
    figures(sonnet,opus)
    captions = {
        "classification_accuracy": "Classifiers on the same 1,000 development messages; whiskers are marginal 95% Wilson accuracy intervals. The three supervised classifiers used 5,992 labeled fitting examples; LLM classifiers used zero-shot intent names. These are different task-specific training conditions.",
        "llm_confidence": "Within each Claude model, the original, verifier and learned scores retain identical selected labels. Whiskers are conditional 95% bootstrap intervals from 5,000 shared resamples. Learned scores combine extra same-model calls and supervised TF-IDF evidence; no feature ablation identifies the responsible ingredient.",
        "risk_coverage": "Descriptive risk–coverage curves on development data; tied scores enter together. The 1% line marks observed risk, while threshold selection uses the upper endpoint of a Wilson interval. The highlighted operating point was selected on these outcomes and requires independent validation.",
        "calibration_mappings": "Scalar mappings fitted only on the 1,000 calibration examples. Logistic on score and on log-odds learn slope and intercept with C=1. Isotonic fits a non-decreasing map with clipped out-of-range inputs. Temperature cannot be drawn as one universal map of top confidence because it transforms the full class distribution.",
        "calibration_reliability": "The fitted calibration methods evaluated on 1,000 separate development examples. Marker size grows with bin count; thin whiskers are nominal pointwise 95% Wilson intervals. Sparse bins remain uncertain. All methods within a panel retain identical selected labels.",
        **PROB_CAPTIONS,
        "reliability_counts": builder.CAPTIONS["reliability_counts"].removeprefix("Figure 4. "),
        **(cost["captions"] if cost else {}),
    }
    builder.CAPTIONS = captions
    if opus:
        captions.update({"opus_calibration_mappings":"Opus native and learned correctness mappings fitted only on the 1,000 calibration outcomes, using the same logistic and isotonic definitions as the earlier classifiers.",
                         "opus_calibration_reliability":"Opus calibration methods evaluated on the separate 1,000 development outcomes. Marker size reflects bin count; thin whiskers are nominal 95% Wilson intervals. Selected labels remain fixed within each panel.",
                         "opus_probability_reliability":"Opus original and learned scores on the same 1,000 development messages, with selected labels held fixed. Points use ten shared equal-width bins, with nominal 95% Wilson intervals; count histograms show the evidence in each bin. Empty bins are omitted.",
                         "opus_probability_outcomes":"Opus confidence distributions normalized separately within correct and incorrect development outcomes. Legend counts give the denominators. Separation measures review-ranking potential, while reliability must be assessed in the companion figure."})
        for key in ('probability_reliability','probability_outcomes'):
            captions[key]=captions[key].replace('Haiku and Sonnet','All three Claude models’').replace('within Haiku or Sonnet','within each Claude model')
    builder.INSERT_BEFORE = {}
    body, navigation = builder.render(markdown)
    # Place independent figures at complete section boundaries.
    inserts = {
        "learning-correctness-materially-improved-haiku-confidence": ["classification_accuracy"],
        "better-error-ranking-did-not-guarantee-low-risk-automation": ["llm_confidence"],
        "temperature-logistic-and-isotonic-calibration": ["risk_coverage"],
        "comparing-the-probability-curves": ["calibration_mappings", "calibration_reliability"],
        "why-the-embedding-curves-looked-volatile": ["probability_reliability", "probability_outcomes"],
        "what-the-findings-justify-next": ["reliability_counts"],
    }
    if cost:
        inserts.update({"cost-and-pareto-frontiers":["reliability_counts"],
                        "local-cost-sensitivity-and-study-spending":["cost_pareto"],
                        "what-the-findings-justify-next":["cost_sensitivity"]})
    if opus:
        inserts['comparing-the-probability-curves'] += ['opus_calibration_mappings','opus_calibration_reliability']
        inserts['better-error-ranking-did-not-guarantee-low-risk-automation'] += ['opus_probability_reliability','opus_probability_outcomes']
    for anchor, names in inserts.items():
        marker = f'<h2 id="{anchor}">'
        assert body.count(marker) == 1, anchor
        body = body.replace(marker, "\n".join(builder.figure(name) for name in names) + marker, 1)
    style = builder.STYLE + "\nfigure svg text{font-family:'Segoe UI',Arial,sans-serif!important}\n@media print{figure{max-width:100%}figure svg{max-height:225mm}thead{display:table-header-group}tr{break-inside:avoid}}"
    page = '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>' + TITLE + '</title><style>' + style + '</style></head><body><div class="topbar"><span>BANKING77 · Complete findings · 9 October 2026</span><button onclick="window.print()">Print or save as PDF</button></div><main>' + body + '<nav aria-label="Report sections">' + navigation + '</nav><footer>Evidence: <a href="consolidated-evidence.json">consolidated source record</a> · <a href="analysis.json">original paired analysis</a> · <a href="sonnet-comparison.json">Sonnet replication</a> · <a href="../calibrator-comparison/analysis.json">calibration comparisons</a> · <a href="report.md">report source</a><br>All findings use development data. The official test set remains reserved.</footer></main></body></html>'
    (OUT / "report.html").write_text(page, encoding="utf-8")
    write_json(OUT / "figure_captions.json", captions)
    sources.update(prob["source_sha256"])
    for path in (OUT / "analysis.json", OUT / "sonnet-comparison.json", CAL / "analysis.json", CAL / "calibrators.joblib", Path(__file__), ROOT/'scripts/compare_calibrators.py',ROOT/'scripts/add_probability_comparisons.py', *([cost_path, OUT/"cost-section.md", ROOT/"scripts/analyze_cost_frontiers.py"] if cost else []), *([opus_path,ROOT/"scripts/analyze_opus_comparison.py",ROOT/"scripts/report_opus.py",ROOT/'scripts/finish_opus_replication.py'] if opus else [])):
        sources[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    for relative, expected in sources.items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected, relative
    write_json(OUT / "consolidated-evidence.json", {"updated_utc":datetime.now(timezone.utc).isoformat(),
        "official_test_evaluated":False, "sonnet_included":True, "isotonic_included":True,
        "logistic_score_included":True, "logistic_logit_included":True, "calibration_variants":sum(len(v) for v in calibration['results'].values()),
        "opus_included":bool(opus),
        "cost_and_pareto_included":bool(cost),
        "source_sha256":sources, "figure_captions":captions,
        "report_sha256":hashlib.sha256((OUT/"report.md").read_bytes()).hexdigest()})
    write_json(OUT / "page-update-content.json", {"markdown":markdown, "captions":captions,
        "inserts":inserts, "figures":{name:(OUT/"figures"/f"{name}.svg").read_text(encoding="utf-8")
                                       for name in captions}})
    print(f"Complete report: {OUT/'report.html'}")
    print(f"Verified {len(sources)} source hashes; {len(captions)} figures; {len(markdown.split())} words.")


if __name__ == "__main__":
    main()
