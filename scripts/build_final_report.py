"""Independent test report from completed frozen arms; no fitting or API calls."""
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import re
import shutil
import subprocess

import joblib
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from confidence_experiment.io import read_json, write_json
from confidence_experiment.metrics import evaluate, wilson
from final_protocol import ROOT, OUT as RUN, PLAN, preflight, completed, sha, transform
from analyze_cost_frontiers import frontier

OUT = ROOT / "reports/banking77-findings"
FIG = OUT / "figures"
NAMES = {"embedding_lr": "EmbeddingGemma LR", "embedding_mlp": "EmbeddingGemma MLP", "tfidf": "TF-IDF LR",
         "decisions_confidence": "OpenAI confidence", "decisions_probability": "OpenAI probability"}
METHODS = {"raw": "Raw", "temperature": "Temperature", "logistic_score": "Logistic score",
           "logistic_logit": "Logistic log-odds", "isotonic": "Isotonic"}
COLORS = {"raw": "#87909e", "temperature": "#b85b22", "logistic_score": "#4e7189",
          "logistic_logit": "#256b48", "isotonic": "#8c47a2"}
CAPTIONS = {
    "classification_accuracy": "The same frozen classifiers on development (1,000 messages) and official test (3,080 messages). Whiskers are marginal 95% Wilson intervals. The two splits contain different messages; changes between splits are not paired comparisons. Local models used 5,992 fitting labels; OpenAI used zero-shot intent names.",
    "risk_coverage": "Descriptive official-test risk curves with score ties preserved. The highlighted point uses the primary threshold fixed on development. The 1% line is the intended error target. Curves do not authorize selecting a replacement threshold from test outcomes.",
    "calibration_mappings": "Unchanged scalar calibrators fitted on the 1,000 calibration messages, shown for the confidence sources tested here. Logistic regression learns slope and intercept; isotonic introduces plateaus and ties. Temperature transforms all class logits and cannot be drawn as one universal map of top confidence.",
    "calibration_reliability": "Every calibration variant on the official test set, using the original fitted objects. Panels hold selected labels fixed; marker sizes reflect bin counts. Thin whiskers are nominal pointwise 95% Wilson intervals. Shared equal-width bins expose sparse regions without smoothing.",
    "probability_reliability": "Raw versus frozen calibrated correctness probabilities on the same 3,080 test messages. Bin counts and nominal 95% Wilson intervals show the evidence behind each point. Empty bins are omitted. All transformations preserve classifier-selected labels.",
    "probability_outcomes": "Test confidence split by correct and incorrect outcomes, normalized separately within each outcome group. Legend counts show the denominators. Separation describes error-ranking ability; the two curves do not represent overall error prevalence.",
    "cost_pareto": "Official-test Pareto frontiers for the 24 completed local/OpenAI variants only. Prices count unique recorded API requests per 1,000 messages; local compute and labeling are unpriced. Rings indicate nondominated point estimates. Claude development results are excluded from these test frontiers.",
    "cost_sensitivity": "Illustrative accuracy frontiers with assumed all-in local embedding costs of $0, $0.10, $1, or $3 per 1,000 messages. These are assumptions rather than electricity or deployment-cost measurements. TF-IDF local cost is also unpriced and is omitted from this sensitivity projection.",
    "llm_confidence": "Historical development results only: Haiku, Sonnet and Opus on 1,000 messages, with original, verifier and learned confidence retaining identical selected labels within each model. Learned confidence combines additional API calls and supervised TF-IDF evidence. Claude official-test replication is deferred.",
}
INSERTS = {
    "classification-on-independent-data": ["classification_accuracy"],
    "temperature-logistic-and-isotonic-calibration": ["risk_coverage"],
    "comparing-the-probability-curves": ["calibration_mappings", "calibration_reliability"],
    "cost-and-pareto-frontiers": ["probability_reliability", "probability_outcomes"],
    "claude-findings-remain-development-only": ["cost_pareto", "cost_sensitivity"],
    "what-the-findings-justify-next": ["llm_confidence"],
}


def table(headers, rows):
    return "| " + " | ".join(headers) + " |\n|" + "|".join("---" for _ in headers) + "|\n" + "\n".join("| " + " | ".join(map(str, row)) + " |" for row in rows)


def pct(value):
    return f"{100 * value:.2f}%"


def ci(values, percentage=True):
    if values is None:
        return "—"
    scale = 100 if percentage else 1
    suffix = "%" if percentage else ""
    return f"{scale * values[0]:.2f}–{scale * values[1]:.2f}{suffix}"


def export(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    for extension in ("svg", "png", "pdf"):
        fig.savefig(FIG / f"{name}.{extension}", dpi=160, bbox_inches="tight")
    plt.close(fig)


def reliability(ax, frame, color, label, counts=False):
    q, correct = frame.confidence.to_numpy(), frame.correct.to_numpy()
    ids = np.minimum((q * 10).astype(int), 9)
    xs, ys, ns, low, high = [], [], [], [], []
    for index in range(10):
        mask = ids == index
        n = int(mask.sum())
        if not n:
            continue
        y = correct[mask].mean()
        interval = wilson(int(correct[mask].sum()), n)
        xs.append(q[mask].mean()); ys.append(y); ns.append(n)
        low.append(max(0, y - interval[0])); high.append(max(0, interval[1] - y))
    ax.errorbar(xs, ys, yerr=[low, high], fmt="none", ecolor=color, alpha=.6, capsize=2)
    ax.scatter(xs, ys, s=12 + np.sqrt(ns) * 3, color=color, label=label, alpha=.8)
    if counts:
        for x, y, n in zip(xs, ys, ns):
            ax.annotate(str(n), (x, y), xytext=(-3 if x > .9 else 3, 6 if y < .1 else -10),
                        ha="right" if x > .9 else "left", textcoords="offset points", fontsize=7)
    ax.plot([0, 1], [0, 1], "--", color="#bbb", linewidth=.8)
    ax.set(xlim=(0, 1), ylim=(0, 1.03), xlabel="Reported correctness probability", ylabel="Observed correctness")
    ax.grid(alpha=.15)


def make_figures(metrics, frames, plan, pipelines, frontiers, development):
    keys = ["embedding_lr.temperature", "embedding_mlp.temperature", "tfidf.temperature", "decisions_probability.logistic_logit"]
    titles = ["Embedding LR", "Embedding MLP", "TF-IDF LR", "OpenAI Decisions"]
    fig, ax = plt.subplots(figsize=(8.5, 4))
    for offset, role, values, color in ((-.12, "Development", [plan["variants"][k]["development_metrics"] for k in keys], "#87909e"),
                                      (.12, "Official test", [metrics[k] for k in keys], "#256b48")):
        x = np.arange(4) + offset
        y = np.array([m["accuracy"] for m in values])
        intervals = np.array([m["accuracy_ci95"] for m in values])
        ax.errorbar(x, y, yerr=[y - intervals[:, 0], intervals[:, 1] - y], fmt="o", capsize=4, color=color, label=role)
        for a, b in zip(x, y):
            ax.annotate(pct(b), (a, b), xytext=(-4 if role == "Development" else 4, 7),
                        textcoords="offset points", ha="right" if role == "Development" else "left", fontsize=8)
    ax.set(xticks=np.arange(4), xticklabels=titles, ylabel="Accuracy", ylim=(.60, 1.01))
    ax.yaxis.set_major_formatter(PercentFormatter(1)); ax.legend(); ax.grid(axis="y", alpha=.2)
    export(fig, "classification_accuracy")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    full_risk_max = 0.0
    for key in keys:
        curve = pd.read_csv(RUN / plan["variants"][key]["arm"] / f"{key}.risk_coverage.csv")
        full_risk_max = max(full_risk_max, float(curve.risk.max()))
        for ax in axes:
            ax.plot(curve.coverage, curve.risk, label=NAMES[plan["variants"][key]["stream"]], linewidth=1.3)
    a = metrics[plan["primary"]]["acceptance"]
    for ax in axes:
        ax.scatter(a["coverage"], a["risk"], color="#b85b22", zorder=5)
        ax.axhline(.01, linestyle="--", color="#b85b22", linewidth=.8)
        ax.set(xlabel="Accepted fraction", ylabel="Error rate among accepted")
        ax.xaxis.set_major_formatter(PercentFormatter(1)); ax.yaxis.set_major_formatter(PercentFormatter(1)); ax.grid(alpha=.2)
    axes[0].set(title="Full test risk curves", ylim=(0, min(1, full_risk_max + .02)))
    axes[1].set(title="Low-risk region and frozen primary point", ylim=(0, .04))
    axes[0].legend(fontsize=8)
    export(fig, "risk_coverage")

    objects = joblib.load(ROOT / "reports/calibrator-comparison/calibrators.joblib")
    chosen = ["embedding_lr", "embedding_mlp", "tfidf", "decisions_probability"]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7), constrained_layout=True)
    grid = np.linspace(0, 1, 501)
    for ax, key in zip(axes.flat, chosen):
        for method, values in transform(key, grid, ["label"] * len(grid), objects).items():
            ax.plot(grid, values, label=METHODS[method], color=COLORS[method], linewidth=1.2)
        ax.set(title=NAMES[key], xlabel="Raw confidence", ylabel="Calibrated confidence", xlim=(0, 1), ylim=(0, 1)); ax.grid(alpha=.2)
    axes.flat[0].legend(fontsize=8)
    export(fig, "calibration_mappings")
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7), constrained_layout=True)
    for ax, key in zip(axes.flat, chosen):
        for method in METHODS:
            name = key + "." + method
            if name in frames: reliability(ax, frames[name], COLORS[method], METHODS[method])
        ax.set_title(NAMES[key])
    axes.flat[0].legend(fontsize=8)
    export(fig, "calibration_reliability")
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7), constrained_layout=True)
    pairs = [("embedding_lr.raw", "Embedding LR raw"), ("embedding_lr.temperature", "Embedding LR temperature"),
             ("decisions_probability.raw", "OpenAI probability raw"), ("decisions_probability.logistic_logit", "OpenAI probability logistic log-odds")]
    for ax, (name, title) in zip(axes.flat, pairs):
        reliability(ax, frames[name], "#256b48", title, counts=True); ax.set_title(title)
    export(fig, "probability_reliability")
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7), constrained_layout=True)
    edges = np.linspace(0, 1, 11)
    for ax, key, title in zip(axes.flat, keys, titles):
        f = frames[key]
        for correct, color, label in ((True, "#256b48", "Correct"), (False, "#b85b22", "Incorrect")):
            q = f.loc[f.correct == correct, "confidence"].to_numpy()
            hist, _ = np.histogram(q, edges)
            ax.step(edges[:-1] + .05, hist / len(q), where="mid", color=color, label=f"{label} (n={len(q)})")
        ax.set(title=title, xlabel="Correctness probability", ylabel="Share within outcome", xlim=(0, 1)); ax.grid(alpha=.2); ax.legend(fontsize=8)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
    export(fig, "probability_outcomes")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    palette = {"embedding_lr":"#256b48", "embedding_mlp":"#4e7189", "tfidf":"#87909e", "decisions_confidence":"#b85b22", "decisions_probability":"#b85b22"}
    for ax, field, title in zip(axes, ("accuracy", "brier", "error_auroc"), ("Accuracy", "Brier error", "Error AUROC")):
        for row in pipelines:
            ax.scatter(row["api_usd_per_1000"], row[field], color=palette[row["stream"]], alpha=.65, s=25)
            if row["id"] in frontiers[field]:
                ax.scatter(row["api_usd_per_1000"], row[field], facecolors="none", edgecolors="#20252b", s=100)
        ax.set(title=title, xlabel="API USD per 1,000 test messages", ylabel=title); ax.grid(alpha=.2)
    legend = [Line2D([], [], marker="o", linestyle="none", color=palette[key], label=label)
              for key, label in (("embedding_lr", "Embedding LR"), ("embedding_mlp", "Embedding MLP"),
                                 ("tfidf", "TF-IDF LR"), ("decisions_probability", "OpenAI"))]
    axes[0].legend(handles=legend, fontsize=7, loc="lower right")
    export(fig, "cost_pareto")
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.5), sharey=True, constrained_layout=True)
    for ax, price in zip(axes, (0, .10, 1, 3)):
        points = []
        for key in ("embedding_lr.temperature", "embedding_mlp.temperature", "decisions_probability.logistic_logit"):
            row = next(r for r in pipelines if r["id"] == key).copy()
            if row["stream"].startswith("embedding"): row["api_usd_per_1000"] = price
            points.append(row)
        ids = frontier(points, "accuracy")
        for row in points:
            ax.scatter(row["api_usd_per_1000"], row["accuracy"], color=palette[row["stream"]], s=45)
            if row["id"] in ids: ax.scatter(row["api_usd_per_1000"], row["accuracy"], facecolors="none", edgecolors="#20252b", s=140)
        ax.set(title=f"Local assumption ${price:g}", xlabel="USD per 1,000"); ax.grid(alpha=.2)
    axes[0].set_ylabel("Test accuracy"); axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[-1].legend(handles=[legend[0], legend[1], legend[3]], fontsize=7, loc="lower right")
    export(fig, "cost_sensitivity")


def markdown_html(markdown):
    # Small deterministic renderer for this report's headings, paragraphs and GFM tables.
    def inline(text):
        text = html.escape(text)
        text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
        text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', text)
        return text
    answer = []
    for block in re.split(r"\n\s*\n", markdown.strip()):
        if block.startswith("#"):
            marks, title = block.split(" ", 1)
            slug = re.sub("[^a-z0-9]+", "-", title.lower()).strip("-")
            if len(marks) == 2:
                for key in INSERTS.get(slug, []):
                    svg = (FIG / (key + ".svg")).read_text(encoding="utf-8")
                    svg = svg[svg.index("<svg"):svg.index("</svg>") + 6]
                    answer.append(f'<figure>{svg}<figcaption>{inline(CAPTIONS[key])}</figcaption></figure>')
            answer.append(f'<h{len(marks)} id="{slug}">{inline(title)}</h{len(marks)}>')
        elif block.startswith("|"):
            lines = block.splitlines()
            cells = lambda line: [c.strip() for c in line.strip("|").split("|")]
            answer.append("<table><thead><tr>" + "".join("<th>" + inline(c) + "</th>" for c in cells(lines[0])) + "</tr></thead><tbody>" +
                          "".join("<tr>" + "".join("<td>" + inline(c) + "</td>" for c in cells(line)) + "</tr>" for line in lines[2:]) + "</tbody></table>")
        else:
            answer.append("<p>" + inline(block) + "</p>")
    return "\n".join(answer)


def main():
    plan = preflight()
    scope = read_json(RUN / "execution-scope.json")
    if not all(completed(arm) for arm in scope["active_arms"]):
        raise ValueError("Requested test scope is incomplete; no partial numbers substituted")
    if set(scope["active_arms"]) != {"embeddings", "tfidf", "decisions"}:
        raise ValueError("This report currently supports the explicitly requested local/OpenAI scope")
    OUT.mkdir(parents=True, exist_ok=True)
    # Retain the complete earlier development report and its Page authoring source.
    for name in ("report.md", "report.html", "report.pdf", "page-update-content.json", "publication.json"):
        original = OUT / name
        archived = OUT / (name.rsplit(".", 1)[0] + ".development." + name.rsplit(".", 1)[1])
        if original.exists() and not archived.exists(): shutil.copy2(original, archived)
    metrics, frames, costs, inputs = {}, {}, {}, dict(plan["input_sha256"])
    reference = pd.read_csv(ROOT / "data/banking77/test.csv", keep_default_na=False)
    for arm in scope["active_arms"]:
        folder = RUN / arm
        current = read_json(folder / "metrics.json")
        metadata = read_json(folder / "metadata.json")
        costs[arm] = metadata["costs"]
        for path in folder.glob("*"):
            if path.is_file() and path.name not in {"status.json", "request_progress.json", ".lock"}:
                inputs[path.relative_to(ROOT).as_posix()] = sha(path)
        for key, m in current.items():
            f = pd.read_csv(folder / f"{key}.predictions.csv", float_precision="round_trip", keep_default_na=False)
            if len(f) != 3080 or not f.id.is_unique or not np.array_equal(f.id, reference.id) or not np.array_equal(f.truth, reference.label):
                raise ValueError(f"Wrong test rows: {key}")
            if not np.array_equal(f.correct, f.truth == f.prediction): raise ValueError("Invalid correctness flags")
            checked, _, _ = evaluate(f.truth, f.prediction, f.confidence, plan["variants"][key]["threshold"], labels=read_json(ROOT / "data/banking77/manifest.json")["labels"])
            for field in ("accuracy", "brier", "binary_log_loss", "error_auroc"):
                if not np.isclose(checked[field], m[field], atol=1e-7, rtol=0): raise ValueError("Saved metric mismatch")
            metrics[key], frames[key] = m, f
    if len(metrics) != 24: raise ValueError("Expected 24 completed test variants")
    boot = np.random.default_rng(plan["bootstrap"]["seed"]).integers(0, 3080, size=(5000, 3080))
    comparisons = {}
    pairs = [("lr_minus_mlp_accuracy", "accuracy", "embedding_lr.temperature", "embedding_mlp.temperature"),
             ("lr_minus_tfidf_accuracy", "accuracy", "embedding_lr.temperature", "tfidf.temperature"),
             ("lr_minus_openai_accuracy", "accuracy", "embedding_lr.temperature", "decisions_probability.logistic_logit"),
             ("lr_isotonic_minus_temperature_brier", "brier", "embedding_lr.isotonic", "embedding_lr.temperature"),
             ("lr_logistic_score_minus_temperature_brier", "brier", "embedding_lr.logistic_score", "embedding_lr.temperature"),
             ("openai_isotonic_minus_logit_brier", "brier", "decisions_probability.isotonic", "decisions_probability.logistic_logit")]
    for name, metric, a, b in pairs:
        if metric == "accuracy": difference = frames[a].correct.to_numpy(dtype=float) - frames[b].correct.to_numpy(dtype=float)
        else: difference = ((frames[a].confidence - frames[a].correct) ** 2 - (frames[b].confidence - frames[b].correct) ** 2).to_numpy()
        interval = np.quantile(difference[boot].mean(axis=1), [.025, .975])
        comparisons[name] = {"metric": metric, "first": a, "second": b, "difference": float(difference.mean()), "ci95": interval.tolist()}
    del boot
    distinct = {r["request_sha256"]: r for r in costs["decisions"]}
    api_cost = sum(r["usd"] for r in distinct.values())
    pipelines = []
    for key, m in metrics.items():
        stream = plan["variants"][key]["stream"]
        pipelines.append({"id": key, "stream": stream, "api_usd_per_1000": api_cost / 3080 * 1000 if stream.startswith("decisions") else 0,
                          "accuracy": m["accuracy"], "brier": m["brier"], "error_auroc": m["error_auroc"],
                          "coverage": m["acceptance"]["coverage"]})
    frontiers = {field: frontier(pipelines, field, minimize=field == "brier") for field in ("accuracy", "brier", "error_auroc", "coverage")}
    development = read_json(ROOT / "reports/calibrator-comparison/analysis.json")
    make_figures(metrics, frames, plan, pipelines, frontiers, development)
    primary = metrics[plan["primary"]]; a = primary["acceptance"]
    accuracy_rows = []
    for key in ("embedding_lr.temperature", "embedding_mlp.temperature", "tfidf.temperature", "decisions_probability.logistic_logit"):
        m = metrics[key]; spec = plan["variants"][key]
        accuracy_rows.append([NAMES[spec["stream"]], pct(spec["development_metrics"]["accuracy"]), pct(m["accuracy"]), ci(m["accuracy_ci95"]), m["errors"], f"{m['macro_f1']:.3f}"])
    policy_keys = ["embedding_lr.raw", "embedding_lr.temperature", "embedding_lr.logistic_logit", "embedding_lr.isotonic",
                   "embedding_mlp.raw", "embedding_mlp.temperature", "embedding_mlp.isotonic", "tfidf.temperature", "decisions_probability.logistic_logit"]
    policy_rows = []
    for key in policy_keys:
        m = metrics[key]; accept = m["acceptance"]; spec = plan["variants"][key]
        policy_rows.append([NAMES[spec["stream"]] + " " + METHODS[spec["variant"]], accept["accepted"], accept["errors"],
                            pct(accept["coverage"]), pct(accept["risk"]) if accept["risk"] is not None else "—", ci(accept["risk_ci95"])])
    calibration_rows = [[NAMES[stream]] + [f"{metrics[stream+'.'+variant]['brier']:.5f}" if stream + "." + variant in metrics else "—" for variant in METHODS]
                        for stream in NAMES]
    chosen = ["embedding_lr.temperature", "embedding_mlp.temperature", "tfidf.temperature", "decisions_probability.logistic_logit"]
    cost_rows = [[NAMES[plan["variants"][key]["stream"]], f"${next(r for r in pipelines if r['id']==key)['api_usd_per_1000']:.4f}",
                 pct(metrics[key]["accuracy"]), f"{metrics[key]['brier']:.5f}", f"{metrics[key]['error_auroc']:.3f}"] for key in chosen]
    claude_rows = []
    dev_cost = read_json(OUT / "cost-frontiers.json")
    for family in ("haiku", "sonnet", "opus"):
        raw = development["results"][family + "_same"]["raw"]
        learned = development["results"][family + "_correctness"]
        cost = next(r["api_usd_per_1000"] for r in dev_cost["pipelines"] if r["id"] == family + "_correctness.logistic_score")
        claude_rows.append([family.title() + " 5.5", pct(raw["accuracy"]), f"{raw['brier']:.5f}",
                            f"{learned['raw']['brier']:.5f}", f"{learned['logistic_score']['brier']:.5f}", f"${cost:.2f}"])
    brier_ci = comparisons["lr_isotonic_minus_temperature_brier"]
    accuracy_ci = comparisons["lr_minus_mlp_accuracy"]
    decision = metrics["decisions_probability.logistic_logit"]
    refusals = int((frames["decisions_probability.raw"].prediction == "__refusal__").sum())
    lr_iso = metrics["embedding_lr.isotonic"]["acceptance"]
    encoding = read_json(RUN / "embeddings/metadata.json")["details"]
    markdown = f"""# Learning When to Trust a Classifier

**BANKING77 findings · 13 October 2026 · Independent test validation for local models and OpenAI**

The independent test set changed the most important conclusion. The primary EmbeddingGemma classifier retained high accuracy, **{pct(primary['accuracy'])}**, but its frozen automation policy accepted **{a['accepted']:,} messages with {a['errors']} errors**. That is **{pct(a['risk'])} error**, with a nominal 95% interval of **{ci(a['risk_ci95'])}**. The intended maximum was 1%. **The policy failed independent validation.**

Development had suggested that temperature-scaled embedding LR could accept 89.5% of messages with 3 errors in 895 accepted predictions. On the official test set, coverage was **{pct(a['coverage'])}**, with substantially more accepted errors. The development result was useful for choosing a candidate; it overstated the evidence for dependable 1% automation.

All **24 local and OpenAI score variants** were evaluated on the same **3,080 official test messages**, using the existing fitted models, calibrators, and development-selected thresholds. No test outcomes were used to refit a calibrator or choose a replacement threshold. Claude test evaluation is deferred; Haiku, Sonnet, and Opus findings below remain development-only.

## The independent evaluation protocol

BANKING77 contains short banking messages spanning 77 closely related intents. Its data audit removed seven normalized training texts overlapping the official test set and four duplicate training rows. Stratified seed-42 partitions separated classifier fitting, correctness learning, calibration, and development. The official test labels were first used for model evaluation in this final stage. The test retains one normalized duplicate text pair, so independence-based uncertainty intervals are approximate. [Original dataset paper](https://aclanthology.org/2020.nlp4convai-1.5/)

{table(['Data role','Examples','Use'],[['Classifier fitting',5992,'Fit TF-IDF and embedding heads'],['Correctness training',2000,'Fit the historical Claude correctness estimators'],['Calibration',1000,'Fit temperature, logistic and isotonic calibrators'],['Development',1000,'Compare candidates and select thresholds'],['Official test',3080,'Independent evaluation with all choices fixed']])}

The final plan sealed 60 variants before test inference. At the user's request, execution was restricted to the local models and OpenAI: 24 variants across five confidence streams. The remaining 36 Claude variants retain their frozen specifications but have no test results. This execution change did not alter any fitted model, calibrator, or threshold.

## Classification on independent data

{table(['Classifier','Development accuracy','Test accuracy','Test accuracy 95% interval','Test errors','Test macro F1'],accuracy_rows)}

Embedding LR remained more accurate than the tested alternatives. Its paired test accuracy difference from the MLP was **{100*accuracy_ci['difference']:+.2f} percentage points**, with a nominal 95% interval of **{ci(accuracy_ci['ci95'])}**. These models use the same fixed embedding representation, so this is a comparison of the fitted classifier heads, not evidence that nonlinear heads cannot help with other settings or seeds.

The embedding LR made **{primary['errors']} test errors**, compared with 29 development errors. The two datasets differ in size and difficulty: the error rate rose from 2.9% to {pct(1-primary['accuracy'])}. TF-IDF remained near its earlier 87.1% accuracy. The official test results therefore make the earlier embedding advantage smaller without eliminating it.

OpenAI Decisions correctly classified **{decision['n']-decision['errors']:,} of 3,080 messages ({pct(decision['accuracy'])})**. Its **{refusals} refusals** remain in the denominator as unsuccessful classifications and always require review. API-selected labels are preserved even when a returned probability distribution has a different argmax. The local classifiers received 5,992 fitting labels; OpenAI received zero-shot intent names, so cross-family accuracy differences combine model choice and access to supervision.

## The frozen automation policy failed its target

{table(['Frozen policy','Accepted','Errors','Test coverage','Accepted error rate','95% error interval'],policy_rows)}

The primary temperature threshold was fixed at **{plan['variants'][plan['primary']]['threshold']['threshold']:.8f}** before any test prediction was scored. Its error interval lies entirely above 1%, so this failure is more than an upper interval merely crossing the target. The system would send {3080-a['accepted']} of 3,080 messages to review while leaving {a['errors']} incorrect classifications among the automatically accepted messages.

Isotonic embedding LR accepted fewer messages, **{lr_iso['accepted']:,}**, but still made **{lr_iso['errors']} errors ({pct(lr_iso['risk'])})**. Every tested embedding policy had a 95% upper error endpoint above 1%. A more flexible calibrator did not rescue the existing automation rule.

TF-IDF and all OpenAI variants retained null acceptance thresholds because no threshold met the development criterion. Their zero test coverage follows that frozen policy. It does not establish that every possible test threshold would fail; searching the test curves for a new threshold would turn this final set into tuning data.

The original selection rule searched development thresholds for the largest acceptance set whose upper 95% Wilson error endpoint was at most 1%. A nominal interval at a threshold chosen using the same outcomes does not account for that search. The independent test exposed the gap between a promising development operating point and a validated operational policy.

## What calibration did and did not fix

The table compares binary correctness Brier error on official test data. Within each row, every variant retains the same predicted labels. Smaller values are better. Temperature optimizes multiclass calibration loss; the scalar methods target selected-label correctness. Their objectives differ, so a binary Brier result does not establish superiority of the full class distribution.

{table(['Confidence source']+list(METHODS.values()),calibration_rows)}

For embedding LR, temperature reduced Brier error from **{metrics['embedding_lr.raw']['brier']:.5f} to {primary['brier']:.5f}**. Its test binary log loss was **{primary['binary_log_loss']:.5f}**. Isotonic minus temperature Brier was **{brier_ci['difference']:+.5f}**, with a paired nominal 95% interval of **[{brier_ci['ci95'][0]:+.5f}, {brier_ci['ci95'][1]:+.5f}]**. Positive differences favor temperature. This comparison uses frozen fits and provides no license to choose a new deployment configuration after looking at test outcomes.

The result supports the usefulness of simple scaling while showing its limit: improved average probability quality did not ensure that a high-confidence subset had less than 1% error. Temperature cannot repair missing information or guarantee that mistakes will be ranked below every accepted case. Logistic maps can adjust slope and intercept; positive slopes preserve raw score ranking. Isotonic can introduce tied groups and exact zero/one probabilities. Log loss clips at 10⁻⁷ consistently with the earlier analyses.

All calibrators still learned from only 1,000 calibration examples. More test mistakes improve evaluation precision, but they do not add training evidence to an already fitted isotonic model. The calibration partition contained only 39 embedding LR errors, which remains a constraint on flexible fitted mappings.

## Reading the probability curves

Reliability plots compare a reported correctness probability with the observed fraction correct. They use ten shared equal-width bins, sample-size markers, and nominal pointwise Wilson intervals. Empty bins are omitted; points are neither smoothed nor joined into a visually authoritative curve. Sparse bins remain uncertain even with 3,080 test examples because most scores concentrate near one.

The score distributions answer a different question: whether correct and incorrect predictions receive different scores. Each outcome is normalized separately, with its sample count retained in the legend. Error AUROC summarizes ranking across all thresholds. For embedding LR it was **{primary['error_auroc']:.3f}** after temperature scaling; for embedding MLP temperature it was **{metrics['embedding_mlp.temperature']['error_auroc']:.3f}**. Ranking, average probability accuracy, and performance at the frozen acceptance threshold should be assessed separately.

## Cost and Pareto frontiers on test

{table(['Fixed representative pipeline','API USD per 1,000','Test accuracy','Test Brier','Test error AUROC'],cost_rows)}

The OpenAI test cohort used **{len(distinct):,} distinct requests** and recorded **${api_cost:.4f}** in API fees. Duplicate-message cache reuse is counted once for cohort costs. The per-1,000 figure divides those unique recorded fees by all 3,080 evaluated messages. All OpenAI calibrators reuse the same predictions and add no API requests. Fee amounts are client-ledger estimates computed from returned token usage at the configured rate; they are not invoice totals. No Claude test batch was accepted or charged.

Local provider fees are zero; local computation, labeling, setup, and human review are not free. On this CPU, encoding the 3,080 test messages took **{encoding['encoding_seconds']:.1f} seconds** with two threads and batch size eight, excluding **{encoding['model_load_seconds']:.1f} seconds** to load the pinned encoder. This is one observed batch-throughput measurement, not a service-latency or electricity-cost estimate. The embedding heads reuse those encodings.

The test Pareto sets include the 24 completed variants only. A point is dominated when another costs no more, performs at least as well, and improves at least one selected objective. Accuracy, Brier, error ranking, and coverage have separate frontiers. Exact equivalents remain together; small differences are descriptive point estimates. Claude development points are excluded from test frontiers because mixing evaluation datasets would create a misleading comparison.

For a deployment of N messages, fixed setup/annotation cost F, local variable cost v per message, API fees A per 1,000, review fraction r, and review cost R per message, total USD per 1,000 is **A + 1,000v + 1,000F/N + 1,000rR**. F, v, and R remain unmeasured. A policy's review savings cannot justify claiming a 1% quality target that failed independent validation.

## Claude findings remain development only

{table(['Claude model','Development accuracy','Native Brier','Learned Brier','Learned plus logistic Brier','Learned API USD per 1,000'],claude_rows)}

Learned correctness improved the development confidence scores for Haiku, Sonnet, and Opus while keeping their selected labels unchanged. It combined original confidence, verifier confidence, blind-classification agreement, length/missing-content signals, and supervised TF-IDF support. Those gains cannot be attributed to learning alone because extra calls and supervised features were added together, with no feature ablation.

The historical Opus configuration used adaptive thinking and a 2,048-token output allowance; Sonnet and Haiku used their supported thinking settings and 256 tokens. These are comparisons of implemented configurations. The learned pipeline requires three calls per message; full Claude test replication was estimated near $56 for all three models combined, predominantly Opus. These development findings have not yet been tested on the official test messages and should not be compared directly with local/OpenAI test metrics as a provider ranking.

The full illustrated development study is preserved as [HTML](https://github.com/miegielsen/confidence-experiment/blob/main/reports/banking77-findings/report.development.html), [PDF](https://github.com/miegielsen/confidence-experiment/blob/main/reports/banking77-findings/report.development.pdf), and [Markdown](https://github.com/miegielsen/confidence-experiment/blob/main/reports/banking77-findings/report.development.md), including all 60 calibration variants, Opus, Sonnet, probability diagnostics, and the separate development cost frontier. The historical observations remain useful; the independent test changes the automation recommendation.

## What should happen next

EmbeddingGemma LR remains the strongest tested classifier by accuracy in this implementation. Temperature scaling remains useful for average probability quality. **The existing high-coverage policy should not be deployed as a validated 1% error system.** Improving classification, gathering more representative calibration/threshold-selection data, or using a selection procedure with explicit risk-control assumptions are reasonable next experiments. Any revised policy requires new independent validation data; the official test outcomes are now known for the completed arms.

A harder or more representative task such as LEDGAR can test whether the calibration findings transfer beyond short banking queries. Multiple training/split seeds and feature ablations would separate stable effects from one fitted configuration. Full encoder fine-tuning remains deferred; this experiment supplies no evidence about its benefit.

The paired intervals use 5,000 shared test-example resamples and condition on the frozen fits. They omit retraining/calibration-training variability and are not adjusted for all comparisons. The primary policy was declared before testing; the remaining tables and Pareto frontiers describe the prespecified variants rather than establishing universal winners. Public benchmark exposure, ambiguous intent labels, and differences in supervision constrain transfer to enterprise documents.

The final runner validates model, calibrator, dataset, dependency, and scientific-code hashes before API work. It records completion only after result artifacts are written; report generation is separate and can be repeated without inference or paid requests. All results retain per-example predictions, reliability bins, risk curves, fixed thresholds, source hashes, and the sealed plan.
"""
    # Retain already-correct explanations from the development study verbatim.
    # This also preserves native Page block identities during the broad update.
    old_blocks = re.split(r"\n\s*\n", (OUT / "report.development.md").read_text(encoding="utf-8").strip())
    old_payload = read_json(OUT / "page-update-content.development.json")
    CAPTIONS["calibration_mappings"] = old_payload["captions"]["calibration_mappings"]
    def old(prefix):
        matches = [block for block in old_blocks if block.startswith(prefix)]
        if len(matches) != 1: raise ValueError(f"Missing historical explanation: {prefix}")
        return matches[0]
    blocks = re.split(r"\n\s*\n", markdown.strip())
    for i, block in enumerate(blocks):
        if block.startswith("BANKING77 contains"):
            blocks[i] = old("BANKING77 asks") + "\n\n" + old("The data audit removed") + "\n\nThe official test retains one normalized duplicate pair; independence-based uncertainty intervals are approximate. Test labels first entered model evaluation during this final stage."
        elif block.startswith("The table compares binary"):
            blocks[i] = old("Brier error is") + "\n\n" + old("The scalar comparison adds") + "\n\n" + block
        elif block.startswith("Reliability plots compare"):
            blocks[i] = old("Reliability plots answer")
        elif block.startswith("The score distributions answer"):
            tail = "For embedding LR it was" + block.split("For embedding LR it was", 1)[1]
            blocks[i] = old("The second figure separates") + "\n\n## Why the embedding curves looked volatile\n\n" + old("Connecting these estimates") + "\n\n" + old("The appropriate response is") + "\n\n" + tail
        elif block.startswith("For a deployment of N messages, fixed"):
            blocks[i] = old("For a deployment of N messages, let F") + "\n\nA policy's review savings cannot justify claiming a 1% quality target that failed independent validation."
        elif block.startswith("A harder or more representative task"):
            blocks[i] = old("**Expand the evaluation around") + "\n\nMultiple training/split seeds and feature ablations would separate stable effects from one fitted configuration. Full encoder fine-tuning remains deferred; this experiment supplies no evidence about its benefit."
        elif block.startswith("The paired intervals use"):
            blocks[i] = "## Scope and reproducibility\n\n" + block
    markdown = "\n\n".join(blocks) + "\n"
    for before, after in (("The independent evaluation protocol", "The experiment in brief"),
                          ("What calibration did and did not fix", "Temperature logistic and isotonic calibration"),
                          ("Reading the probability curves", "Comparing the probability curves"),
                          ("Cost and Pareto frontiers on test", "Cost and Pareto frontiers"),
                          ("What should happen next", "What the findings justify next")):
        markdown = markdown.replace("## " + before + "\n", "## " + after + "\n")
    (OUT / "report.md").write_text(markdown, encoding="utf-8")
    body = markdown_html(markdown)
    document = '<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Learning When to Trust a Classifier</title><style>body{font:16px/1.6 Arial,sans-serif;max-width:1080px;margin:40px auto;padding:0 24px;color:#202b32}h1,h2{line-height:1.2}h2{margin-top:40px}table{border-collapse:collapse;width:100%;font-size:12px;line-height:1.35;margin:20px 0}th,td{padding:8px;border-bottom:1px solid #ddd;text-align:left}th{background:#f1f5f2}figure{margin:30px 0;break-inside:avoid}svg{max-width:100%;height:auto}figcaption{font-size:12px;color:#53616a;margin-top:8px}a{color:#246447}@media print{body{font-size:11px;max-width:none;margin:0}h2{break-after:avoid}table{font-size:9px}figure{margin:16px 0}}</style></head><body>' + body + '</body></html>'
    (OUT / "report.html").write_text(document, encoding="utf-8")
    for name in CAPTIONS:
        if not (FIG / f"{name}.svg").exists(): raise ValueError(f"Missing figure {name}")
    write_json(OUT / "figure_captions.json", CAPTIONS)
    write_json(OUT / "page-update-content.json", {"markdown": markdown, "captions": CAPTIONS, "inserts": INSERTS,
                "figures": {name: (FIG / f"{name}.svg").read_text(encoding="utf-8") for name in CAPTIONS}})
    evidence = {"plan_sha256": plan["sha256"], "evaluation_role": "official_test", "n": 3080, "test_variants": 24,
                "active_arms": scope["active_arms"], "deferred_arms": scope["deferred_arms"], "thresholds_refitted": False,
                "metrics": metrics, "paired_comparisons": comparisons, "pipelines": pipelines, "frontiers": frontiers,
                "test_api_cost_usd": api_cost, "source_sha256": inputs, "report_sha256": sha(OUT / "report.md"),
                "primary_target_met": a["risk_ci95"][1] <= .01, "created_utc": datetime.now(timezone.utc).isoformat()}
    inputs[Path(__file__).relative_to(ROOT).as_posix()] = sha(__file__)
    for path in [ROOT / "scripts/analyze_cost_frontiers.py", OUT / "cost-frontiers.json"]:
        inputs[path.relative_to(ROOT).as_posix()] = sha(path)
    evidence["figures_sha256"] = {key: sha(FIG / (key + ".svg")) for key in CAPTIONS}
    write_json(OUT / "final-evidence.json", evidence)
    pd.DataFrame(pipelines).to_csv(OUT / "test-cost-frontiers.csv", index=False)
    pd.DataFrame([{ "method": key, **{field: m[field] for field in ("accuracy", "brier", "binary_log_loss", "error_auroc")},
                    **{"accepted_" + field: m["acceptance"][field] for field in ("coverage", "risk", "errors", "accepted")}} for key, m in metrics.items()]).to_csv(OUT / "test-summary.csv", index=False)
    chrome = Path("C:/Program Files/Google/Chrome/Application/chrome.exe")
    if chrome.exists():
        import os
        profile = Path(os.environ["LOCALAPPDATA"]) / "Temp/confidence-final-report-preview"
        subprocess.run([str(chrome), "--headless=new", "--disable-gpu", "--no-first-run", "--disable-extensions", "--disable-background-networking",
                        "--no-pdf-header-footer", "--user-data-dir=" + str(profile), "--print-to-pdf=" + str(OUT / "report.pdf"),
                        (OUT / "report.html").as_uri()], check=True, timeout=120)
    pdf = (OUT / "report.pdf").read_bytes()
    if not pdf.startswith(b"%PDF-") or b"%%EOF" not in pdf[-4096:]: raise ValueError("Invalid report PDF")
    publication = read_json(OUT / "publication.development.json")
    publication.update(official_test_evaluated=True, test_scope=["local", "openai_decisions"], claude_test_evaluated=False,
                       test_variants=24, frozen_plan_sha256=plan["sha256"], primary_target_met=evidence["primary_target_met"],
                       verified_text_matches_local=False, page_sync_pending=True, pdf_sha256=sha(OUT / "report.pdf"), html_sha256=sha(OUT / "report.html"),
                       local_figures_visually_inspected=False, local_html_preview_inspected=False,
                       consolidated_source_hashes_verified=len(inputs),
                       pdf_pages=len(re.findall(rb"/Type\s*/Page\b", pdf)), updated_utc=datetime.now(timezone.utc).isoformat())
    write_json(OUT / "publication.json", publication)
    print(json.dumps({"test_variants": 24, "primary_risk": a["risk"], "primary_target_met": evidence["primary_target_met"], "test_api_cost_usd": api_cost, "pdf_pages": publication["pdf_pages"]}), flush=True)


if __name__ == "__main__":
    main()
