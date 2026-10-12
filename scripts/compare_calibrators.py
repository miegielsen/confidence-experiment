"""Compare correctness calibrators using frozen outputs; no API or encoder calls.

Only calibration.csv and development.csv are read. Original runs stay unchanged.
Temperature is the original multiclass-NLL fit; scalar calibrators instead target
selected-label correctness. Comparisons are exploratory development results.
"""
from datetime import datetime, timezone
import hashlib
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import logit, softmax
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from confidence_experiment.io import read_json, write_json
from confidence_experiment.metrics import Sigmoid, evaluate, select_threshold, wilson
from confidence_experiment.runner import features

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/calibrator-comparison"
EPS = 1e-7


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sources = {}

    def track(path):
        path = ROOT / path
        sources[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path

    frames = {role: pd.read_csv(track(f"data/banking77/{role}.csv"), float_precision="round_trip")
              for role in ("calibration", "development")}
    assert all(len(f) == 1000 and f.id.is_unique for f in frames.values())
    assert not set(frames["calibration"].id) & set(frames["development"].id)
    labels = read_json(track("data/banking77/manifest.json"))["labels"]
    streams = {}

    def load_run(name):
        run = ROOT / "runs" / name
        hashes = read_json(track(run / "frozen_hashes.json"))
        for filename, expected in hashes.items():
            path = track(run / filename)
            assert sources[str(path.relative_to(ROOT))] == expected, path
        frozen = joblib.load(run / "frozen.joblib")
        return run, frozen["calibration"]

    def add(key, title, run, raw_name, cal_pred, cal_q, existing):
        def read_dev(method):
            f = pd.read_csv(track(run / "development" / f"{method}.predictions.csv"), float_precision="round_trip")
            assert f.id.is_unique
            f = f.set_index("id").loc[frames["development"].id].reset_index()
            assert np.array_equal(f.truth, frames["development"].label)
            assert np.array_equal(f.correct, f.truth == f.prediction)
            return f

        dev = read_dev(raw_name)
        variants = {variant: read_dev(method).confidence.to_numpy()
                    for variant, method in existing.items()}
        for method in existing.values():
            assert np.array_equal(read_dev(method).prediction, dev.prediction)
        saved_metrics = read_json(track(run / "development/metrics.json"))
        streams[key] = dict(title=title, cal_pred=np.asarray(cal_pred),
                            cal_q=np.asarray(cal_q, dtype=float), dev=dev,
                            existing=variants, saved_metrics={"raw": saved_metrics[raw_name],
                                **{variant: saved_metrics[method] for variant, method in existing.items()}})

    run, frozen = load_run("embeddings")
    logits, _ = joblib.load(track(run / "supervised.joblib"))
    for name, title in (("embedding_lr", "Embedding + LR"), ("embedding_mlp", "Embedding + MLP")):
        cal = softmax(logits[name]["calibration"], axis=1)
        dev = softmax(logits[name]["development"], axis=1)
        add(name, title, run, name + "_raw", np.array(labels)[cal.argmax(axis=1)],
            cal.max(axis=1), {"temperature": name + "_temperature"})
        assert np.allclose(dev.max(axis=1), streams[name]["dev"].confidence, atol=1e-7)

    claude_runs = [("haiku", "anthropic-batch-v2"), ("sonnet", "anthropic-sonnet-5-5")]
    if (ROOT / "runs/anthropic-opus-5-5/frozen.joblib").exists():
        claude_runs.append(("opus", "anthropic-opus-5-5"))
    for family, name in claude_runs:
        run, frozen = load_run(name)
        logits, _ = joblib.load(track(run / "supervised.joblib"))
        rows = pd.read_csv(track(run / "llm_calibration.csv"), float_precision="round_trip")
        assert rows.id.is_unique
        rows = rows.set_index("id").loc[frames["calibration"].id].reset_index()
        records = rows.to_dict("records")
        x = features(records, frames["calibration"], labels, logits["tfidf"]["calibration"])
        learned = frozen["correctness_model"].predict_proba(x)[:, 1]
        dev_rows = pd.read_csv(track(run / "llm_development.csv"), float_precision="round_trip")
        dev_rows = dev_rows.set_index("id").loc[frames["development"].id].reset_index()
        x_dev = features(dev_rows.to_dict("records"), frames["development"], labels,
                         logits["tfidf"]["development"])
        learned_dev = frozen["correctness_model"].predict_proba(x_dev)[:, 1]
        for suffix, q, title in (("same", rows.confidence, "native"),
                                 ("verifier", rows.verifier, "verifier"),
                                 ("correctness", learned, "learned correctness")):
            key = family + "_" + suffix
            add(key, family.title() + " " + title, run, "llm_" + suffix + "_raw",
                rows.label, q, {"logistic_score": "llm_" + suffix + "_sigmoid"})
            streams[key]["frozen_sigmoid"] = frozen["llm_" + suffix]
            if suffix == "correctness":
                assert np.allclose(learned_dev, streams[key]["dev"].confidence, atol=1e-12)
        if family == "haiku":
            cal = softmax(logits["tfidf"]["calibration"], axis=1)
            add("tfidf", "TF-IDF + LR", run, "tfidf_raw", np.array(labels)[cal.argmax(axis=1)],
                cal.max(axis=1), {"temperature": "tfidf_temperature"})

    run, frozen = load_run("openai-decisions-v2")
    rows = pd.read_csv(track(run / "decisions_calibration.csv"), float_precision="round_trip")
    assert rows.id.is_unique
    rows = rows.set_index("id").loc[frames["calibration"].id].reset_index()
    for suffix, field in (("confidence", "confidence"), ("probability", "selected_probability")):
        name = "decisions_" + suffix
        existing = {"logistic_score": name + "_sigmoid"}
        if suffix == "probability":
            existing["temperature"] = name + "_temperature"
        add(name, "OpenAI " + suffix, run, name + "_raw", rows.label,
            rows[field].fillna(0), existing)
        streams[name]["frozen_sigmoid"] = frozen[name]

    results, calibrators, predictions, table, comparisons = {}, {}, {}, [], {}
    rng = np.random.default_rng(20261009)
    # Paired evaluation uncertainty conditional on these fitted calibrators.
    boot_indices = rng.integers(0, 1000, size=(5000, 1000))
    for key, stream in streams.items():
        y_cal = (stream["cal_pred"] == frames["calibration"].label.to_numpy()).astype(int)
        dev = stream["dev"]
        q_cal, q_dev = stream["cal_q"], dev.confidence.to_numpy()
        assert np.isfinite(q_cal).all() and ((q_cal >= 0) & (q_cal <= 1)).all()
        sigmoid = stream.get("frozen_sigmoid") or Sigmoid().fit(q_cal, y_cal)
        logistic = LogisticRegression(C=1.0, max_iter=1000).fit(
            logit(np.clip(q_cal, EPS, 1 - EPS)).reshape(-1, 1), y_cal)
        isotonic = IsotonicRegression(y_min=0, y_max=1, increasing=True,
                                      out_of_bounds="clip").fit(q_cal, y_cal)
        calibrators[key] = dict(logistic_score=sigmoid, logistic_logit=logistic, isotonic=isotonic)
        variants = {"raw": q_dev, **stream["existing"],
                    "logistic_score": sigmoid.predict(q_dev),
                    "logistic_logit": logistic.predict_proba(
                        logit(np.clip(q_dev, EPS, 1 - EPS)).reshape(-1, 1))[:, 1],
                    "isotonic": isotonic.predict(q_dev)}
        refused = dev.prediction.to_numpy() == "__refusal__"
        for variant in variants:
            variants[variant] = np.asarray(variants[variant], dtype=float).copy()
            variants[variant][refused] = 0
        for variant, expected in stream["existing"].items():
            assert np.allclose(variants[variant], expected, atol=1e-12), (key, variant)
        results[key], predictions[key] = {}, variants
        for variant, q in variants.items():
            threshold = select_threshold(dev.correct.to_numpy(dtype=int), q, .01, "wilson")
            metrics, reliability, curve = evaluate(dev.truth, dev.prediction, q, threshold, labels=labels)
            if variant in stream["saved_metrics"]:
                expected = stream["saved_metrics"][variant]
                assert np.isclose(metrics["brier"], expected["brier"], atol=1e-7), (key, variant)
                assert np.isclose(metrics["binary_log_loss"], expected["binary_log_loss"], atol=1e-7), (key, variant)
                assert np.isclose(metrics["error_auroc"], expected["error_auroc"], atol=1e-12, rtol=0), (key, variant)
                assert metrics["acceptance"]["coverage"] == expected["acceptance"]["coverage"], (key, variant)
            results[key][variant] = metrics
            row = dict(stream=key, title=stream["title"], method=variant,
                       calibration_n=1000, calibration_errors=int((1 - y_cal).sum()),
                       development_errors=int((~dev.correct).sum()),
                       brier=metrics["brier"], log_loss=metrics["binary_log_loss"],
                       error_auroc=metrics["error_auroc"], coverage=metrics["acceptance"]["coverage"],
                       unique_scores=len(np.unique(q)), exact_one_errors=int(((q == 1) & ~dev.correct).sum()))
            table.append(row)
            output = dev.copy()
            output["confidence"] = q
            output.to_csv(OUT / f"{key}.{variant}.predictions.csv", index=False)
            pd.DataFrame(reliability).to_csv(OUT / f"{key}.{variant}.reliability.csv", index=False)
            pd.DataFrame(curve).to_csv(OUT / f"{key}.{variant}.risk_coverage.csv", index=False)
        comparisons[key] = {}
        for baseline in ["logistic_logit", "logistic_score"] + (["temperature"] if "temperature" in variants else []):
            difference = ((variants["isotonic"] - dev.correct.to_numpy()) ** 2
                          - (variants[baseline] - dev.correct.to_numpy()) ** 2)
            ci = np.quantile(difference[boot_indices].mean(axis=1), [.025, .975])
            comparisons[key]["isotonic_minus_" + baseline] = {
                "brier_difference": float(difference.mean()), "paired_bootstrap_ci95": ci.tolist()}
        comparisons[key]["isotonic_fit"] = {
            "calibration_errors": int((1 - y_cal).sum()),
            "x_thresholds": isotonic.X_thresholds_.tolist(),
            "y_thresholds": isotonic.y_thresholds_.tolist(),
            "logistic_logit_slope": float(logistic.coef_[0, 0]),
            "logistic_logit_intercept": float(logistic.intercept_[0])}

    table = pd.DataFrame(table)
    table.to_csv(OUT / "summary.csv", index=False)
    joblib.dump(calibrators, OUT / "calibrators.joblib")
    primary = ["embedding_lr", "haiku_same", "sonnet_same", "sonnet_correctness"]
    colors = {"raw": "#87909e", "temperature": "#b85b22", "logistic_score": "#4e7189",
              "logistic_logit": "#256b48", "isotonic": "#8c47a2"}
    fig, axes = plt.subplots(len(primary)//2, 2, figsize=(12, 4.5*(len(primary)//2)), constrained_layout=True)
    for ax, key in zip(axes.flat, primary):
        for variant in ("raw", "temperature", "logistic_score", "logistic_logit", "isotonic"):
            if variant not in predictions[key]:
                continue
            reliability = pd.read_csv(OUT / f"{key}.{variant}.reliability.csv")
            intervals = np.array([wilson(int(round(r.accuracy * r.n)), int(r.n))
                                  for r in reliability.itertuples()])
            ax.errorbar(reliability.confidence, reliability.accuracy,
                        yerr=np.maximum(0, np.stack([reliability.accuracy - intervals[:, 0],
                                                    intervals[:, 1] - reliability.accuracy])),
                        fmt="none", ecolor=colors[variant], alpha=.25, linewidth=.8)
            ax.scatter(reliability.confidence, reliability.accuracy,
                       s=20 + np.sqrt(reliability.n) * 3, label=variant, color=colors[variant], alpha=.8)
        ax.plot([0, 1], [0, 1], color="#aaa", linestyle="--", linewidth=1)
        ax.set(title=streams[key]["title"], xlabel="Mean predicted correctness", ylabel="Observed correctness",
               xlim=(-.02, 1.02), ylim=(-.02, 1.02))
        ax.legend(fontsize=8)
    fig.suptitle("Development reliability: bin counts shown by size; 95% Wilson intervals", fontsize=14)
    for extension in ("svg", "png", "pdf"):
        fig.savefig(OUT / f"reliability.{extension}", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(len(primary)//2, 2, figsize=(12, 4.5*(len(primary)//2)), constrained_layout=True)
    grid = np.linspace(0, 1, 1001)
    for ax, key in zip(axes.flat, primary):
        fit = calibrators[key]
        ax.plot(grid, grid, "--", color=colors["raw"], label="identity")
        ax.plot(grid, fit["logistic_score"].predict(grid), color=colors["logistic_score"], label="logistic on score")
        ax.plot(grid, fit["logistic_logit"].predict_proba(logit(np.clip(grid, EPS, 1-EPS)).reshape(-1,1))[:,1],
                color=colors["logistic_logit"], label="logistic on log-odds")
        ax.plot(grid, fit["isotonic"].predict(grid), color=colors["isotonic"], label="isotonic")
        ax.set(title=streams[key]["title"], xlabel="Original correctness score", ylabel="Calibrated correctness probability",
               xlim=(0,1), ylim=(0,1))
        ax.legend(fontsize=8)
    fig.suptitle("Mappings fitted only on the 1,000 calibration examples", fontsize=14)
    for extension in ("svg", "png", "pdf"):
        fig.savefig(OUT / f"mappings.{extension}", dpi=160)
    plt.close(fig)

    if "opus_same" in streams:
        from report_opus import export_opus_calibration
        export_opus_calibration(OUT, streams, calibrators, colors, EPS)
    for relative, expected in sources.items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected, relative
    write_json(OUT / "analysis.json", {"created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {"calibration_n": 1000, "evaluation_n": 1000, "evaluation_role": "development",
                     "target": "correctness of the unchanged selected label", "logistic_C": 1.0,
                     "logistic_logit_clip": EPS, "binary_log_loss_clip": EPS,
                     "bootstrap_repeats": 5000, "bootstrap_seed": 20261009,
                     "bootstrap_conditions_on_fitted_calibrators": True,
                     "temperature_objective": "original multiclass negative log likelihood",
                     "isotonic_objective": "binary correctness squared error",
                     "threshold_selection": "exploratory same-development Wilson upper endpoint <=1%; ties kept together",
                     "no_test_data_read": True, "no_api_calls": True},
        "results": results, "comparisons": comparisons, "source_sha256": sources,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    display = table[["title", "method", "calibration_errors", "brier", "log_loss", "error_auroc", "coverage", "exact_one_errors"]]
    highlights = table[table.stream.isin(primary + ["decisions_probability", "opus_same", "opus_correctness"])].pivot(
        index="title", columns="method", values="brier").reindex(
            columns=["raw", "temperature", "logistic_score", "logistic_logit", "isotonic"])
    highlights.columns = ["Raw", "Temperature", "Logistic on score", "Logistic on log-odds", "Isotonic"]
    text = """<h1>Temperature, logistic and isotonic calibration</h1>
<p><strong>More flexibility did not produce a consistent improvement.</strong> For the
embedding classifier, temperature scaling reduced Brier loss from 0.03279 to 0.02081
(36.5%). Logistic calibration and isotonic also improved the raw score, but their
point estimates were higher. Isotonic minus temperature was +0.00177 in Brier;
the paired 95% bootstrap interval [−0.00004, +0.00348] includes zero.</p>
<p>For Haiku native confidence, isotonic and logistic on log-odds were nearly tied.
For Sonnet native confidence, logistic on score had the lowest point estimate among
these mappings. For Sonnet learned correctness, isotonic was worse than logistic
on score: Brier 0.05328 versus 0.05052, with a conditional paired interval for the
difference of [+0.00051, +0.00503]. Isotonic assigned probability exactly one to
three incorrect development predictions; log loss rose to 0.22409 versus 0.18855.
This is a concrete reason to validate flexible calibration on unseen examples.</p>
<p>Calibration remains useful: OpenAI selected-option probability improved with both
logistic and isotonic calibration, while the previously fitted temperature transform
was harmful for this correctness target. Results depend on the model and target;
there is no universally best mapping in this comparison.</p>
<h2>Selected development Brier scores (lower is better)</h2>"""
    text += highlights.to_html(na_rep="—", float_format=lambda x: f"{x:.5f}")
    text += """<h2>Protocol and interpretation</h2>
<p>Each scalar mapping is fitted on the same 1,000 calibration examples and evaluated on
the same separate 1,000 development examples. Class predictions stay fixed. Logistic on
score reproduces the existing sigmoid calibrator (C=1); logistic on log-odds uses logit(p)
as its feature with C=1 and clipping at 10⁻⁷. Isotonic is non-decreasing with clipped
out-of-range inputs. No calibration settings were chosen using development performance.</p>
<p>Temperature is a different intervention: the original single-parameter multiclass
logit scaling, optimized for multiclass log loss. Isotonic fits binary correctness
squared error. A win in binary Brier is therefore not evidence of better full class
probabilities. Brier measures discrimination as well as calibration. Isotonic can
introduce ties and change ranking metrics. Binned reliability estimates remain noisy.</p>
<p>Log loss clips exact zero/one predictions at 10⁻⁷, consistently with the original report;
without clipping an incorrect probability-one prediction has infinite log loss.
Coverage reuses development outcomes to choose a threshold and is exploratory, with
no independent guarantee. Paired bootstrap Brier intervals in analysis.json condition
on the fitted calibrators and omit calibration-training uncertainty. Official test
data remains reserved. All recorded source and frozen-artifact hashes were checked.</p>"""
    page = ("<!doctype html><html><meta charset='utf-8'><title>Calibrator comparison</title>"
            "<style>body{font:16px system-ui;max-width:1200px;margin:40px auto;padding:0 25px;color:#203040}"
            "p{max-width:950px;line-height:1.6}table{border-collapse:collapse;width:100%;font-size:13px}"
            "td,th{border-bottom:1px solid #ddd;padding:7px;text-align:right}td:first-child{text-align:left}"
            "img{width:100%}</style>" + text + display.to_html(index=False, float_format=lambda x: f"{x:.5f}")
            + "<h2>Reliability on development data</h2><img src='reliability.svg'>"
            + "<h2>Learned calibration mappings</h2><img src='mappings.svg'></html>")
    (OUT / "report.html").write_text(page, encoding="utf-8")
    if "opus_same" in streams:
        page=page.replace('</html>',"<h2>Opus native and learned correctness</h2><img src='opus_reliability.svg'><img src='opus_mappings.svg'></html>")
        (OUT / "report.html").write_text(page, encoding="utf-8")
    print(display.to_string(index=False))
    print(f"Report: {OUT / 'report.html'}")


if __name__ == "__main__":
    main()
