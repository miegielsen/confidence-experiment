"""Report the frozen Haiku test sweep; optionally wait for accepted batches.

This process never constructs an API client, fits a model, or selects a threshold.
The existing local/OpenAI report remains a separately versioned comparison.
"""
import argparse
from datetime import datetime, timezone
import html
import json
import time
import traceback

import numpy as np
import pandas as pd

from confidence_experiment.io import read_json, write_json
from confidence_experiment.metrics import evaluate
from final_protocol import ROOT, OUT as RUN, completed, preflight, sha
from analyze_cost_frontiers import frontier

OUT = ROOT / "reports/haiku-test"
STATE = RUN / "haiku-report/status.json"


def report():
    plan = preflight()
    if not completed("haiku"):
        raise ValueError("Haiku test results are incomplete")
    folder = RUN / "haiku"
    metrics = read_json(folder / "metrics.json")
    expected = {key for key, spec in plan["variants"].items() if spec["arm"] == "haiku"}
    if set(metrics) != expected:
        raise ValueError("Haiku variants differ from the sealed plan")
    test = pd.read_csv(ROOT / "data/banking77/test.csv", keep_default_na=False)
    labels = read_json(ROOT / "data/banking77/manifest.json")["labels"]
    frames = {}
    for key, metric in metrics.items():
        frame = pd.read_csv(folder / f"{key}.predictions.csv", float_precision="round_trip", keep_default_na=False)
        if not np.array_equal(frame.id, test.id) or not np.array_equal(frame.truth, test.label):
            raise ValueError("Test example identity or labels changed")
        if not np.array_equal(frame.correct, frame.prediction == frame.truth):
            raise ValueError("Incorrect saved correctness flags")
        checked, _, _ = evaluate(frame.truth, frame.prediction, frame.confidence,
                                 plan["variants"][key]["threshold"], labels=labels)
        for field in ("accuracy", "brier", "binary_log_loss", "error_auroc"):
            if not np.isclose(checked[field], metric[field], atol=1e-7, rtol=0):
                raise ValueError(f"Metric mismatch: {key}/{field}")
        frames[key] = frame
    for frame in frames.values():
        if not np.array_equal(frame.prediction, frames["haiku_same.raw"].prediction):
            raise ValueError("Confidence variants changed selected labels")
    costs = read_json(folder / "metadata.json")["costs"]
    stages = {"haiku_same": {"original"}, "haiku_verifier": {"original", "verifier"},
              "haiku_correctness": {"original", "blind", "verifier"}}
    stream_cost = {stream: sum(row["usd"] for row in
                   {row["request_sha256"]: row for row in costs if row["stage"] in selected}.values())
                   for stream, selected in stages.items()}
    unique = {row["request_sha256"]: row for row in costs}
    comparisons = {}
    boot = np.random.default_rng(plan["bootstrap"]["seed"]).integers(0, len(test), (5000, len(test)))
    for key in ("haiku_correctness.raw", "haiku_correctness.logistic_score",
                "haiku_correctness.logistic_logit", "haiku_correctness.isotonic"):
        first, second = frames[key], frames["haiku_same.raw"]
        diff = ((first.confidence - first.correct) ** 2 - (second.confidence - second.correct) ** 2).to_numpy()
        comparisons[key + "_minus_native_brier"] = {
            "difference": float(diff.mean()), "ci95": np.quantile(diff[boot].mean(axis=1), [.025, .975]).tolist()}
    del boot
    baseline = read_json(ROOT / "reports/banking77-findings/final-evidence.json")
    if baseline["plan_sha256"] != plan["sha256"]:
        raise ValueError("Baseline report belongs to another plan")
    for arm in ("embeddings", "tfidf", "decisions"):
        if not completed(arm):
            raise ValueError("Baseline results failed verification")
    pipelines = list(baseline["pipelines"])
    rows = []
    for key, metric in metrics.items():
        spec = plan["variants"][key]
        dev = spec["development_metrics"]
        accept = metric["acceptance"]
        price = stream_cost[spec["stream"]] / len(test) * 1000
        pipelines.append({"id": key, "stream": spec["stream"], "api_usd_per_1000": price,
                          **{field: metric[field] for field in ("accuracy", "brier", "error_auroc")},
                          "coverage": accept["coverage"]})
        rows.append({"variant": key, "development_accuracy": dev["accuracy"], "test_accuracy": metric["accuracy"],
                     "development_brier": dev["brier"], "test_brier": metric["brier"],
                     "development_error_auroc": dev["error_auroc"], "test_error_auroc": metric["error_auroc"],
                     "accepted": accept["accepted"], "accepted_errors": accept["errors"],
                     "coverage": accept["coverage"], "accepted_risk": accept["risk"], "api_usd_per_1000": price})
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "comparison.csv", index=False)
    pd.DataFrame(pipelines).to_csv(OUT / "test-cost-frontiers.csv", index=False)
    raw = metrics["haiku_same.raw"]
    dev = plan["variants"]["haiku_same.raw"]["development_metrics"]
    pct = lambda number: f"{100 * number:.2f}%"
    table = "| Confidence variant | Dev Brier | Test Brier | Test error AUROC | Test coverage | Accepted errors / accepted |\n|---|---:|---:|---:|---:|---:|\n"
    for row in rows:
        table += f"| {row['variant']} | {row['development_brier']:.5f} | {row['test_brier']:.5f} | {row['test_error_auroc']:.3f} | {pct(row['coverage'])} | {row['accepted_errors']} / {row['accepted']} |\n"
    native = metrics["haiku_same.raw"]["brier"]
    learned = metrics["haiku_correctness.logistic_score"]["brier"]
    interval = comparisons["haiku_correctness.logistic_score_minus_native_brier"]["ci95"]
    text = f"""# Haiku: independent test validation

This smaller sweep evaluates **12 frozen Haiku confidence variants on {len(test):,} official test messages**. Sonnet and Opus remain deferred. Models, calibrators, prompts, and development-selected thresholds are unchanged. No test threshold search or refitting was performed.

Classification accuracy was **{pct(dev['accuracy'])} on development and {pct(raw['accuracy'])} on test**, a change of **{100 * (raw['accuracy'] - dev['accuracy']):+.2f} percentage points**. The nominal test accuracy interval is **{pct(raw['accuracy_ci95'][0])}–{pct(raw['accuracy_ci95'][1])}**. All confidence variants retain the same classifications.

Native confidence test Brier was **{native:.5f}**; learned correctness plus the already-fitted logistic-on-score calibrator was **{learned:.5f}**. Their paired difference, learned minus native, has nominal 95% interval **[{interval[0]:+.5f}, {interval[1]:+.5f}]**. Negative differences favor learned correctness. That pipeline combines extra API calls and supervised TF-IDF features, so this comparison does not isolate the effect of learning alone.

{table}

Zero acceptance means the frozen development policy selected no threshold. It does not establish that no possible threshold could work. Any nonzero acceptance policy must be assessed using its saved error interval in evidence.json. Low Brier error does not itself establish a 1% accepted-error guarantee.

The sweep recorded **{len(unique):,} unique requests** and an estimated **${stream_cost['haiku_correctness']:.4f}** in batch API fees. Native-only classification costs **${stream_cost['haiku_same'] / len(test) * 1000:.4f} per 1,000 messages**; classification plus verification costs **${stream_cost['haiku_verifier'] / len(test) * 1000:.4f}**; the full learned-correctness pipeline costs **${stream_cost['haiku_correctness'] / len(test) * 1000:.4f}**. Calibrators reuse responses and add no calls. These estimates use returned usage at configured rates, not invoice totals; labeling, fitting, compute, and review costs are excluded.

The accompanying test-cost-frontiers.csv and evidence.json combine these 12 variants with the previous 24 local/OpenAI test variants. Frontiers use point estimates, retain ties, and exclude Sonnet/Opus development results. Paired intervals use 5,000 shared example resamples and condition on fitted models; they omit fitting variability and are not adjusted for multiple comparisons. The test includes a normalized duplicate pair, making independence assumptions approximate.

The main local/OpenAI report is preserved at ../banking77-findings/report.md. This addendum supplies the newly validated Haiku results without changing the original primary-policy conclusion.
"""
    (OUT / "report.md").write_text(text, encoding="utf-8")
    document = '<!doctype html><meta charset="utf-8"><title>Haiku test validation</title><style>body{font:14px/1.6 Arial;max-width:1100px;margin:40px auto}pre{white-space:pre-wrap}</style><pre>' + html.escape(text) + '</pre>'
    (OUT / "report.html").write_text(document, encoding="utf-8")
    sources = {"scripts/report_haiku_test.py": sha(__file__), "runs/final-evaluation/frozen-plan.json": sha(RUN / "frozen-plan.json"),
               "reports/banking77-findings/final-evidence.json": sha(ROOT / "reports/banking77-findings/final-evidence.json")}
    for path in folder.iterdir():
        if path.is_file() and path.name not in {"status.json", ".lock"}:
            sources[path.relative_to(ROOT).as_posix()] = sha(path)
    write_json(OUT / "evidence.json", {"plan_sha256": plan["sha256"], "n": len(test), "test_variants": len(metrics),
               "metrics": metrics, "paired_comparisons": comparisons, "test_api_usd": stream_cost['haiku_correctness'],
               "unique_requests": len(unique), "cost_by_stream_usd": stream_cost, "pipelines": pipelines,
               "frontiers": {field: frontier(pipelines, field, minimize=field == "brier") for field in ("accuracy", "brier", "error_auroc", "coverage")},
               "source_sha256": sources, "report_sha256": sha(OUT / "report.md"), "thresholds_refitted": False})
    write_json(STATE, {"state": "complete", "report": str(OUT / "report.md"),
               "updated_utc": datetime.now(timezone.utc).isoformat()})
    sweep = read_json(RUN / "haiku-sweep.json")
    sweep.update(state="complete", completed_utc=datetime.now(timezone.utc).isoformat(),
                 test_api_usd=stream_cost['haiku_correctness'])
    write_json(RUN / "haiku-sweep.json", sweep)
    print(json.dumps({"accuracy": raw['accuracy'], "development_accuracy": dev['accuracy'], "cost_usd": stream_cost['haiku_correctness'],
                      "report": str(OUT / "report.md")}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    try:
        while args.wait and not completed("haiku"):
            status = read_json(RUN / "haiku/status.json")
            if status["state"] == "failed":
                raise RuntimeError("Haiku inference failed; no partial report generated")
            write_json(STATE, {"state": "waiting_for_batches"})
            time.sleep(20)
        report()
    except Exception:
        write_json(STATE, {"state": "failed", "error": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
