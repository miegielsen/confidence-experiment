"""Budgeted matched Sonnet/Opus test inference with frozen scientific choices."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import threading
import traceback

import joblib
import numpy as np
import pandas as pd

from confidence_experiment.io import digest, read_json, write_json
from confidence_experiment.llm import CLASSIFY, BLIND, VERIFY, BudgetExhausted
from confidence_experiment.metrics import evaluate
from confidence_experiment.runner import features
from final_protocol import ROOT, OUT as FINAL, preflight, config_for, frozen_for, completed, sha, transform, now
from run_anthropic_batch import Batcher, specification

OUT = ROOT / "runs/claude-sample-20261014"
REPORT = ROOT / "reports/claude-sample-test"
PLAN = OUT / "plan.json"
LEDGER = OUT / "spending.json"
FAMILIES = ("sonnet", "opus")
BUDGET = 12.0
LEDGER_LOCK = threading.RLock()


def sample_rows(frame, per_intent=8, seed=20261014):
    """Outcome-independent, balanced, interleaved random sample of test IDs."""
    rng = np.random.default_rng(seed)
    groups = [group.iloc[rng.permutation(len(group))[:per_intent]]
              for _, group in frame.groupby("label", sort=True)]
    if any(len(group) != per_intent for group in groups):
        raise ValueError("Insufficient examples in an intent")
    rows = [groups[i].iloc[round_index] for round_index in range(per_intent)
            for i in rng.permutation(len(groups))]
    return pd.DataFrame(rows).reset_index(drop=True)


def freeze():
    base = preflight()
    if PLAN.exists():
        return validate()
    for arm in ("embeddings", "tfidf", "decisions", "haiku"):
        if not completed(arm):
            raise ValueError(f"Matched baseline is incomplete: {arm}")
    frame = sample_rows(pd.read_csv(ROOT / "data/banking77/test.csv", keep_default_na=False))
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / "sample.csv", index=False)
    plan = {"protocol": 1, "created_utc": now(), "parent_plan_sha256": base["sha256"],
            "runner_sha256": sha(__file__), "sample_sha256": sha(OUT / "sample.csv"),
            "seed": 20261014, "per_intent": 8, "n": len(frame), "ids": frame.id.tolist(),
            "families": list(FAMILIES), "budget_usd": BUDGET,
            "expected_usd": len(frame) / 1000 * (5.626964 + 12.342798),
            "rules": ["No fitting or threshold selection on test", "Same sampled IDs for every model",
                      "Stop before a new block if worst-case reservations cannot fit remaining budget",
                      "No full-test Sonnet/Opus completion markers are written",
                      "Budget stopping depends on usage only; never examine scores to decide continuation"],
            "prices_verified_url": "https://platform.claude.com/docs/en/about-claude/pricing",
            "prices_verified_client_date": "2026-10-14"}
    plan["sha256"] = digest(plan)
    write_json(PLAN, plan)
    return plan


def validate():
    base = preflight()
    plan = read_json(PLAN)
    if digest({k: v for k, v in plan.items() if k != "sha256"}) != plan["sha256"]:
        raise ValueError("Sample plan changed")
    if plan["parent_plan_sha256"] != base["sha256"] or plan["runner_sha256"] != sha(__file__):
        raise ValueError("Scientific inputs or sample runner changed")
    if plan["budget_usd"] != BUDGET or plan["families"] != list(FAMILIES):
        raise ValueError("Budget or model scope changed")
    if sha(OUT / "sample.csv") != plan["sample_sha256"]:
        raise ValueError("Sample rows changed")
    return plan


class BudgetBatcher(Batcher):
    """Serialize both models' shared ledger transactions, including submissions."""
    def __init__(self, family, labels, chunk):
        settings = dict(config_for(family)["llm"])
        settings.update(enforce_budget=True, budget_usd=BUDGET, ledger_path=str(LEDGER))
        super().__init__(settings, labels, chunk / family)
        # Original full-test prepared jobs contain IDs outside this sample.
        # Share parsed responses, but keep sample batch ownership separate.
        self.jobs_dir = OUT / family / "batches"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)

    def submit(self, specs):
        with LEDGER_LOCK:
            self.ledger = read_json(LEDGER)
            return super().submit(specs)

    def import_results(self, job):
        with LEDGER_LOCK:
            self.ledger = read_json(LEDGER)
            return super().import_results(job)


def specifications(frame, settings, labels, verifier=False):
    label_text = "\n".join(labels)
    longest_label = max(labels, key=lambda label: len(json.dumps(label, ensure_ascii=False).encode()))
    records = []
    for row in frame.itertuples(index=False):
        original = specification(settings, settings["model"], CLASSIFY.format(labels=label_text), row.text, labels)
        blind = specification(settings, settings["verifier_model"], BLIND.format(labels=label_text), row.text, labels)
        selected = longest_label
        cache = ROOT / settings["cache_dir"] / (original["key"] + ".json")
        if verifier:
            selected = read_json(cache)["parsed"]["label"]
        user = json.dumps({"message": row.text, "proposed_intent": selected}, ensure_ascii=False)
        verify = specification(settings, settings["verifier_model"], VERIFY.format(labels=label_text), user, labels, True)
        records.append((row, original, blind, verify))
    return records


def affordable_end(bounds, start, remaining):
    amount, end = 0.0, start
    for bound in bounds[start:]:
        if amount + bound > remaining - 1e-8:
            break
        amount += bound
        end += 1
    return end


def run_block(frame, labels, folder):
    validate()  # Must precede both client constructors and every new paid block.
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = []
        for family in FAMILIES:
            batcher = BudgetBatcher(family, labels, folder)
            records = specifications(frame, batcher.settings, labels)
            futures.append(pool.submit(batcher.wave, [spec for _, a, b, _ in records for spec in (a, b)], "sample_classification_and_blind"))
        for future in futures:
            future.result()
    validate()
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = []
        for family in FAMILIES:
            batcher = BudgetBatcher(family, labels, folder)
            records = specifications(frame, batcher.settings, labels, verifier=True)
            futures.append(pool.submit(batcher.wave, [v for _, _, _, v in records], "sample_verification"))
        for future in futures:
            future.result()


def score(family, frame, labels, base, plan):
    settings = config_for(family)["llm"]
    rows, costs = [], []
    for row, original, blind, verifier in specifications(frame, settings, labels, verifier=True):
        values = [read_json(ROOT / settings["cache_dir"] / (spec["key"] + ".json"))
                  for spec in (original, blind, verifier)]
        a, b, v = values
        rows.append({"id": row.id, "label": a["parsed"]["label"], "confidence": a["parsed"]["confidence"],
                     "blind_label": b["parsed"]["label"], "blind_confidence": b["parsed"]["confidence"], "verifier": v["parsed"]["confidence"]})
        for stage, spec, value in zip(("original", "blind", "verifier"), (original, blind, verifier), values):
            costs.append({"id": row.id, "stage": stage, "request_sha256": spec["key"], "usd": value["usd"],
                          "usage": value["usage"], "model": value["model"]})
    vectorizer, classifier = joblib.load(ROOT / config_for(family)["run_dir"] / "tfidf.joblib")
    logits = classifier.decision_function(vectorizer.transform(frame.text))
    learned = frozen_for(family)["calibration"]["correctness_model"].predict_proba(features(rows, frame, labels, logits))[:, 1]
    prediction = np.asarray([row["label"] for row in rows])
    objects = joblib.load(ROOT / "reports/calibrator-comparison/calibrators.joblib")
    streams = {family + "_same": [row["confidence"] for row in rows], family + "_verifier": [row["verifier"] for row in rows],
               family + "_correctness": learned}
    folder = OUT / family / "evaluation"
    folder.mkdir(parents=True, exist_ok=True)
    metrics = {}
    for stream, q in streams.items():
        for variant, scores in transform(stream, q, prediction, objects).items():
            key = stream + "." + variant
            threshold = base["variants"][key]["threshold"]
            m, reliability, curve = evaluate(frame.label, prediction, scores, threshold, labels=labels)
            metrics[key] = m
            pd.DataFrame({"id": frame.id, "text": frame.text, "truth": frame.label,
                          "prediction": prediction, "confidence": scores, "correct": frame.label.to_numpy() == prediction}).to_csv(folder / (key + ".predictions.csv"), index=False)
            pd.DataFrame(reliability).to_csv(folder / (key + ".reliability.csv"), index=False)
            pd.DataFrame(curve).to_csv(folder / (key + ".risk_coverage.csv"), index=False)
    pd.DataFrame(rows).to_csv(folder / "llm_sample.csv", index=False)
    write_json(folder / "metrics.json", metrics)
    write_json(folder / "metadata.json", {"sample_plan_sha256": plan["sha256"], "n": len(frame), "costs": costs,
               "evaluation_role": "official_test_subset", "thresholds_refitted": False})
    write_json(folder / "complete.json", {"sample_plan_sha256": plan["sha256"],
               "outputs_sha256": {p.name: sha(p) for p in folder.iterdir() if p.is_file() and p.name != "complete.json"}})


def run():
    plan = validate()
    frame = pd.read_csv(OUT / "sample.csv", keep_default_na=False)
    labels = read_json(ROOT / "data/banking77/manifest.json")["labels"]
    if not LEDGER.exists():
        write_json(LEDGER, {"usd": 0.0, "calls": 0, "reservations": {}})
    # Bound all six requests before starting a matched block. The verifier bound
    # uses the longest possible label; pricing and max output match development.
    bounds = np.zeros(len(frame))
    for family in FAMILIES:
        settings = config_for(family)["llm"]
        records = specifications(frame, settings, labels)
        for index, (_, a, b, v) in enumerate(records):
            bounds[index] += sum(spec["reserve"] for spec in (a, b, v))
    chunks_path = OUT / "chunks.json"
    chunks = read_json(chunks_path) if chunks_path.exists() else []
    start = 0
    for chunk in chunks:
        if chunk["start"] != start:
            raise ValueError("Sample chunk order changed")
        if chunk["state"] != "complete":
            run_block(frame.iloc[chunk["start"]:chunk["end"]], labels, OUT / "chunks" / str(chunk["start"]))
            chunk["state"] = "complete"
            write_json(chunks_path, chunks)
        start = chunk["end"]
    while start < len(frame):
        ledger = read_json(LEDGER)
        remaining = BUDGET - ledger["usd"] - sum(ledger["reservations"].values())
        end = affordable_end(bounds, start, remaining)
        if end == start:
            break
        chunk = {"start": start, "end": end, "state": "running", "worst_case_usd": float(bounds[start:end].sum())}
        chunks.append(chunk)
        write_json(chunks_path, chunks)
        write_json(OUT / "status.json", {"state": "running", "completed_messages_per_model": start,
                   "target_messages_per_model": len(frame), "current_chunk": chunk, "spent_usd": ledger["usd"], "updated_utc": now()})
        run_block(frame.iloc[start:end], labels, OUT / "chunks" / str(start))
        chunk["state"] = "complete"
        write_json(chunks_path, chunks)
        start = end
    if not start:
        raise BudgetExhausted("No complete matched example fits remaining budget")
    base = preflight()
    for family in FAMILIES:
        score(family, frame.iloc[:start].reset_index(drop=True), labels, base, plan)
    write_json(OUT / "status.json", {"state": "inference_complete", "n": start, "target_n": len(frame),
               "budget_limited": start != len(frame), "spent_usd": read_json(LEDGER)["usd"], "report_pending": True, "updated_utc": now()})
    report()


def report():
    plan = validate()
    base = read_json(FINAL / "frozen-plan.json")
    status = read_json(OUT / "status.json")
    frame = pd.read_csv(OUT / "sample.csv", keep_default_na=False).iloc[:status["n"]]
    labels = read_json(ROOT / "data/banking77/manifest.json")["labels"]
    metrics, frames, costs, inputs = {}, {}, {}, {"runs/claude-sample-20261014/plan.json": sha(PLAN),
                                                "scripts/run_claude_sample.py": sha(__file__)}
    for arm in ("embeddings", "tfidf", "decisions", "haiku", "sonnet", "opus"):
        folder = OUT / arm / "evaluation" if arm in FAMILIES else FINAL / arm
        marker = read_json(folder / "complete.json")
        for name, value in marker["outputs_sha256"].items():
            if sha(folder / name) != value:
                raise ValueError(f"Changed result: {arm}/{name}")
        inputs[(folder / "complete.json").relative_to(ROOT).as_posix()] = sha(folder / "complete.json")
        costs[arm] = [row for row in read_json(folder / "metadata.json")["costs"] if row["id"] in set(frame.id)]
        for key in read_json(folder / "metrics.json"):
            source = folder / (key + ".predictions.csv")
            f = pd.read_csv(source, float_precision="round_trip", keep_default_na=False).set_index("id").loc[frame.id].reset_index()
            if not np.array_equal(f.truth, frame.label) or not np.array_equal(f.correct, f.prediction == f.truth):
                raise ValueError("Matched prediction identity changed")
            metrics[key], _, _ = evaluate(f.truth, f.prediction, f.confidence, base["variants"][key]["threshold"], labels=labels)
            frames[key] = f
            inputs[source.relative_to(ROOT).as_posix()] = sha(source)
    if set(metrics) != set(base["variants"]):
        raise ValueError("Expected all 60 matched confidence variants")
    rng = np.random.default_rng(plan["seed"])
    boot = np.concatenate([rng.choice(group.index.to_numpy(), (5000, len(group)), replace=True)
                           for _, group in frame.reset_index(drop=True).groupby("label", sort=True)], axis=1)
    comparisons = {}
    pairs = [("opus_minus_sonnet_accuracy", "opus_same.raw", "sonnet_same.raw", "accuracy")]
    for family in ("haiku", "sonnet", "opus"):
        for variant in ("raw", "logistic_score", "logistic_logit", "isotonic"):
            pairs.append((family + "_learned_" + variant + "_minus_native_brier", family + "_correctness." + variant, family + "_same.raw", "brier"))
    for name, a, b, metric in pairs:
        first, second = frames[a], frames[b]
        diff = (first.correct.astype(float) - second.correct.astype(float)).to_numpy() if metric == "accuracy" else ((first.confidence-first.correct)**2-(second.confidence-second.correct)**2).to_numpy()
        comparisons[name] = {"difference": float(diff.mean()), "ci95": np.quantile(diff[boot].mean(axis=1), [.025, .975]).tolist(), "metric": metric}
    del boot
    pipelines = []
    for key, metric in metrics.items():
        spec = base["variants"][key]; stream, arm = spec["stream"], spec["arm"]
        selected = {"original"} if stream.endswith("_same") else {"original", "verifier"} if stream.endswith("_verifier") else {"original", "blind", "verifier"}
        price = sum(row["usd"] for row in {row["request_sha256"]: row for row in costs[arm] if row["stage"] in selected}.values())
        pipelines.append({"id": key, "stream": stream, "api_usd_per_1000": price / len(frame) * 1000,
                          **{field: metric[field] for field in ("accuracy", "brier", "error_auroc")}, "coverage": metric["acceptance"]["coverage"]})
    from analyze_cost_frontiers import frontier
    frontiers = {field: frontier(pipelines, field, minimize=field == "brier") for field in ("accuracy", "brier", "error_auroc", "coverage")}
    REPORT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(pipelines).to_csv(REPORT / "cost-frontiers.csv", index=False)
    summary = []
    for key, m in metrics.items():
        summary.append({"variant": key, **{field: m[field] for field in ("accuracy", "brier", "binary_log_loss", "error_auroc")},
                        "coverage": m["acceptance"]["coverage"], "accepted_errors": m["acceptance"]["errors"], "accepted_risk": m["acceptance"]["risk"]})
        frames[key].to_csv(REPORT / (key + ".predictions.csv"), index=False)
    pd.DataFrame(summary).to_csv(REPORT / "comparison.csv", index=False)
    from build_final_report import table, pct
    keys = ["embedding_lr.temperature", "embedding_mlp.temperature", "tfidf.temperature", "decisions_probability.logistic_logit", "haiku_same.raw", "sonnet_same.raw", "opus_same.raw"]
    names = ["Embedding LR", "Embedding MLP", "TF-IDF", "OpenAI", "Haiku", "Sonnet", "Opus"]
    accuracy_rows = [[name, pct(base["variants"][key]["development_metrics"]["accuracy"]), pct(metrics[key]["accuracy"]),
                      f"{pct(metrics[key]['accuracy_ci95'][0])}–{pct(metrics[key]['accuracy_ci95'][1])}"] for name, key in zip(names, keys)]
    confidence_rows = []
    for family in ("haiku", "sonnet", "opus"):
        for source in ("same", "verifier", "correctness"):
            stream = family + "_" + source
            for variant in ("raw", "logistic_score", "logistic_logit", "isotonic"):
                m = metrics[stream + "." + variant]
                confidence_rows.append([stream + "." + variant, f"{m['brier']:.5f}", f"{m['error_auroc']:.3f}", pct(m['acceptance']['coverage'])])
    spent = read_json(LEDGER)["usd"]
    text = f"""# Sonnet and Opus: matched independent sample validation

**{len(frame)} official test messages per model; {len(metrics)} frozen confidence variants; new API fees ${spent:.4f}.**

The planned sample selects eight messages randomly per intent (616 total), using seed 20261014 without reference to prediction outcomes. All models below use the same selected IDs. Budget-limited stopping: **{status['budget_limited']}**. The shared new-call limit is $12; the cohort size is determined by the frozen sample order and usage, never by observed accuracy or calibration. The independent full-test local/OpenAI report and full-test Haiku addendum remain separately available.

## Classification

{table(['Model','Development accuracy (1,000)','Matched test accuracy','Nominal 95% interval'],accuracy_rows)}

The local heads used 5,992 fitting labels, whereas the hosted classifications use zero-shot intent names. All classifier, correctness-model, calibrator, prompt and threshold choices are inherited unchanged from the original sealed plan. No sample outcomes are used for refitting or threshold selection. Development and sample accuracy use different examples.

## Confidence and calibration

{table(['Frozen confidence variant','Test Brier (lower is better)','Error AUROC','Frozen policy coverage'],confidence_rows)}

Learned confidence combines native confidence, verifier output, blind agreement and supervised TF-IDF support. Its improvements cannot be attributed solely to a calibrator or to extra inference without a feature ablation. Selected labels stay fixed across confidence variants. Zero policy coverage means development selected no acceptance threshold; it does not show that every possible threshold fails. This sample is a generalization check and does not establish useful automation at 1% error.

## Cost and uncertainty

The spending ledger records **${spent:.4f}** in new Sonnet/Opus batch fees, against a **$12** cap. Costs are usage-based estimates at the verified configured rates, not invoice totals. Native confidence uses one call, verifier confidence two, and learned correctness three; recalibration adds no API calls. Baseline predictions are reused and incur no new fees. Cohort API cost frontiers cover all 60 variants on this same subset and exclude unpriced labeling, fitting, compute and review costs.

Paired confidence comparisons and their intervals are in evidence.json. They use 5,000 resamples within intent strata, condition on the fitted objects, and are not corrected for multiple comparisons. Small differences and rare accepted errors remain uncertain. If the cap truncated the sample, intent counts may differ and that usage-based stopping limits interpretation. Public benchmark familiarity and distribution shift remain further limitations.

The larger local/OpenAI evaluation is in ../banking77-findings/report.md; the full Haiku evaluation is in ../haiku-test/report.md. These sample results must not be ranked against other models' full-test results without matching cohorts.
"""
    (REPORT / "report.md").write_text(text, encoding="utf-8")
    import build_final_report as renderer
    renderer.INSERTS = {}
    html = '<!doctype html><meta charset="utf-8"><title>Matched Claude test sample</title><style>body{font:14px/1.6 Arial;max-width:1200px;margin:40px auto;padding:20px}table{border-collapse:collapse;width:100%;font-size:12px}td,th{padding:7px;border-bottom:1px solid #ddd;text-align:left}th{background:#eef3ef}</style>' + renderer.markdown_html(text)
    (REPORT / "report.html").write_text(html, encoding="utf-8")
    write_json(REPORT / "evidence.json", {"sample_plan_sha256": plan["sha256"], "parent_plan_sha256": base["sha256"],
               "n": len(frame), "budget_limited": status["budget_limited"], "metrics": metrics, "paired_comparisons": comparisons,
               "pipelines": pipelines, "frontiers": frontiers, "new_spending_usd": spent, "source_sha256": inputs,
               "report_sha256": sha(REPORT / "report.md"), "thresholds_refitted": False})
    write_json(OUT / "status.json", {"state": "complete", "n": len(frame), "target_n": plan["n"],
               "budget_limited": status["budget_limited"], "spent_usd": spent, "report": str(REPORT / "report.md"), "updated_utc": now()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "run", "report"))
    args = parser.parse_args()
    if args.command == "freeze":
        plan = freeze()
        print(json.dumps({"n": plan["n"], "expected_usd": plan["expected_usd"], "cap_usd": BUDGET, "sha256": plan["sha256"]}), flush=True)
        return
    validate()
    if args.command == "report":
        report()
        return
    lock = OUT / ".lock"
    with lock.open("x") as handle:
        handle.write(str(os.getpid()))
    try:
        if (OUT / "status.json").exists() and read_json(OUT / "status.json")["state"] == "complete":
            print("Sample evaluation already complete; no API work repeated.", flush=True)
            return
        run()
    except Exception:
        write_json(OUT / "status.json", {"state": "failed", "error": traceback.format_exc(), "updated_utc": now()})
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
