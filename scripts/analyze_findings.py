"""Read completed development outputs and quantify the report's comparisons.

Does not refit models, select new thresholds, call APIs, or evaluate test data.
Bootstrap intervals condition on the fitted models and calibration objects.
"""
from datetime import datetime, timezone
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from confidence_experiment.io import read_json, write_json

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports" / "banking77-findings"
RUNS = {"embeddings": "embeddings", "tfidf": "pilot", "haiku": "anthropic-batch-v2", "openai": "openai-decisions-v2"}
PRIMARY = {
    "embedding_lr_raw": "embeddings", "embedding_lr_temperature": "embeddings",
    "embedding_mlp_raw": "embeddings", "embedding_mlp_temperature": "embeddings",
    "tfidf_temperature": "tfidf", "llm_same_raw": "haiku", "llm_same_sigmoid": "haiku",
    "llm_verifier_raw": "haiku", "llm_correctness_raw": "haiku", "llm_correctness_sigmoid": "haiku",
    "decisions_confidence_raw": "openai", "decisions_probability_raw": "openai",
    "decisions_probability_temperature": "openai",
}


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    metrics, frames, hashes = {}, {}, {}
    manifest_path = ROOT / "data/banking77/manifest.json"
    manifest = read_json(manifest_path)
    source_paths = [manifest_path]
    for name, run_name in RUNS.items():
        directory = ROOT / "runs" / run_name / "development"
        source_paths += [directory / "metrics.json", directory / "metadata.json"]
        metrics[name] = read_json(directory / "metrics.json")
    reference = None
    for method, name in PRIMARY.items():
        path = ROOT / "runs" / RUNS[name] / "development" / f"{method}.predictions.csv"
        source_paths.append(path)
        frame = pd.read_csv(path).sort_values("id").reset_index(drop=True)
        assert len(frame) == 1000 and frame.id.is_unique, method
        assert np.array_equal(frame.correct, frame.truth == frame.prediction), method
        if reference is None:
            reference = frame[["id", "truth"]]
        else:
            assert frame[["id", "truth"]].equals(reference), method
        frames[method] = frame
    for family in ([m for m in PRIMARY if m.startswith("llm_")], [m for m in PRIMARY if m.startswith("decisions_")]):
        assert all(frames[m].prediction.equals(frames[family[0]].prediction) for m in family)
    for path in source_paths:
        hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()

    # All comparisons use identical resamples, preserving pairing by example ID.
    seed, repeats, batch = 20261009, 5000, 250
    rng = np.random.default_rng(seed)
    boot = {m: {key: [] for key in ("accuracy", "brier", "error_auroc")} for m in PRIMARY}
    for start in range(0, repeats, batch):
        indices = rng.integers(0, 1000, size=(min(batch, repeats - start), 1000))
        for method, frame in frames.items():
            correct = frame.correct.to_numpy(dtype=float)[indices]
            confidence = frame.confidence.to_numpy(dtype=float)[indices]
            error = 1 - correct
            positives = error.sum(axis=1)
            ranks = rankdata(1 - confidence, axis=1, method="average")
            numerator = (ranks * error).sum(axis=1) - positives * (positives + 1) / 2
            denominator = positives * (1000 - positives)
            auroc = np.divide(numerator, denominator, out=np.full(len(indices), np.nan), where=denominator != 0)
            boot[method]["accuracy"].append(correct.mean(axis=1))
            boot[method]["brier"].append(((confidence - correct) ** 2).mean(axis=1))
            boot[method]["error_auroc"].append(auroc)
    boot = {m: {k: np.concatenate(v) for k, v in values.items()} for m, values in boot.items()}

    def interval(values):
        return [float(x) for x in np.nanquantile(values, [0.025, 0.975])]

    estimates = {}
    for method, name in PRIMARY.items():
        frame = frames[method]
        estimates[method] = {"run": RUNS[name], "n": len(frame),
            "mean_confidence": float(frame.confidence.mean()),
            **{key: {"estimate": metrics[name][method][key], "paired_bootstrap_ci95": interval(boot[method][key])}
                for key in ("accuracy", "brier", "error_auroc")}}

    comparisons = {}
    specifications = [
        ("embedding_minus_haiku_accuracy", "accuracy", "embedding_lr_temperature", "llm_same_raw"),
        ("embedding_minus_openai_accuracy", "accuracy", "embedding_lr_temperature", "decisions_confidence_raw"),
        ("embedding_lr_minus_mlp_accuracy", "accuracy", "embedding_lr_temperature", "embedding_mlp_raw"),
        ("haiku_original_minus_learned_brier", "brier", "llm_same_raw", "llm_correctness_raw"),
        ("haiku_original_minus_verifier_brier", "brier", "llm_same_raw", "llm_verifier_raw"),
        ("haiku_original_minus_sigmoid_brier", "brier", "llm_same_raw", "llm_same_sigmoid"),
        ("embedding_raw_minus_temperature_brier", "brier", "embedding_lr_raw", "embedding_lr_temperature"),
        ("openai_raw_minus_temperature_brier", "brier", "decisions_probability_raw", "decisions_probability_temperature"),
        ("haiku_learned_minus_original_auroc", "error_auroc", "llm_correctness_raw", "llm_same_raw"),
        ("haiku_verifier_minus_original_auroc", "error_auroc", "llm_verifier_raw", "llm_same_raw"),
    ]
    for name, metric, first, second in specifications:
        comparisons[name] = {"metric": metric, "first": first, "second": second,
            "estimate": estimates[first][metric]["estimate"] - estimates[second][metric]["estimate"],
            "paired_bootstrap_ci95": interval(boot[first][metric] - boot[second][metric])}

    coverage = {}
    for method in ("embedding_lr_temperature", "llm_same_raw", "llm_verifier_raw", "llm_correctness_raw", "decisions_confidence_raw"):
        directory = ROOT / "runs" / RUNS[PRIMARY[method]] / "development"
        curve = pd.read_csv(directory / f"{method}.risk_coverage.csv")
        coverage[method] = {}
        for target in (0.5, 0.8, 0.9):
            eligible = curve[curve.coverage <= target + 1e-12]
            if len(eligible):
                row = eligible.iloc[-1]
                coverage[method][str(target)] = {"accepted": int(row.accepted), "errors": int(row.errors),
                    "actual_coverage": float(row.coverage), "risk": float(row.risk), "threshold": float(row.threshold)}

    original, learned, embedding = frames["llm_same_raw"], frames["llm_correctness_raw"], frames["embedding_lr_temperature"]
    mask = (~original.correct) & (original.confidence >= 0.95) & (learned.confidence < 0.25) & embedding.correct
    examples = []
    for i in original.index[mask][:3]:
        examples.append({"id": original.loc[i, "id"], "text": original.loc[i, "text"], "truth": original.loc[i, "truth"],
            "haiku_prediction": original.loc[i, "prediction"], "haiku_confidence": float(original.loc[i, "confidence"]),
            "learned_confidence": float(learned.loc[i, "confidence"]), "embedding_prediction": embedding.loc[i, "prediction"]})

    spending = []
    for name in ("data/anthropic_cache/spending.json", "data/anthropic_spending/spending.json"):
        ledger = read_json(ROOT / name)
        source_paths.append(ROOT / name)
        spending.append({"path": name, "usd": ledger["usd"], "reservations_usd": sum(ledger.get("reservations", {}).values())})
    openai_cost = sum(read_json(p)["response"]["usage"]["input_tokens"] * 0.10 / 1e6
        for p in (ROOT / "data/openai_decisions_cache").glob("*.raw.json"))
    data = {"created_utc": datetime.now(timezone.utc).isoformat(), "evaluation_role": "development", "n": 1000,
        "bootstrap": {"seed": seed, "resamples": repeats, "interval": "Paired percentile bootstrap over shared example IDs, conditional on fitted models and calibrators; development only"},
        "manifest": manifest, "estimates": estimates, "comparisons": comparisons, "risk_at_coverage": coverage,
        "illustrative_high_confidence_errors": examples, "all_metrics": metrics,
        "spending": {"ledgers": spending, "combined_usd": sum(v["usd"] for v in spending), "openai_input_usage_usd": openai_cost},
        "source_sha256": hashes}
    write_json(OUTPUT / "analysis.json", data)
    np.savez_compressed(OUTPUT / "bootstrap_samples.npz", **{f"{m}__{k}": v for m, values in boot.items() for k, v in values.items()})
    print("Verified 1,000 matching development IDs and unchanged predictions within each LLM confidence family.")
    for name, result in comparisons.items():
        print(name, result["estimate"], result["paired_bootstrap_ci95"])
    print("Risk at coverage:", coverage)
    print("Examples:", examples)
    print("API spending:", data["spending"])


if __name__ == "__main__":
    main()
