"""Paired development analysis after the Sonnet replication completes.

No API calls, model fitting, threshold selection, or official-test evaluation.
"""
from datetime import datetime, timezone
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from confidence_experiment.io import read_json, write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/banking77-findings"
RUNS = {"haiku": "anthropic-batch-v2", "sonnet": "anthropic-sonnet-5-5", "embedding": "embeddings"}
FAMILIES = ("llm_same_raw", "llm_verifier_raw", "llm_correctness_raw")


def main():
    previous = read_json(OUT / "analysis.json")
    for relative, expected in previous["source_sha256"].items():
        actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"The original evidence changed: {relative}")
    metrics = {family: read_json(ROOT / "runs" / run / "development/metrics.json") for family, run in RUNS.items()}
    metadata = {family: read_json(ROOT / "runs" / run / "development/metadata.json") for family, run in RUNS.items()}
    frames, hashes, reference = {}, {}, None
    for family, run in RUNS.items():
        methods = ("embedding_lr_temperature",) if family == "embedding" else FAMILIES
        for method in methods:
            path = ROOT / "runs" / run / "development" / f"{method}.predictions.csv"
            frame = pd.read_csv(path).sort_values("id").reset_index(drop=True)
            assert len(frame) == 1000 and frame.id.is_unique
            assert np.array_equal(frame.correct.to_numpy(), frame.truth == frame.prediction)
            if reference is None:
                reference = frame[["id", "truth"]]
            else:
                assert frame[["id", "truth"]].equals(reference)
            frames[f"{family}:{method}"] = frame
            hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        if family != "embedding":
            assert all(frames[f"{family}:{m}"].prediction.equals(frames[f"{family}:{FAMILIES[0]}"].prediction) for m in FAMILIES)

    # Only provider/model/thinking, storage, and matching token rates may differ.
    h, s = metadata["haiku"]["config"], metadata["sonnet"]["config"]
    scientific_keys = ("seed", "data_dir", "methods", "max_error", "threshold_rule", "supervised")
    assert all(h[k] == s[k] for k in scientific_keys)
    for key in ("transport", "max_output_tokens", "batch_size"):
        assert h["llm"][key] == s["llm"][key]
    assert h["llm"]["model"] == h["llm"]["verifier_model"] == "claude-haiku-5-5"
    assert s["llm"]["model"] == s["llm"]["verifier_model"] == "claude-sonnet-5-5"
    assert h["llm"]["thinking"] == "disabled" and s["llm"]["thinking"] == "between_tools"
    assert metadata["haiku"]["dataset"] == metadata["sonnet"]["dataset"]
    # The same labeled TF-IDF support is part of both learned estimators.
    for artifact in ("tfidf.joblib",):
        paths = [ROOT / "runs" / RUNS[family] / artifact for family in ("haiku", "sonnet")]
        objects = []
        import joblib
        for path in paths:
            objects.append(joblib.load(path))
        # Training is deterministic, but pickle byte equality is not assumed.
        (v0, c0), (v1, c1) = objects
        assert v0.vocabulary_ == v1.vocabulary_
        assert np.allclose(v0.idf_, v1.idf_, rtol=0, atol=1e-10)
        assert np.array_equal(c0.classes_, c1.classes_)
        assert np.allclose(c0.coef_, c1.coef_, rtol=0, atol=1e-10)
        assert np.allclose(c0.intercept_, c1.intercept_, rtol=0, atol=1e-10)

    boot = {key: {metric: [] for metric in ("accuracy", "brier", "error_auroc")} for key in frames}
    rng, repeats = np.random.default_rng(20261009), 5000
    for start in range(0, repeats, 250):
        indices = rng.integers(0, 1000, size=(min(250, repeats-start), 1000))
        for key, frame in frames.items():
            correct = frame.correct.to_numpy(dtype=float)[indices]
            confidence = frame.confidence.to_numpy(dtype=float)[indices]
            errors = 1-correct
            positives = errors.sum(axis=1)
            ranks = rankdata(1-confidence, axis=1, method="average")
            numerator = (ranks*errors).sum(axis=1)-positives*(positives+1)/2
            denominator = positives*(1000-positives)
            boot[key]["accuracy"].append(correct.mean(axis=1))
            boot[key]["brier"].append(((confidence-correct)**2).mean(axis=1))
            boot[key]["error_auroc"].append(np.divide(numerator, denominator,
                out=np.full(len(indices), np.nan), where=denominator != 0))
    boot = {key: {metric: np.concatenate(values) for metric, values in row.items()} for key, row in boot.items()}
    estimates = {}
    for key in frames:
        family, method = key.split(":")
        estimates[key] = {metric: {"estimate": metrics[family][method][metric],
            "ci95": [float(v) for v in np.nanquantile(boot[key][metric], [.025, .975])]}
            for metric in boot[key]}

    comparisons = {}
    pairs = [
        ("sonnet_minus_haiku_accuracy", "accuracy", "sonnet:llm_same_raw", "haiku:llm_same_raw"),
        ("embedding_minus_sonnet_accuracy", "accuracy", "embedding:embedding_lr_temperature", "sonnet:llm_same_raw"),
        ("sonnet_original_minus_learned_brier", "brier", "sonnet:llm_same_raw", "sonnet:llm_correctness_raw"),
        ("sonnet_original_minus_verifier_brier", "brier", "sonnet:llm_same_raw", "sonnet:llm_verifier_raw"),
        ("sonnet_learned_minus_original_auroc", "error_auroc", "sonnet:llm_correctness_raw", "sonnet:llm_same_raw"),
        ("haiku_original_minus_learned_brier", "brier", "haiku:llm_same_raw", "haiku:llm_correctness_raw"),
    ]
    for name, metric, first, second in pairs:
        comparisons[name] = {"metric": metric, "first": first, "second": second,
            "estimate": estimates[first][metric]["estimate"]-estimates[second][metric]["estimate"],
            "ci95": [float(v) for v in np.nanquantile(boot[first][metric]-boot[second][metric], [.025, .975])]}
    risk = {}
    for method in FAMILIES:
        curve = pd.read_csv(ROOT / "runs" / RUNS["sonnet"] / "development" / f"{method}.risk_coverage.csv")
        risk[method] = {}
        for target in (.5, .8, .9):
            eligible = curve[curve.coverage <= target+1e-12]
            if len(eligible):
                row = eligible.iloc[-1]
                risk[method][str(target)] = {"accepted": int(row.accepted), "errors": int(row.errors),
                    "coverage": float(row.coverage), "risk": float(row.risk)}
    ledgers = []
    for relative in ("data/anthropic_cache/spending.json", "data/anthropic_spending/spending.json", "data/anthropic_sonnet_cache/spending.json"):
        ledger = read_json(ROOT / relative)
        ledgers.append({"path": relative, "usd": ledger["usd"], "reservations_usd": sum(ledger.get("reservations", {}).values())})
        hashes[relative] = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
    for family, run in RUNS.items():
        for filename in ("metrics.json", "metadata.json"):
            path = ROOT / "runs" / run / "development" / filename
            hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    result = {"created_utc": datetime.now(timezone.utc).isoformat(), "role": "development", "n": 1000,
        "bootstrap": {"seed": 20261009, "resamples": repeats,
            "scope": "Paired percentile intervals conditional on fitted models; no refitting or selection adjustment"},
        "config_match_verified": True, "tfidf_support_match_verified": True,
        "thinking_difference": {"haiku": "disabled", "sonnet": "between_tools", "tools_used": False},
        "metrics": metrics, "estimates": estimates, "comparisons": comparisons, "risk_at_coverage": risk,
        "spending": {"ledgers": ledgers, "combined_usd": sum(x["usd"] for x in ledgers)}, "source_sha256": hashes}
    write_json(OUT / "sonnet-comparison.json", result)
    for key, value in comparisons.items():
        print(key, value, flush=True)
    print("Sonnet confidence metrics:", {k:v for k,v in metrics["sonnet"].items() if k.startswith("llm_")}, flush=True)


if __name__ == "__main__":
    main()
