import importlib.metadata
import hashlib
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

from .data import load
from .io import digest, read_json, write_json
from .metrics import Sigmoid, evaluate, fit_temperature, select_threshold
from .models import train_supervised
from .report import save_report


def features(rows, frame, labels, alternative_logits=None, neighborhood=None):
    selected = np.array([labels.index(row["label"]) for row in rows])
    q1 = np.array([row["confidence"] for row in rows])
    q2 = np.array([row["verifier"] for row in rows])
    agree = np.array([row["label"] == row["blind_label"] for row in rows], dtype=float)
    data = [q1, q2, agree, np.log1p(frame.text.str.len().to_numpy()), (frame.text.str.strip() == "").to_numpy(dtype=float)]
    if alternative_logits is not None:
        probabilities = softmax(alternative_logits, axis=1)
        ordered = np.sort(probabilities, axis=1)
        data += [probabilities[np.arange(len(rows)), selected], ordered[:, -1] - ordered[:, -2]]
    if neighborhood is not None:
        distance, neighbor_labels = neighborhood
        data += [distance, (neighbor_labels == selected[:, None]).mean(axis=1)]
    # Few errors and 77 categories: omit category one-hot features in the first model.
    return np.column_stack(data)


def identity(config, manifest):
    # Operational caps can be changed to resume without changing the experiment.
    scientific = {**config, "llm": {k: v for k, v in config.get("llm", {}).items()
        if k not in {"enabled", "max_new_calls", "budget_usd"}}}
    source = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")}
    return digest({"config": scientific, "data": manifest, "source": source, "protocol": 1})


def file_hash(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def frozen_hashes(output):
    paths = [output / "frozen.joblib"]
    paths += [output / name for name in ("tfidf.joblib", "embedding_lr.joblib", "embedding_mlp.joblib", "embedding_model.json", "encoder_model.json") if (output / name).exists()]
    if (output / "modernbert").exists():
        paths += [p for p in (output / "modernbert").rglob("*") if p.is_file()]
    return {str(p.relative_to(output)): file_hash(p) for p in paths}


def run(config, final=False):
    start = time.perf_counter()
    manifest, frames = load(config["data_dir"])
    labels = manifest["labels"]
    output = Path(config["run_dir"])
    output.mkdir(parents=True, exist_ok=True)
    signature = identity(config, manifest)
    frozen_path = output / "frozen.joblib"
    checkpoint = output / "supervised.joblib"
    identity_path = output / "identity.json"
    if identity_path.exists() and read_json(identity_path)["sha256"] != signature:
        raise ValueError("This run directory belongs to different settings/data. Choose a new run_dir.")
    write_json(identity_path, {"sha256": signature})
    if final and not frozen_path.exists():
        raise ValueError("Complete a development run first, then evaluate its frozen models.")
    if not final and frozen_path.exists():
        raise ValueError("This experiment is frozen. Use --final, or choose a new run_dir to develop a different experiment.")
    if final and (output / "final" / "metrics.json").exists():
        raise ValueError("Final evaluation already exists. Read the existing report.")
    if final:
        if frozen_hashes(output) != read_json(output / "frozen_hashes.json"):
            raise ValueError("Frozen model artifacts were changed; final evaluation refused.")
        frozen = joblib.load(frozen_path)
        logits, artifacts = train_supervised(config, frames, labels, output, include_test=True, frozen=True)
    elif checkpoint.exists():
        logits, artifacts = joblib.load(checkpoint)
    else:
        logits, artifacts = train_supervised(config, frames, labels, output)
        joblib.dump((logits, artifacts), checkpoint)
    roles = ["test"] if final else ["correctness", "calibration", "development"]
    signals, calibration = {}, {} if not final else frozen["calibration"]
    label_id = {label: i for i, label in enumerate(labels)}
    for method, partitions in logits.items():
        if not final:
            truth = np.array([label_id[x] for x in frames["calibration"].label])
            calibration[method] = fit_temperature(partitions["calibration"], truth)
        for variant, temperature in ((method + "_raw", 1.0), (method + "_temperature", calibration[method])):
            signals[variant] = {}
            for role in roles:
                probabilities = softmax(partitions[role] / temperature, axis=1)
                prediction = probabilities.argmax(axis=1)
                signals[variant][role] = (np.array(labels)[prediction], probabilities[np.arange(len(prediction)), prediction])
    ledger = None
    if "llm" in config["methods"]:
        from .llm import generate
        llm_rows, ledger = generate(config, frames, labels, roles)
        llm_features = {}
        for role in roles:
            pd.DataFrame(llm_rows[role]).to_csv(output / f"llm_{role}.csv", index=False)
            support_name = next((m for m in ("embedding_lr", "tfidf", "modernbert") if m in logits), None)
            support = logits[support_name][role] if support_name else None
            neighborhood = artifacts.get("embedding_features", {}).get(role)
            llm_features[role] = features(llm_rows[role], frames[role], labels, support, neighborhood)
        if not final:
            predictions = np.array([row["label"] for row in llm_rows["correctness"]])
            correct = (predictions == frames["correctness"].label.to_numpy()).astype(int)
            if len(np.unique(correct)) != 2:
                raise ValueError("Correctness model needs both correct and incorrect predictions.")
            if (correct == 0).sum() < 100:
                print(f"Only {(correct == 0).sum()} errors for correctness training; treat results as exploratory.", flush=True)
            correctness_model = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=1000, random_state=config["seed"]))
            correctness_model.fit(llm_features["correctness"], correct)
            calibration["correctness_model"] = correctness_model
        correctness_model = calibration["correctness_model"]
        for variant in ("llm_same_raw", "llm_same_sigmoid", "llm_verifier_raw", "llm_verifier_sigmoid", "llm_correctness_raw", "llm_correctness_sigmoid"):
            signals[variant] = {}
        for role in roles:
            rows = llm_rows[role]
            pred = np.array([row["label"] for row in rows])
            signals["llm_same_raw"][role] = (pred, np.array([row["confidence"] for row in rows]))
            signals["llm_verifier_raw"][role] = (pred, np.array([row["verifier"] for row in rows]))
            signals["llm_correctness_raw"][role] = (pred, correctness_model.predict_proba(llm_features[role])[:, 1])
        for name in ("llm_same", "llm_verifier", "llm_correctness"):
            if not final:
                pred, q = signals[name + "_raw"]["calibration"]
                calibration[name] = Sigmoid().fit(q, (pred == frames["calibration"].label.to_numpy()).astype(int))
            for role in roles:
                pred, q = signals[name + "_raw"][role]
                signals[name + "_sigmoid"][role] = (pred, calibration[name].predict(q))
    evaluation_role = "test" if final else "development"
    report_dir = output / ("final" if final else "development")
    report_dir.mkdir(exist_ok=True)
    results, thresholds = {}, {} if not final else frozen["thresholds"]
    for method, partitions in signals.items():
        truth = frames[evaluation_role].label.to_numpy()
        pred, q = partitions[evaluation_role]
        if not final:
            thresholds[method] = select_threshold((truth == pred).astype(int), q, config["max_error"], config.get("threshold_rule", "wilson"))
        metrics, reliability, curve = evaluate(truth, pred, q, thresholds[method], labels=labels)
        results[method] = metrics
        pd.DataFrame(reliability).to_csv(report_dir / f"{method}.reliability.csv", index=False)
        pd.DataFrame(curve).to_csv(report_dir / f"{method}.risk_coverage.csv", index=False)
        predictions = pd.DataFrame({"id": frames[evaluation_role].id, "text": frames[evaluation_role].text,
            "truth": truth, "prediction": pred, "confidence": q, "correct": truth == pred})
        predictions.to_csv(report_dir / f"{method}.predictions.csv", index=False)
        predictions[~predictions.correct].to_csv(report_dir / f"{method}.disagreements.csv", index=False)
        write_json(report_dir / f"{method}.per_category.json", classification_report(truth, pred, labels=labels, output_dict=True, zero_division=0))
    if not final:
        joblib.dump({"calibration": calibration, "thresholds": thresholds, "identity": signature}, frozen_path)
        write_json(output / "frozen_hashes.json", frozen_hashes(output))
    versions = {}
    for name in ("numpy", "pandas", "scipy", "scikit-learn", "anthropic", "torch", "transformers", "sentence-transformers"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    metadata = {"created_utc": datetime.now(timezone.utc).isoformat(), "config": config, "dataset": manifest,
        "identity": signature, "evaluation_role": evaluation_role, "python": platform.python_version(), "packages": versions,
        "runtime_seconds": time.perf_counter() - start,
        "model_details": {k: v for k, v in artifacts.items() if k != "embedding_features"}, "api_ledger": ledger,
        "limitations": ["LLM prompts use intent names as category definitions and no labeled examples (zero-shot).",
                         "Prompt-order stability features, LEDGAR and CLINC150 are deferred extensions.",
                         "Development acceptance metrics reuse threshold-selection data; only frozen final evaluation estimates held-out performance."]}
    table = save_report(report_dir, results, metadata, evaluation_role)
    print(table.to_string(index=False), flush=True)
    print(f"Report: {(report_dir / 'report.html').resolve()}", flush=True)
    return results
