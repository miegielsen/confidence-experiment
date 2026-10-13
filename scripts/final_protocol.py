"""Frozen scientific protocol for the complete BANKING77 test comparison.

Historical package files stay byte-identical to preserve their run identities.
No fitting or threshold selection is permitted in this module.
"""
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time
import tomllib

import joblib
import numpy as np
import pandas as pd
from scipy.special import logit, softmax

from confidence_experiment.data import load
from confidence_experiment.io import digest, read_json, write_json
from confidence_experiment.metrics import evaluate
from confidence_experiment.runner import features, identity

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/final-evaluation"
PLAN = OUT / "frozen-plan.json"
ARMS = {
    "embeddings": ("embeddings", ["embedding_lr", "embedding_mlp"]),
    "tfidf": ("anthropic-batch-v2", ["tfidf"]),
    "haiku": ("anthropic-batch-v2", ["haiku_same", "haiku_verifier", "haiku_correctness"]),
    "sonnet": ("anthropic-sonnet-5-5", ["sonnet_same", "sonnet_verifier", "sonnet_correctness"]),
    "opus": ("anthropic-opus-5-5", ["opus_same", "opus_verifier", "opus_correctness"]),
    "decisions": ("openai-decisions", ["decisions_confidence", "decisions_probability"]),
}
THRESHOLD_FIELDS = ("threshold", "development_coverage", "development_risk", "rule", "target_error")


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def config_for(arm):
    return tomllib.loads((ROOT / "configs" / (ARMS[arm][0] + ".toml")).read_text(encoding="utf-8"))


def frozen_for(arm):
    return joblib.load(ROOT / config_for(arm)["run_dir"] / "frozen.joblib")


def transform(key, q, prediction, calibrators, temperature_q=None):
    """Preserve selected labels; refusals are hard review cases at every score."""
    q = np.asarray(q, dtype=float)
    objects = calibrators[key]
    variants = {
        "raw": q.copy(),
        "logistic_score": objects["logistic_score"].predict(q),
        "logistic_logit": objects["logistic_logit"].predict_proba(
            logit(np.clip(q, 1e-7, 1 - 1e-7)).reshape(-1, 1))[:, 1],
        "isotonic": objects["isotonic"].predict(q),
    }
    if temperature_q is not None:
        variants["temperature"] = np.asarray(temperature_q, dtype=float)
    refused = np.asarray(prediction) == "__refusal__"
    for name, values in variants.items():
        values = np.asarray(values, dtype=float).copy()
        if values.shape != q.shape or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError(f"Invalid probabilities: {key}.{name}")
        values[refused] = 0
        variants[name] = values
    return variants


def temperature_scores(logits, temperature, selected=None):
    probabilities = softmax(logits / temperature, axis=1)
    if selected is None:
        selected = np.asarray(logits).argmax(axis=1)
    return probabilities[np.arange(len(probabilities)), selected]


def freeze():
    if PLAN.exists():
        return preflight()
    analysis = read_json(ROOT / "reports/calibrator-comparison/analysis.json")
    calibrators_path = ROOT / "reports/calibrator-comparison/calibrators.joblib"
    calibrators = joblib.load(calibrators_path)
    manifest = read_json(ROOT / "data/banking77/manifest.json")
    expected = analysis["source_sha256"]
    inputs = {}

    def track(relative, wanted=None):
        p = ROOT / relative
        actual = sha(p)
        if wanted is not None and actual != wanted:
            raise ValueError(f"Development evidence changed: {relative}")
        inputs[p.relative_to(ROOT).as_posix()] = actual

    for name, wanted in expected.items():
        track(name, wanted)
    track("reports/calibrator-comparison/analysis.json")
    track(calibrators_path)
    track("data/banking77/manifest.json")
    for role, wanted in manifest["split_sha256"].items():
        track(f"data/banking77/{role}.csv", wanted)
    # Scientific inference and request-construction code is sealed. Presentation
    # scripts are recorded with their own outputs and can be repaired separately.
    for p in (ROOT / "src/confidence_experiment").glob("*.py"):
        track(p)
    for name in ("final_protocol.py", "run_anthropic_batch.py", "run_openai_decisions.py", "final_transport.py"):
        track(ROOT / "scripts" / name)
    versions = {name: importlib.metadata.version(name) for name in
                ("numpy", "pandas", "scipy", "scikit-learn", "torch", "transformers", "sentence-transformers", "anthropic")}
    for arm in ARMS:
        config = config_for(arm)
        run = ROOT / config["run_dir"]
        track(ROOT / "configs" / (ARMS[arm][0] + ".toml"))
        if arm == "decisions":
            from run_openai_decisions import signature
            current = signature(config, manifest)
        else:
            current = identity(config, manifest)
        frozen = joblib.load(run / "frozen.joblib")
        if current != frozen["identity"] or current != read_json(run / "identity.json")["sha256"]:
            raise ValueError(f"Historical run identity changed: {arm}")
        for name, wanted in read_json(run / "frozen_hashes.json").items():
            track(run / name, wanted)
        package_versions = read_json(run / "development/metadata.json").get("packages", {})
        for name, version in package_versions.items():
            if name in versions and versions[name] != version:
                raise ValueError(f"Package changed since development: {name}")

    dev = pd.read_csv(ROOT / "data/banking77/development.csv", keep_default_na=False)
    labels = manifest["labels"]
    specifications = {}
    for arm, (_, keys) in ARMS.items():
        run = ROOT / config_for(arm)["run_dir"]
        frozen = frozen_for(arm)["calibration"]
        for key in keys:
            raw_path = ROOT / f"reports/calibrator-comparison/{key}.raw.predictions.csv"
            raw = pd.read_csv(raw_path, float_precision="round_trip").set_index("id").loc[dev.id].reset_index()
            if not np.array_equal(raw.truth, dev.label):
                raise ValueError(f"Development row mismatch: {key}")
            tq = None
            if "temperature" in analysis["results"][key]:
                if arm == "decisions":
                    rows = pd.read_csv(run / "decisions_development.csv").set_index("id").loc[dev.id]
                    probabilities = np.array([json.loads(p) if label != "__refusal__" else [1 / len(labels)] * len(labels)
                                              for p, label in zip(rows.probabilities, rows.label)])
                    logits = np.log(np.clip(probabilities, 1e-12, 1))
                    selected = np.array([labels.index(p) if p in labels else 0 for p in raw.prediction])
                    tq = temperature_scores(logits, frozen["temperature"], selected)
                else:
                    logits, _ = joblib.load(run / "supervised.joblib")
                    tq = temperature_scores(logits[key]["development"], frozen[key])
            variants = transform(key, raw.confidence, raw.prediction, calibrators, tq)
            if set(variants) != set(analysis["results"][key]):
                raise ValueError(f"Variant mismatch: {key}")
            for variant, scores in variants.items():
                relative = f"reports/calibrator-comparison/{key}.{variant}.predictions.csv"
                saved = pd.read_csv(ROOT / relative, float_precision="round_trip").set_index("id").loc[dev.id]
                # Historical float32 CSV scores were written with about eight
                # significant digits; their rounding is below 1e-7.
                if not np.array_equal(saved.prediction, raw.prediction) or not np.allclose(saved.confidence, scores, rtol=0, atol=1e-7):
                    raise ValueError(f"Frozen calibrator fails development reproduction: {key}.{variant}")
                track(relative)
                metrics = analysis["results"][key][variant]
                threshold = {field: metrics["acceptance"][field] for field in THRESHOLD_FIELDS}
                if (raw.prediction == "__refusal__").any() and threshold["threshold"] is not None and threshold["threshold"] <= 0:
                    raise ValueError("A refusal cannot be accepted by the frozen threshold")
                checked, _, _ = evaluate(dev.label, raw.prediction, scores, threshold, labels=labels)
                for field in ("accuracy", "brier", "binary_log_loss", "error_auroc", "aurc"):
                    if not np.isclose(checked[field], metrics[field], atol=1e-7, rtol=0):
                        raise ValueError(f"Development metric mismatch: {key}.{variant}:{field}")
                specifications[key + "." + variant] = {"arm": arm, "stream": key, "variant": variant,
                                                        "threshold": threshold, "development_metrics": metrics}
    if len(specifications) != 60:
        raise ValueError("Expected exactly 60 prespecified variants")
    plan = {"protocol": 1, "created_utc": now(), "evaluation_role": "official_test", "test_n": 3080,
            "primary": "embedding_lr.temperature", "variants": specifications, "input_sha256": inputs,
            "packages": versions, "development_reproduction_atol": 1e-7,
            "rules": ["No fitting or threshold selection on test outcomes",
                "All 60 variants evaluated; no test-dependent configuration changes",
                "Use saved development thresholds, including null thresholds that accept nothing",
                "Refusals always require review", "Paired bootstrap intervals are nominal and conditional on frozen fits"],
            "bootstrap": {"seed": 20261013, "resamples": 5000}}
    plan["sha256"] = digest(plan)
    write_json(PLAN, plan)
    return plan


def preflight():
    plan = read_json(PLAN)
    if digest({k: v for k, v in plan.items() if k != "sha256"}) != plan["sha256"]:
        raise ValueError("Frozen final plan changed")
    for name, wanted in plan["input_sha256"].items():
        if sha(ROOT / name) != wanted:
            raise ValueError(f"Frozen final input changed: {name}")
    for name, wanted in plan["packages"].items():
        if importlib.metadata.version(name) != wanted:
            raise ValueError(f"Frozen final dependency changed: {name}")
    return plan


def save_arm(arm, frame, streams, costs=None, details=None):
    plan = preflight()
    folder = OUT / arm
    folder.mkdir(parents=True, exist_ok=True)
    calibrators = joblib.load(ROOT / "reports/calibrator-comparison/calibrators.joblib")
    labels = read_json(ROOT / "data/banking77/manifest.json")["labels"]
    results, hashes = {}, {}
    for key, (prediction, q, tq) in streams.items():
        for variant, scores in transform(key, q, prediction, calibrators, tq).items():
            name = key + "." + variant
            specification = plan["variants"][name]
            if specification["arm"] != arm:
                raise ValueError("Wrong arm")
            threshold = specification["threshold"]
            refused = np.asarray(prediction) == "__refusal__"
            if threshold["threshold"] is not None and np.any(refused & (scores >= threshold["threshold"])):
                raise ValueError("Frozen threshold would accept a refusal")
            metrics, reliability, curve = evaluate(frame.label, prediction, scores, threshold, labels=labels)
            results[name] = metrics
            output = pd.DataFrame({"id": frame.id.to_numpy(), "text": frame.text.to_numpy(), "truth": frame.label.to_numpy(),
                                   "prediction": prediction, "confidence": scores, "correct": frame.label.to_numpy() == prediction})
            output.to_csv(folder / f"{name}.predictions.csv", index=False)
            pd.DataFrame(reliability).to_csv(folder / f"{name}.reliability.csv", index=False)
            pd.DataFrame(curve).to_csv(folder / f"{name}.risk_coverage.csv", index=False)
    expected = {name for name, spec in plan["variants"].items() if spec["arm"] == arm}
    if set(results) != expected:
        raise ValueError("Incomplete arm")
    write_json(folder / "metrics.json", results)
    write_json(folder / "metadata.json", {"plan_sha256": plan["sha256"], "evaluation_role": "official_test",
        "n": len(frame), "created_utc": now(), "costs": costs or [], "details": details or {}, "thresholds_refitted": False})
    for path in folder.glob("*"):
        if path.is_file() and path.name not in {"complete.json", "status.json", ".lock"}:
            hashes[path.name] = sha(path)
    # Completion is an atomic marker written only after every result artifact.
    write_json(folder / "complete.json", {"plan_sha256": plan["sha256"], "outputs_sha256": hashes, "completed_utc": now()})
    return results


def completed(arm):
    marker = OUT / arm / "complete.json"
    if not marker.exists():
        return False
    saved = read_json(marker)
    if saved["plan_sha256"] != read_json(PLAN)["sha256"]:
        raise ValueError("Completed arm belongs to another final plan")
    for name, wanted in saved["outputs_sha256"].items():
        if sha(marker.parent / name) != wanted:
            raise ValueError(f"Completed result changed: {arm}/{name}")
    return True


def local(arm):
    preflight()
    _, frames = load(ROOT / "data/banking77")
    frame = frames["test"]
    run = ROOT / config_for(arm)["run_dir"]
    frozen = frozen_for(arm)["calibration"]
    labels = read_json(ROOT / "data/banking77/manifest.json")["labels"]
    details = {}
    streams = {}
    if arm == "tfidf":
        vectorizer, classifier = joblib.load(run / "tfidf.joblib")
        tick = time.perf_counter()
        logits = classifier.decision_function(vectorizer.transform(frame.text))
        details["inference_seconds"] = time.perf_counter() - tick
        streams["tfidf"] = (np.asarray(labels)[logits.argmax(axis=1)], softmax(logits, axis=1).max(axis=1),
                             temperature_scores(logits, frozen["tfidf"]))
    else:
        import gc
        import torch
        from sentence_transformers import SentenceTransformer
        torch.set_num_threads(2)
        settings = read_json(run / "embedding_model.json")
        folder = OUT / arm
        path = folder / "test-embeddings.npy"
        if path.exists():
            if sha(path) != read_json(folder / "test-embeddings.json")["sha256"]:
                raise ValueError("Test embedding cache changed")
            embedding = np.load(path)
            details = read_json(folder / "test-embeddings.json")
        else:
            tick = time.perf_counter()
            model = SentenceTransformer(settings["model"], revision=settings["revision"], device="cpu", local_files_only=True)
            load_seconds = time.perf_counter() - tick
            tick = time.perf_counter()
            embedding = model.encode(["task: classification | query: " + text for text in frame.text], prompt="",
                                     batch_size=8, normalize_embeddings=True, show_progress_bar=True)
            details = {"model": settings, "model_load_seconds": load_seconds,
                       "encoding_seconds": time.perf_counter() - tick, "cpu_threads": 2, "batch_size": 8}
            temporary = folder / "test-embeddings.tmp.npy"
            np.save(temporary, embedding)
            temporary.replace(path)
            details["sha256"] = sha(path)
            write_json(folder / "test-embeddings.json", details)
            del model
            gc.collect()
        if embedding.shape != (3080, 768) or not np.isfinite(embedding).all():
            raise ValueError("Unexpected test embeddings")
        for key in ARMS[arm][1]:
            classifier = joblib.load(run / (key + ".joblib"))
            tick = time.perf_counter()
            logits = (classifier.decision_function(embedding) if key == "embedding_lr" else
                      np.log(np.clip(classifier.predict_proba(embedding), 1e-30, 1)))
            details[key + "_head_seconds"] = time.perf_counter() - tick
            streams[key] = (np.asarray(labels)[logits.argmax(axis=1)], softmax(logits, axis=1).max(axis=1),
                            temperature_scores(logits, frozen[key]))
    return save_arm(arm, frame, streams, details=details)
