"""Durable API transport for the sealed final comparison, with early preflight."""
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.special import softmax

from confidence_experiment.data import load
from confidence_experiment.io import read_json, write_json
from confidence_experiment.llm import CLASSIFY, BLIND, VERIFY
from confidence_experiment.runner import features
from final_protocol import ROOT, OUT, preflight, config_for, frozen_for, save_arm, temperature_scores
from run_anthropic_batch import Batcher, specification
from run_openai_decisions import DecisionCaller
import joblib


def anthropic(arm):
    # This must precede Batcher construction, cache imports, and both paid waves.
    preflight()
    config = config_for(arm)
    settings = config["llm"]
    manifest, frames = load(ROOT / config["data_dir"])
    frame, labels = frames["test"], manifest["labels"]
    folder = OUT / arm
    batcher = Batcher(settings, labels, folder)
    category_text = "\n".join(labels)
    first, originals = [], []
    for row in frame.itertuples(index=False):
        original = specification(settings, settings["model"], CLASSIFY.format(labels=category_text), row.text, labels)
        blind = specification(settings, settings["verifier_model"], BLIND.format(labels=category_text), row.text, labels)
        first += [original, blind]
        originals.append((row, original, blind))
    batcher.wave(first, "official_test_classification_and_blind")
    preflight()
    second = []
    for row, original, blind in originals:
        selected = batcher.cached(original)["parsed"]["label"]
        user = json.dumps({"message": row.text, "proposed_intent": selected}, ensure_ascii=False)
        second.append(specification(settings, settings["verifier_model"], VERIFY.format(labels=category_text), user, labels, True))
    batcher.wave(second, "official_test_verification")
    rows, costs = [], []
    for (row, original, blind), verifier in zip(originals, second):
        cached = [batcher.cached(spec) for spec in (original, blind, verifier)]
        if any(value is None for value in cached):
            raise ValueError("Final batch wave did not yield all results")
        a, b, v = cached
        rows.append({"id": row.id, "label": a["parsed"]["label"], "confidence": a["parsed"]["confidence"],
                     "blind_label": b["parsed"]["label"], "blind_confidence": b["parsed"]["confidence"],
                     "verifier": v["parsed"]["confidence"], "usd": sum(value["usd"] for value in cached)})
        for stage, spec, value in zip(("original", "blind", "verifier"), (original, blind, verifier), cached):
            costs.append({"id": row.id, "stage": stage, "request_sha256": spec["key"], "usd": value["usd"],
                          "usage": value["usage"], "model": value["model"], "transport": value.get("transport", "synchronous")})
    pd.DataFrame(rows).to_csv(folder / "llm_test.csv", index=False)
    vectorizer, classifier = joblib.load(ROOT / config["run_dir"] / "tfidf.joblib")
    logits = classifier.decision_function(vectorizer.transform(frame.text))
    prediction = np.asarray([row["label"] for row in rows])
    learned = frozen_for(arm)["calibration"]["correctness_model"].predict_proba(features(rows, frame, labels, logits))[:, 1]
    streams = {arm + "_same": (prediction, [row["confidence"] for row in rows], None),
               arm + "_verifier": (prediction, [row["verifier"] for row in rows], None),
               arm + "_correctness": (prediction, learned, None)}
    return save_arm(arm, frame, streams, costs, {"api_ledger_path": str(batcher.ledger_path)})


def decisions():
    preflight()
    config = config_for("decisions")
    manifest, frames = load(ROOT / config["data_dir"])
    frame, labels = frames["test"], manifest["labels"]
    folder = OUT / "decisions"
    caller = DecisionCaller(config["decisions"])
    rows, costs = [], []
    for index, row in enumerate(frame.itertuples(index=False), 1):
        result = caller.call(row.text, labels)
        rows.append({"id": row.id, **result["parsed"]})
        costs.append({"id": row.id, "stage": "original", "request_sha256": result["request_sha256"], "usd": result["usd"],
                      "usage": result["usage"], "model": result["model"], "transport": "synchronous"})
        if index % 50 == 0:
            write_json(folder / "request_progress.json", {"completed": index, "total": len(frame)})
            print(f"Decisions test: {index}/{len(frame)}", flush=True)
    saved = pd.DataFrame(rows)
    saved["probabilities"] = saved.probabilities.map(json.dumps)
    saved.to_csv(folder / "decisions_test.csv", index=False)
    prediction = np.asarray([r["label"] for r in rows])
    probabilities = np.array([r["probabilities"] if r.get("probabilities") is not None else [1 / len(labels)] * len(labels) for r in rows])
    selected = np.array([labels.index(p) if p in labels else 0 for p in prediction])
    tq = temperature_scores(np.log(np.clip(probabilities, 1e-12, 1)), frozen_for("decisions")["calibration"]["temperature"], selected)
    streams = {"decisions_confidence": (prediction, [r["confidence"] or 0 for r in rows], None),
               "decisions_probability": (prediction, [r["selected_probability"] or 0 for r in rows], tq)}
    return save_arm("decisions", frame, streams, costs)
