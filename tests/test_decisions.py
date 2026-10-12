import importlib.util
from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import pytest

from confidence_experiment.io import read_json, write_json

spec = importlib.util.spec_from_file_location("decisions_worker", Path(__file__).parents[1] / "scripts/run_openai_decisions.py")
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


def response(kind="choice"):
    return {"model": "test-model", "answers": [{"name": "intent", "type": kind,
        "choice": "a", "confidence": 0.72,
        "probabilities": [{"value": "b", "probability": 0.2}, {"value": "a", "probability": 0.8}]}],
        "usage": {"input_tokens": 100, "input_tokens_details": {"cached_tokens": 10, "cache_write_tokens": 5}, "output_tokens": 99}}


def settings(tmp_path, **overrides):
    return {"cache_dir": str(tmp_path / "cache"), "ledger_path": str(tmp_path / "shared/spending.json"),
        "model": "test-model", "enabled": True, "max_new_calls": 10,
        "budget_usd": 25, "input_usd_per_million": 0.1, **overrides}


def fake_http(monkeypatch, body):
    import io
    requests = []
    def urlopen(request, **kwargs):
        requests.append(json.loads(request.data))
        return io.BytesIO(json.dumps(body).encode())
    monkeypatch.setattr(worker.urllib.request, "urlopen", urlopen)
    monkeypatch.setenv("OPENAI_API_KEY", "fake-unit-test-key")
    return requests


def test_choice_keeps_separate_confidence_and_probability_and_label_order():
    parsed = worker.parse_choice(response(), ["a", "b"])
    assert parsed == {"label": "a", "confidence": 0.72, "selected_probability": 0.8, "probabilities": [0.8, 0.2]}
    with pytest.raises(ValueError, match="exactly once"):
        worker.parse_choice(response(), ["a", "b", "c"])


def test_shared_budget_cache_and_accounting_survive_interrupted_cache_write(tmp_path, monkeypatch):
    config = settings(tmp_path)
    write_json(config["ledger_path"], {"usd": 24, "calls": 100, "reservations": {}, "accounted_batches": ["old-anthropic-batch"]})
    requests = fake_http(monkeypatch, response())
    caller = worker.DecisionCaller(config)
    result = caller.call("message", ["a", "b"])
    assert caller.ledger["usd"] == pytest.approx(24 + 85 * 0.1 / 1e6)
    assert caller.ledger["accounted_batches"] == ["old-anthropic-batch"]
    assert requests[0]["input"] == "message"
    # Raw response and accounted ledger survive, parsed-cache write is lost.
    (Path(config["cache_dir"]) / f"{result['request_sha256']}.json").unlink()
    restart = worker.DecisionCaller({**config, "enabled": False})
    assert restart.call("message", ["a", "b"]) == result
    assert restart.ledger["calls"] == 101
    assert len(requests) == 1


def test_shared_reservations_prevent_overspend(tmp_path, monkeypatch):
    config = settings(tmp_path)
    write_json(config["ledger_path"], {"usd": 24.9, "calls": 1, "reservations": {"anthropic-pending": 0.1}})
    requests = fake_http(monkeypatch, response())
    with pytest.raises(RuntimeError, match="spending limit"):
        worker.DecisionCaller(config).call("message", ["a", "b"])
    assert requests == []


def test_abstention_reuses_billed_refusal_and_budget_can_be_disabled(tmp_path, monkeypatch):
    requests = fake_http(monkeypatch, response("refusal"))
    config = settings(tmp_path, enforce_budget=False, budget_usd=0)
    with pytest.raises(RuntimeError, match="refused"):
        worker.DecisionCaller(config).call("message", ["a", "b"])
    caller = worker.DecisionCaller({**config, "refusal_policy": "abstain"})
    result = caller.call("message", ["a", "b"])
    assert result["parsed"]["refused"] is True
    assert result["parsed"]["label"] == "__refusal__"
    assert result["parsed"]["confidence"] is None
    assert len(requests) == 1
    assert caller.ledger["calls"] == 1


def test_ambiguous_failure_is_not_retried_and_refusal_is_billed(tmp_path, monkeypatch):
    requests = fake_http(monkeypatch, response("refusal"))
    config = settings(tmp_path)
    caller = worker.DecisionCaller(config)
    with pytest.raises(RuntimeError, match="refused"):
        caller.call("message", ["a", "b"])
    assert caller.ledger["calls"] == 1
    with pytest.raises(RuntimeError, match="refused"):
        worker.DecisionCaller(config).call("message", ["a", "b"])
    assert len(requests) == 1
    def fail(*args, **kwargs):
        raise worker.urllib.error.URLError("lost connection")
    monkeypatch.setattr(worker.urllib.request, "urlopen", fail)
    with pytest.raises(RuntimeError, match="outcome is unknown"):
        caller.call("different message", ["a", "b"])
    with pytest.raises(RuntimeError, match="Unresolved"):
        worker.DecisionCaller(config).call("different message", ["a", "b"])


def test_frozen_final_reuses_calibrators_and_does_not_refit(tmp_path, monkeypatch):
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from test_runner import dataset
    config = {"data_dir": str(dataset(tmp_path)), "run_dir": str(tmp_path / "run"),
        "seed": 42, "max_error": 0.01, "decisions": settings(tmp_path)}
    seen = []
    def generate(caller, frames, labels, roles, output):
        seen.append(roles)
        # Include correct/incorrect cases and a non-argmax selected category.
        return {role: [{"id": row.id, "label": "__refusal__" if i == 0 else "b" if i % 4 == 0 else row.label,
            "confidence": 0.3 if i % 4 == 0 else 0.85, "selected_probability": 1 / 3,
            "probabilities": None if i == 0 else [1 / 3] * 3, "refused": i == 0,
            "usd": 0.00001, "latency_seconds": 0.1, "model": "fake"}
            for i, row in enumerate(frames[role].itertuples(index=False))] for role in roles}
    monkeypatch.setattr(worker, "generate", generate)
    development = worker.run(config)
    assert seen == [["calibration", "development"]]
    monkeypatch.setattr(worker, "fit_temperature", lambda *args: pytest.fail("Final evaluation must not refit temperature"))
    monkeypatch.setattr(worker.Sigmoid, "fit", lambda *args: pytest.fail("Final evaluation must not refit sigmoid"))
    final = worker.run(config, final=True)
    assert seen[-1] == ["test"]
    for method in final:
        assert final[method]["acceptance"]["threshold"] == development[method]["acceptance"]["threshold"]
    assert len({v["accuracy"] for v in final.values()}) == 1
    assert all(v["n"] == 24 for v in final.values())
    assert all(v["acceptance"]["threshold"] is None or v["acceptance"]["threshold"] > 0 for v in final.values())
    assert read_json(tmp_path / "run/final/metadata.json")["refusals_by_role"] == {"test": 1}
    with pytest.raises(ValueError, match="already exists"):
        worker.run(config, final=True)


def test_report_tolerates_rounded_wilson_endpoint(tmp_path):
    from confidence_experiment.metrics import evaluate, select_threshold
    from confidence_experiment.report import save_report
    truth, prediction, q = np.array(["a", "b"]), np.array(["a", "a"]), np.array([1.0, 0.2])
    metrics, reliability, curve = evaluate(truth, prediction, q, select_threshold(truth == prediction, q))
    # Represent the CSV rounding edge case that stopped the embedding report.
    reliability[-1]["accuracy_ci95"][1] = 1 - 1e-15
    import pandas as pd
    pd.DataFrame(reliability).to_csv(tmp_path / "method.reliability.csv", index=False)
    pd.DataFrame(curve).to_csv(tmp_path / "method.risk_coverage.csv", index=False)
    save_report(tmp_path, {"method": metrics}, {"config": {"max_error": 0.01}}, "development")
    assert (tmp_path / "report.html").exists()
