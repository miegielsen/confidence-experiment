"""A sample must preserve selection independence and the shared spending cap."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_claude_sample as sample
from confidence_experiment.io import read_json, write_json
from confidence_experiment.llm import BudgetExhausted
from run_anthropic_batch import Batcher, specification


def test_sample_is_balanced_reproducible_and_independent_of_model_outcomes():
    frame = pd.DataFrame({"id": range(30), "label": ["a"] * 10 + ["b"] * 10 + ["c"] * 10,
                          "text": ["message"] * 30, "correct": [False] * 30})
    first = sample.sample_rows(frame, per_intent=4, seed=10)
    changed = sample.sample_rows(frame.assign(correct=True), per_intent=4, seed=10)
    assert first.id.tolist() == changed.id.tolist()
    assert first.id.is_unique
    assert first.groupby("label").size().tolist() == [4, 4, 4]
    assert first.iloc[:3].label.nunique() == 3
    assert first.id.tolist() != sample.sample_rows(frame, per_intent=4, seed=11).id.tolist()


def test_rejected_preflight_prevents_both_paid_sample_clients(monkeypatch, tmp_path):
    def reject():
        raise ValueError("changed sealed input")
    monkeypatch.setattr(sample, "validate", reject)
    monkeypatch.setattr(sample, "BudgetBatcher", lambda *a: pytest.fail("Client constructed before validation"))
    with pytest.raises(ValueError, match="sealed input"):
        sample.run_block(pd.DataFrame(), [], tmp_path)


def test_concurrent_models_share_one_reservation_cap(tmp_path, monkeypatch):
    ledger = tmp_path / "spending.json"
    write_json(ledger, {"usd": 0.0, "calls": 0, "reservations": {}})
    monkeypatch.setattr(sample, "LEDGER", ledger)
    calls = []
    class FakeBatches:
        def create(self, requests):
            calls.append(requests)
            return SimpleNamespace(id="batch" + str(len(calls)), processing_status="in_progress")
    client = SimpleNamespace(messages=SimpleNamespace(batches=FakeBatches()))
    workers, specs = [], []
    for family in ("sonnet", "opus"):
        settings = {"cache_dir": str(tmp_path / family), "ledger_path": str(ledger), "enabled": True,
                    "enforce_budget": True, "budget_usd": 1.0, "max_new_calls": 10,
                    "input_usd_per_million": 2., "output_usd_per_million": 10.}
        worker = sample.BudgetBatcher.__new__(sample.BudgetBatcher)
        Batcher.__init__(worker, settings, ["a", "b"], tmp_path / (family + "out"), client=client)
        spec = specification(settings, family, "classify", "message", ["a", "b"])
        spec["reserve"] = .7
        workers.append(worker); specs.append(spec)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(worker.submit, [spec]) for worker, spec in zip(workers, specs)]
        results = []
        for future in futures:
            try:
                results.append(future.result())
            except BudgetExhausted:
                results.append("capped")
    assert results.count("capped") == 1
    assert len(calls) == 1
    assert sum(read_json(ledger)["reservations"].values()) == pytest.approx(.7)


def test_new_block_requires_budget_for_both_models_and_all_three_stages():
    assert sample.affordable_end([.4, .4, .4], 0, .9) == 2
    assert sample.affordable_end([.4, .4, .4], 2, .3) == 2
