from types import SimpleNamespace
import json

import pytest

from confidence_experiment.llm import BudgetExhausted, Caller


class Usage:
    def model_dump(self):
        return {"input_tokens": 100, "output_tokens": 20, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}


class Response:
    model = "test-model"
    usage = Usage()
    stop_reason = "end_turn"
    content = [SimpleNamespace(type="text", text='{"label":"a","confidence":0.8}')]

    def model_dump(self, **kwargs):
        return {"model": self.model, "usage": self.usage.model_dump()}


def settings(**overrides):
    return {"enabled": True, "input_usd_per_million": 1, "output_usd_per_million": 5,
            "max_new_calls": 10, "budget_usd": 25, **overrides}


def test_cache_and_spend_survive_restart(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-for-unit-test")
    calls = []
    def create(**kwargs):
        calls.append(kwargs)
        return Response()
    caller = Caller(settings(), tmp_path)
    caller.client = SimpleNamespace(messages=SimpleNamespace(create=create))
    first = caller.call("test-model", "system", "message", ["a", "b"])
    assert len(calls) == 1
    assert caller.ledger["usd"] == pytest.approx(0.0002)
    restart = Caller(settings(enabled=False), tmp_path)
    assert restart.call("test-model", "system", "message", ["a", "b"]) == first
    assert restart.ledger["usd"] == caller.ledger["usd"]


def test_no_call_when_budget_too_small(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-for-unit-test")
    caller = Caller(settings(budget_usd=0.00001), tmp_path)
    with pytest.raises(BudgetExhausted):
        caller.call("test-model", "system", "message", ["a", "b"])
    assert caller.ledger["calls"] == 0


def test_failed_call_retains_budget_reservation(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-for-unit-test")
    def fail(**kwargs):
        raise RuntimeError("network failure")
    caller = Caller(settings(), tmp_path)
    caller.client = SimpleNamespace(messages=SimpleNamespace(create=fail))
    with pytest.raises(RuntimeError, match="network failure"):
        caller.call("test-model", "system", "message", ["a", "b"])
    restart = Caller(settings(), tmp_path)
    assert len(restart.ledger["reservations"]) == 1
    with pytest.raises(RuntimeError, match="Unresolved API reservation"):
        restart.call("test-model", "system", "message", ["a", "b"])
