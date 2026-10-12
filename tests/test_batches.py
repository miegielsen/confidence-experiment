import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from confidence_experiment.io import read_json, write_json
from confidence_experiment.llm import Caller, BudgetExhausted

path = Path(__file__).parents[1] / "scripts" / "run_anthropic_batch.py"
module_spec = importlib.util.spec_from_file_location("batch_transport", path)
batch = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(batch)


class Result:
    def __init__(self, key, failed=False):
        self.key, self.failed = key, failed

    def model_dump(self, **kwargs):
        result = {"type": "errored", "error": {"type": "invalid_request_error"}} if self.failed else {
            "type": "succeeded", "message": {"model": "test-model", "stop_reason": "end_turn",
            "content": [{"type": "text", "text": '{"label":"a","confidence":0.8}'}],
            "usage": {"input_tokens": 100, "output_tokens": 20}}}
        return {"custom_id": self.key, "result": result}


class Endpoint:
    def __init__(self):
        self.created = []
        self.failed = False

    def create(self, requests):
        self.created.append(requests)
        return SimpleNamespace(id=f"batch-{len(self.created)}", processing_status="in_progress")

    def retrieve(self, batch_id):
        return SimpleNamespace(processing_status="ended", request_counts=SimpleNamespace(model_dump=lambda: {"succeeded": 2}))

    def results(self, batch_id):
        group = self.created[int(batch_id.split("-")[1]) - 1]
        return [Result(row["custom_id"], failed=self.failed) for row in reversed(group)]


def setup(tmp_path, **overrides):
    settings = {"enabled": True, "cache_dir": str(tmp_path / "cache"), "budget_usd": 25,
        "max_new_calls": 100, "max_output_tokens": 256, "input_usd_per_million": 1,
        "output_usd_per_million": 5, "batch_size": 2, "poll_seconds": 0, **overrides}
    endpoint = Endpoint()
    client = SimpleNamespace(messages=SimpleNamespace(batches=endpoint))
    instance = batch.Batcher(settings, ["a", "b"], tmp_path / "out", client)
    specs = [batch.specification(settings, "test-model", "system", f"message {i}", ["a", "b"]) for i in range(2)]
    return instance, specs, endpoint


def test_out_of_order_results_match_keys_and_charge_discount_once(tmp_path):
    instance, specs, endpoint = setup(tmp_path)
    instance.wave(specs, "first")
    assert instance.ledger["usd"] == pytest.approx(0.0002)
    assert instance.ledger["calls"] == 2
    assert not instance.ledger["reservations"]
    for spec in specs:
        assert instance.cached(spec)["request_sha256"] == spec["key"]
        assert instance.cached(spec)["latency_seconds"] is None
    instance.wave(specs, "first")
    assert len(endpoint.created) == 1
    assert instance.ledger["usd"] == pytest.approx(0.0002)


def test_budget_blocks_before_remote_submission(tmp_path):
    instance, specs, endpoint = setup(tmp_path, budget_usd=0.000001)
    with pytest.raises(BudgetExhausted):
        instance.submit(specs)
    assert not endpoint.created


def test_independent_ledger_and_disabled_budget_leave_shared_spending_untouched(tmp_path):
    shared = tmp_path / "cache/spending.json"
    original = {"usd": 1.25, "calls": 50, "reservations": {"openai-inflight": 0.001}}
    write_json(shared, original)
    independent = tmp_path / "anthropic_spending/spending.json"
    instance, specs, endpoint = setup(tmp_path, ledger_path=str(independent),
        enforce_budget=False, budget_usd=0)
    instance.wave(specs, "first")
    assert len(endpoint.created) == 1
    assert read_json(shared) == original
    assert read_json(independent)["usd"] == pytest.approx(0.0002)
    assert not read_json(independent)["reservations"]


def test_resume_does_not_regroup_partially_imported_batch(tmp_path):
    instance, specs, endpoint = setup(tmp_path)
    job = instance.submit(specs)
    write_json(instance.cache / f"{specs[0]['key']}.json", {"parsed": {"label": "a", "confidence": 0.8}})
    resumed = batch.Batcher(instance.settings, ["a", "b"], tmp_path / "out", instance.client)
    resumed.wave(specs, "first")
    assert len(endpoint.created) == 1
    assert resumed.ledger["calls"] == 2


def test_sync_cache_keys_are_identical(tmp_path, monkeypatch):
    instance, specs, endpoint = setup(tmp_path)
    instance.wave(specs, "first")
    caller = Caller({**instance.settings, "enabled": False}, instance.cache)
    cached = caller.call("test-model", "system", "message 0", ["a", "b"])
    assert cached["request_sha256"] == specs[0]["key"]


def test_ambiguous_submission_is_not_repeated(tmp_path):
    instance, specs, endpoint = setup(tmp_path)
    job = instance.submit(specs)
    job.update(batch_id=None, state="submitting")
    write_json(instance.jobs_dir / f"{job['fingerprint']}.json", job)
    with pytest.raises(RuntimeError, match="Ambiguous"):
        instance.wave(specs, "first")
    assert len(endpoint.created) == 1


def test_failed_batch_requests_are_not_billed(tmp_path):
    instance, specs, endpoint = setup(tmp_path)
    endpoint.failed = True
    with pytest.raises(RuntimeError, match="failed results"):
        instance.wave(specs, "first")
    assert instance.ledger["usd"] == 0
    assert not instance.ledger["reservations"]


def test_thinking_setting_is_sent_and_has_its_own_cache_key(tmp_path):
    instance, specs, endpoint = setup(tmp_path, thinking="disabled")
    assert specs[0]["request"]["thinking"] == {"type": "disabled"}
    prior = batch.specification({k: v for k, v in instance.settings.items() if k != "thinking"},
        "test-model", "system", "message 0", ["a", "b"])
    assert prior["key"] != specs[0]["key"]
    instance.wave(specs, "first")
    caller = Caller({**instance.settings, "enabled": False}, instance.cache)
    assert caller.call("test-model", "system", "message 0", ["a", "b"])["request_sha256"] == specs[0]["key"]
