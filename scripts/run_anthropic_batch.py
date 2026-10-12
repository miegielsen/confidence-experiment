"""Two-wave Anthropic batch transport feeding the common prediction cache."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
import tomllib
from typing import Literal

from anthropic import Anthropic, transform_schema
from pydantic import Field, create_model

from confidence_experiment.data import load
from confidence_experiment.io import digest, read_json, write_json
from confidence_experiment.llm import CLASSIFY, BLIND, VERIFY, BudgetExhausted
from confidence_experiment.runner import identity, run
from confidence_experiment.report import save_report


def specification(settings, model, system, user, labels, confidence_only=False):
    schema = (create_model("Confidence", confidence=(float, Field(ge=0, le=1))) if confidence_only else
              create_model("Classification", label=(Literal[tuple(labels)], ...), confidence=(float, Field(ge=0, le=1))))
    request = {"model": model, "system": system, "messages": [{"role": "user", "content": user}],
               "max_tokens": settings.get("max_output_tokens", 256)}
    if settings.get("thinking"):
        request["thinking"] = {"type": settings["thinking"]}
    raw_schema = schema.model_json_schema()
    key = digest({**request, "schema": raw_schema, "protocol": 1})
    bound = len(json.dumps({**request, "schema": raw_schema}, ensure_ascii=False).encode()) + 4096
    reserve = (bound * settings["input_usd_per_million"] + request["max_tokens"] * settings["output_usd_per_million"]) / 2e6
    return {"key": key, "request": request, "schema": raw_schema, "reserve": reserve,
            "confidence_only": confidence_only, "validator": schema}


class Batcher:
    def __init__(self, settings, labels, output, client=None):
        self.settings, self.labels = settings, labels
        self.cache = Path(settings.get("cache_dir", "data/anthropic_cache"))
        self.jobs_dir = self.cache / "batches"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.ledger_path = Path(settings.get("ledger_path", self.cache / "spending.json"))
        self.ledger = read_json(self.ledger_path) if self.ledger_path.exists() else {"usd": 0.0, "calls": 0, "reservations": {}}
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.client = client or Anthropic(max_retries=0, timeout=120.0)
        self.new_requests = 0

    def cached(self, spec):
        path = self.cache / f"{spec['key']}.json"
        if path.exists():
            value = read_json(path)
            spec["validator"].model_validate(value["parsed"])
            return value
        return None

    def submit(self, specs):
        # The identical hash is used by synchronous calls, regardless of transport.
        specs = list({s["key"]: s for s in specs if self.cached(s) is None}.values())
        if not specs:
            return None
        fingerprint = digest([s["key"] for s in specs])
        path = self.jobs_dir / f"{fingerprint}.json"
        reserve_key = "batch:" + fingerprint
        if path.exists():
            job = read_json(path)
            if job.get("batch_id"):
                return job
            if job["state"] != "prepared":
                raise RuntimeError(f"Ambiguous batch submission {path}. Recover its batch ID from Anthropic before resuming; it will not be submitted twice.")
        else:
            if not self.settings.get("enabled"):
                raise RuntimeError("Batch submissions are disabled.")
            if self.new_requests + len(specs) > self.settings.get("max_new_calls", 0):
                raise BudgetExhausted("Per-invocation request cap reached before batch submission.")
            amount = sum(s["reserve"] for s in specs)
            if self.settings.get("enforce_budget", True) and self.ledger["usd"] + sum(self.ledger["reservations"].values()) + amount > self.settings.get("budget_usd", 0):
                raise BudgetExhausted("Batch reservation would exceed the shared spending limit.")
            job = {"fingerprint": fingerprint, "batch_id": None, "state": "prepared", "reserved_usd": amount,
                   "keys": [s["key"] for s in specs], "kinds": {s["key"]: s["confidence_only"] for s in specs},
                   "created_utc": datetime.now(timezone.utc).isoformat()}
            write_json(path, job)
        if reserve_key not in self.ledger["reservations"]:
            # This branch also safely resumes a prepared, never-submitted job.
            amount = job["reserved_usd"]
            if self.settings.get("enforce_budget", True) and self.ledger["usd"] + sum(self.ledger["reservations"].values()) + amount > self.settings.get("budget_usd", 0):
                raise BudgetExhausted("Shared spending limit reached.")
            self.ledger["reservations"][reserve_key] = amount
            write_json(self.ledger_path, self.ledger)
        requests = [{"custom_id": s["key"], "params": {**s["request"],
                    "output_config": {"format": {"type": "json_schema", "schema": transform_schema(s["validator"])}}}} for s in specs]
        if len(json.dumps({"requests": requests}).encode()) > 250_000_000:
            raise ValueError("Batch payload too large. Reduce batch_size.")
        write_json(self.jobs_dir / f"{fingerprint}.requests.json", requests)
        job["state"] = "submitting"
        write_json(path, job)
        response = self.client.messages.batches.create(requests=requests)
        job.update(batch_id=response.id, state=response.processing_status)
        write_json(path, job)
        self.new_requests += len(specs)
        print(f"Submitted {response.id}: {len(specs)} requests; reserved ${job['reserved_usd']:.3f}", flush=True)
        return job

    def import_results(self, job):
        fingerprint, batch_id = job["fingerprint"], job["batch_id"]
        results_path = self.jobs_dir / f"{fingerprint}.results.json"
        if results_path.exists():
            results = read_json(results_path)
        else:
            results = [r.model_dump(mode="json") for r in self.client.messages.batches.results(batch_id)]
            write_json(results_path, results)
        if len(results) != len(job["keys"]) or {r["custom_id"] for r in results} != set(job["keys"]):
            raise RuntimeError("Batch results do not match submitted custom IDs; reservation retained.")
        total, succeeded = 0.0, 0
        cached, errors = {}, []
        for item in results:
            key, result = item["custom_id"], item["result"]
            if result["type"] != "succeeded":
                errors.append({"key": key, "type": result["type"], "error": result.get("error")})
                continue
            succeeded += 1
            message = result["message"]
            usage = message["usage"]
            usd = (usage["input_tokens"] * self.settings["input_usd_per_million"]
                + usage["output_tokens"] * self.settings["output_usd_per_million"]
                + (usage.get("cache_read_input_tokens") or 0) * self.settings.get("cached_input_usd_per_million", self.settings["input_usd_per_million"])
                + (usage.get("cache_creation_input_tokens") or 0) * self.settings["input_usd_per_million"] * 2) / 2e6
            total += usd
            try:
                if message["stop_reason"] != "end_turn":
                    raise ValueError("Nonterminal or refused response")
                body = "".join(block["text"] for block in message["content"] if block["type"] == "text")
                parsed = json.loads(body)
                confidence_only = job["kinds"][key]
                schema = (create_model("Confidence", confidence=(float, Field(ge=0, le=1))) if confidence_only else
                    create_model("Classification", label=(Literal[tuple(self.labels)], ...), confidence=(float, Field(ge=0, le=1))))
                parsed = schema.model_validate(parsed).model_dump()
                cached[key] = {"parsed": parsed, "model": message["model"], "usage": usage, "usd": usd,
                    "latency_seconds": None, "transport": "batch", "batch_id": batch_id, "request_sha256": key}
            except (ValueError, TypeError, KeyError) as exc:
                errors.append({"key": key, "type": "invalid_response", "error": type(exc).__name__})
        # Account once for the entire durable results file, even if importing cache files is interrupted.
        accounted = self.ledger.setdefault("accounted_batches", {})
        if batch_id not in accounted:
            self.ledger["usd"] += total
            self.ledger["calls"] += succeeded
            self.ledger["reservations"].pop("batch:" + fingerprint, None)
            accounted[batch_id] = {"usd": total, "successful_requests": succeeded}
            write_json(self.ledger_path, self.ledger)
        for key, value in cached.items():
            write_json(self.cache / f"{key}.json", value)
        job.update(state="imported" if not errors else "imported_with_errors", usd=total, errors=errors)
        write_json(self.jobs_dir / f"{fingerprint}.json", job)
        print(f"Imported {batch_id}: {len(cached)} valid results, ${total:.3f}", flush=True)
        if errors:
            raise RuntimeError(f"Batch {batch_id} has {len(errors)} failed results. Successful responses are cached; inspect its job file before retrying.")

    def wave(self, specs, phase):
        by_key = {s["key"]: s for s in specs}
        missing = {k for k, s in by_key.items() if self.cached(s) is None}
        jobs, covered = [], set()
        # Resume existing jobs before regrouping remaining requests. A crash during
        # cache import must never move in-flight requests into a second paid batch.
        for path in self.jobs_dir.glob("*.json"):
            if path.name.count(".") != 1:
                continue
            job = read_json(path)
            if not missing.intersection(job["keys"]):
                continue
            if not job.get("batch_id"):
                if job["state"] != "prepared":
                    raise RuntimeError(f"Ambiguous batch submission {path}; recover its ID before resuming.")
                job = self.submit([by_key[k] for k in job["keys"]])
            jobs.append(job)
            covered.update(job["keys"])
        specs = [by_key[k] for k in by_key if k in missing and k not in covered]
        size = self.settings.get("batch_size", 2000)
        jobs += [self.submit(specs[i:i + size]) for i in range(0, len(specs), size)]
        pending = {j["batch_id"]: j for j in jobs if j is not None}
        while pending:
            states = []
            for batch_id, job in list(pending.items()):
                response = self.client.messages.batches.retrieve(batch_id)
                states.append({"id": batch_id, "state": response.processing_status, "counts": response.request_counts.model_dump()})
                if response.processing_status == "ended":
                    self.import_results(job)
                    del pending[batch_id]
            write_json(self.output / "batch_status.json", {"phase": phase, "batches": states, "pending_batches": len(pending),
                       "spent_usd": self.ledger["usd"], "reserved_usd": sum(self.ledger["reservations"].values()),
                       "updated_utc": datetime.now(timezone.utc).isoformat()})
            if pending:
                time.sleep(self.settings.get("poll_seconds", 30))


def execute(config, final=False):
    manifest, frames = load(config["data_dir"])
    settings, labels = config["llm"], manifest["labels"]
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("Set ANTHROPIC_API_KEY locally before submitting batches.")
    if min(settings["input_usd_per_million"], settings["output_usd_per_million"]) <= 0:
        raise ValueError("Configure positive model prices.")
    if not 1 <= settings.get("batch_size", 2000) <= 100000:
        raise ValueError("Invalid batch_size.")
    output = Path(config["run_dir"])
    cache = Path(settings.get("cache_dir", "data/anthropic_cache"))
    ledger_path = Path(settings.get("ledger_path", cache / "spending.json"))
    held = []
    try:
        for path in (output / ".running.lock", ledger_path.parent / ".running.lock"):
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x") as handle:
                handle.write(str(os.getpid()))
            held.append(path)
        signature = identity(config, manifest)
        identity_path = output / "identity.json"
        if identity_path.exists() and read_json(identity_path)["sha256"] != signature:
            raise ValueError("Batch run settings/data/source changed. Choose a new run directory.")
        write_json(identity_path, {"sha256": signature})
        if final and not (output / "frozen.joblib").exists():
            raise ValueError("Complete development before final test batches.")
        batcher = Batcher(settings, labels, output)
        category_text = "\n".join(labels)
        roles = ["test"] if final else ["correctness", "calibration", "development"]
        first, originals = [], []
        for role in roles:
            for row in frames[role].itertuples(index=False):
                original = specification(settings, settings["model"], CLASSIFY.format(labels=category_text), row.text, labels)
                blind = specification(settings, settings.get("verifier_model") or settings["model"], BLIND.format(labels=category_text), row.text, labels)
                first += [original, blind]
                originals.append((row.text, original))
        batcher.wave(first, "classification_and_blind")
        second = []
        for text, original in originals:
            prediction = batcher.cached(original)
            if prediction is None:
                raise RuntimeError("Missing initial classification after first batch wave.")
            user = json.dumps({"message": text, "proposed_intent": prediction["parsed"]["label"]}, ensure_ascii=False)
            second.append(specification(settings, settings.get("verifier_model") or settings["model"], VERIFY.format(labels=category_text), user, labels, True))
        batcher.wave(second, "verification")
        write_json(output / "batch_status.json", {"phase": "complete", "spent_usd": batcher.ledger["usd"], "pending_batches": 0})
        # Cached-only scoring ensures a missing result never triggers a synchronous fallback.
        cached_config = deepcopy(config)
        cached_config["llm"]["enabled"] = False
        results = run(cached_config, final=final)
        # Cache reads still use the original ledger; report this transport's
        # independent accounting without changing any cached predictions.
        report_dir = output / ("final" if final else "development")
        metadata = read_json(report_dir / "metadata.json")
        metadata.update(api_ledger=batcher.ledger, api_ledger_path=str(batcher.ledger_path),
            api_spending_scope="This Anthropic ledger; historical charges remain in the original shared ledger.")
        save_report(report_dir, results, metadata, "test" if final else "development")
    finally:
        for path in held:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/anthropic-batch.toml")
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args()
    execute(tomllib.loads(Path(args.config).read_text(encoding="utf-8")), args.final)
