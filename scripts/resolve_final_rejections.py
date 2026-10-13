"""Record definite Anthropic credit rejections and release their reservations.

Only HTTP 400 insufficient-credit failures from this final run qualify. Ambiguous
network errors and accepted batches are never made retryable by this command.
"""
import json
import os
import re
from pathlib import Path

from confidence_experiment.io import read_json, write_json
from final_protocol import OUT, ROOT, PLAN, config_for, now


def main():
    plan = read_json(PLAN)
    for arm in ("haiku", "sonnet", "opus"):
        folder = OUT / arm
        status = read_json(folder / "status.json")
        error = status.get("error", "")
        if not (status.get("state") == "failed" and "Error code: 400" in error and
                "Your credit balance is too low" in error):
            raise ValueError(f"No definite credit rejection for {arm}")
        lock = folder / ".lock"
        with lock.open("x") as handle:
            handle.write(str(os.getpid()))
        try:
            settings = config_for(arm)["llm"]
            cache = ROOT / settings["cache_dir"]
            ledger_path = ROOT / settings.get("ledger_path", cache / "spending.json")
            ledger = read_json(ledger_path)
            resolved = []
            for path in (cache / "batches").glob("*.json"):
                if path.name.count(".") != 1:
                    continue
                job = read_json(path)
                if (job.get("batch_id") or job.get("state") != "submitting" or
                        job.get("created_utc", "") < plan["created_utc"]):
                    continue
                requests_path = path.with_name(path.stem + ".requests.json")
                requests = read_json(requests_path)
                if {r["custom_id"] for r in requests} != set(job["keys"]):
                    raise ValueError("Rejected request payload does not match job")
                if any((cache / (key + ".json")).exists() for key in job["keys"]):
                    raise ValueError("Rejected job has results; inspect manually")
                reservation = ledger["reservations"].pop("batch:" + job["fingerprint"], None)
                record = {"arm": arm, "fingerprint": job["fingerprint"], "provider_http_status": 400,
                          "reason": "insufficient_credits", "request_ids": re.findall(r"req_[A-Za-z0-9]+", error),
                          "provider_error": error, "released_reservation_usd": reservation,
                          "batch_id": None, "charged_usd": 0, "recorded_utc": now()}
                write_json(folder / "rejections" / (job["fingerprint"] + ".json"), record)
                job.update(state="prepared", definite_rejection=record)
                write_json(path, job)
                resolved.append(job["fingerprint"])
            if len(resolved) != 1:
                raise ValueError(f"Expected one rejected final job for {arm}, found {len(resolved)}")
            write_json(ledger_path, ledger)
            write_json(folder / "status.json", {"state": "deferred", "reason": "user_requested_local_and_openai_only",
                       "definite_credit_rejection_recorded": True, "updated_utc": now()})
            print(f"Recorded rejected {arm} submission; no charge; reservation released.")
        finally:
            lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
