"""Finish reports after all requested test arms complete; never submit requests."""
import subprocess
import os
import sys
import time
import traceback

from confidence_experiment.io import read_json, write_json
from final_protocol import ROOT, OUT, ARMS, completed, now


def finish():
    try:
        while True:
            scope = read_json(OUT / "execution-scope.json")["active_arms"]
            ready = all(completed(arm) for arm in scope)
            if ready and not (OUT / ".supervisor.lock").exists():
                break
            write_json(OUT / "report_status.json", {"state": "waiting", "active_arms": scope, "updated_utc": now()})
            time.sleep(20)
        subprocess.run([sys.executable, "scripts/build_final_report.py"], cwd=ROOT, check=True)
        write_json(OUT / "report_status.json", {"state": "complete", "completed_utc": now()})
        write_json(OUT / "status.json", {"state": "active_scope_complete", "completed_utc": now(),
                   "active_arms": scope, "deferred_arms": [arm for arm in ARMS if arm not in scope], "page_sync_pending": True})
    except Exception:
        write_json(OUT / "report_status.json", {"state": "failed", "error": traceback.format_exc(), "updated_utc": now()})
        raise


def main():
    lock = OUT / ".report.lock"
    with lock.open("x") as handle:
        handle.write(str(os.getpid()))
    try:
        finish()
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
