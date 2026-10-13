"""Freeze and run all 60 final variants without changing historical run identities."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import subprocess
import sys
import traceback

from confidence_experiment.io import read_json, write_json
from final_protocol import ROOT, OUT, ARMS, freeze, preflight, completed, local, now


def run_arm(arm):
    preflight()
    if completed(arm):
        print(f"Verified completed final arm: {arm}", flush=True)
        return
    folder = OUT / arm
    folder.mkdir(parents=True, exist_ok=True)
    lock = folder / ".lock"
    with lock.open("x") as handle:
        handle.write(str(os.getpid()))
    try:
        write_json(folder / "status.json", {"state": "running", "started_utc": now(), "pid": os.getpid()})
        if arm in {"embeddings", "tfidf"}:
            local(arm)
        else:
            from final_transport import anthropic, decisions
            decisions() if arm == "decisions" else anthropic(arm)
        write_json(folder / "status.json", {"state": "complete", "completed_utc": now()})
        print(f"Completed final arm: {arm}", flush=True)
    except Exception:
        write_json(folder / "status.json", {"state": "failed", "error": traceback.format_exc(), "updated_utc": now()})
        raise
    finally:
        lock.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "run", "arm"))
    parser.add_argument("--arm", choices=tuple(ARMS))
    args = parser.parse_args()
    if args.command == "freeze":
        plan = freeze()
        print(f"Frozen {len(plan['variants'])} variants; plan {plan['sha256']}")
        return
    preflight()
    if args.command == "arm":
        if not args.arm:
            parser.error("--arm required")
        run_arm(args.arm)
        return
    lock = OUT / ".supervisor.lock"
    with lock.open("x") as handle:
        handle.write(str(os.getpid()))
    try:
        write_json(OUT / "status.json", {"state": "running", "started_utc": now(), "pid": os.getpid()})
        # Remote jobs use one shared Python process. The encoder uses a separate
        # process so its memory is returned to Windows as soon as inference ends.
        scope_path = OUT / "execution-scope.json"
        scope = read_json(scope_path)["active_arms"] if scope_path.exists() else list(ARMS)
        with ThreadPoolExecutor(max_workers=4) as pool:
            hosted = {pool.submit(run_arm, arm): arm for arm in scope if arm not in {"embeddings", "tfidf"}}
            if "tfidf" in scope:
                run_arm("tfidf")
            if "embeddings" in scope:
                subprocess.run([sys.executable, str(Path(__file__).resolve()), "arm", "--arm", "embeddings"], cwd=ROOT, check=True)
            for future in as_completed(hosted):
                future.result()
        if not all(completed(arm) for arm in scope):
            raise RuntimeError("Final comparison is incomplete")
        write_json(OUT / "status.json", {"state": "evaluation_complete", "completed_utc": now(), "report_pending": True})
        subprocess.run([sys.executable, "scripts/build_final_report.py"], cwd=ROOT, check=True)
        write_json(OUT / "status.json", {"state": "active_scope_complete", "completed_utc": now(), "report_pending": False,
                   "deferred_arms": [arm for arm in ARMS if arm not in scope]})
    except Exception:
        write_json(OUT / "status.json", {"state": "failed", "error": traceback.format_exc(), "updated_utc": now()})
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
