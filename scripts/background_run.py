"""Run a long experiment with durable logs and a readable status file."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys
import time
import tomllib

from confidence_experiment.io import read_json, write_json


def now():
    return datetime.now(timezone.utc).isoformat()


def available_memory_gib():
    if os.name != "nt":
        import psutil
        return psutil.virtual_memory().available / 2**30
    import ctypes
    class MemoryStatus(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
            *[(name, ctypes.c_ulonglong) for name in ("total_physical", "available_physical", "total_page",
                "available_page", "total_virtual", "available_virtual", "available_extended")]]
    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError("Could not measure available physical memory")
    return status.available_physical / 2**30


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--download-embedding", action="store_true")
    parser.add_argument("--download-encoder", action="store_true")
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--then-config", help="Start another background configuration only after this one succeeds")
    args = parser.parse_args()
    if args.cpu_threads < 1:
        parser.error("cpu-threads must be positive")
    if os.name == "nt":
        import ctypes
        # Let foreground apps take precedence over the experiment worker.
        ctypes.windll.kernel32.SetPriorityClass(ctypes.c_void_p(-1), 0x4000)
    config = tomllib.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.then_config:
        next_config = tomllib.loads(Path(args.then_config).read_text(encoding="utf-8"))
        if Path(next_config["run_dir"]).resolve() == Path(config["run_dir"]).resolve():
            parser.error("then-config must use a different run directory")
    output = Path(config["run_dir"])
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / "background_status.json"
    if status_path.exists() and read_json(status_path).get("state") == "complete" and (output / "frozen.joblib").exists():
        print(f"Already completed: {output}", flush=True)
        return 0
    supervisor_lock = output / ".supervisor.lock"
    try:
        with supervisor_lock.open("x") as handle:
            handle.write(str(os.getpid()))
    except FileExistsError:
        print(f"A background supervisor already owns {output}; no duplicate was started.", flush=True)
        return 0
    try:
        return supervise(args, config)
    finally:
        supervisor_lock.unlink(missing_ok=True)


def supervise(args, config):
    output = Path(config["run_dir"])
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / "background_status.json"
    status = {"state": "starting", "started_utc": now(), "config": args.config, "pid": os.getpid(),
              "log": str(output / "background.log"), "budget_usd": config.get("decisions", config.get("llm", {})).get("budget_usd"),
              "cpu_threads": args.cpu_threads, "priority": "below_normal" if os.name == "nt" else "default"}
    status["budget_enforced"] = config.get("decisions", config.get("llm", {})).get("enforce_budget", True)
    if args.then_config:
        status["next_config"] = args.then_config
        status["next_state"] = "queued"
    write_json(status_path, status)
    with (output / "background.log").open("a", encoding="utf-8", buffering=1) as log:
        log.write(f"\nStarted {now()}\n")
        commands = []
        if args.download_embedding:
            metadata = read_json(output / "embedding_model.json")
            commands.append(("downloading", [sys.executable, "scripts/download_model_weights.py", metadata["model"], "--revision", metadata["revision"]]))
        if getattr(args, "download_encoder", False):
            encoder = config["encoder"]
            commands.append(("downloading", [sys.executable, "scripts/download_model_weights.py", encoder["model"],
                "--revision", encoder["revision"], "--workers", "2"]))
        if "openai_decisions" in config.get("methods", []):
            commands.append(("running", [sys.executable, "scripts/run_openai_decisions.py", "run", "--config", args.config]))
        elif config.get("llm", {}).get("transport") == "batch":
            commands.append(("running", [sys.executable, "scripts/run_anthropic_batch.py", "--config", args.config]))
        elif config.get("methods") == ["modernbert"]:
            commands.append(("running", [sys.executable, "scripts/run_modernbert_cpu.py", "--config", args.config]))
        else:
            commands.append(("running", [sys.executable, "-m", "confidence_experiment", "run", "--config", args.config]))
        for state, command in commands:
            minimum_memory = config.get("runtime", {}).get("min_available_memory_gib", 0) if state == "running" else 0
            if minimum_memory:
                reported_wait = False
                while True:
                    available = available_memory_gib()
                    status.update(available_memory_gib=round(available, 3), required_available_memory_gib=minimum_memory, updated_utc=now())
                    if available >= minimum_memory:
                        break
                    status["state"] = "waiting_memory"
                    write_json(status_path, status)
                    if not reported_wait:
                        log.write(f"Waiting for {minimum_memory:.1f} GiB available RAM; currently {available:.2f} GiB. Model training has not started.\n")
                        reported_wait = True
                    time.sleep(15)
            environment = dict(os.environ, PYTHONUNBUFFERED="1", OMP_NUM_THREADS=str(args.cpu_threads),
                               MKL_NUM_THREADS=str(args.cpu_threads), OPENBLAS_NUM_THREADS=str(args.cpu_threads))
            child = subprocess.Popen(command, stdout=log, stderr=log, env=environment)
            status.update(state=state, child_pid=child.pid, updated_utc=now())
            write_json(status_path, status)
            while child.poll() is None:
                time.sleep(15)
                status["updated_utc"] = now()
                write_json(status_path, status)
            code = child.returncode
            if code:
                status.update(state="failed", exit_code=code, completed_utc=now())
                write_json(status_path, status)
                return code
    status.update(state="complete", exit_code=0, completed_utc=now(), report=str(output / "development" / "report.html"))
    write_json(status_path, status)
    if args.then_config:
        try:
            following = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--config", args.then_config,
                "--cpu-threads", str(args.cpu_threads)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            status.update(next_state="started", next_pid=following.pid)
        except OSError as error:
            status.update(next_state="launch_failed", next_error=str(error))
        write_json(status_path, status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
