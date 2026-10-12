import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tomllib

from .data import prepare, load


def main():
    parser = argparse.ArgumentParser(description="BANKING77 classification confidence experiment")
    parser.add_argument("command", choices=("prepare", "plan", "run"))
    parser.add_argument("--config", default="configs/pilot.toml")
    parser.add_argument("--final", action="store_true", help="Evaluate previously frozen models on the official test set")
    args = parser.parse_args()
    config = tomllib.loads(Path(args.config).read_text(encoding="utf-8"))
    allowed = {"tfidf", "embedding_lr", "embedding_mlp", "modernbert", "llm"}
    if not config.get("methods") or set(config["methods"]) - allowed:
        parser.error("methods must contain supported method names")
    if config.get("threshold_rule", "wilson") not in {"wilson", "empirical"}:
        parser.error("threshold_rule must be wilson or empirical")
    if not 0 < config["max_error"] < 1:
        parser.error("max_error must be between zero and one")
    if args.command == "prepare":
        print(json.dumps(prepare(config["data_dir"], config["seed"]), indent=2))
        return
    manifest, frames = load(config["data_dir"])
    if args.command == "plan":
        roles = ["test"] if args.final else ["correctness", "calibration", "development"]
        print(json.dumps({"methods": config["methods"], "sizes": manifest["sizes"],
            "evaluation": "official test" if args.final else "development only", "anthropic_key_present": bool(os.environ.get("ANTHROPIC_API_KEY")),
            "anthropic_calls_without_cache": 3 * sum(len(frames[r]) for r in roles) if "llm" in config["methods"] else 0,
            "budget_usd": config.get("llm", {}).get("budget_usd", 0), "run_dir": config["run_dir"]}, indent=2))
        return
    if config.get("llm", {}).get("transport") == "batch":
        script = Path(__file__).resolve().parents[2] / "scripts" / "run_anthropic_batch.py"
        command = [sys.executable, str(script), "--config", str(Path(args.config).resolve())]
        if args.final:
            command.append("--final")
        parser.exit(subprocess.run(command).returncode)
    # One process per run; one API process across all runs sharing a ledger.
    paths = [Path(config["run_dir"]) / ".running.lock"]
    if "llm" in config["methods"]:
        paths.append(Path(config["llm"].get("cache_dir", "data/anthropic_cache")) / ".running.lock")
    held = []
    try:
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x") as handle:
                handle.write(str(os.getpid()))
            held.append(path)
        from .runner import run
        run(config, final=args.final)
    except FileExistsError:
        parser.exit(1, "A run lock already exists. Stop the other runner, or remove a stale .running.lock after checking its PID.\n")
    except (ValueError, RuntimeError) as error:
        parser.exit(1, f"{error}\n")
    finally:
        for path in held:
            path.unlink(missing_ok=True)
