"""Check 10 development examples with the real three-call protocol and shared budget."""
import time
import tomllib
from pathlib import Path

from confidence_experiment.data import load
from confidence_experiment.io import write_json
from confidence_experiment.llm import generate

config = tomllib.loads(Path("configs/anthropic.toml").read_text(encoding="utf-8"))
manifest, frames = load(config["data_dir"])
frames["development"] = frames["development"].head(10)
tick = time.perf_counter()
rows, ledger = generate(config, frames, manifest["labels"], ["development"])
summary = {"status": "connectivity and schema smoke test only; not a benchmark result", "model": config["llm"]["model"],
           "examples": 10, "seconds": time.perf_counter() - tick, "ledger": ledger, "rows": rows["development"]}
write_json("runs/anthropic/smoke.json", summary)
print(f"Smoke test passed: 10 examples, ${ledger['usd']:.4f} spent in total. Cached calls will be reused.", flush=True)
