"""Check model access and the exact three-call protocol before batch submission.

Successful responses use the production cache keys and are reused by batches.
"""
import argparse
from pathlib import Path
import time
import tomllib

from anthropic import Anthropic

from confidence_experiment.data import load
from confidence_experiment.io import write_json
from confidence_experiment.llm import generate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--examples", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.examples <= 10:
        parser.error("examples must be between 1 and 10")
    config = tomllib.loads(Path(args.config).read_text(encoding="utf-8"))
    settings = config["llm"]
    # The synchronous Caller uses cache_dir/spending.json. Keep smoke and batch
    # accounting together, rather than silently introducing another ledger.
    if Path(settings["ledger_path"]) != Path(settings["cache_dir"]) / "spending.json":
        parser.error("This smoke check requires ledger_path = cache_dir/spending.json")
    client = Anthropic(max_retries=0, timeout=120.0)
    metadata = client.models.retrieve(settings["model"]).model_dump(mode="json")
    write_json(Path(config["run_dir"]) / "model_access.json", metadata)
    print(f"Model access verified: {metadata['id']}", flush=True)
    manifest, frames = load(config["data_dir"])
    frames["development"] = frames["development"].head(args.examples)
    tick = time.perf_counter()
    rows, ledger = generate(config, frames, manifest["labels"], ["development"])
    summary = {"status": "connectivity and schema smoke check; not a benchmark result",
        "model": settings["model"], "thinking": settings.get("thinking"),
        "examples": args.examples, "seconds": time.perf_counter() - tick,
        "ledger": ledger, "rows": rows["development"]}
    write_json(Path(config["run_dir"]) / "smoke.json", summary)
    print(f"Three-call protocol passed for {args.examples} examples. "
          f"Recorded spending ${ledger['usd']:.4f}; cached calls will be reused.", flush=True)


if __name__ == "__main__":
    main()
