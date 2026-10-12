"""Measure actual full-model training steps while retaining the standard trainer."""
import argparse
from collections import deque
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import sys
import time
import tomllib

import torch

from confidence_experiment.data import load
from confidence_experiment.io import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/modernbert.toml")
    args = parser.parse_args()
    config = tomllib.loads(Path(args.config).read_text(encoding="utf-8"))
    threads = int(os.environ.get("OMP_NUM_THREADS", "2"))
    torch.set_num_threads(threads)
    _, frames = load(config["data_dir"])
    settings = config["encoder"]
    expected_steps = math.ceil(len(frames["fit"]) / settings["batch_size"]) * settings["epochs"]
    output = Path(config["run_dir"])
    output.mkdir(parents=True, exist_ok=True)
    original_adamw = torch.optim.AdamW

    class TimedAdamW(original_adamw):
        def __init__(self, *optimizer_args, **kwargs):
            super().__init__(*optimizer_args, **kwargs)
            self.completed_steps = 0
            self.recent_seconds = deque(maxlen=50)
            self.previous = time.perf_counter()
            self.parameter_count = sum(p.numel() for group in self.param_groups for p in group["params"] if p.requires_grad)
            print(f"Full-model CPU training: {self.parameter_count:,} trainable parameters; {expected_steps:,} optimizer steps; {threads} threads.", flush=True)

        def step(self, *step_args, **kwargs):
            result = super().step(*step_args, **kwargs)
            tick = time.perf_counter()
            self.completed_steps += 1
            seconds = tick - self.previous
            self.previous = tick
            if self.completed_steps > 2:
                self.recent_seconds.append(seconds)
            if self.completed_steps in {1, 10} or self.completed_steps % 50 == 0 or self.completed_steps == expected_steps:
                mean_seconds = sum(self.recent_seconds) / len(self.recent_seconds) if self.recent_seconds else seconds
                remaining = (expected_steps - self.completed_steps) * mean_seconds
                write_json(output / "training_progress.json", {
                    "updated_utc": datetime.now(timezone.utc).isoformat(), "completed_steps": self.completed_steps,
                    "total_steps": expected_steps, "seconds_per_step_recent": mean_seconds,
                    "remaining_training_seconds_estimate": remaining, "trainable_parameters": self.parameter_count,
                    "cpu_threads": threads, "note": "Includes real forward/backward/optimizer steps; inference and reporting are additional."})
                print(f"ModernBERT step {self.completed_steps}/{expected_steps}: {mean_seconds:.2f} seconds/step; estimated training remaining {remaining / 3600:.2f} hours.", flush=True)
            return result

    torch.optim.AdamW = TimedAdamW
    try:
        from confidence_experiment.cli import main as run_experiment
        sys.argv = [sys.argv[0], "run", "--config", args.config]
        run_experiment()
    finally:
        torch.optim.AdamW = original_adamw


if __name__ == "__main__":
    main()
