import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from confidence_experiment.io import read_json


def script(name):
    path = Path(__file__).parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_memory_gate_waits_before_launching_full_model(tmp_path, monkeypatch):
    supervisor = script("background_run.py")
    memory = iter([1.5, 4.5])
    monkeypatch.setattr(supervisor, "available_memory_gib", lambda: next(memory))
    sleeps, launches = [], []
    monkeypatch.setattr(supervisor.time, "sleep", sleeps.append)
    def popen(command, **kwargs):
        launches.append(command)
        return SimpleNamespace(pid=123, returncode=0, poll=lambda: 0)
    monkeypatch.setattr(supervisor.subprocess, "Popen", popen)
    config = {"run_dir": str(tmp_path), "methods": ["modernbert"], "runtime": {"min_available_memory_gib": 4.0}}
    args = SimpleNamespace(config="config.toml", cpu_threads=2, then_config=None, download_embedding=False)
    assert supervisor.supervise(args, config) == 0
    assert sleeps == [15]
    assert launches[0][1] == "scripts/run_modernbert_cpu.py"
    assert "training has not started" in (tmp_path / "background.log").read_text()
    assert read_json(tmp_path / "background_status.json")["state"] == "complete"


def test_training_timing_preserves_adam_updates(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    worker = script("run_modernbert_cpu.py")
    original_adam = torch.optim.AdamW
    config_path = tmp_path / "config.toml"
    config_path.write_text(f'data_dir = "unused"\nrun_dir = "{tmp_path.as_posix()}"\n'
        '[encoder]\nbatch_size = 1\nepochs = 1\n', encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["worker", "--config", str(config_path)])
    monkeypatch.setattr(worker, "load", lambda *args: ({}, {"fit": list(range(10))}))
    def optimize(optimizer_type):
        weights = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
        optimizer = optimizer_type([weights], lr=0.01)
        for _ in range(10):
            optimizer.zero_grad()
            weights.square().sum().backward()
            optimizer.step()
        return weights.detach()
    expected = optimize(original_adam)
    observed = []
    monkeypatch.setattr("confidence_experiment.cli.main", lambda: observed.append(optimize(torch.optim.AdamW)))
    worker.main()
    assert torch.equal(expected, observed[0])
    assert torch.optim.AdamW is original_adam
    progress = read_json(tmp_path / "training_progress.json")
    assert progress["completed_steps"] == progress["total_steps"] == 10
    assert progress["trainable_parameters"] == 2
