import hashlib

import numpy as np
import pandas as pd
import pytest

from confidence_experiment.data import ROLES
from confidence_experiment.io import write_json
from confidence_experiment.runner import run


def dataset(tmp_path):
    directory = tmp_path / "data"
    directory.mkdir()
    hashes = {}
    for role in ROLES:
        frame = pd.DataFrame({"id": [f"{role}:{i}" for i in range(24)],
            "text": [f"{role} message {i}" for i in range(24)], "label": ["a", "b", "c"] * 8})
        path = directory / f"{role}.csv"
        frame.to_csv(path, index=False)
        hashes[role] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(directory / "manifest.json", {"labels": ["a", "b", "c"], "seed": 42, "split_sha256": hashes})
    return directory


def test_development_freezes_before_test_and_final_reuses_calibration(tmp_path, monkeypatch):
    directory = dataset(tmp_path)
    modes = []
    def supervised(config, frames, labels, output, include_test=False, frozen=False):
        modes.append((include_test, frozen))
        matrix = np.tile(np.array([[2, 1, 0], [1, 0, 2], [0, 1, 2]]), (8, 1))
        return {"tfidf": {r: matrix for r in ROLES if include_test or r != "test"}}, {}
    monkeypatch.setattr("confidence_experiment.runner.train_supervised", supervised)
    config = {"data_dir": str(directory), "run_dir": str(tmp_path / "run"), "methods": ["tfidf"],
              "seed": 42, "max_error": 0.01, "threshold_rule": "wilson"}
    development = run(config)
    assert not (tmp_path / "run" / "final").exists()
    final = run(config, final=True)
    assert modes == [(False, False), (True, True)]
    assert final["tfidf_temperature"]["acceptance"]["threshold"] == development["tfidf_temperature"]["acceptance"]["threshold"]
    with pytest.raises(ValueError, match="already exists"):
        run(config, final=True)


def test_changed_scientific_config_rejected(tmp_path, monkeypatch):
    directory = dataset(tmp_path)
    from confidence_experiment.runner import identity
    from confidence_experiment.io import read_json
    config = {"data_dir": str(directory), "run_dir": str(tmp_path / "run"), "methods": ["tfidf"],
              "seed": 42, "max_error": 0.01}
    signature = identity(config, read_json(directory / "manifest.json"))
    write_json(tmp_path / "run" / "identity.json", {"sha256": signature})
    with pytest.raises(ValueError, match="different settings"):
        run({**config, "max_error": 0.02})
