import hashlib
from io import BytesIO

import pandas as pd
import pytest

from confidence_experiment.data import load, prepare


def test_prepare_enforces_disjoint_texts_and_detects_tampering(tmp_path, monkeypatch):
    train = pd.DataFrame({"text": [f"train message {i}" for i in range(10003)],
                          "category": [f"intent_{i % 77:02}" for i in range(10003)]})
    test = pd.DataFrame({"text": [f"test message {i}" for i in range(3080)],
                         "category": [f"intent_{i % 77:02}" for i in range(3080)]})
    train.loc[77, "text"] = train.loc[0, "text"]  # same-label duplicate
    train.loc[1, "text"] = "TEST MESSAGE 0"  # overlap, regardless of label
    train.loc[3, "text"] = train.loc[2, "text"]  # conflicting labels
    def download(url, **kwargs):
        return BytesIO((train if url.endswith("train.csv") else test).to_csv(index=False).encode())
    monkeypatch.setattr("confidence_experiment.data.urlopen", download)
    manifest = prepare(tmp_path)
    assert manifest["audit"]["total_train_rows_excluded"] == 4
    _, frames = load(tmp_path)
    seen = set()
    for role, frame in frames.items():
        keys = set(frame.text.str.casefold().str.split().str.join(" "))
        assert not keys & seen
        seen |= keys
    assert len(frames["test"]) == 3080
    assert prepare(tmp_path) == manifest
    with (tmp_path / "fit.csv").open("a") as handle:
        handle.write("changed\n")
    with pytest.raises(ValueError, match="Dataset split changed"):
        load(tmp_path)
