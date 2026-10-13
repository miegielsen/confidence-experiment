"""Final inference must never fit, retune, or submit before validating evidence."""
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.special import softmax
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import final_protocol as protocol
import final_transport as transport
import final_evaluation as worker
from confidence_experiment.metrics import Sigmoid
from confidence_experiment.io import digest, write_json


def test_invalid_plan_prevents_both_api_clients(monkeypatch):
    def reject():
        raise ValueError("bad frozen input")
    def unexpected(*args, **kwargs):
        pytest.fail("API client constructed before final validation")
    monkeypatch.setattr(transport, "preflight", reject)
    monkeypatch.setattr(transport, "Batcher", unexpected)
    monkeypatch.setattr(transport, "DecisionCaller", unexpected)
    with pytest.raises(ValueError, match="bad frozen input"):
        transport.anthropic("haiku")
    with pytest.raises(ValueError, match="bad frozen input"):
        transport.decisions()


def test_scalar_calibrators_never_refit_and_refusals_stay_zero(monkeypatch):
    q = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 0.9])
    y = np.array([0, 0, 1, 0, 1, 1])
    objects = {"x": {"logistic_score": Sigmoid().fit(q, y),
                     "logistic_logit": LogisticRegression().fit(np.log(q / (1 - q)).reshape(-1, 1), y),
                     "isotonic": IsotonicRegression(out_of_bounds="clip").fit(q, y)}}
    def unexpected(*args, **kwargs):
        pytest.fail("Final evaluation refitted a calibrator")
    monkeypatch.setattr(LogisticRegression, "fit", unexpected)
    monkeypatch.setattr(IsotonicRegression, "fit", unexpected)
    scores = protocol.transform("x", [.01, .99], ["a", "__refusal__"], objects, [.1, .8])
    assert set(scores) == {"raw", "logistic_score", "logistic_logit", "isotonic", "temperature"}
    assert all(values[1] == 0 for values in scores.values())


def test_temperature_scores_keep_api_selected_label():
    logits = np.array([[4., 1., 0.]])
    scores = protocol.temperature_scores(logits, 2., selected=np.array([1]))
    assert scores[0] == pytest.approx(softmax(logits / 2, axis=1)[0, 1])
    assert scores[0] < softmax(logits / 2, axis=1).max()


def test_completed_results_skip_api_work_when_report_needs_retry(tmp_path, monkeypatch):
    monkeypatch.setattr(worker, "preflight", lambda: {})
    monkeypatch.setattr(worker, "completed", lambda arm: True)
    monkeypatch.setattr(worker, "local", lambda arm: pytest.fail("Repeated inference"))
    monkeypatch.setattr(transport, "anthropic", lambda arm: pytest.fail("Repeated API work"))
    worker.run_arm("embeddings")
    worker.run_arm("haiku")


def test_completion_requires_intact_results(tmp_path, monkeypatch):
    monkeypatch.setattr(protocol, "OUT", tmp_path)
    monkeypatch.setattr(protocol, "PLAN", tmp_path / "plan.json")
    write_json(protocol.PLAN, {"sha256": "sealed"})
    folder = tmp_path / "haiku"
    folder.mkdir()
    (folder / "metrics.json").write_text("{}")
    assert not protocol.completed("haiku")
    write_json(folder / "complete.json", {"plan_sha256": "sealed", "outputs_sha256": {"metrics.json": protocol.sha(folder / "metrics.json")}})
    assert protocol.completed("haiku")
    (folder / "metrics.json").write_text('{"changed":true}')
    with pytest.raises(ValueError, match="Completed result changed"):
        protocol.completed("haiku")
