import numpy as np
import pytest
from scipy.special import softmax

from confidence_experiment.metrics import evaluate, fit_temperature, risk_curve, select_threshold, wilson


def test_threshold_does_not_split_tied_confidences():
    curve = risk_curve([1, 0, 1], [0.9, 0.9, 0.2])
    assert curve["accepted"].tolist() == [2, 3]
    threshold = select_threshold([1, 0, 1], [0.9, 0.9, 0.2], max_error=0.1, rule="empirical")
    assert threshold["threshold"] is None


def test_wilson_rejects_small_error_free_samples():
    assert wilson(0, 100)[1] > 0.01
    assert wilson(0, 1000)[1] < 0.01
    assert select_threshold(np.ones(100), np.full(100, 0.9))["threshold"] is None


def test_constant_score_has_no_error_ranking_power():
    metrics, bins, curve = evaluate([0, 0, 1, 1], [0, 1, 1, 0], [0.5] * 4, {"threshold": 0.5})
    assert metrics["error_auroc"] == 0.5
    assert metrics["brier"] == 0.25
    assert len(curve["coverage"]) == 1
    assert bins[0]["n"] == 4


def test_temperature_preserves_selected_labels():
    logits = np.array([[10, 0], [8, 0], [0, 8], [0, 10]], dtype=float)
    temperature = fit_temperature(logits, np.array([0, 1, 1, 0]))
    assert temperature > 1
    assert np.array_equal(logits.argmax(1), softmax(logits / temperature, axis=1).argmax(1))


def test_no_accepted_examples_has_undefined_risk():
    metrics, _, _ = evaluate([0, 1], [0, 0], [0.8, 0.7], {"threshold": None})
    assert metrics["acceptance"]["coverage"] == 0
    assert metrics["acceptance"]["risk"] is None


@pytest.mark.parametrize("scores", [[float('nan')], [1.1], [-0.1]])
def test_invalid_confidence_rejected(scores):
    with pytest.raises(ValueError):
        risk_curve([1], scores)
