"""All confidence scores estimate correctness of the selected label."""
import numpy as np
from scipy.special import softmax
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, log_loss, roc_auc_score


def wilson(errors, n, z=1.959963984540054):
    if n == 0:
        return [0.0, 1.0]
    p = errors / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [float(max(0, center - half)), float(min(1, center + half))]


def fit_temperature(logits, labels):
    result = minimize_scalar(
        lambda log_t: log_loss(labels, softmax(logits / np.exp(log_t), axis=1), labels=np.arange(logits.shape[1])),
        bounds=(-4, 4), method="bounded",
    )
    return float(np.exp(result.x))


class Sigmoid:
    def fit(self, scores, correct):
        if len(np.unique(correct)) < 2:
            raise ValueError("Calibration requires both correct and incorrect examples.")
        self.model = LogisticRegression(C=1.0).fit(np.asarray(scores).reshape(-1, 1), correct)
        return self

    def predict(self, scores):
        return self.model.predict_proba(np.asarray(scores).reshape(-1, 1))[:, 1]


def risk_curve(correct, confidence):
    correct, confidence = np.asarray(correct), np.asarray(confidence)
    if not len(correct) or not np.isfinite(confidence).all() or ((confidence < 0) | (confidence > 1)).any():
        raise ValueError("Confidence must contain finite probabilities for a nonempty dataset.")
    order = np.argsort(-confidence, kind="stable")
    sorted_q = confidence[order]
    # End ties together: thresholding cannot accept a fraction of identical scores.
    ends = np.r_[np.flatnonzero(sorted_q[:-1] != sorted_q[1:]), len(correct) - 1]
    accepted = ends + 1
    errors = np.cumsum(1 - correct[order])[ends]
    return {"threshold": sorted_q[ends], "accepted": accepted, "errors": errors,
            "coverage": accepted / len(correct), "risk": errors / accepted}


def select_threshold(correct, confidence, max_error=0.01, rule="wilson"):
    curve = risk_curve(correct, confidence)
    upper = np.array([wilson(int(e), int(n))[1] for e, n in zip(curve["errors"], curve["accepted"])])
    eligible = (upper if rule == "wilson" else curve["risk"]) <= max_error
    if not eligible.any():
        return {"threshold": None, "development_coverage": 0.0, "development_risk": None,
                "rule": rule, "target_error": max_error}
    i = np.flatnonzero(eligible)[-1]
    return {"threshold": float(curve["threshold"][i]), "development_coverage": float(curve["coverage"][i]),
            "development_risk": float(curve["risk"][i]), "rule": rule, "target_error": max_error}


def evaluate(truth, prediction, confidence, threshold, bins=10, labels=None):
    truth, prediction, confidence = np.asarray(truth), np.asarray(prediction), np.asarray(confidence)
    correct = (truth == prediction).astype(int)
    curve = risk_curve(correct, confidence)
    error = 1 - correct
    clipped = np.clip(confidence, 1e-7, 1 - 1e-7)
    reliability = []
    bin_id = np.minimum((confidence * bins).astype(int), bins - 1)
    for index in range(bins):
        mask = bin_id == index
        n = int(mask.sum())
        if n:
            interval = wilson(int(correct[mask].sum()), n)
            reliability.append({"bin": index, "n": n, "confidence": float(confidence[mask].mean()),
                                "accuracy": float(correct[mask].mean()), "accuracy_ci95": interval})
    accepted = np.zeros(len(correct), dtype=bool) if threshold["threshold"] is None else confidence >= threshold["threshold"]
    n, e = int(accepted.sum()), int(error[accepted].sum())
    metrics = {
        "n": len(correct), "accuracy": float(accuracy_score(truth, prediction)),
        "accuracy_ci95": wilson(int(correct.sum()), len(correct)),
        "macro_f1": float(f1_score(truth, prediction, labels=labels, average="macro", zero_division=0)),
        "errors": int(error.sum()), "brier": float(np.mean((confidence - correct) ** 2)),
        "binary_log_loss": float(-np.mean(correct * np.log(clipped) + error * np.log1p(-clipped))),
        "ece_10_equal_width": float(sum(r["n"] * abs(r["accuracy"] - r["confidence"]) for r in reliability) / len(correct)),
        "error_auroc": float(roc_auc_score(error, 1 - confidence)) if len(np.unique(error)) == 2 else None,
        "error_average_precision": float(average_precision_score(error, 1 - confidence)) if error.any() else None,
        "error_prevalence": float(error.mean()),
        "aurc": float(np.sum(curve["risk"] * np.diff(np.r_[0, curve["coverage"]]))),
        "acceptance": {**threshold, "accepted": n, "errors": e, "coverage": n / len(correct),
                       "risk": e / n if n else None, "risk_ci95": wilson(e, n) if n else None},
    }
    return metrics, reliability, curve
