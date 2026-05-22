

import numpy as np
from scipy import stats
from typing import Sequence
from src.utils import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────
# Core selective metrics
# ─────────────────────────────────────────────

def selective_risk(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    predicted_mask: np.ndarray,
) -> float:
    """
    Fraction of mis-classifications among *predicted* samples.

    selective_risk = (# errors on predicted) / (# predicted)
    """
    if predicted_mask.sum() == 0:
        return np.nan
    return float((y_pred[predicted_mask] != y_true[predicted_mask]).mean())


def coverage(predicted_mask: np.ndarray) -> float:
    """Fraction of samples that received a prediction."""
    return float(predicted_mask.mean())


def selective_precision_recall(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    predicted_mask: np.ndarray,
) -> dict:
    """
    Compute precision and recall restricted to the predicted subset.
    """
    yt = y_true[predicted_mask]
    yp = y_pred[predicted_mask]
    tp = ((yt == 1) & (yp == 1)).sum()
    fp = ((yt == 0) & (yp == 1)).sum()
    fn = ((yt == 1) & (yp == 0)).sum()
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return {"precision": float(prec), "recall": float(rec), "f1": float(f1)}


# ─────────────────────────────────────────────
# Operational cost model
# ─────────────────────────────────────────────

def expected_operational_cost(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    actions: Sequence[str],
    cost_fp: float = 1.0,
    cost_fn: float = 5.0,
    cost_abstain: float = 0.5,
    cost_defer: float = 1.5,
) -> dict:
    """
    Compute the expected cost under a simple linear cost model.

    Cost components:
        False positive:  cost_fp   (e.g., wasted review)
        False negative:  cost_fn   (e.g., missed fraud)
        Abstain:         cost_abstain (e.g., manual review)
        Defer:           cost_defer   (e.g., slower model or expert)

    Returns total cost and per-unit cost.
    """
    total = 0.0
    n = len(y_true)

    for i, action in enumerate(actions):
        if action == "predict":
            if y_pred[i] == 1 and y_true[i] == 0:
                total += cost_fp
            elif y_pred[i] == 0 and y_true[i] == 1:
                total += cost_fn
            # correct prediction: zero cost
        elif action == "abstain":
            total += cost_abstain
        elif action == "defer":
            total += cost_defer

    return {
        "total_cost": float(total),
        "per_sample_cost": float(total / n),
        "n_samples": n,
    }


def full_prediction_cost(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cost_fp: float = 1.0,
    cost_fn: float = 5.0,
) -> dict:
    """Baseline: cost when model predicts on everything (no abstention)."""
    total = 0.0
    for yt, yp in zip(y_true, y_pred):
        if yp == 1 and yt == 0:
            total += cost_fp
        elif yp == 0 and yt == 1:
            total += cost_fn
    n = len(y_true)
    return {
        "total_cost": float(total),
        "per_sample_cost": float(total / n),
    }


# ─────────────────────────────────────────────
# Risk–Coverage curve area (AURC)
# ─────────────────────────────────────────────

def area_under_risk_coverage_curve(
    sweep_records: list[dict],
) -> float:
    """
    Compute the area under the risk–coverage curve via the trapezoidal rule.

    Lower AURC is better (less risk for the same coverage).
    """
    records = sorted(sweep_records, key=lambda r: r["coverage"])
    coverages = [r["coverage"] for r in records]
    risks = [r["selective_risk"] for r in records]

    # Drop NaN entries
    valid = [(c, r) for c, r in zip(coverages, risks) if not np.isnan(r)]
    if len(valid) < 2:
        return np.nan

    cov_arr = np.array([v[0] for v in valid])
    risk_arr = np.array([v[1] for v in valid])
    return float(np.trapezoid(risk_arr, cov_arr))


# ─────────────────────────────────────────────
# Bootstrap confidence intervals
# ─────────────────────────────────────────────

def bootstrap_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict:
    """
    Return bootstrap mean and confidence interval for `metric_fn(y_true, y_pred)`.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        scores.append(metric_fn(y_true[idx], y_pred[idx]))

    alpha = (1.0 - ci) / 2
    lo = float(np.quantile(scores, alpha))
    hi = float(np.quantile(scores, 1.0 - alpha))
    return {"mean": float(np.mean(scores)), "ci_lo": lo, "ci_hi": hi, "ci": ci}
