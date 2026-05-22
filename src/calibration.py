

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from src.utils import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────
# Calibration metrics
# ─────────────────────────────────────────────

def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Compute the Expected Calibration Error (ECE).

    ECE = Σ_b (|B_b| / n) * |acc(B_b) - conf(B_b)|

    where B_b is the set of samples whose predicted probability falls
    in the b-th bin.
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        bin_acc  = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


def maximum_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Maximum Calibration Error across all bins."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    mce = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        bin_acc  = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        mce = max(mce, abs(bin_acc - bin_conf))
    return float(mce)


def reliability_diagram_data(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """
    Return fraction_of_positives and mean_predicted_value for each bin.
    Ready for plotting a reliability diagram.
    """
    frac_pos, mean_pred = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy="uniform"
    )
    return {"fraction_of_positives": frac_pos, "mean_predicted_value": mean_pred}


def calibration_summary(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "",
) -> dict:
    """Return a dict of all calibration metrics for a model."""
    ece  = expected_calibration_error(y_true, y_prob)
    mce  = maximum_calibration_error(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    logger.info(
        f"{model_name or 'Model'} — ECE: {ece:.4f}, MCE: {mce:.4f}, "
        f"Brier: {brier:.4f}"
    )
    return {"ece": ece, "mce": mce, "brier": brier}
