

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
)
from src.selective_prediction import (
    static_threshold_selection,
    cost_aware_selection,
    dynamic_threshold_selection,
    tri_action_selection,
    threshold_sweep,
)
from src.risk_metrics import (
    selective_risk, coverage, selective_precision_recall,
    expected_operational_cost, full_prediction_cost,
    area_under_risk_coverage_curve,
)
from src.calibration import calibration_summary
from src.utils import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────
# Baseline (no abstention) metrics
# ─────────────────────────────────────────────

def evaluate_baseline(model, X: np.ndarray, y_true: np.ndarray) -> dict:
    y_pred  = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    return {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc":   float(roc_auc_score(y_true, y_proba)),
        "avg_prec":  float(average_precision_score(y_true, y_proba)),
        **full_prediction_cost(y_true, y_pred),
        **calibration_summary(y_true, y_proba),
    }


# ─────────────────────────────────────────────
# Single-strategy evaluation
# ─────────────────────────────────────────────

def evaluate_strategy(
    result,          # SelectionResult
    y_true: np.ndarray,
    cost_fp: float = 1.0,
    cost_fn: float = 5.0,
    cost_abstain: float = 0.5,
    cost_defer: float = 1.5,
) -> dict:
    """Compute all metrics for a SelectionResult object."""
    predicted_mask = np.array(result.actions) == "predict"

    if predicted_mask.sum() == 0:
        return {
            "strategy": result.strategy,
            "coverage": 0.0,
            "abstention_rate": result.abstention_rate,
            "defer_rate": result.defer_rate,
            "selective_risk": np.nan,
            "selective_precision": np.nan,
            "selective_recall": np.nan,
            "selective_f1": np.nan,
            "total_cost": np.nan,
            "per_sample_cost": np.nan,
        }

    sr = selective_risk(y_true, result.predictions, predicted_mask)
    spr = selective_precision_recall(y_true, result.predictions, predicted_mask)
    cost = expected_operational_cost(
        y_true, result.predictions, result.actions,
        cost_fp=cost_fp, cost_fn=cost_fn,
        cost_abstain=cost_abstain, cost_defer=cost_defer,
    )
    return {
        "strategy":           result.strategy,
        "coverage":           result.coverage,
        "abstention_rate":    result.abstention_rate,
        "defer_rate":         result.defer_rate,
        "selective_risk":     sr,
        "selective_precision": spr["precision"],
        "selective_recall":   spr["recall"],
        "selective_f1":       spr["f1"],
        **cost,
    }


# ─────────────────────────────────────────────
# Full model evaluation across all strategies
# ─────────────────────────────────────────────

def evaluate_model_all_strategies(
    model,
    X: np.ndarray,
    y_true: np.ndarray,
    static_threshold: float = 0.70,
    target_coverage: float = 0.80,
    cost_fp: float = 1.0,
    cost_fn: float = 5.0,
    cost_abstain: float = 0.5,
    cost_defer: float = 1.5,
) -> dict:
    """
    Run baseline + all three selective strategies on the model.
    Returns a dict with keys: "baseline", "static", "cost_aware", "dynamic",
    "tri_action", "sweep_static", "sweep_dynamic".
    """
    logger.info(f"Evaluating {type(model).__name__} …")

    # Baseline
    baseline = evaluate_baseline(model, X, y_true)

    # Strategy 1 — static threshold
    res_static = static_threshold_selection(model, X, threshold=static_threshold)
    eval_static = evaluate_strategy(res_static, y_true, cost_fp, cost_fn, cost_abstain)

    # Strategy 2 — cost-aware
    res_cost = cost_aware_selection(
        model, X, cost_fp=cost_fp, cost_fn=cost_fn, cost_abstain=cost_abstain
    )
    eval_cost = evaluate_strategy(res_cost, y_true, cost_fp, cost_fn, cost_abstain)

    # Strategy 3 — dynamic threshold
    res_dynamic = dynamic_threshold_selection(model, X, target_coverage=target_coverage)
    eval_dynamic = evaluate_strategy(res_dynamic, y_true, cost_fp, cost_fn, cost_abstain)

    # Tri-action framework
    res_tri = tri_action_selection(model, X)
    eval_tri = evaluate_strategy(
        res_tri, y_true, cost_fp, cost_fn, cost_abstain, cost_defer
    )

    # Threshold sweeps for risk–coverage curves
    sweep_st = threshold_sweep(model, X, y_true, strategy="static")
    sweep_dy = threshold_sweep(
        model, X, y_true,
        strategy="dynamic",
        thresholds=np.linspace(0.50, 0.99, 50),
    )

    aurc_static  = area_under_risk_coverage_curve(sweep_st)
    aurc_dynamic = area_under_risk_coverage_curve(sweep_dy)

    return {
        "baseline":      baseline,
        "static":        eval_static,
        "cost_aware":    eval_cost,
        "dynamic":       eval_dynamic,
        "tri_action":    eval_tri,
        "sweep_static":  sweep_st,
        "sweep_dynamic": sweep_dy,
        "aurc_static":   aurc_static,
        "aurc_dynamic":  aurc_dynamic,
    }


# ─────────────────────────────────────────────
# Multi-model comparison table
# ─────────────────────────────────────────────

def build_comparison_table(all_results: dict) -> pd.DataFrame:
    """
    all_results: {model_name: evaluate_model_all_strategies(…)}

    Returns a DataFrame with one row per (model, strategy).
    """
    rows = []
    for model_name, res in all_results.items():
        # Baseline row
        b = res["baseline"]
        rows.append({
            "model": model_name,
            "strategy": "none (baseline)",
            "coverage": 1.0,
            "selective_risk": 1.0 - b["accuracy"],
            "selective_f1": b["f1"],
            "per_sample_cost": b["per_sample_cost"],
            "ece": b.get("ece", np.nan),
        })
        # Selective strategies
        for strat in ("static", "cost_aware", "dynamic", "tri_action"):
            s = res[strat]
            rows.append({
                "model":           model_name,
                "strategy":        strat,
                "coverage":        s["coverage"],
                "selective_risk":  s["selective_risk"],
                "selective_f1":    s["selective_f1"],
                "per_sample_cost": s["per_sample_cost"],
                "ece":             np.nan,
            })

    df = pd.DataFrame(rows)
    # Round for display
    float_cols = ["coverage", "selective_risk", "selective_f1", "per_sample_cost", "ece"]
    df[float_cols] = df[float_cols].round(4)
    return df
