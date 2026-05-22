

import numpy as np
from dataclasses import dataclass, field
from typing import Literal
from src.utils import get_logger

logger = get_logger(__name__)

Action = Literal["predict", "abstain", "defer"]


# ─────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────

@dataclass
class SelectionResult:
    """Holds outputs for a selective prediction run."""
    predictions:      np.ndarray          # hard labels for predicted samples (-1 = abstained)
    confidences:      np.ndarray          # max predicted probability for each sample
    actions:          list[Action]        # "predict" | "abstain" | "defer"
    threshold:        float               # threshold used
    strategy:         str                 # name of the strategy
    coverage:         float = field(init=False)
    abstention_rate:  float = field(init=False)
    defer_rate:       float = field(init=False)

    def __post_init__(self):
        n = len(self.actions)
        self.coverage        = sum(a == "predict" for a in self.actions) / n
        self.abstention_rate = sum(a == "abstain" for a in self.actions) / n
        self.defer_rate      = sum(a == "defer"   for a in self.actions) / n


# ─────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────

def get_confidence_scores(
    model,
    X: np.ndarray,
    positive_class_index: int = 1,
) -> np.ndarray:
    """
    Return the model's confidence in the positive class.

    For binary classification we use P(y=1|x) directly rather than
    max-confidence because the positive class is the meaningful uncertain one.
    """
    proba = model.predict_proba(X)
    return proba[:, positive_class_index]


def get_max_confidence(model, X: np.ndarray) -> np.ndarray:
    """Return max P(y|x) across all classes (works for multi-class too)."""
    return model.predict_proba(X).max(axis=1)


# ─────────────────────────────────────────────
# Strategy 1 — Static threshold
# ─────────────────────────────────────────────

def static_threshold_selection(
    model,
    X: np.ndarray,
    threshold: float = 0.7,
    positive_class_index: int = 1,
) -> SelectionResult:
    """
    Abstain when the model's confidence in its prediction falls below `threshold`.

    A sample is "confident" if P(positive) >= threshold  OR
                               P(negative) >= threshold.
    i.e., we abstain on the fuzzy middle ground.
    """
    proba = model.predict_proba(X)
    max_conf = proba.max(axis=1)
    hard_preds = proba.argmax(axis=1)

    actions: list[Action] = []
    preds = np.full(len(X), -1, dtype=int)

    for i, (conf, pred) in enumerate(zip(max_conf, hard_preds)):
        if conf >= threshold:
            actions.append("predict")
            preds[i] = pred
        else:
            actions.append("abstain")

    confidences = proba[:, positive_class_index]
    return SelectionResult(
        predictions=preds,
        confidences=confidences,
        actions=actions,
        threshold=threshold,
        strategy="static",
    )


# ─────────────────────────────────────────────
# Strategy 2 — Cost-aware abstention
# ─────────────────────────────────────────────

def cost_aware_selection(
    model,
    X: np.ndarray,
    cost_fp: float = 1.0,
    cost_fn: float = 5.0,
    cost_abstain: float = 0.5,
    positive_class_index: int = 1,
) -> SelectionResult:
    """
    Abstain when the expected cost of predicting exceeds the abstention cost.

    Expected cost of predicting:
        E[cost | x] = P(y=0|x) * cost_FP * I(predict_pos)
                    + P(y=1|x) * cost_FN * I(predict_neg)

    Decision rule:
        - If model predicts positive: E[cost] = P(y=0|x) * cost_fp
        - If model predicts negative: E[cost] = P(y=1|x) * cost_fn
        - Abstain if E[cost] > cost_abstain
    """
    proba = model.predict_proba(X)
    p_pos = proba[:, positive_class_index]
    p_neg = 1.0 - p_pos
    hard_preds = proba.argmax(axis=1)

    actions: list[Action] = []
    preds = np.full(len(X), -1, dtype=int)
    thresholds = np.empty(len(X))

    for i, (pp, pn, pred) in enumerate(zip(p_pos, p_neg, hard_preds)):
        if pred == 1:
            expected_cost = pn * cost_fp       # risk of FP
        else:
            expected_cost = pp * cost_fn       # risk of FN

        thresholds[i] = expected_cost
        if expected_cost <= cost_abstain:
            actions.append("predict")
            preds[i] = pred
        else:
            actions.append("abstain")

    return SelectionResult(
        predictions=preds,
        confidences=p_pos,
        actions=actions,
        threshold=cost_abstain,
        strategy="cost_aware",
    )


# ─────────────────────────────────────────────
# Strategy 3 — Dynamic threshold
# ─────────────────────────────────────────────

def dynamic_threshold_selection(
    model,
    X: np.ndarray,
    target_coverage: float = 0.80,
    positive_class_index: int = 1,
) -> SelectionResult:
    """
    Choose the threshold τ that achieves exactly `target_coverage`
    of the dataset being predicted.

    The threshold is calibrated on the same X (unsupervised) — in
    production you'd calibrate on a held-out set.
    """
    proba = model.predict_proba(X)
    max_conf = proba.max(axis=1)
    hard_preds = proba.argmax(axis=1)

    # Find τ such that coverage ≈ target
    threshold = float(np.quantile(max_conf, 1.0 - target_coverage))

    actions: list[Action] = []
    preds = np.full(len(X), -1, dtype=int)

    for i, (conf, pred) in enumerate(zip(max_conf, hard_preds)):
        if conf >= threshold:
            actions.append("predict")
            preds[i] = pred
        else:
            actions.append("abstain")

    return SelectionResult(
        predictions=preds,
        confidences=proba[:, positive_class_index],
        actions=actions,
        threshold=threshold,
        strategy="dynamic",
    )


# ─────────────────────────────────────────────
# Tri-action framework (Predict / Abstain / Defer)
# ─────────────────────────────────────────────

def tri_action_selection(
    model,
    X: np.ndarray,
    threshold_predict: float = 0.75,
    threshold_defer: float = 0.55,
    positive_class_index: int = 1,
) -> SelectionResult:
    """
    Three-way action framework:

        confidence >= threshold_predict  → PREDICT (high confidence)
        confidence  < threshold_defer    → ABSTAIN  (very low confidence, reject)
        otherwise                        → DEFER    (moderate confidence: route
                                                    to a more expensive model or
                                                    human reviewer)

    This formalises the real-world idea that "I don't know" has two
    distinct sub-cases: "completely unsure" vs "almost sure, need a
    second opinion".
    """
    proba = model.predict_proba(X)
    max_conf = proba.max(axis=1)
    hard_preds = proba.argmax(axis=1)

    actions: list[Action] = []
    preds = np.full(len(X), -1, dtype=int)

    for i, (conf, pred) in enumerate(zip(max_conf, hard_preds)):
        if conf >= threshold_predict:
            actions.append("predict")
            preds[i] = pred
        elif conf < threshold_defer:
            actions.append("abstain")
        else:
            actions.append("defer")

    return SelectionResult(
        predictions=preds,
        confidences=proba[:, positive_class_index],
        actions=actions,
        threshold=threshold_predict,
        strategy="tri_action",
    )


# ─────────────────────────────────────────────
# Threshold sweep (for risk–coverage curves)
# ─────────────────────────────────────────────

def threshold_sweep(
    model,
    X: np.ndarray,
    y_true: np.ndarray,
    thresholds: np.ndarray | None = None,
    strategy: str = "static",
    positive_class_index: int = 1,
    **cost_kwargs,
) -> list[dict]:
    """
    Sweep over threshold values and collect coverage / selective_risk at each point.

    Returns a list of dicts, one per threshold value.
    """
    if thresholds is None:
        thresholds = np.linspace(0.50, 0.99, 50)

    records = []
    for tau in thresholds:
        if strategy == "static":
            result = static_threshold_selection(
                model, X, threshold=tau,
                positive_class_index=positive_class_index
            )
        elif strategy == "dynamic":
            result = dynamic_threshold_selection(
                model, X, target_coverage=tau,
                positive_class_index=positive_class_index
            )
        elif strategy == "cost_aware":
            result = cost_aware_selection(
                model, X,
                cost_abstain=tau,
                positive_class_index=positive_class_index,
                **cost_kwargs,
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy!r}")

        predicted_mask = np.array(result.actions) == "predict"
        if predicted_mask.sum() == 0:
            sel_risk = np.nan
        else:
            sel_risk = (result.predictions[predicted_mask] != y_true[predicted_mask]).mean()

        records.append({
            "threshold": tau,
            "coverage": result.coverage,
            "selective_risk": sel_risk,
            "abstention_rate": result.abstention_rate,
        })

    return records
