

import joblib
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from src.utils import get_models_dir, get_logger, set_seed

logger = get_logger(__name__)
RANDOM_STATE = 42


# ─────────────────────────────────────────────
# Model factory
# ─────────────────────────────────────────────

def build_model(name: str, class_weight: str | None = "balanced", **kwargs):
    """
    Instantiate a classifier by name.

    Parameters
    ----------
    name : one of {"lr", "rf", "gb", "xgb"}
    class_weight : passed to models that support it
    **kwargs : forwarded to the underlying constructor
    """
    set_seed(RANDOM_STATE)

    if name == "lr":
        return LogisticRegression(
            max_iter=1000,
            class_weight=class_weight,
            solver="lbfgs",
            random_state=RANDOM_STATE,
            **kwargs,
        )
    elif name == "rf":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            class_weight=class_weight,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            **kwargs,
        )
    elif name == "gb":
        # GradientBoosting doesn't support class_weight natively
        return GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            random_state=RANDOM_STATE,
            **kwargs,
        )
    elif name == "xgb":
        if not XGBOOST_AVAILABLE:
            raise ImportError("xgboost is not installed.")
        scale_pos_weight = kwargs.pop("scale_pos_weight", 10)
        return XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            verbosity=0,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown model name: {name!r}. Choose from lr, rf, gb, xgb.")


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────

def train_model(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    calibrate: bool = False,
    calibration_method: str = "isotonic",
) -> object:
    """
    Fit `model` on (X_train, y_train).

    If `calibrate=True`, wraps the trained model in a
    CalibratedClassifierCV using the provided val set (or cross-val
    on the training set if val is not supplied).

    Returns the fitted (and optionally calibrated) model.
    """
    logger.info(f"Training {type(model).__name__} …")
    model.fit(X_train, y_train)

    if calibrate:
        logger.info(f"Applying probability calibration ({calibration_method}) ...")
        if X_val is not None and y_val is not None:
            X_cal = np.vstack([X_train, X_val])
            y_cal = np.concatenate([y_train, y_val])
        else:
            X_cal, y_cal = X_train, y_train
        cal_model = CalibratedClassifierCV(
            model, method=calibration_method, cv=5
        )
        cal_model.fit(X_cal, y_cal)
        return cal_model

    return model


# ─────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────

def save_model(model, name: str) -> Path:
    path = get_models_dir() / f"{name}.pkl"
    joblib.dump(model, path)
    logger.info(f"Model saved → {path}")
    return path


def load_model(name: str) -> object:
    path = get_models_dir() / f"{name}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"No saved model at {path}")
    return joblib.load(path)


# ─────────────────────────────────────────────
# Batch training helper
# ─────────────────────────────────────────────

def train_all_models(
    splits: dict,
    model_names: list[str] = ("lr", "rf", "gb"),
    calibrate: bool = True,
) -> dict:
    """
    Train and optionally calibrate all requested models.

    Returns a dict {model_name: fitted_model}.
    """
    trained = {}
    for name in model_names:
        model = build_model(name)
        fitted = train_model(
            model,
            splits["X_train"],
            splits["y_train"],
            X_val=splits["X_val"],
            y_val=splits["y_val"],
            calibrate=calibrate,
        )
        save_model(fitted, name)
        trained[name] = fitted
        logger.info(f"  ✓ {name} trained and saved.")
    return trained
