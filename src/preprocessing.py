
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from src.utils import get_project_root, get_logger, set_seed

logger = get_logger(__name__)
RANDOM_STATE = 42


# ─────────────────────────────────────────────
# Synthetic dataset (fallback)
# ─────────────────────────────────────────────

def make_synthetic_fraud_dataset(
    n_samples: int = 20_000,
    fraud_ratio: float = 0.05,
    noise_std: float = 0.0,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """
    Generate a synthetic credit-card-fraud-like dataset.

    The generator deliberately creates overlapping class distributions in
    certain feature pairs so that uncertainty is *real*, not trivial.
    """
    rng = np.random.default_rng(random_state)
    n_fraud = int(n_samples * fraud_ratio)
    n_legit = n_samples - n_fraud

    def _block(n, label, rng):
        # 10 PCA-style latent features + 2 raw features (Amount, Hour)
        if label == 0:  # legitimate
            feats = rng.normal(loc=0.0, scale=1.0, size=(n, 10))
            amount = rng.lognormal(mean=4.5, sigma=1.2, size=(n, 1))
            hour = rng.integers(0, 24, size=(n, 1)).astype(float)
        else:           # fraudulent
            feats = rng.normal(loc=0.5, scale=1.3, size=(n, 10))  # overlapping!
            amount = rng.lognormal(mean=3.0, sigma=1.5, size=(n, 1))
            hour = rng.integers(0, 24, size=(n, 1)).astype(float)

        # Optional additive noise (for distribution-shift experiments)
        if noise_std > 0:
            feats += rng.normal(0, noise_std, size=feats.shape)

        cols = [f"V{i+1}" for i in range(10)] + ["Amount", "Hour"]
        df = pd.DataFrame(
            np.hstack([feats, amount, hour]), columns=cols
        )
        df["Class"] = label
        return df

    df = pd.concat([_block(n_legit, 0, rng), _block(n_fraud, 1, rng)], ignore_index=True)
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    return df


# ─────────────────────────────────────────────
# Real dataset loader (Kaggle creditcard.csv)
# ─────────────────────────────────────────────

def load_raw_dataset(csv_path: str | None = None) -> pd.DataFrame:
    """
    Try to load the Kaggle Credit Card Fraud CSV.
    Fall back to the synthetic dataset if the file is not present.
    """
    if csv_path is None:
        csv_path = get_project_root() / "data" / "raw" / "creditcard.csv"

    if Path(csv_path).exists():
        logger.info(f"Loading real dataset from {csv_path}")
        df = pd.read_csv(csv_path)
        # The Kaggle dataset uses 'Time' not 'Hour'; rename for consistency
        if "Time" in df.columns:
            df["Hour"] = (df["Time"] // 3600) % 24
            df.drop(columns=["Time"], inplace=True)
        return df
    else:
        logger.warning(
            "Real dataset not found. Generating synthetic fallback "
            "(20 000 samples, 5 %% fraud rate)."
        )
        return make_synthetic_fraud_dataset()


# ─────────────────────────────────────────────
# Feature engineering
# ─────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Light feature engineering on top of the raw columns."""
    df = df.copy()
    df["log_Amount"] = np.log1p(df["Amount"])
    if "Amount" in df.columns:
        df.drop(columns=["Amount"], inplace=True)
    return df


# ─────────────────────────────────────────────
# Split + scale
# ─────────────────────────────────────────────

def split_and_scale(
    df: pd.DataFrame,
    target_col: str = "Class",
    test_size: float = 0.2,
    val_size: float = 0.1,
    apply_smote: bool = False,
    random_state: int = RANDOM_STATE,
) -> dict:
    """
    Returns a dict with keys:
        X_train, X_val, X_test, y_train, y_val, y_test, feature_names, scaler
    """
    set_seed(random_state)

    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols].values
    y = df[target_col].values

    # First split off test set
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    # Then split val from train+val
    val_frac = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=val_frac, stratify=y_tv, random_state=random_state
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    if apply_smote:
        logger.info("Applying SMOTE to training set …")
        sm = SMOTE(random_state=random_state)
        X_train, y_train = sm.fit_resample(X_train, y_train)
        logger.info(f"After SMOTE: {np.bincount(y_train)}")

    return {
        "X_train": X_train,
        "X_val":   X_val,
        "X_test":  X_test,
        "y_train": y_train,
        "y_val":   y_val,
        "y_test":  y_test,
        "feature_names": feature_cols,
        "scaler": scaler,
    }


# ─────────────────────────────────────────────
# Distribution-shift utilities
# ─────────────────────────────────────────────

def inject_noise(X: np.ndarray, noise_std: float = 0.5, seed: int = 0) -> np.ndarray:
    """Add Gaussian noise to every feature (simulates sensor drift)."""
    rng = np.random.default_rng(seed)
    return X + rng.normal(0, noise_std, size=X.shape)


def perturb_features(
    X: np.ndarray,
    feature_indices: list[int],
    scale: float = 2.0,
    seed: int = 0,
) -> np.ndarray:
    """Multiply selected features by a random scale (simulates covariate shift)."""
    rng = np.random.default_rng(seed)
    X_out = X.copy()
    for idx in feature_indices:
        X_out[:, idx] *= rng.uniform(1.0, scale)
    return X_out


def imbalance_shift(
    X: np.ndarray,
    y: np.ndarray,
    target_fraud_ratio: float = 0.20,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Over-sample the minority class to simulate a higher-fraud-rate environment.
    """
    rng = np.random.default_rng(seed)
    idx_legit = np.where(y == 0)[0]
    idx_fraud = np.where(y == 1)[0]

    n_legit = len(idx_legit)
    n_fraud_target = int(n_legit * target_fraud_ratio / (1 - target_fraud_ratio))
    if n_fraud_target <= len(idx_fraud):
        return X, y  # nothing to do

    extra = rng.choice(idx_fraud, size=n_fraud_target - len(idx_fraud), replace=True)
    X_out = np.vstack([X, X[extra]])
    y_out = np.concatenate([y, y[extra]])
    shuffle = rng.permutation(len(y_out))
    return X_out[shuffle], y_out[shuffle]


# ─────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────

if __name__ == "__main__":
    df = load_raw_dataset()
    df = engineer_features(df)
    splits = split_and_scale(df)
    logger.info(
        f"Train: {splits['X_train'].shape}, "
        f"Val: {splits['X_val'].shape}, "
        f"Test: {splits['X_test'].shape}"
    )
    logger.info(f"Fraud rate (test): {splits['y_test'].mean():.3f}")
