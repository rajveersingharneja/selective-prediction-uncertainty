
import os
import json
import random
import logging
import numpy as np
from pathlib import Path
from datetime import datetime


# ─────────────────────────────────────────────
# Project root resolution
# ─────────────────────────────────────────────

def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return Path(__file__).resolve().parent.parent


def get_data_dir(subdir: str = "processed") -> Path:
    return get_project_root() / "data" / subdir


def get_reports_dir(subdir: str = "figures") -> Path:
    path = get_project_root() / "reports" / subdir
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_models_dir() -> Path:
    path = get_project_root() / "experiments" / "saved_models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_outputs_dir() -> Path:
    path = get_project_root() / "experiments" / "outputs"
    path.mkdir(parents=True, exist_ok=True)
    return path


# ─────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create a named logger with a clean format."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "[%(asctime)s] %(levelname)s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# ─────────────────────────────────────────────
# Config loading
# ─────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    """Load a JSON experiment config file."""
    with open(config_path, "r") as f:
        return json.load(f)


def save_config(config: dict, config_path: str) -> None:
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)


# ─────────────────────────────────────────────
# Timestamp
# ─────────────────────────────────────────────

def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")
