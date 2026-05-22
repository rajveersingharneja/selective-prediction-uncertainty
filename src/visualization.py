

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path
from src.utils import get_reports_dir

# ── consistent style ──────────────────────────────────────────────────────────
PALETTE   = ["#2563EB", "#16A34A", "#DC2626", "#D97706", "#7C3AED"]
GRAY_DARK = "#1e1e2e"
GRAY_MID  = "#374151"
GRID_COL  = "#e5e7eb"

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.color":        GRID_COL,
    "grid.linewidth":    0.6,
    "axes.labelsize":    11,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "figure.dpi":        150,
    "savefig.bbox":      "tight",
    "savefig.dpi":       200,
})

FIG_DIR = get_reports_dir("figures")


def _save(fig, name: str) -> Path:
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


# ─────────────────────────────────────────────
# 1. Confidence distribution
# ─────────────────────────────────────────────

def plot_confidence_distribution(
    confidences: np.ndarray,
    y_true: np.ndarray,
    model_name: str = "Model",
    threshold: float = 0.70,
) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, color, lbl in zip([0, 1], ["#2563EB", "#DC2626"], ["Legit", "Fraud"]):
        mask = y_true == label
        ax.hist(
            confidences[mask], bins=50, density=True,
            alpha=0.55, color=color, label=lbl, edgecolor="none",
        )
    ax.axvline(threshold, color=GRAY_MID, lw=1.5, ls="--",
               label=f"τ = {threshold:.2f}")
    ax.set_xlabel("P(fraud | x)")
    ax.set_ylabel("Density")
    ax.set_title(f"Confidence Distribution — {model_name}")
    ax.legend()
    return _save(fig, f"confidence_distribution_{model_name.lower().replace(' ', '_')}")


# ─────────────────────────────────────────────
# 2. Risk–coverage curve
# ─────────────────────────────────────────────

def plot_risk_coverage_curve(
    sweep_records_per_strategy: dict,  # {strategy_name: list[dict]}
    model_name: str = "Model",
) -> Path:
    fig, ax = plt.subplots(figsize=(7, 5))
    for (strategy, records), color in zip(
        sweep_records_per_strategy.items(), PALETTE
    ):
        valid = [(r["coverage"], r["selective_risk"])
                 for r in records if not np.isnan(r["selective_risk"])]
        if not valid:
            continue
        cov, risk = zip(*sorted(valid))
        ax.plot(cov, risk, lw=2, color=color,
                label=strategy.replace("_", " ").title())

    ax.set_xlabel("Coverage")
    ax.set_ylabel("Selective Risk (error rate)")
    ax.set_title(f"Risk–Coverage Curves — {model_name}")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(bottom=0)
    ax.legend()
    return _save(fig, f"risk_coverage_{model_name.lower().replace(' ', '_')}")


# ─────────────────────────────────────────────
# 3. Reliability diagram
# ─────────────────────────────────────────────

def plot_reliability_diagram(
    reliability_data_per_model: dict,
    title: str = "Reliability Diagram",
) -> Path:
    n = len(reliability_data_per_model)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, (name, rdata), color in zip(axes, reliability_data_per_model.items(), PALETTE):
        frac = rdata["fraction_of_positives"]
        mean = rdata["mean_predicted_value"]
        ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Perfect calibration")
        ax.plot(mean, frac, "o-", color=color, lw=1.8, ms=5, label=name)
        ax.fill_between(mean, mean, frac, alpha=0.12, color=color)
        ax.set_xlabel("Mean predicted probability")
        ax.set_title(name)
        ax.legend(loc="upper left")

    axes[0].set_ylabel("Fraction of positives")
    fig.suptitle(title, fontsize=12, y=1.02)
    return _save(fig, "reliability_diagram")


# ─────────────────────────────────────────────
# 4. Distribution-shift comparison
# ─────────────────────────────────────────────

def plot_distribution_shift_comparison(
    shift_results: dict,   # {shift_name: {strategy: metric_value}}
    metric_key: str = "selective_risk",
    model_name: str = "Model",
) -> Path:
    strategies = list(next(iter(shift_results.values())).keys())
    shift_names = list(shift_results.keys())
    x = np.arange(len(shift_names))
    width = 0.22
    offsets = np.linspace(-(len(strategies) - 1) / 2, (len(strategies) - 1) / 2, len(strategies)) * width

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (strategy, color, off) in enumerate(zip(strategies, PALETTE, offsets)):
        vals = [shift_results[sn].get(strategy, {}).get(metric_key, np.nan)
                for sn in shift_names]
        ax.bar(x + off, vals, width, label=strategy.replace("_", " ").title(),
               color=color, alpha=0.85, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", "\n") for s in shift_names], fontsize=9)
    ax.set_ylabel(metric_key.replace("_", " ").title())
    ax.set_title(f"Distribution Shift — {metric_key.replace('_', ' ').title()} | {model_name}")
    ax.legend(ncol=2)
    return _save(fig, f"dist_shift_{metric_key}_{model_name.lower().replace(' ', '_')}")


# ─────────────────────────────────────────────
# 5. Cost comparison bar chart
# ─────────────────────────────────────────────

def plot_cost_comparison(
    cost_data: dict,   # {model_name: {strategy: per_sample_cost}}
) -> Path:
    models = list(cost_data.keys())
    strategies = list(next(iter(cost_data.values())).keys())
    x = np.arange(len(models))
    width = 0.18
    offsets = np.linspace(-(len(strategies) - 1) / 2, (len(strategies) - 1) / 2, len(strategies)) * width

    fig, ax = plt.subplots(figsize=(9, 5))
    for strategy, color, off in zip(strategies, PALETTE, offsets):
        vals = [cost_data[m].get(strategy, np.nan) for m in models]
        ax.bar(x + off, vals, width, label=strategy.replace("_", " ").title(),
               color=color, alpha=0.85, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Per-sample operational cost")
    ax.set_title("Operational Cost Comparison by Model & Strategy")
    ax.legend(ncol=2)
    return _save(fig, "cost_comparison")


# ─────────────────────────────────────────────
# 6. Threshold sensitivity
# ─────────────────────────────────────────────

def plot_threshold_sensitivity(
    sweep_records: list[dict],
    model_name: str = "Model",
    strategy: str = "static",
) -> Path:
    valid = [r for r in sweep_records if not np.isnan(r["selective_risk"])]
    thresholds  = [r["threshold"] for r in valid]
    risks       = [r["selective_risk"] for r in valid]
    coverages   = [r["coverage"] for r in valid]
    abstentions = [r["abstention_rate"] for r in valid]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    ax1.plot(thresholds, risks, color=PALETTE[2], lw=2, label="Selective risk")
    ax1.set_ylabel("Selective Risk")
    ax1.set_title(f"Threshold Sensitivity — {strategy.title()} | {model_name}")
    ax1_twin = ax1.twinx()
    ax1_twin.plot(thresholds, coverages, color=PALETTE[0], lw=1.8, ls="--", label="Coverage")
    ax1_twin.set_ylabel("Coverage", color=PALETTE[0])
    ax1_twin.tick_params(axis="y", labelcolor=PALETTE[0])
    ax1.legend(loc="upper left"); ax1_twin.legend(loc="upper right")

    ax2.plot(thresholds, abstentions, color=PALETTE[3], lw=2)
    ax2.set_ylabel("Abstention Rate")
    ax2.set_xlabel("Confidence Threshold τ")

    fig.tight_layout()
    return _save(fig, f"threshold_sensitivity_{strategy}_{model_name.lower().replace(' ', '_')}")


# ─────────────────────────────────────────────
# 7. Coverage vs F1 pareto
# ─────────────────────────────────────────────

def plot_coverage_f1_pareto(
    sweep_records_per_model: dict,
    model_name: str = "Model",
) -> Path:
    fig, ax = plt.subplots(figsize=(7, 5))
    for (label, records), color in zip(sweep_records_per_model.items(), PALETTE):
        valid = [(r["coverage"], r.get("selective_f1", np.nan)) for r in records
                 if not np.isnan(r.get("selective_f1", np.nan))]
        if not valid:
            continue
        cov, f1 = zip(*valid)
        ax.plot(cov, f1, lw=2, color=color,
                label=label.replace("_", " ").title())

    ax.set_xlabel("Coverage")
    ax.set_ylabel("Selective F1")
    ax.set_title(f"Coverage–F1 Trade-off — {model_name}")
    ax.legend()
    return _save(fig, f"coverage_f1_{model_name.lower().replace(' ', '_')}")


# ─────────────────────────────────────────────
# 8. Tri-action action distribution pie
# ─────────────────────────────────────────────

def plot_action_distribution(
    actions: list[str],
    model_name: str = "Model",
) -> Path:
    from collections import Counter
    counts = Counter(actions)
    labels = [k.title() for k in counts.keys()]
    sizes  = list(counts.values())
    colors = [PALETTE[0], PALETTE[2], PALETTE[3]][:len(labels)]

    fig, ax = plt.subplots(figsize=(5, 5))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors,
        autopct="%1.1f%%", startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    for at in autotexts:
        at.set_fontsize(10)
    ax.set_title(f"Tri-Action Distribution — {model_name}")
    return _save(fig, f"action_dist_{model_name.lower().replace(' ', '_')}")
