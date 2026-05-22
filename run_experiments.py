"""
run_experiments.py
------------------
End-to-end experiment runner.

Usage:
    python run_experiments.py
    python run_experiments.py --config experiments/configs/default_config.json
    python run_experiments.py --smote
"""

import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# Allow importing from project root
sys.path.insert(0, str(Path(__file__).parent))

from src.utils import set_seed, get_logger, get_outputs_dir, load_config
from src.preprocessing import (
    load_raw_dataset, engineer_features, split_and_scale,
    inject_noise, perturb_features, imbalance_shift,
)
from src.training import build_model, train_model, save_model, load_model
from src.calibration import calibration_summary, reliability_diagram_data
from src.selective_prediction import (
    threshold_sweep, static_threshold_selection,
    cost_aware_selection, dynamic_threshold_selection,
    tri_action_selection,
)
from src.evaluation import (
    evaluate_baseline, evaluate_strategy, evaluate_model_all_strategies,
    build_comparison_table,
)
from src.risk_metrics import expected_operational_cost, full_prediction_cost
from src.visualization import (
    plot_confidence_distribution,
    plot_risk_coverage_curve,
    plot_reliability_diagram,
    plot_distribution_shift_comparison,
    plot_cost_comparison,
    plot_threshold_sensitivity,
    plot_action_distribution,
)

logger = get_logger("run_experiments")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        default="experiments/configs/default_config.json",
        help="Path to JSON experiment config",
    )
    p.add_argument("--smote", action="store_true", help="Apply SMOTE to training set")
    p.add_argument("--no-save", action="store_true", help="Skip saving models")
    return p.parse_args()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    args = parse_args()
    cfg  = load_config(args.config)
    seed = cfg.get("seed", 42)
    set_seed(seed)

    sp_cfg = cfg["selective_prediction"]
    cost_fp      = sp_cfg["cost_fp"]
    cost_fn      = sp_cfg["cost_fn"]
    cost_abstain = sp_cfg["cost_abstain"]
    cost_defer   = sp_cfg["cost_defer"]

    # ── 1. Data ───────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 1: Data preparation")
    df = load_raw_dataset()
    df = engineer_features(df)
    splits = split_and_scale(
        df,
        apply_smote=args.smote or cfg["dataset"].get("apply_smote", False),
        random_state=seed,
    )
    logger.info(
        f"  Train: {splits['X_train'].shape[0]} | "
        f"Val: {splits['X_val'].shape[0]} | "
        f"Test: {splits['X_test'].shape[0]}"
    )
    logger.info(f"  Fraud rate (test): {splits['y_test'].mean():.3%}")

    # ── 2. Train models ───────────────────────────────────────────────────────
    logger.info("STEP 2: Training models")
    model_names = cfg.get("models", ["lr", "rf", "gb"])
    calibrate   = cfg.get("calibrate", True)

    trained_models = {}
    for mname in model_names:
        model = build_model(mname)
        fitted = train_model(
            model,
            splits["X_train"], splits["y_train"],
            splits["X_val"],   splits["y_val"],
            calibrate=calibrate,
        )
        if not args.no_save:
            save_model(fitted, mname)
        trained_models[mname] = fitted

    # ── 3. Full evaluation ────────────────────────────────────────────────────
    logger.info("STEP 3: Selective prediction evaluation")
    all_results = {}
    for mname, model in trained_models.items():
        all_results[mname] = evaluate_model_all_strategies(
            model,
            splits["X_test"],
            splits["y_test"],
            static_threshold = sp_cfg["static_threshold"],
            target_coverage  = sp_cfg["target_coverage"],
            cost_fp=cost_fp, cost_fn=cost_fn,
            cost_abstain=cost_abstain, cost_defer=cost_defer,
        )

    # ── 4. Results tables ─────────────────────────────────────────────────────
    logger.info("STEP 4: Saving results tables")
    out_dir = get_outputs_dir()
    table = build_comparison_table(all_results)
    table.to_csv(out_dir / "comparison_table.csv", index=False)
    logger.info(f"\n{table.to_string(index=False)}\n")

    # Per-model JSON dumps
    for mname, res in all_results.items():
        subset = {
            k: v for k, v in res.items()
            if k not in ("sweep_static", "sweep_dynamic")
        }
        with open(out_dir / f"results_{mname}.json", "w") as f:
            json.dump(subset, f, indent=2, default=str)

    # ── 5. Visualizations ─────────────────────────────────────────────────────
    logger.info("STEP 5: Generating figures")

    # 5a. Confidence distributions
    for mname, model in trained_models.items():
        proba = model.predict_proba(splits["X_test"])[:, 1]
        plot_confidence_distribution(
            proba, splits["y_test"], mname,
            threshold=sp_cfg["static_threshold"]
        )

    # 5b. Risk–coverage curves (per model, combining static & dynamic)
    for mname, res in all_results.items():
        plot_risk_coverage_curve(
            {"static": res["sweep_static"], "dynamic": res["sweep_dynamic"]},
            model_name=mname,
        )

    # 5c. Threshold sensitivity
    for mname, res in all_results.items():
        plot_threshold_sensitivity(
            res["sweep_static"], model_name=mname, strategy="static"
        )

    # 5d. Calibration / reliability diagrams
    rel_data = {}
    for mname, model in trained_models.items():
        proba = model.predict_proba(splits["X_test"])[:, 1]
        rel_data[mname] = reliability_diagram_data(splits["y_test"], proba)
    plot_reliability_diagram(rel_data)

    # 5e. Cost comparison
    cost_data = {}
    for mname, res in all_results.items():
        cost_data[mname] = {
            "no abstention": res["baseline"]["per_sample_cost"],
            "static":        res["static"]["per_sample_cost"],
            "cost_aware":    res["cost_aware"]["per_sample_cost"],
            "dynamic":       res["dynamic"]["per_sample_cost"],
        }
    plot_cost_comparison(cost_data)

    # 5f. Tri-action distribution (use first model)
    first_model = trained_models[model_names[0]]
    tri_res = tri_action_selection(
        first_model, splits["X_test"],
        threshold_predict=sp_cfg["tri_action"]["threshold_predict"],
        threshold_defer=sp_cfg["tri_action"]["threshold_defer"],
    )
    plot_action_distribution(tri_res.actions, model_name=model_names[0])

    # ── 6. Distribution shift ─────────────────────────────────────────────────
    logger.info("STEP 6: Distribution shift robustness experiments")
    shift_cfg  = cfg.get("distribution_shift", {})
    noise_stds = shift_cfg.get("noise_stds", [0.0, 0.5, 1.0, 2.0])

    shift_results = {}    # {shift_label: {strategy: {"selective_risk": …}}}
    X_test_base = splits["X_test"]
    y_test      = splits["y_test"]

    for noise in noise_stds:
        label = f"noise={noise}"
        X_shifted = inject_noise(X_test_base, noise_std=noise, seed=seed)
        row = {}
        for strategy_name, selector_fn in [
            ("static",     lambda m, X: static_threshold_selection(m, X, threshold=0.70)),
            ("cost_aware", lambda m, X: cost_aware_selection(m, X, cost_fp=cost_fp, cost_fn=cost_fn, cost_abstain=cost_abstain)),
            ("dynamic",    lambda m, X: dynamic_threshold_selection(m, X, target_coverage=0.80)),
        ]:
            # Use first trained model for shift experiments
            res = selector_fn(first_model, X_shifted)
            strat_eval = evaluate_strategy(res, y_test, cost_fp, cost_fn, cost_abstain)
            row[strategy_name] = strat_eval
        shift_results[label] = row

    # Imbalance shift
    imb_ratios = shift_cfg.get("imbalance_ratios", [0.05, 0.10, 0.20])
    for ratio in imb_ratios:
        label = f"imbalance={ratio:.0%}"
        X_sh, y_sh = imbalance_shift(X_test_base, y_test, target_fraud_ratio=ratio, seed=seed)
        row = {}
        for strategy_name, selector_fn in [
            ("static",     lambda m, X: static_threshold_selection(m, X, threshold=0.70)),
            ("cost_aware", lambda m, X: cost_aware_selection(m, X, cost_fp=cost_fp, cost_fn=cost_fn, cost_abstain=cost_abstain)),
            ("dynamic",    lambda m, X: dynamic_threshold_selection(m, X, target_coverage=0.80)),
        ]:
            res = selector_fn(first_model, X_sh)
            strat_eval = evaluate_strategy(res, y_sh, cost_fp, cost_fn, cost_abstain)
            row[strategy_name] = strat_eval
        shift_results[label] = row

    # Save shift results
    with open(out_dir / "distribution_shift_results.json", "w") as f:
        json.dump(shift_results, f, indent=2, default=str)

    # Plot shift comparison
    plot_distribution_shift_comparison(
        shift_results, metric_key="selective_risk",
        model_name=model_names[0]
    )
    plot_distribution_shift_comparison(
        shift_results, metric_key="coverage",
        model_name=model_names[0]
    )

    # ── 7. Summary ────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("EXPERIMENT COMPLETE")
    logger.info(f"  Results  → {out_dir}")
    from src.utils import get_reports_dir
    logger.info(f"  Figures  → {get_reports_dir('figures')}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
