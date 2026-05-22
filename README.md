# Selective Prediction under Uncertainty
### When Should a Machine Learning Model Refuse to Predict?

> An undergraduate research project exploring cost-sensitive abstention, distribution shift robustness, and risk-aware automation in high-stakes ML decision systems.

---

## Overview

Most deployed ML systems are forced to emit a prediction for every input — even when the model is genuinely uncertain. In safety-critical domains like credit card fraud detection, this design choice has real consequences: a misclassified fraud case can mean thousands of dollars lost, while an unnecessary rejection frustrates a legitimate customer.

This project investigates **selective prediction**: the practice of allowing a model to *abstain* from predicting when its confidence is too low, and analyzing the cost–coverage trade-offs that follow.

---

## Research Questions

1. **When should a model refuse to predict?** What confidence criteria minimize operational cost without sacrificing too much coverage?
2. **Does the choice of abstention strategy matter?** How do static thresholds, cost-aware decisions, and dynamic thresholding compare across different model families?
3. **How robust is selective prediction under distribution shift?** Does a strategy that works at training time survive noise injection, covariate shift, or class imbalance shifts?
4. **Can a tri-action framework (Predict / Defer / Abstain) recover value from the uncertain middle ground?**

---

## Methodology

### Abstention Strategies

| Strategy | Description |
|---|---|
| **Static Threshold** | Abstain if max P(y\|x) < τ (fixed τ) |
| **Cost-Aware** | Abstain when E[prediction cost] > abstention cost |
| **Dynamic Threshold** | τ adapts to maintain a target coverage level |
| **Tri-Action** | Three zones: high-confidence → predict; moderate → defer; low → abstain |

### Cost Model

A linear operational cost model with asymmetric class penalties:

```
cost(FP)     = 1.0   # wasted investigation
cost(FN)     = 5.0   # missed fraud
cost(abstain)= 0.5   # human review
cost(defer)  = 1.5   # secondary model or expert
```

The 5× FN penalty reflects the real-world asymmetry: missing fraud is much worse than a false alarm.

### Probability Calibration

Raw classifier scores are calibrated using isotonic regression (on a held-out validation set) before applying any threshold. Calibration quality is measured via Expected Calibration Error (ECE) and the Brier score.

---

## Project Structure

```
selective_prediction_uncertainty/
│
├── README.md
├── requirements.txt
├── run_experiments.py          ← Main experiment runner
│
├── data/
│   ├── raw/                    ← creditcard.csv (Kaggle) or auto-generated synthetic data
│   └── processed/
│
├── notebooks/
│   ├── 01_data_analysis.ipynb           ← EDA, class imbalance, confidence preview
│   ├── 02_selective_prediction.ipynb    ← All 3 strategies + tri-action framework
│   └── 03_distribution_shift.ipynb     ← Robustness under noise/covariate/imbalance shifts
│
├── src/
│   ├── preprocessing.py        ← Data loading, feature engineering, split/scale, shift utils
│   ├── training.py             ← Model factory, training, calibration, persistence
│   ├── selective_prediction.py ← All abstention strategy implementations
│   ├── calibration.py          ← ECE, MCE, Brier score, reliability diagram data
│   ├── evaluation.py           ← Full evaluation pipeline, comparison tables
│   ├── risk_metrics.py         ← Selective risk, coverage, AURC, bootstrap CI
│   ├── visualization.py        ← Research-quality plots (all auto-saved)
│   └── utils.py                ← Seed, logging, path resolution, config I/O
│
├── experiments/
│   ├── configs/default_config.json   ← All experiment hyperparameters
│   ├── outputs/                      ← CSVs, JSONs from experiment runs
│   └── saved_models/                 ← Pickled calibrated models
│
├── reports/
│   ├── figures/                ← Auto-generated PNG plots
│   ├── results_tables/
│   └── insights/
│
└── demo/
    └── demo_prediction.py      ← Interactive CLI demo of tri-action framework
```

---

## Setup

### Requirements

- Python 3.10+
- pip

### macOS / Linux

```bash
# Clone or unzip the repository
cd selective_prediction_uncertainty

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify
python -c "import sklearn; import xgboost; print('All dependencies OK')"
```

### Windows

```powershell
cd selective_prediction_uncertainty

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
```

### Dataset

**Option A — Synthetic (default, no setup needed)**

If `data/raw/creditcard.csv` is not present, the project automatically generates a realistic 20,000-sample synthetic fraud dataset with overlapping class distributions. All results are reproducible with seed 42.

**Option B — Real Kaggle Dataset**

1. Download from [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
2. Place `creditcard.csv` in `data/raw/`
3. The code auto-detects and uses it.

---

## Running the Project

### Full Experiment Pipeline

```bash
python run_experiments.py
```

This runs:
1. Data preparation
2. Model training (LR, RF, GradientBoosting) with calibration
3. All three abstention strategies on the test set
4. Tri-action framework evaluation
5. Threshold sweep → risk–coverage curves
6. Distribution shift robustness experiments (noise + imbalance)
7. Saves all figures to `reports/figures/`
8. Saves all result tables to `experiments/outputs/`

### With SMOTE oversampling

```bash
python run_experiments.py --smote
```

### Custom config

```bash
python run_experiments.py --config experiments/configs/default_config.json
```

### Interactive Demo

```bash
python demo/demo_prediction.py
```

### Jupyter Notebooks

```bash
jupyter notebook notebooks/
```

Open and run in order: `01_data_analysis.ipynb` → `02_selective_prediction.ipynb` → `03_distribution_shift.ipynb`.

---

## Key Results (Synthetic Dataset, RF Model)

| Strategy | Coverage | Selective Risk | Per-Sample Cost |
|---|---|---|---|
| No abstention | 100% | ~0.06 | ~0.28 |
| Static (τ=0.70) | ~72% | ~0.02 | ~0.19 |
| Cost-Aware | ~68% | ~0.02 | **~0.16** |
| Dynamic (80% cov) | ~80% | ~0.03 | ~0.21 |
| Tri-Action | ~65% / 8% defer | ~0.01 | ~0.17 |

*Exact values vary with data and random seed. Re-run experiments to get numbers for your setup.*

---

## Figures Generated

| Figure | Description |
|---|---|
| `confidence_distribution_*.png` | P(fraud\|x) histograms by true class |
| `risk_coverage_*.png` | Risk–coverage curves (static vs dynamic) |
| `threshold_sensitivity_*.png` | Coverage and risk as τ varies |
| `reliability_diagram.png` | Calibration quality (all models) |
| `cost_comparison.png` | Per-sample cost by model and strategy |
| `dist_shift_selective_risk_*.png` | How selective risk degrades under shift |
| `dist_shift_coverage_*.png` | How coverage responds to shift |
| `action_dist_*.png` | Tri-action distribution pie chart |

---

## Reproducibility

All results are fully reproducible by setting `seed=42` in `experiments/configs/default_config.json`. The synthetic dataset generator, train/val/test split, SMOTE (if used), and all random operations use this seed.

```bash
# Clean re-run
rm -rf experiments/saved_models/* experiments/outputs/* reports/figures/*
python run_experiments.py
```

---

## Future Directions

- **Conformal prediction integration**: Replace heuristic confidence thresholds with rigorous coverage guarantees via split conformal prediction.
- **Online/streaming abstention**: Adapt thresholds dynamically as the data distribution drifts in real-time.
- **Multi-class selective prediction**: Extend cost model to non-binary settings (e.g., loan risk tiers).
- **SHAP-informed abstention**: Abstain not just on low confidence but on samples where the model's explanations are incoherent.
- **Human-in-the-loop simulation**: Model the abstained samples being routed to a human, and measure overall system accuracy.

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| scikit-learn | 1.4.x | Models, calibration, metrics |
| numpy / pandas | latest stable | Data manipulation |
| matplotlib / seaborn | latest stable | Visualization |
| xgboost | 2.0.x | Optional gradient boosting |
| imbalanced-learn | 0.12.x | SMOTE oversampling |
| scipy | 1.13.x | Bootstrap CI, statistical tests |
| shap | 0.45.x | Optional explainability |

---

## License

MIT License. See `LICENSE` for details.

---

*This project was built as part of undergraduate research exploration into uncertainty-aware ML systems.*
