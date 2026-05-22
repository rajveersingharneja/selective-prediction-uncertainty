
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import get_logger, set_seed
from src.preprocessing import load_raw_dataset, engineer_features, split_and_scale
from src.training import build_model
from src.selective_prediction import tri_action_selection, static_threshold_selection

logger = get_logger("demo")


def main():
    set_seed(42)
    print("\n" + "=" * 65)
    print("   Selective Prediction Demo — Credit Card Fraud Detection")
    print("=" * 65)

    print("\n[1/3] Preparing data and training demo model ...")
    df = load_raw_dataset()
    df = engineer_features(df)
    splits = split_and_scale(df)

    model = build_model("rf")
    model.fit(splits["X_train"], splits["y_train"])
    print("      ✓ Random Forest trained.")

    # Pick a balanced set: 5 legit + 5 fraud
    y_test = splits["y_test"]
    X_test = splits["X_test"]
    rng = np.random.default_rng(7)
    idx_legit = rng.choice(np.where(y_test == 0)[0], size=5, replace=False)
    idx_fraud = rng.choice(np.where(y_test == 1)[0], size=5, replace=False)
    idx = np.concatenate([idx_legit, idx_fraud])
    rng.shuffle(idx)

    X_demo = X_test[idx]
    y_demo = y_test[idx]

    print("\n[2/3] Running tri-action selective prediction on 10 samples (5 legit + 5 fraud) ...")

    result = tri_action_selection(
        model, X_demo,
        threshold_predict=0.80,
        threshold_defer=0.40,
    )

    print("\n[3/3] Results:\n")
    print(f"  {'#':<4} {'True':<10} {'Predicted':<12} {'Action':<10} {'P(fraud)':>10}")
    print("  " + "-" * 52)
    for i, (action, conf, pred, true) in enumerate(
        zip(result.actions, result.confidences, result.predictions, y_demo)
    ):
        true_lbl = "FRAUD" if true == 1 else "LEGIT"
        if action == "predict":
            pred_lbl = "FRAUD" if pred == 1 else "LEGIT"
        else:
            pred_lbl = f"--{action}--"
        marker = " ✓" if (action == "predict" and pred == true) else (" ✗" if action == "predict" else "  ")
        print(f"  {i+1:<4} {true_lbl:<10} {pred_lbl:<12} {action.upper():<10} {conf:>10.3f}{marker}")

    print(f"\n  Summary:")
    print(f"    Predicted (auto):  {result.coverage:.0%}")
    print(f"    Deferred (review): {result.defer_rate:.0%}")
    print(f"    Abstained (reject):{result.abstention_rate:.0%}")

    static_res = static_threshold_selection(model, X_demo, threshold=0.70)
    print(f"\n  vs Static threshold (τ=0.70):")
    print(f"    Predicted: {static_res.coverage:.0%}  |  Abstained: {static_res.abstention_rate:.0%}")
    print(f"\n  Tri-action routes uncertain samples to DEFER rather than")
    print(f"  discarding them, preserving value from the 'grey zone'.")
    print("\n" + "=" * 65 + "\n")


if __name__ == "__main__":
    main()
