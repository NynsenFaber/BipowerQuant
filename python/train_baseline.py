"""
Train the Logistic Regression baseline on Order Flow Imbalance alone.

    cd python
    python train_baseline.py --bars-cache ../data/bars_full.npz

This is the control, not a contender. It sees **one** number per window — OFI —
so whatever it scores is what pure order-flow momentum is worth on this target,
and the gap between it and XGBoost is what the jump-diffusion features bought.
Pass `--all-features` to give it the full 7-feature matrix instead, which is the
right comparison if you want to know how much of XGBoost's score needs a
non-linear model rather than just the extra inputs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import sequence_matrix as seq
from data_feeder import FILE_PATH
from ml_matrix import FEATURE_NAMES, build_from_csv
from train_xgboost import log_results, purged_split

# Column 3 of the 7-feature matrix; see sequence_matrix.TABULAR_FEATURE_NAMES.
OFI_COLUMN = FEATURE_NAMES.index("Order Flow Imbalance")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=FILE_PATH)
    parser.add_argument("--bars-cache", default=None, help="reuse/write a .npz second-bar cache")
    parser.add_argument("--hours", type=float, default=None, help="None = the whole file")
    parser.add_argument("--label-mode", default="triple_barrier", choices=seq.LABEL_MODES)
    parser.add_argument("--all-features", action="store_true", help="use all 7, not just OFI")
    parser.add_argument("--no-log", action="store_true")
    args = parser.parse_args()

    print("1. Building the feature matrix ...")
    X_full, y, starts, bars = build_from_csv(
        args.csv, hours=args.hours, cache=args.bars_cache, label_mode=args.label_mode
    )
    n_windows = seq.valid_window_starts(bars["price"].size).size

    if args.all_features:
        X, description = X_full, f"all {X_full.shape[1]} features"
    else:
        X, description = X_full[:, [OFI_COLUMN]], "Order Flow Imbalance only"
    print(f"   {X.shape[0]:,} windows | {description} | positive rate {y.mean():.2%}")

    print("2. Splitting chronologically, purging the boundaries ...")
    masks = purged_split(starts, n_windows)
    X_train, y_train = X[masks["train"]], y[masks["train"]]
    X_test, y_test = X[masks["test"]], y[masks["test"]]
    print(f"   train {len(y_train):,} | test {len(y_test):,}")

    print("3. Training Logistic Regression ...")
    # The features span ~20 orders of magnitude (RV is ~1e-7, OFI is ~1e2), so an
    # unscaled solver either fails to converge or converges to the wrong place.
    # Scaling is monotone per feature, so it cannot change a single-feature AUC.
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=1000),
    )
    model.fit(X_train, y_train)

    print("4. Scoring ...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)

    ci = seq.block_bootstrap_auc(y_test, y_prob)

    print("\n✅ Logistic Regression evaluated")
    print("=========================================")
    print(
        f"ROC-AUC:   {roc_auc:.4f}   95% CI [{ci['lo']:.4f}, {ci['hi']:.4f}], "
        f"P(<=0.5) = {ci['p_le_half']:.3f}"
    )
    print(f"Precision: {precision:.4f}   (base rate: {y_test.mean():.4f})")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"Accuracy:  {acc:.4f}   (always-0 baseline: {1 - y_test.mean():.4f})")
    print("=========================================")

    if not args.no_log:
        extra = (
            f"Input: {description} | Label: {args.label_mode} at "
            f"+/-{seq.BARRIER:.4%} over {seq.HORIZON}s\n"
            f"ROC-AUC 95% CI: [{ci['lo']:.4f}, {ci['hi']:.4f}] | "
            f"P(AUC <= 0.5) = {ci['p_le_half']:.3f}\n"
            f"Precision: {precision:.4f} | Recall: {recall:.4f} | "
            f"Test windows: {len(y_test):,}"
        )
        log_results(f"Logistic Regression ({description}, triple barrier)", acc, f1, roc_auc, extra)
        print(f"\nAppended to {Path(__file__).with_name('training_logs.txt')}")


if __name__ == "__main__":
    main()
