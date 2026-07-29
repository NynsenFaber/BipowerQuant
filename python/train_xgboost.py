"""
Train the XGBoost baseline on the 7-feature jump-diffusion matrix.

    cd python
    python train_xgboost.py --bars-cache ../data/bars_full.npz

Uses the same purged chronological split as the PatchTST pipeline, so the two
sets of numbers are comparable: identical bars, identical windows, identical
triple-barrier labels.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import sequence_matrix as seq
from data_feeder import FILE_PATH
from ml_matrix import FEATURE_NAMES, build_from_csv

LOG_PATH = Path(__file__).with_name("training_logs.txt")


def log_results(model_name: str, acc: float, f1: float, roc_auc: float, extra: str = "") -> None:
    with open(LOG_PATH, "a") as handle:
        handle.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {model_name}\n")
        handle.write(f"Accuracy: {acc:.4f} | F1: {f1:.4f} | ROC-AUC: {roc_auc:.4f}\n")
        if extra:
            handle.write(f"{extra}\n")
        handle.write("-" * 50 + "\n")


def purged_split(
    starts: np.ndarray,
    n_windows: int,
    window: int = seq.WINDOW_SIZE,
    horizon: int = seq.HORIZON,
    train_frac: float = 0.7,
    val_frac: float = 0.1,
) -> dict[str, np.ndarray]:
    """Boolean masks over `starts`, reproducing `seq.chronological_split` exactly.

    Windows overlap by `window - 1` bars and are labelled up to `horizon` bars
    past their end, so a naive cut puts training labels inside the next split's
    lookback. Dropping `window + horizon - 1` windows before each boundary
    removes the overlap completely.

    Two details make this match the sequence pipeline rather than merely
    resemble it, and both matter for the numbers to be comparable:

    * `n_windows` is the **unfiltered** window count, not `starts.size`. The
      triple barrier resolves only ~21% of windows, so deriving the cut from the
      survivors would put the boundaries at different moments in time.
    * The trees are given the 70% training split only, leaving the validation
      block unused. They do not need it — there is no early stopping — but
      handing them the extra 10% would mean beating PatchTST on data volume
      rather than on modelling.
    """
    purge = window + horizon - 1
    train_end = int(n_windows * train_frac)
    val_end = int(n_windows * (train_frac + val_frac))
    return {
        "train": starts < train_end - purge,
        "val": (starts >= train_end) & (starts < val_end - purge),
        "test": starts >= val_end,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=FILE_PATH)
    parser.add_argument("--bars-cache", default=None, help="reuse/write a .npz second-bar cache")
    parser.add_argument("--hours", type=float, default=None, help="None = the whole file")
    parser.add_argument("--label-mode", default="triple_barrier", choices=seq.LABEL_MODES)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--no-log", action="store_true")
    args = parser.parse_args()

    print("1. Building the feature matrix ...")
    X, y, starts, bars = build_from_csv(
        args.csv, hours=args.hours, cache=args.bars_cache, label_mode=args.label_mode
    )
    n_windows = seq.valid_window_starts(bars["price"].size).size
    print(
        f"   {X.shape[0]:,} of {n_windows:,} windows resolved by a barrier "
        f"({X.shape[0] / n_windows:.1%}) | positive rate {y.mean():.2%}"
    )

    print("2. Splitting chronologically, purging the boundaries ...")
    masks = purged_split(starts, n_windows)
    X_train, y_train = X[masks["train"]], y[masks["train"]]
    X_test, y_test = X[masks["test"]], y[masks["test"]]
    print(
        f"   train {len(y_train):,} | val {int(masks['val'].sum()):,} (unused by the trees) "
        f"| test {len(y_test):,}"
    )

    n_pos, n_neg = int((y_train == 1).sum()), int((y_train == 0).sum())
    if n_pos == 0 or n_neg == 0:
        raise SystemExit("The training split has only one class — stream more tape.")
    imbalance_ratio = n_neg / n_pos

    print("3. Training XGBoost ...")
    print(f"   positive class: {n_pos:,} / {len(y_train):,} ({n_pos / len(y_train):.2%})")
    print(f"   scale_pos_weight: {imbalance_ratio:.3f}")

    model = xgb.XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        scale_pos_weight=imbalance_ratio,
        eval_metric="logloss",
        random_state=42,
        # Pinned: XGBoost's histogram reduction is order-dependent, so the same
        # data on a different core count gives a slightly different tree and an
        # AUC that moves in the third decimal. On a ~0.55 signal that is enough
        # to make two scripts disagree about the same model, so every fit in this
        # repo (here, benchmark_inference, make_figures) uses one thread.
        n_jobs=1,
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

    print("\n✅ XGBoost evaluated")
    print("=========================================")
    print(f"ROC-AUC:   {roc_auc:.4f}   95% CI [{ci['lo']:.4f}, {ci['hi']:.4f}], "
          f"P(<=0.5) = {ci['p_le_half']:.3f}")
    print(f"Precision: {precision:.4f}   (base rate: {y_test.mean():.4f})")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"Accuracy:  {acc:.4f}   (always-0 baseline: {1 - y_test.mean():.4f})")
    print("=========================================")

    print("\n📈 Precision in the confident tail:")
    order = np.argsort(-y_prob)
    for pct in (1, 5, 10, 25):
        k = max(int(pct * order.size / 100), 1)
        hit = y_test[order[:k]].mean()
        print(f"   top {pct:>2}%  {hit:.4f}   ({hit / max(y_test.mean(), 1e-12):.2f}x base rate)")

    print("\n🔍 Feature importance:")
    importances = model.feature_importances_
    for name, imp in sorted(zip(FEATURE_NAMES, importances), key=lambda p: -p[1]):
        print(f"{name:<25}: {imp:.4f}")

    if not args.no_log:
        importances_str = "\n".join(
            f"{name}: {imp:.4f}" for name, imp in zip(FEATURE_NAMES, importances)
        )
        extra = (
            f"Label: {args.label_mode} at +/-{seq.BARRIER:.4%} over {seq.HORIZON}s | "
            f"Positive rate (train): {n_pos / len(y_train):.2%} | "
            f"scale_pos_weight: {imbalance_ratio:.3f}\n"
            f"ROC-AUC 95% CI: [{ci['lo']:.4f}, {ci['hi']:.4f}] | "
            f"P(AUC <= 0.5) = {ci['p_le_half']:.3f}\n"
            f"Precision: {precision:.4f} | Recall: {recall:.4f} | "
            f"Test windows: {len(y_test):,}\n"
            f"Feature Importances:\n{importances_str}"
        )
        log_results("XGBoost (7-feature matrix, triple barrier)", acc, f1, roc_auc, extra)
        print(f"\nAppended to {LOG_PATH}")


if __name__ == "__main__":
    main()
