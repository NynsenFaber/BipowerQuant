import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

# Import from your existing pipeline
from data_feeder import get_lazy_feeder, stream_hourly_chunks, FILE_PATH
from ml_matrix import build_training_matrix, FEATURE_NAMES, FEE_THRESHOLD

from datetime import datetime

def log_results(model_name, acc, f1, roc_auc, extra=""):
    with open("training_logs.txt", "a") as f:
        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {model_name}\n")
        f.write(f"Accuracy: {acc:.4f} | F1: {f1:.4f} | ROC-AUC: {roc_auc:.4f}\n")
        if extra:
            f.write(f"{extra}\n")
        f.write("-" * 50 + "\n")

def train_xgboost():
    print("1. Loading data and building full feature matrix...")
    lazy_pipeline = get_lazy_feeder(FILE_PATH)
    # Stream 8 hours to give the trees more data to split on
    feeder = stream_hourly_chunks(lazy_pipeline, chunk_hours=8)
    chunk = next(feeder)

    X, y = build_training_matrix(chunk)
    print(f"   Matrix: {X.shape[0]:,} rows x {X.shape[1]} features")

    print("2. Splitting data chronologically...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    print("3. Training XGBoost Classifier...")
    # The profit threshold makes the positive class rare, so scale_pos_weight is
    # recomputed from the actual train split rather than assumed near 1.0
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    if n_pos == 0:
        raise ValueError(
            f"No profitable moves (> {FEE_THRESHOLD:.4%}) in the training split. "
            "Lower FEE_THRESHOLD or stream a more volatile chunk."
        )
    imbalance_ratio = n_neg / n_pos

    print(f"   Positive class: {n_pos:,} / {len(y_train):,} ({n_pos / len(y_train):.2%})")
    print(f"   scale_pos_weight: {imbalance_ratio:.3f}")

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=imbalance_ratio,
        eval_metric='logloss',
        random_state=42
    )

    model.fit(X_train, y_train)

    print("4. Calculating Metrics...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)

    print("\n✅ XGBoost Model Evaluated Successfully")
    print("=========================================")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print("=========================================")

    print("\n🔍 Feature Importance:")
    importances = model.feature_importances_
    for name, imp in zip(FEATURE_NAMES, importances):
        print(f"{name:<25}: {imp:.4f}")

    # Format the feature importances into a single string
    importances_str = "\n".join([f"{name}: {imp:.4f}" for name, imp in zip(FEATURE_NAMES, importances)])
    extra = (
        f"Target: forward 1m return > {FEE_THRESHOLD:.4%} | "
        f"Positive rate (train): {n_pos / len(y_train):.2%} | scale_pos_weight: {imbalance_ratio:.3f}\n"
        f"Precision: {precision:.4f} | Recall: {recall:.4f}\n"
        f"Feature Importances:\n{importances_str}"
    )
    log_results("XGBoost (Contextual Matrix + Fee Threshold)", acc, f1, roc_auc, extra=extra)

if __name__ == "__main__":
    train_xgboost()
