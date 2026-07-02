import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from data_feeder import get_lazy_feeder, stream_hourly_chunks, FILE_PATH
from ml_matrix import build_training_matrix

from datetime import datetime

def log_results(model_name, acc, f1, roc_auc, extra=""):
    with open("training_logs.txt", "a") as f:
        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {model_name}\n")
        f.write(f"Accuracy: {acc:.4f} | F1: {f1:.4f} | ROC-AUC: {roc_auc:.4f}\n")
        if extra:
            f.write(f"{extra}\n")
        f.write("-" * 50 + "\n")

def train_and_evaluate():
    print("1. Loading data and building feature matrix...")
    lazy_pipeline = get_lazy_feeder(FILE_PATH)
    # Stream a larger chunk (e.g., 4 hours) to get a solid training sample
    feeder = stream_hourly_chunks(lazy_pipeline, chunk_hours=4) 
    chunk = next(feeder)
    
    X_full, y_full = build_training_matrix(chunk)
    
    print("2. Isolating the OFI feature...")
    # In ml_matrix.py, X is stacked as (RV, BPV, Jumps, OFI)
    # OFI is at index 3
    X_ofi = X_full[:, 3].reshape(-1, 1)
    
    # Split chronologically (80% train, 20% test) to prevent time-series leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X_ofi, y_full, test_size=0.2, shuffle=False
    )
    
    print("3. Training Logistic Regression Baseline...")
    # class_weight='balanced' handles any imbalance between up/down directional ticks
    model = LogisticRegression(class_weight='balanced')
    model.fit(X_train, y_train)
    
    print("4. Calculating Metrics...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    print("\n✅ Baseline Model Evaluated Successfully")
    print("=========================================")
    print(f"Accuracy:  {acc:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print("=========================================")

    # store the logs
    log_results("Logistic Regression (Baseline)", acc, f1, roc_auc)

if __name__ == "__main__":
    train_and_evaluate()