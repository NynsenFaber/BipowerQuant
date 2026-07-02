import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Import from your existing pipeline
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

def train_xgboost():
    print("1. Loading data and building full feature matrix...")
    lazy_pipeline = get_lazy_feeder(FILE_PATH)
    # Stream 8 hours to give the trees more data to split on
    feeder = stream_hourly_chunks(lazy_pipeline, chunk_hours=8) 
    chunk = next(feeder)
    
    X, y = build_training_matrix(chunk)
    
    print("2. Splitting data chronologically...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    
    print("3. Training XGBoost Classifier...")
    # Scale_pos_weight helps balance asymmetric market regimes
    imbalance_ratio = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
    
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
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    print("\n✅ XGBoost Model Evaluated Successfully")
    print("=========================================")
    print(f"Accuracy:  {acc:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print("=========================================")
    
    print("\n🔍 Feature Importance:")
    feature_names = ["Realized Variance", "Bipower Variation", "Jumps", "Order Flow Imbalance"]
    importances = model.feature_importances_
    for name, imp in zip(feature_names, importances):
        print(f"{name:<25}: {imp:.4f}")
    
    # Format the feature importances into a single string
    importances_str = "\n".join([f"{name}: {imp:.4f}" for name, imp in zip(feature_names, model.feature_importances_)])
    log_results("XGBoost (Full Matrix)", acc, f1, roc_auc, extra=f"Feature Importances:\n{importances_str}")

if __name__ == "__main__":
    train_xgboost()