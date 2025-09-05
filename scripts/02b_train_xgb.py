# scripts/02b_train_xgb.py
import joblib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    classification_report, confusion_matrix
)
from xgboost import XGBClassifier

# paths
DATA_PATH = Path("data/raw/creditcard.csv")
ARTIFACTS_DIR = Path("artifacts")
MODEL_PATH = ARTIFACTS_DIR / "model_xgb.joblib"
SCALER_PATH = ARTIFACTS_DIR / "scaler_xgb.joblib"

RANDOM_STATE = 42

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    print(f"Loading dataset: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    # features/labels
    X = df.drop(columns=["Class"])
    y = df["Class"]

    # stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    # scale Time + Amount only
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[["Time", "Amount"]] = scaler.fit_transform(X_train[["Time", "Amount"]])
    X_test_scaled[["Time", "Amount"]] = scaler.transform(X_test[["Time", "Amount"]])

    # xgboost model
    clf = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=(len(y_train) - sum(y_train)) / sum(y_train),  # handle imbalance
        random_state=RANDOM_STATE,
        n_jobs=-1,
        use_label_encoder=False,
        eval_metric="logloss"
    )

    clf.fit(X_train_scaled, y_train)

    # eval
    probs = clf.predict_proba(X_test_scaled)[:, 1]
    preds = (probs >= 0.5).astype(int)

    roc = roc_auc_score(y_test, probs)
    pr = average_precision_score(y_test, probs)
    cm = confusion_matrix(y_test, preds)
    report = classification_report(y_test, preds, digits=4)

    print("\n=== Evaluation Metrics (XGBoost) ===")
    print(f"ROC-AUC: {roc:.4f}")
    print(f"PR-AUC : {pr:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)

    # save artifacts
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    print(f"\nModel saved at  : {MODEL_PATH}")
    print(f"Scaler saved at : {SCALER_PATH}")

if __name__ == "__main__":
    main()
