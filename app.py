# app.py
from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from pydantic import BaseModel
from typing import List
import pandas as pd
import joblib
import io, base64
import matplotlib.pyplot as plt

# ================== Paths ==================
RF_MODEL_PATH = "artifacts/model_rf.joblib"
RF_SCALER_PATH = "artifacts/scaler.joblib"
XGB_MODEL_PATH = "artifacts/model_xgb.joblib"
XGB_SCALER_PATH = "artifacts/scaler_xgb.joblib"

FEATURE_ORDER = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
SCALE_COLS = ["Time", "Amount"]

# ================== Schemas ==================
class Transaction(BaseModel):
    Time: float
    Amount: float
    V1: float; V2: float; V3: float; V4: float; V5: float; V6: float; V7: float
    V8: float; V9: float; V10: float; V11: float; V12: float; V13: float; V14: float
    V15: float; V16: float; V17: float; V18: float; V19: float; V20: float; V21: float
    V22: float; V23: float; V24: float; V25: float; V26: float; V27: float; V28: float

class PredictRequest(BaseModel):
    transactions: List[Transaction]

# ================== Helpers ==================
def load_artifacts(path_model, path_scaler):
    model = joblib.load(path_model)
    scaler = joblib.load(path_scaler)
    return model, scaler

rf_model, rf_scaler = load_artifacts(RF_MODEL_PATH, RF_SCALER_PATH)
xgb_model, xgb_scaler = load_artifacts(XGB_MODEL_PATH, XGB_SCALER_PATH)

app = FastAPI(title="Credit Card Fraud Detection API", version="2.0")

@app.get("/health")
def health():
    return {"status": "ok", "models": ["rf", "xgb"]}

# ================== Single Transaction ==================
@app.post("/predict")
def predict(req: PredictRequest, model: str = Query("rf", enum=["rf", "xgb"])):
    try:
        # pick model
        if model == "rf":
            clf, scaler, model_name = rf_model, rf_scaler, "RandomForest"
        else:
            clf, scaler, model_name = xgb_model, xgb_scaler, "XGBoost"

        # dataframe
        df = pd.DataFrame([t.model_dump() for t in req.transactions])[FEATURE_ORDER]
        df[SCALE_COLS] = scaler.transform(df[SCALE_COLS])

        probs = clf.predict_proba(df)[:, 1]
        preds = (probs >= 0.5).astype(int)

        # Graph: legit vs fraud prob
        prob_legit = 1 - probs[0]
        prob_fraud = probs[0]

        plt.figure(figsize=(4, 4))
        bars = plt.bar(["Legit", "Fraud"], [prob_legit, prob_fraud],
                       color=["green", "red"])
        for bar, val in zip(bars, [prob_legit, prob_fraud]):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                     f"{val:.2f}", ha="center", va="center", color="white", fontsize=12)
        plt.title("Fraud Probability")
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        graph_b64 = base64.b64encode(buf.read()).decode("utf-8")

        return {
            "model_used": model_name,
            "probability": float(probs[0]),
            "prediction": int(preds[0]),
            "classification": "Fraudulent" if preds[0] == 1 else "Legit",
            "graph": graph_b64
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# ================== Bulk CSV ==================
@app.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...), model: str = Query("rf", enum=["rf", "xgb"])):
    try:
        df = pd.read_csv(file.file)[FEATURE_ORDER]

        if model == "rf":
            clf, scaler, model_name = rf_model, rf_scaler, "RandomForest"
        else:
            clf, scaler, model_name = xgb_model, xgb_scaler, "XGBoost"

        df[SCALE_COLS] = scaler.transform(df[SCALE_COLS])
        probs = clf.predict_proba(df)[:, 1]
        preds = (probs >= 0.5).astype(int)

        df["Probability"] = probs
        df["Prediction"] = preds

        # Convert to CSV string
        csv_buf = io.StringIO()
        df.to_csv(csv_buf, index=False)
        csv_str = csv_buf.getvalue()

        # Graph: Class distribution
        fraud_counts = df["Prediction"].value_counts(normalize=True) * 100
        plt.figure(figsize=(5, 4))
        ax = fraud_counts.plot(kind="bar", color=["green", "red"])
        for i, val in enumerate(fraud_counts):
            ax.text(i, val/2, f"{val:.1f}%", ha="center", va="center", color="white", fontsize=12)
        plt.title("Fraud vs Legit Predictions (%)")
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        graph_b64 = base64.b64encode(buf.read()).decode("utf-8")

        return {
            "model_used": model_name,
            "processed_csv": csv_str,
            "graph": graph_b64
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
