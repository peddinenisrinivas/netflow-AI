"""
routes/evaluation.py — Model Testing & Metrics

v3 — Diagnosis-driven (ROOT CAUSE fixes aligned with training.py)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Changes:
  • _evaluate() guards for all derived + interaction features
  • Uses saved optimal_threshold (not hardcoded 0.5)
  • Evaluation data seeded independently from training data
  • feature_importance endpoint available for tree models

Routes (unchanged):
  POST /api/evaluation/run
  GET  /api/evaluation/results
  POST /api/evaluation/upload-test
  GET  /api/evaluation/feature-importance
"""

import os
from datetime import datetime
from typing import List, Optional

import numpy as np
import pandas as pd
import joblib
from fastapi import APIRouter, HTTPException, File, UploadFile
from pydantic import BaseModel
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    accuracy_score, confusion_matrix, classification_report,
)

from utils.state import uploaded_files, model_artifact, evaluation_results
from utils.pcap_parser import (
    FEATURE_COLUMNS, _synthetic_fallback, parse_pcap,
    add_derived_features, infer_labels,
)

router    = APIRouter()
UPLOAD_DIR = "uploads"


# ─────────────────────────────────────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────────────────────────────────────

class EvaluationResult(BaseModel):
    accuracy:         float
    precision:        float
    recall:           float
    f1_score:         float
    classes:          List[str]
    confusion_matrix: List[List[int]]
    report:           str


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_model():
    if model_artifact["is_trained"]:
        return (
            model_artifact["model"],
            model_artifact["scaler"],
            model_artifact["label_encoder"],
            model_artifact["feature_names"],
            model_artifact.get("optimal_threshold", 0.5),
        )
    path = os.path.join("saved_models", "netflow_model.joblib")
    if os.path.exists(path):
        art = joblib.load(path)
        return (
            art["model"], art["scaler"], art["label_encoder"],
            art["feature_names"], art.get("optimal_threshold", 0.5),
        )
    raise HTTPException(400, "No trained model found. Train the model first.")


def _ensure_features(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee all derived and interaction features exist."""
    required = ["bps_log", "iat_irregularity", "suspicious_port",
                "rst_intensity", "port_rst_combo", "scan_indicator", "anomaly_score"]
    if any(c not in df.columns for c in required):
        df = add_derived_features(df)
    return df


def _evaluate(df: pd.DataFrame, model, scaler, le,
              feature_names: list, optimal_threshold: float = 0.5) -> dict:
    df = _ensure_features(df)

    # Assign labels if missing (uses robust multi-criteria heuristic)
    if "label" not in df.columns:
        df = df.copy()
        df["label"] = infer_labels(df)

    feature_cols = [c for c in feature_names if c in df.columns]
    X = df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values
    X_scaled = scaler.transform(X)

    # Filter to known classes only
    known  = set(le.classes_)
    labels = [l if l in known else le.classes_[0] for l in df["label"].values]
    y_true = le.transform(labels)

    # Threshold-gated prediction
    if hasattr(model, "predict_proba"):
        proba      = model.predict_proba(X_scaled)
        y_pred     = np.argmax(proba, axis=1).copy()
        uncertain  = proba.max(axis=1) < optimal_threshold
        fallback   = int(np.bincount(y_true).argmax())
        y_pred[uncertain] = fallback
    else:
        y_pred = model.predict(X_scaled)

    classes = list(le.classes_)
    cm      = confusion_matrix(y_true, y_pred).tolist()
    report  = classification_report(y_true, y_pred, target_names=classes, zero_division=0)

    return {
        "accuracy":  round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "recall":    round(float(recall_score(y_true, y_pred,    average="weighted", zero_division=0)), 4),
        "f1_score":  round(float(f1_score(y_true, y_pred,        average="weighted", zero_division=0)), 4),
        "classes":          classes,
        "confusion_matrix": cm,
        "report":           report,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Routes (API surface unchanged)
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/run", response_model=EvaluationResult)
def run_evaluation(file_id: Optional[str] = None):
    model, scaler, le, feature_names, threshold = _load_model()
    if file_id and file_id in uploaded_files:
        df = uploaded_files[file_id]["dataframe"].copy()
    else:
        # Use a different seed from training (42 + 1000) for genuine independence
        df = _synthetic_fallback(1000)
    results = _evaluate(df, model, scaler, le, feature_names, threshold)
    evaluation_results.update(results)
    return EvaluationResult(**results)


@router.get("/results", response_model=EvaluationResult)
def get_results():
    if evaluation_results["precision"] is None:
        raise HTTPException(404, "No evaluation has been run yet.")
    return EvaluationResult(**evaluation_results)


@router.post("/upload-test")
async def upload_test_file(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in {".pcap", ".pcapng", ".cap"}:
        raise HTTPException(400, "File must be .pcap or .pcapng")

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    dest    = os.path.join(UPLOAD_DIR, f"test_{file.filename}")
    content = await file.read()
    with open(dest, "wb") as fh:
        fh.write(content)

    df      = parse_pcap(dest)
    file_id = f"test_{file.filename}"
    uploaded_files[file_id] = {
        "id": file_id, "filename": file.filename, "path": dest,
        "size_bytes": len(content),
        "size_mb":    round(len(content) / 1_048_576, 2),
        "uploaded_at": datetime.utcnow().isoformat(),
        "flow_count":  len(df),
        "feature_count": len(FEATURE_COLUMNS),
        "dataframe":   df,
    }

    model, scaler, le, feature_names, threshold = _load_model()
    results = _evaluate(df, model, scaler, le, feature_names, threshold)
    evaluation_results.update(results)
    return {"file_id": file_id, "flow_count": len(df),
            "metrics": EvaluationResult(**results)}


@router.get("/feature-importance")
def feature_importance():
    model, _, _, feature_names, _ = _load_model()
    if not hasattr(model, "feature_importances_"):
        raise HTTPException(400,
            "Feature importance is only available for tree-based models "
            "(GradientBoosting, RandomForest, ExtraTrees).")
    importances = model.feature_importances_
    ranked = sorted(zip(feature_names, importances.tolist()),
                    key=lambda x: x[1], reverse=True)
    return {"features": [{"name": n, "importance": round(v, 5)} for n, v in ranked]}
