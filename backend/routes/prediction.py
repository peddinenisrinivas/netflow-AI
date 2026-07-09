"""
routes/prediction.py — Real-time Prediction Output

v3 — All derived + interaction features computed inline.
API surface is unchanged.

Routes:
  POST /api/prediction/predict
  POST /api/prediction/predict-batch
  GET  /api/prediction/stream
  GET  /api/prediction/recent
  POST /api/prediction/predict-file/{id}
"""

import os
import json
import asyncio
import random
import joblib
from datetime import datetime
from typing import List, Optional, Dict, Any

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from utils.state import uploaded_files, model_artifact
from utils.pcap_parser import FEATURE_COLUMNS, _synthetic_fallback, add_derived_features

router = APIRouter()
_prediction_log: List[Dict[str, Any]] = []
MAX_LOG = 200

_SUSPICIOUS_PORTS = {23, 4444, 6667, 31337, 1337, 8888, 9999, 1234, 12345, 6666}


# ─────────────────────────────────────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────────────────────────────────────

class FlowFeatures(BaseModel):
    duration_ms:      float = 100.0
    pkt_count:        float = 10.0
    byte_count:       float = 5000.0
    avg_pkt_size:     float = 500.0
    std_pkt_size:     float = 50.0
    min_pkt_size:     float = 64.0
    max_pkt_size:     float = 1500.0
    avg_iat_ms:       float = 10.0
    std_iat_ms:       float = 2.0
    protocol_tcp:     float = 1.0
    protocol_udp:     float = 0.0
    protocol_icmp:    float = 0.0
    protocol_other:   float = 0.0
    src_port:         float = 443.0
    dst_port:         float = 8080.0
    flag_syn:         float = 1.0
    flag_ack:         float = 1.0
    flag_fin:         float = 0.0
    flag_rst:         float = 0.0
    flag_psh:         float = 0.0
    bytes_per_second: float = 50000.0
    pkts_per_second:  float = 10.0
    src_ip:           Optional[str] = "192.168.1.1"
    dst_ip:           Optional[str] = "10.0.0.1"


class PredictionResult(BaseModel):
    timestamp:      str
    src_ip:         str
    dst_ip:         str
    label:          str
    confidence:     float
    predicted_mbps: float
    status:         str
    allocated_mbps: Optional[float] = None
    rl_action:      Optional[str]   = None


class BatchPredictionRequest(BaseModel):
    flows: List[FlowFeatures]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_model():
    if model_artifact["is_trained"]:
        return (model_artifact["model"], model_artifact["scaler"],
                model_artifact["label_encoder"], model_artifact["feature_names"])
    path = os.path.join("saved_models", "netflow_model.joblib")
    if os.path.exists(path):
        art = joblib.load(path)
        return art["model"], art["scaler"], art["label_encoder"], art["feature_names"]
    raise HTTPException(400, "No trained model. Train first.")


def _load_rl_policy() -> dict:
    path = os.path.join("saved_models", "rl_agent.joblib")
    if os.path.exists(path):
        return joblib.load(path).get("policy", {})
    return {}


def _flow_to_vector(flow: FlowFeatures, feature_names: list) -> np.ndarray:
    """
    Build complete feature vector including all derived + interaction features.
    Computes derived values analytically from the FlowFeatures fields.
    """
    eps     = 1e-9
    fd      = flow.model_dump()
    bps     = fd.get("bytes_per_second", 0.0)
    avg_pkt = fd.get("avg_pkt_size",     0.0)
    std_pkt = fd.get("std_pkt_size",     0.0)
    avg_iat = fd.get("avg_iat_ms",       0.0)
    std_iat = fd.get("std_iat_ms",       0.0)
    bc      = fd.get("byte_count",       0.0)
    pc      = fd.get("pkt_count",        1.0)
    dur_ms  = fd.get("duration_ms",      1.0)
    dst     = int(fd.get("dst_port",     0))
    flag_r  = fd.get("flag_rst",         0.0)

    # Derived
    bps_log          = float(np.log1p(max(bps, 0)))
    iat_irregularity = std_iat / (avg_iat + eps)
    bytes_per_pkt    = bc / (pc + 1)
    flow_efficiency  = pc / (dur_ms / 1000.0 + eps)
    suspicious_port  = 1.0 if dst in _SUSPICIOUS_PORTS else 0.0
    rst_intensity    = flag_r * (1.0 / float(np.log1p(pc + 1)))
    short_flow_flag  = 1.0 if pc < 10 else 0.0
    # Interaction
    port_rst_combo   = suspicious_port * flag_r
    scan_indicator   = suspicious_port * short_flow_flag
    anomaly_score    = rst_intensity   * iat_irregularity

    derived = {
        "bps_log": bps_log, "iat_irregularity": iat_irregularity,
        "bytes_per_pkt": bytes_per_pkt, "flow_efficiency": flow_efficiency,
        "suspicious_port": suspicious_port, "rst_intensity": rst_intensity,
        "short_flow_flag": short_flow_flag,
        "port_rst_combo": port_rst_combo, "scan_indicator": scan_indicator,
        "anomaly_score": anomaly_score,
    }

    vec = []
    for f in feature_names:
        if f in fd:
            vec.append(float(fd[f]))
        elif f in derived:
            vec.append(derived[f])
        else:
            vec.append(0.0)

    return np.array([vec])


def _make_prediction(model, scaler, le, feature_names, flow: FlowFeatures,
                     rl_policy: dict = None) -> PredictionResult:
    X_sc      = scaler.transform(_flow_to_vector(flow, feature_names))
    label_idx = model.predict(X_sc)[0]
    label     = le.inverse_transform([label_idx])[0]
    confidence = (
        float(np.max(model.predict_proba(X_sc)[0]))
        if hasattr(model, "predict_proba")
        else 0.85 + random.uniform(-0.05, 0.10)
    )
    predicted_mbps = round(flow.bytes_per_second / 1e6, 3)

    allocated_mbps, rl_action = None, None
    if rl_policy:
        action         = rl_policy.get(label, "maintain")
        rl_action      = action
        mults          = {"increase": 1.5, "maintain": 1.0, "decrease": 0.7, "throttle": 0.4}
        allocated_mbps = round(predicted_mbps * mults.get(action, 1.0), 3)

    result = PredictionResult(
        timestamp=datetime.utcnow().strftime("%H:%M:%S.%f")[:-4],
        src_ip=flow.src_ip or "0.0.0.0",
        dst_ip=flow.dst_ip or "0.0.0.0",
        label=label,
        confidence=round(confidence, 4),
        predicted_mbps=predicted_mbps,
        status=label,
        allocated_mbps=allocated_mbps,
        rl_action=rl_action,
    )
    _prediction_log.append(result.model_dump())
    if len(_prediction_log) > MAX_LOG:
        _prediction_log.pop(0)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/predict", response_model=PredictionResult)
def predict_single(flow: FlowFeatures):
    model, scaler, le, feature_names = _load_model()
    return _make_prediction(model, scaler, le, feature_names, flow, _load_rl_policy())


@router.post("/predict-batch", response_model=List[PredictionResult])
def predict_batch(request: BatchPredictionRequest):
    model, scaler, le, feature_names = _load_model()
    policy = _load_rl_policy()
    return [_make_prediction(model, scaler, le, feature_names, f, policy)
            for f in request.flows]


@router.get("/stream")
async def stream_predictions():
    model, scaler, le, feature_names = _load_model()
    policy = _load_rl_policy()

    async def generator():
        while True:
            df  = _synthetic_fallback(1)
            row = df.iloc[0]
            base_fields = [
                "duration_ms", "pkt_count", "byte_count",
                "avg_pkt_size", "std_pkt_size", "min_pkt_size", "max_pkt_size",
                "avg_iat_ms", "std_iat_ms",
                "protocol_tcp", "protocol_udp", "protocol_icmp", "protocol_other",
                "src_port", "dst_port",
                "flag_syn", "flag_ack", "flag_fin", "flag_rst", "flag_psh",
                "bytes_per_second", "pkts_per_second",
            ]
            kwargs = {f: float(row.get(f, 0.0)) for f in base_fields if f in row.index}
            kwargs["src_ip"] = str(row.get("_src_ip", "192.168.1.1"))
            kwargs["dst_ip"] = str(row.get("_dst_ip", "10.0.0.1"))
            flow   = FlowFeatures(**kwargs)
            result = _make_prediction(model, scaler, le, feature_names, flow, policy)
            yield f"data: {json.dumps(result.model_dump())}\n\n"
            await asyncio.sleep(1.0)

    return StreamingResponse(generator(), media_type="text/event-stream")


@router.get("/recent")
def recent_predictions(limit: int = 50):
    return {"count": min(limit, len(_prediction_log)),
            "predictions": _prediction_log[-limit:][::-1]}


@router.post("/predict-file/{file_id}")
def predict_from_file(file_id: str, limit: int = 100):
    if file_id not in uploaded_files:
        raise HTTPException(404, "File ID not found")
    model, scaler, le, feature_names = _load_model()
    policy  = _load_rl_policy()
    df      = uploaded_files[file_id]["dataframe"].copy()
    results = []
    base_fields = [
        "duration_ms", "pkt_count", "byte_count",
        "avg_pkt_size", "std_pkt_size", "min_pkt_size", "max_pkt_size",
        "avg_iat_ms", "std_iat_ms",
        "protocol_tcp", "protocol_udp", "protocol_icmp", "protocol_other",
        "src_port", "dst_port",
        "flag_syn", "flag_ack", "flag_fin", "flag_rst", "flag_psh",
        "bytes_per_second", "pkts_per_second",
    ]
    for _, row in df.head(limit).iterrows():
        kwargs = {f: float(row.get(f, 0.0)) for f in base_fields if f in row.index}
        kwargs["src_ip"] = str(row.get("_src_ip", "0.0.0.0"))
        kwargs["dst_ip"] = str(row.get("_dst_ip", "0.0.0.0"))
        flow = FlowFeatures(**kwargs)
        results.append(_make_prediction(model, scaler, le, feature_names, flow, policy).model_dump())
    return {"file_id": file_id, "count": len(results), "predictions": results}
