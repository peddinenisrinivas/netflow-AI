"""
NetFlow AI - Intelligent Bandwidth Allocation using Reinforcement Learning
Backend API (FastAPI) — ML only, no deep learning.

Run with:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os

from routes.ingestion  import router as ingestion_router
from routes.training   import router as training_router
from routes.evaluation import router as evaluation_router
from routes.prediction import router as prediction_router
from routes.rl_agent   import router as rl_router

app = FastAPI(
    title="NetFlow AI – Intelligent Bandwidth Allocation",
    description=(
        "Reinforcement-Learning + ML backend for real-time bandwidth allocation. "
        "Uses Q-Learning with Random-Forest / Gradient-Boosting state classifiers. "
        "No deep learning."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("uploads",      exist_ok=True)
os.makedirs("saved_models", exist_ok=True)

app.include_router(ingestion_router,  prefix="/api/ingestion",  tags=["1. PCAP Ingestion"])
app.include_router(training_router,   prefix="/api/training",   tags=["2-3. ML Training"])
app.include_router(evaluation_router, prefix="/api/evaluation", tags=["4-5. Evaluation"])
app.include_router(prediction_router, prefix="/api/prediction", tags=["6. Prediction"])
app.include_router(rl_router,         prefix="/api/rl",         tags=["7. RL Agent"])


@app.get("/", tags=["Health"])
def root():
    return {
        "status": "ok",
        "message": "NetFlow AI API is running",
        "project": "Intelligent Bandwidth Allocation using Reinforcement Learning",
    }


@app.get("/api/health", tags=["Health"])
def health():
    return {"status": "healthy"}
