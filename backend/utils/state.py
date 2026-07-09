"""
utils/state.py — Shared in-memory state for all route modules.
In production replace with Redis / a proper database.
"""
from typing import Any, Dict

# ── Uploaded file registry ────────────────────────────────────────────────────
uploaded_files: Dict[str, Dict] = {}

# ── ML model artifact ─────────────────────────────────────────────────────────
model_artifact: Dict[str, Any] = {
    "model":         None,
    "scaler":        None,
    "label_encoder": None,
    "feature_names": [],
    "is_trained":    False,
}

# ── Training job state ────────────────────────────────────────────────────────
training_state: Dict[str, Any] = {
    "status":           "idle",   # idle | running | complete | error | stopped
    "epoch":            0,
    "total_epochs":     0,
    "accuracy_history": [],
    "loss_history":     [],
    "current_accuracy": 0.0,
    "current_loss":     0.0,
    "error":            None,
}

# ── Evaluation results ────────────────────────────────────────────────────────
evaluation_results: Dict[str, Any] = {
    "precision":        None,
    "recall":           None,
    "f1_score":         None,
    "accuracy":         None,
    "confusion_matrix": None,
    "report":           None,
    "classes":          None,
}

# ── RL agent state ────────────────────────────────────────────────────────────
rl_state: Dict[str, Any] = {
    "q_table":              {},   # state_key -> {action: q_value}
    "is_trained":           False,
    "episodes_completed":   0,
    "total_reward_history": [],
    "epsilon":              1.0,  # exploration rate (decays over training)
    "policy":               {},   # state_key -> best_action
    "allocation_log":       [],   # last 500 allocation decisions
    "config":               {},
}
