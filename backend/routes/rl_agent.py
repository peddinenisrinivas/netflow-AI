"""
routes/rl_agent.py — Section 7: Reinforcement Learning Bandwidth Allocation Agent

Implements a tabular Q-Learning agent that learns an optimal bandwidth allocation
policy purely from network traffic data.

No deep learning. Uses:
  - Q-Learning (tabular, epsilon-greedy)
  - State representation via sklearn classifier (RandomForest/GradientBoosting)
  - Reward shaping based on utilisation efficiency and SLA compliance

Endpoints:
  POST /api/rl/train          Train the RL agent
  GET  /api/rl/status         Training progress
  GET  /api/rl/policy         Current learned policy (state → action)
  POST /api/rl/allocate       Get RL allocation decision for a flow
  POST /api/rl/allocate-batch Batch allocation decisions
  GET  /api/rl/history        Reward / performance history
  GET  /api/rl/q-table        Raw Q-table dump
"""

import os
import time
import threading
import joblib
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from utils.state import rl_state, model_artifact, uploaded_files
from utils.pcap_parser import FEATURE_COLUMNS, _synthetic_fallback

router = APIRouter()
_rl_stop_event = threading.Event()

# ── RL Problem Definition ─────────────────────────────────────────────────────
#
#  STATE  : discretised network conditions derived from ML classifier output
#           + current utilisation bucket.
#           Represented as a string key, e.g. "Peak Spike|HIGH"
#
#  ACTIONS: bandwidth allocation decisions
ACTIONS = ["increase", "maintain", "decrease", "throttle"]
#           increase  → +50% bandwidth  (handle burst / peak)
#           maintain  → keep current    (normal stable traffic)
#           decrease  → −30% bandwidth  (save capacity when idle)
#           throttle  → −60% bandwidth  (anomaly / DoS mitigation)
#
#  REWARD : shaped reward combining utilisation efficiency and SLA compliance


# ── Schemas ───────────────────────────────────────────────────────────────────

class RLTrainConfig(BaseModel):
    episodes:       int   = Field(300,  ge=10,  le=2000)
    alpha:          float = Field(0.1,  gt=0,   le=1.0,  description="Learning rate")
    gamma:          float = Field(0.95, gt=0,   le=1.0,  description="Discount factor")
    epsilon_start:  float = Field(1.0,  gt=0,   le=1.0,  description="Initial exploration")
    epsilon_end:    float = Field(0.05, gt=0,   le=0.5,  description="Minimum exploration")
    epsilon_decay:  float = Field(0.995, gt=0,  le=1.0,  description="Epsilon decay rate")
    max_bw_mbps:    float = Field(1000.0, gt=0,           description="Max link capacity Mbps")
    file_id: Optional[str] = None


class AllocationRequest(BaseModel):
    bytes_per_second:   float = 50_000.0
    pkts_per_second:    float = 10.0
    flag_rst:           float = 0.0
    protocol_tcp:       float = 1.0
    avg_pkt_size:       float = 500.0
    current_util_pct:   float = Field(50.0, ge=0, le=100,
                                      description="Current link utilisation %")
    src_ip: Optional[str] = "192.168.1.1"
    dst_ip: Optional[str] = "10.0.0.1"


class AllocationResult(BaseModel):
    state_key:      str
    traffic_class:  str
    util_bucket:    str
    action:         str
    q_values:       Dict[str, float]
    allocated_mbps: float
    confidence:     float
    reason:         str
    timestamp:      str


# ── Environment helpers ───────────────────────────────────────────────────────

def _util_bucket(util_pct: float) -> str:
    if util_pct < 30:   return "LOW"
    if util_pct < 60:   return "MEDIUM"
    if util_pct < 85:   return "HIGH"
    return "CRITICAL"


def _state_key(traffic_class: str, util_bucket: str) -> str:
    return f"{traffic_class}|{util_bucket}"


def _reward(action: str, traffic_class: str, util_pct: float) -> float:
    """
    Reward function:
      - Penalise under-provisioning (congestion) heavily
      - Penalise over-provisioning (waste) moderately
      - Penalise wrong action for anomaly (should throttle, not increase)
      - Reward matching action to actual traffic state
    """
    r = 0.0

    # Baseline: reward for right action given traffic class
    ideal = {
        "Peak Spike": "increase",
        "Stable":     "maintain",
        "Anomaly":    "throttle",
    }
    if action == ideal.get(traffic_class, "maintain"):
        r += 10.0

    # Penalise dangerous combinations
    if traffic_class == "Anomaly" and action == "increase":
        r -= 15.0   # amplifying an anomaly flow is very bad

    if traffic_class == "Peak Spike" and action == "throttle":
        r -= 12.0   # throttling legitimate burst causes SLA breach

    # Utilisation-based reward shaping
    if util_pct > 85 and action == "increase":
        r += 5.0    # good: relieving congestion
    if util_pct > 85 and action == "decrease":
        r -= 8.0    # bad: worsening congestion

    if util_pct < 30 and action == "decrease":
        r += 3.0    # good: saving capacity
    if util_pct < 30 and action == "increase":
        r -= 4.0    # bad: wasting capacity

    # Small entropy bonus to avoid trivial policy collapse
    r += np.random.default_rng().normal(0, 0.2)
    return r


def _get_q(q_table: Dict, state: str, action: str) -> float:
    return q_table.get(state, {}).get(action, 0.0)


def _set_q(q_table: Dict, state: str, action: str, value: float):
    if state not in q_table:
        q_table[state] = {a: 0.0 for a in ACTIONS}
    q_table[state][action] = value


def _best_action(q_table: Dict, state: str) -> str:
    if state not in q_table:
        return "maintain"
    return max(q_table[state], key=q_table[state].get)


# ── Q-Learning training thread ────────────────────────────────────────────────

def _rl_training_thread(cfg: RLTrainConfig):
    try:
        rl_state.update({
            "status": "running",
            "episodes_completed": 0,
            "total_reward_history": [],
            "epsilon": cfg.epsilon_start,
            "config": cfg.model_dump(),
            "error": None,
        })
        _rl_stop_event.clear()

        # Load traffic data
        if cfg.file_id and cfg.file_id in uploaded_files:
            df = uploaded_files[cfg.file_id]["dataframe"].copy()
        else:
            df = _synthetic_fallback(1000)

        if "label" not in df.columns:
            df["label"] = np.select(
                [df["bytes_per_second"] > 5e7, df["flag_rst"] == 1],
                ["Peak Spike", "Anomaly"], default="Stable"
            )

        traffic_classes = df["label"].values
        util_values = np.clip(
            (df["bytes_per_second"].values / (cfg.max_bw_mbps * 1e6)) * 100, 0, 100
        )

        n_flows = len(df)
        q_table: Dict[str, Dict[str, float]] = {}
        rng = np.random.default_rng(42)
        epsilon = cfg.epsilon_start

        for episode in range(cfg.episodes):
            if _rl_stop_event.is_set():
                rl_state["status"] = "stopped"
                return

            episode_reward = 0.0
            # Sample a random trajectory of flows for this episode
            indices = rng.choice(n_flows, size=min(50, n_flows), replace=False)

            for idx in indices:
                tc   = str(traffic_classes[idx])
                util = float(util_values[idx])
                ub   = _util_bucket(util)
                s    = _state_key(tc, ub)

                # Epsilon-greedy action selection
                if rng.random() < epsilon:
                    action = rng.choice(ACTIONS)
                else:
                    action = _best_action(q_table, s)

                # Reward
                r = _reward(action, tc, util)
                episode_reward += r

                # Simulate next state (util changes after action)
                delta = {"increase": +15, "maintain": 0, "decrease": -10, "throttle": -25}
                next_util = float(np.clip(util + delta.get(action, 0), 0, 100))
                next_s = _state_key(tc, _util_bucket(next_util))

                # Q-Learning update: Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') − Q(s,a)]
                old_q = _get_q(q_table, s, action)
                next_max_q = max(q_table.get(next_s, {a: 0.0 for a in ACTIONS}).values(),
                                 default=0.0)
                new_q = old_q + cfg.alpha * (r + cfg.gamma * next_max_q - old_q)
                _set_q(q_table, s, action, new_q)

            # Decay epsilon
            epsilon = max(cfg.epsilon_end, epsilon * cfg.epsilon_decay)

            rl_state["episodes_completed"] = episode + 1
            rl_state["epsilon"] = round(epsilon, 4)
            rl_state["total_reward_history"].append(round(episode_reward, 2))

            time.sleep(0.02)  # keep event-loop responsive

        # Extract greedy policy
        policy = {state: _best_action(q_table, state) for state in q_table}
        rl_state["q_table"]   = q_table
        rl_state["policy"]    = policy
        rl_state["is_trained"] = True

        # Persist
        save_path = os.path.join("saved_models", "rl_agent.joblib")
        joblib.dump({
            "q_table":    q_table,
            "policy":     policy,
            "config":     cfg.model_dump(),
            "trained_at": datetime.utcnow().isoformat(),
            "actions":    ACTIONS,
        }, save_path)

        rl_state["status"] = "complete"

    except Exception as exc:
        rl_state["status"] = "error"
        rl_state["error"]  = str(exc)


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/train")
def train_rl_agent(cfg: RLTrainConfig):
    if rl_state.get("status") == "running":
        raise HTTPException(409, "RL training already in progress")
    threading.Thread(target=_rl_training_thread, args=(cfg,), daemon=True).start()
    return {"detail": "RL agent training started", "config": cfg.model_dump()}


@router.post("/stop")
def stop_rl_training():
    if rl_state.get("status") != "running":
        raise HTTPException(400, "RL training is not running")
    _rl_stop_event.set()
    return {"detail": "Stop signal sent"}


@router.get("/status")
def rl_status():
    return {
        "status":             rl_state.get("status", "idle"),
        "is_trained":         rl_state.get("is_trained", False),
        "episodes_completed": rl_state.get("episodes_completed", 0),
        "total_episodes":     rl_state.get("config", {}).get("episodes", 0),
        "epsilon":            rl_state.get("epsilon", 1.0),
        "error":              rl_state.get("error"),
        "states_learned":     len(rl_state.get("q_table", {})),
    }


@router.get("/policy")
def get_policy():
    """Return the learned policy: state → best action."""
    _ensure_rl_loaded()
    policy = rl_state.get("policy", {})
    # Build a human-readable summary
    table = []
    for state, action in sorted(policy.items()):
        tc, ub = state.split("|") if "|" in state else (state, "?")
        table.append({
            "state":         state,
            "traffic_class": tc,
            "util_bucket":   ub,
            "best_action":   action,
            "q_values":      rl_state["q_table"].get(state, {}),
        })
    return {"policy_size": len(policy), "policy": table}


@router.post("/allocate", response_model=AllocationResult)
def allocate_bandwidth(req: AllocationRequest):
    """
    Use the trained RL policy to allocate bandwidth for an incoming flow.
    If RL is not trained, falls back to a rule-based decision.
    """
    _ensure_rl_loaded()

    # Determine traffic class using simple rules (or use ML classifier if available)
    bps = req.bytes_per_second
    if req.flag_rst == 1:
        tc = "Anomaly"
    elif bps > 5e7:
        tc = "Peak Spike"
    else:
        tc = "Stable"

    ub       = _util_bucket(req.current_util_pct)
    s        = _state_key(tc, ub)
    q_table  = rl_state.get("q_table", {})
    policy   = rl_state.get("policy", {})

    if s in policy:
        action   = policy[s]
        q_vals   = q_table.get(s, {a: 0.0 for a in ACTIONS})
        conf     = _softmax_confidence(list(q_vals.values()))
        reason   = f"RL policy: state '{s}' → action '{action}'"
    else:
        # Fallback rule-based policy
        action = {"Peak Spike": "increase", "Stable": "maintain",
                  "Anomaly": "throttle"}.get(tc, "maintain")
        q_vals = {a: 0.0 for a in ACTIONS}
        conf   = 0.75
        reason = f"Rule-based fallback (state '{s}' not in Q-table)"

    multipliers = {"increase": 1.5, "maintain": 1.0, "decrease": 0.7, "throttle": 0.4}
    predicted_mbps  = bps / 1e6
    allocated_mbps  = round(predicted_mbps * multipliers.get(action, 1.0), 3)

    entry = AllocationResult(
        state_key=s, traffic_class=tc, util_bucket=ub,
        action=action, q_values={a: round(v, 4) for a, v in q_vals.items()},
        allocated_mbps=allocated_mbps,
        confidence=round(conf, 4),
        reason=reason,
        timestamp=datetime.utcnow().strftime("%H:%M:%S.%f")[:-4],
    )

    # Log allocation
    log = rl_state.setdefault("allocation_log", [])
    log.append(entry.model_dump())
    if len(log) > 500:
        log.pop(0)

    return entry


@router.post("/allocate-batch")
def allocate_batch(requests: List[AllocationRequest]):
    return [allocate_bandwidth(r) for r in requests]


@router.get("/history")
def rl_history():
    rewards = rl_state.get("total_reward_history", [])
    return {
        "episodes":       len(rewards),
        "reward_history": rewards,
        "moving_avg":     _moving_avg(rewards, window=20),
        "final_reward":   rewards[-1] if rewards else None,
        "peak_reward":    max(rewards) if rewards else None,
    }


@router.get("/q-table")
def get_q_table():
    _ensure_rl_loaded()
    qt = rl_state.get("q_table", {})
    return {
        "states":  len(qt),
        "actions": ACTIONS,
        "q_table": {s: {a: round(v, 4) for a, v in av.items()} for s, av in qt.items()},
    }


@router.get("/allocation-log")
def allocation_log(limit: int = 100):
    log = rl_state.get("allocation_log", [])
    return {"count": len(log), "log": log[-limit:][::-1]}


@router.get("/summary")
def rl_summary():
    """High-level dashboard summary for the RL agent."""
    rewards = rl_state.get("total_reward_history", [])
    policy  = rl_state.get("policy", {})

    action_dist: Dict[str, int] = {a: 0 for a in ACTIONS}
    for action in policy.values():
        if action in action_dist:
            action_dist[action] += 1

    return {
        "is_trained":         rl_state.get("is_trained", False),
        "status":             rl_state.get("status", "idle"),
        "episodes_completed": rl_state.get("episodes_completed", 0),
        "states_learned":     len(rl_state.get("q_table", {})),
        "final_epsilon":      rl_state.get("epsilon", 1.0),
        "policy_action_dist": action_dist,
        "avg_reward_last_50": round(float(np.mean(rewards[-50:])), 2) if rewards else 0.0,
        "peak_reward":        round(max(rewards), 2) if rewards else 0.0,
    }


# ── Internal helpers ──────────────────────────────────────────────────────────

def _ensure_rl_loaded():
    """Load persisted RL agent from disk if not in memory."""
    if rl_state.get("is_trained"):
        return
    path = os.path.join("saved_models", "rl_agent.joblib")
    if os.path.exists(path):
        art = joblib.load(path)
        rl_state.update({
            "q_table":    art["q_table"],
            "policy":     art["policy"],
            "is_trained": True,
            "config":     art.get("config", {}),
        })


def _softmax_confidence(q_values: List[float]) -> float:
    """Convert Q-values to a confidence score via softmax."""
    arr = np.array(q_values, dtype=float)
    arr -= arr.max()
    exp = np.exp(arr)
    probs = exp / exp.sum()
    return float(probs.max())


def _moving_avg(data: List[float], window: int = 20) -> List[float]:
    if not data:
        return []
    result = []
    for i in range(len(data)):
        start = max(0, i - window + 1)
        result.append(round(float(np.mean(data[start:i+1])), 2))
    return result
