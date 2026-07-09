# NetFlow AI

> **Intelligent Bandwidth Allocation using Reinforcement Learning**

NetFlow AI is a full-stack network intelligence platform that combines classical machine learning with tabular Q-Learning to automatically classify network traffic and make real-time bandwidth allocation decisions — with no deep learning required.

---

## How It Works

NetFlow AI operates in a two-stage pipeline:

**Stage 1 — Traffic Classification**
A scikit-learn classifier is trained on 22 per-flow features extracted from `.pcap` network capture files (or synthetic data if no file is available). It labels each flow as one of:
- `Stable` — normal, predictable traffic
- `Peak Spike` — legitimate traffic burst
- `Anomaly` — suspicious or malicious behaviour (e.g. DoS)

**Stage 2 — RL Bandwidth Allocation**
A tabular Q-Learning agent uses the classifier's output combined with the current link utilisation level as its state. It then selects one of four actions — `increase`, `maintain`, `decrease`, or `throttle` — and learns an optimal allocation policy through shaped rewards over training episodes.

---

## Tech Stack

### Backend
| Package | Version | Purpose |
|---|---|---|
| `fastapi` | 0.111.0 | REST API framework |
| `uvicorn[standard]` | 0.29.0 | ASGI server |
| `python-multipart` | 0.0.9 | File upload support |
| `pydantic` | 2.7.1 | Request/response validation |
| `scikit-learn` | 1.5.0 | ML classifiers (RandomForest, SVM, etc.) |
| `numpy` | 1.26.4 | Numerical computation |
| `pandas` | 2.2.2 | Data manipulation |
| `joblib` | 1.4.2 | Model persistence |
| `scapy` | 2.5.0 | PCAP parsing *(optional — falls back to synthetic data if not installed)* |

### Frontend
Plain HTML/CSS/JavaScript — no build tools or frameworks required. Open directly in any browser.

---

## Project Structure

```
netflow_ai/
├── backend/
│   ├── main.py                  # FastAPI entry point
│   ├── requirements.txt         # Python dependencies
│   ├── uploads/                 # Uploaded PCAP files (auto-created at runtime)
│   ├── saved_models/            # Persisted ML & RL models (auto-created at runtime)
│   ├── routes/
│   │   ├── ingestion.py         # PCAP upload & flow feature extraction
│   │   ├── training.py          # ML classifier training (async)
│   │   ├── evaluation.py        # Accuracy, confusion matrix, feature importance
│   │   ├── prediction.py        # Single/batch predictions + SSE live stream
│   │   └── rl_agent.py          # Q-Learning RL bandwidth allocation agent
│   └── utils/
│       ├── state.py             # Shared in-memory application state
│       └── pcap_parser.py       # PCAP → per-flow feature extraction
│
└── frontend/
    ├── dashboard.html           # Main dashboard — start here
    ├── DataIngestion.html       # Upload PCAP files
    ├── Model_training.html      # Train ML classifier + RL agent
    ├── ModelEvaluation.html     # View metrics, confusion matrix, feature importance
    └── BandwidthPrediction.html # Live prediction stream + manual RL allocation
```

---

## Prerequisites

- **Python 3.9+**
- A modern web browser (Chrome, Firefox, Edge, Safari)
- *(Optional)* `.pcap` / `.pcapng` network capture files for real traffic data

---

## Installation

**1. Clone or unzip the project, then navigate to the backend folder:**

```bash
cd netflow_ai/backend
```

**2. Install Python dependencies:**

```bash
python -m pip install -r requirements.txt
```

This installs FastAPI, scikit-learn, numpy, pandas, scapy, and all other required packages at their pinned versions.

> **Note:** Scapy (for PCAP parsing) is included in `requirements.txt`. If Scapy fails to install on your system, the app will still work using automatically generated synthetic traffic data.

---

## Running the App

**1. Start the backend server** (from inside the `backend/` folder):

```bash
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

Leave this terminal running.

**2. Open the frontend:**

Navigate to the `frontend/` folder and open `dashboard.html` in your browser — either by double-clicking it or dragging it into a browser window. The sidebar links all pages together.

**3. Access the API docs (Swagger UI):**

```
http://localhost:8000/docs
```

---

## Recommended Workflow

Follow these steps in order for a complete end-to-end run:

| Step | Page | Action |
|---|---|---|
| 1 | Dashboard | Upload a `.pcap` file, or skip to use synthetic data |
| 2 | Model Training | Select a classifier (e.g. RandomForest) and click **Start Training** |
| 3 | Model Training | Click **Train RL Agent** (default: 300 episodes) |
| 4 | Model Evaluation | Click **Run Evaluation** to view accuracy, F1, and confusion matrix |
| 5 | Bandwidth Predictions | Click **Start Live Stream** to see real-time allocation decisions |

You can also test manual allocation scenarios directly on the Predictions page.

---

## RL Agent Design

The agent uses **tabular Q-Learning** with epsilon-greedy exploration. No neural networks are involved.

### State Space
State is represented as a string combining the traffic class and the current utilisation bucket:

```
STATE = traffic_class × utilisation_bucket
```

Example states: `"Stable|LOW"`, `"Peak Spike|HIGH"`, `"Anomaly|CRITICAL"`

Utilisation buckets: `LOW` (< 30%) | `MEDIUM` (30–60%) | `HIGH` (60–85%) | `CRITICAL` (> 85%)

This produces **12 possible states** (3 traffic classes × 4 utilisation buckets).

### Actions
| Action | Effect | Use Case |
|---|---|---|
| `increase` | +50% bandwidth | Handle legitimate burst or peak traffic |
| `maintain` | No change | Normal, stable traffic conditions |
| `decrease` | −30% bandwidth | Save capacity during low utilisation |
| `throttle` | −60% bandwidth | Anomaly or DoS mitigation |

### Reward Shaping
| Situation | Reward |
|---|---|
| Correct action taken | +10 |
| Amplifying an anomaly (increasing bandwidth during an attack) | −15 |
| Throttling legitimate burst traffic | −12 |

### Q-Update Rule

```
Q(s,a) ← Q(s,a) + α [ r + γ · max Q(s',a') − Q(s,a) ]
```

### Default Hyperparameters
| Parameter | Default | Range |
|---|---|---|
| Episodes | 300 | 10 – 2000 |
| Learning rate (α) | 0.1 | 0 – 1.0 |
| Discount factor (γ) | 0.95 | 0 – 1.0 |
| Epsilon start | 1.0 | — |
| Epsilon end | 0.05 | — |
| Epsilon decay | 0.995 | — |
| Max bandwidth | 1000 Mbps | — |

---

## ML Models Supported

| Model | Notes |
|---|---|
| `RandomForest` | 200 trees, max depth 20, balanced class weights |
| `GradientBoosting` | 150 estimators, configurable learning rate |
| `ExtraTrees` | 200 trees, max depth 20, balanced class weights |
| `SVM` | RBF kernel, C=10, probability outputs enabled |
| `LogisticRegression` | L-BFGS solver, configurable max iterations |

---

## Flow Features

The PCAP parser extracts **22 features per network flow**:

| # | Feature | Description |
|---|---|---|
| 1 | `duration_ms` | Flow duration in milliseconds |
| 2 | `pkt_count` | Total packet count |
| 3 | `byte_count` | Total bytes transferred |
| 4 | `avg_pkt_size` | Mean packet size |
| 5 | `std_pkt_size` | Standard deviation of packet size |
| 6 | `min_pkt_size` | Minimum packet size |
| 7 | `max_pkt_size` | Maximum packet size |
| 8 | `avg_iat_ms` | Mean inter-arrival time (ms) |
| 9 | `std_iat_ms` | Std deviation of inter-arrival time |
| 10 | `protocol_tcp` | TCP flag (0/1) |
| 11 | `protocol_udp` | UDP flag (0/1) |
| 12 | `protocol_icmp` | ICMP flag (0/1) |
| 13 | `protocol_other` | Other protocol flag (0/1) |
| 14 | `src_port` | Source port number |
| 15 | `dst_port` | Destination port number |
| 16 | `flag_syn` | SYN flag present (0/1) |
| 17 | `flag_ack` | ACK flag present (0/1) |
| 18 | `flag_fin` | FIN flag present (0/1) |
| 19 | `flag_rst` | RST flag present (0/1) |
| 20 | `flag_psh` | PSH flag present (0/1) |
| 21 | `bytes_per_second` | Throughput in bytes/sec |
| 22 | `pkts_per_second` | Packet rate in pkts/sec |

> When Scapy is not installed, all features are generated synthetically using realistic statistical distributions, allowing the full ML and RL pipeline to run without real capture files.
