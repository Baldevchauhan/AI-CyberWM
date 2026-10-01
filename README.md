# AI World Model for Predictive Cyber Defence

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Offline Capable](https://img.shields.io/badge/Offline-100%25%20Air--Gapped-green.svg)]()

A fully open-source, offline-capable AI cybersecurity system that models how computer network states evolve over time, detects ongoing malicious operations, predicts how attacks will progress across future time horizons ($K$-step forward simulation), maps predicted trajectories to the MITRE ATT&CK framework, and explains model predictions using path-integrated attribution.

---

## 1. Project Overview

Modern cyber defence relies heavily on point-in-time signature or flow classification. Conventional intrusion detection systems answer:
> *"Is this specific flow malicious right now?"*

The **AI Cyber World Model** shifts the paradigm from reactive classification to **predictive world dynamics**. It continuously answers:
> **"Given the network behaviour observed up to the current time window, what is likely to happen next, and is the current trajectory progressing toward an attack?"**

---

## 2. Problem Statement

1. **Signatures are brittle**: Adversaries continually rotate domains, payloads, and ports.
2. **Point-in-time classifiers miss the kill-chain**: Multi-stage campaigns (Reconnaissance $\to$ Exploitation $\to$ Lateral Movement $\to$ C2 $\to$ Exfiltration) take place over minutes, hours, or days.
3. **No proactive horizon**: Standard intrusion detection triggers only after an exploit has already fired or data has begun leaking.
4. **Black-box decisions**: Defenders need clear, justifiable model attribution linking alerts to specific network features and historical time windows.

---

## 3. World-Model Concept

Instead of:
```
Traffic ───► Classifier ───► Benign / Malicious
```

The system implements:
```
Traffic Telemetry (CSV / PCAP / NetFlow)
    │
    ▼
Discretized Network State S(t) ∈ ℝ²⁰
    │
    ▼
Temporal World Model  P(S(t+1) | S(t), History)
    │
    ▼
K-Step Autoregressive Rollout: S(t) ──► Ŝ(t+1) ──► Ŝ(t+2) ──► ... ──► Ŝ(t+K)
    │
    ├─────────────────────────────┬─────────────────────────────┐
    ▼                             ▼                             ▼
Attack Probability Timeline    MITRE ATT&CK Stage Forecast   Model Attribution
[p(t), p(t+1), ..., p(t+K)]   (Recon, LatMove, C2, Exfil)   (SHAP / Integrated Gradients)
```

---

## 4. Architecture

```
cyber-world-model/
├── config.yaml                     # Global hyperparameters, features, and thresholds
├── requirements.txt                # Python dependencies
├── README.md                       # Project overview and runbooks
├── docs/
│   └── architecture.md             # In-depth architectural specification
├── data/
│   ├── raw/                        # Ingested raw telemetry uploads
│   ├── processed/                  # Cached state tensors
│   └── sample/                     # Pre-packaged offline sample CSV & PCAP
├── src/
│   ├── data/
│   │   ├── datasets/               # Adapters: generic_csv, cicids2018, ctu13
│   │   ├── parsers/                # PCAP packet/flow parser via Scapy
│   │   └── preprocessing/          # Strict chronological train/val/test splitter
│   ├── features/
│   │   ├── flow_features.py        # IAT, flags, port scan heuristics
│   │   └── state_builder.py        # 20-dim state matrix S(t) and sliding windows
│   ├── models/
│   │   ├── world_model.py          # Base abstract world model interface
│   │   ├── lstm_world_model.py     # Primary LSTM network world model
│   │   ├── transformer_world_model.py # Modular Transformer alternative
│   │   └── baseline.py             # Logistic Regression static baseline
│   ├── prediction/
│   │   ├── rollout.py              # K-step simulation rollout coordinator
│   │   ├── risk.py                 # Graded risk categorization & timeline builder
│   │   └── attack_stage.py         # Multi-stage kill-chain forecaster
│   ├── explainability/
│   │   └── shap_explainer.py       # Integrated Gradients & feature attribution
│   ├── mitre/
│   │   └── attack_mapping.py       # MITRE Enterprise tactics & techniques mapping
│   └── utils/
│       ├── config.py               # YAML configuration loader
│       └── graph.py                # NetworkX topology & Plotly visualizer
├── training/
│   ├── train_world_model.py        # World model composite loss training pipeline
│   └── train_baseline.py           # Logistic Regression baseline training
├── evaluation/
│   └── evaluate.py                 # Benchmarking & report generation
├── api/
│   └── main.py                     # FastAPI REST service
├── dashboard/
│   └── app.py                      # Interactive Streamlit offline dashboard
├── scripts/
│   ├── train.py                    # Unified training CLI
│   ├── predict.py                  # Forward inference CLI
│   └── evaluate.py                 # Evaluation CLI
├── reports/                        # Evaluation metrics, confusion matrix, plots
└── tests/
    └── test_all.py                 # Comprehensive unit & integration tests
```

---

## 5. Dataset Preparation

The pipeline does not hardcode column names. Instead, adapters translate schemas into canonical flow records:
- **Generic CSV Adapter**: `src/data/datasets/generic_csv.py` with auto-column matching or custom config.
- **CIC-IDS-2018 Adapter**: `src/data/datasets/cicids2018.py` for CSE-CIC-IDS2018.
- **CTU-13 Adapter**: `src/data/datasets/ctu13.py` for CTU-13 NetFlow captures.
- **PCAP Parser**: `src/data/parsers/pcap_parser.py` extracts packet statistics (TTL, window sizes, flag sequences, inter-arrival times) from `.pcap` / `.pcapng`.

A pre-packaged realistic synthetic dataset (`data/sample/sample_traffic.csv` and `data/sample/sample_traffic.pcap`) is bundled for immediate offline out-of-the-box operation.

---

## 6. Installation

Ensure Python 3.11+ is installed. Clone the repository and install dependencies:

```bash
# Clone the repository
git clone https://github.com/example/cyber-world-model.git
cd cyber-world-model

# Install dependencies
pip install -r requirements.txt
```

---

## 7. Training

Train both the Neural World Model and the Logistic Regression Baseline using the unified CLI:

```bash
python scripts/train.py
```

Options:
- `--config config.yaml`: Path to configuration file.
- `--data data/sample/sample_traffic.csv`: Optional custom training dataset.

The composite training loss balances physical state prediction with security objectives:
$$\mathcal{L} = \lambda_1 \text{MSE}(\hat{S}_{t+1}, S_{t+1}) + \lambda_2 \text{BCE}(\hat{p}_{\text{risk}}, y_{\text{risk}}) + \lambda_3 \text{CE}(\hat{y}_{\text{stage}}, y_{\text{stage}})$$

---

## 8. Inference

Execute $K$-step forward simulation on network traffic:

```bash
python scripts/predict.py --input data/sample/sample_traffic.csv --horizon 5
```

Example CLI Output:
```text
==================================================
      CYBER WORLD MODEL PREDICTION REPORT
==================================================
Current Attack Probability: 47.96% [MEDIUM]
Predicted Attack Stage:    Lateral Movement (Confidence: 18.7%)
MITRE Tactic:              Lateral Movement (TA0008)

Future Risk Timeline (Forward Rollout):
  Current State S(t)     -> Risk: 47.96% [MEDIUM]
  +1 window (10s)        -> Risk: 47.96% [MEDIUM]
  +2 window (20s)        -> Risk: 48.01% [MEDIUM]
  +3 window (30s)        -> Risk: 48.16% [MEDIUM]
  +4 window (40s)        -> Risk: 48.23% [MEDIUM]
  +5 window (50s)        -> Risk: 48.17% [MEDIUM]

Top Contributing Features (Attribution):
  1. unique_dst_ips           +0.0012 (positive (escalates risk))
  2. avg_iat                  -0.0006 (negative (suppresses risk))
  3. outbound_bytes           -0.0005 (negative (suppresses risk))
  4. unique_src_ips           +0.0005 (positive (escalates risk))
  5. unique_dst_ports         +0.0004 (positive (escalates risk))
```

---

## 9. Dashboard Usage

Launch the offline Streamlit interactive dashboard:

```bash
streamlit run dashboard/app.py
```

Dashboard Features:
1. **Telemetry Upload**: Upload custom CSV or PCAP files or use bundled sample data.
2. **Configurable Windows**: Select time window duration (5s, 10s, 30s, 60s).
3. **Forward Simulation Rollout**: Interactive Plotly projection of attack risk across $K$ future time windows.
4. **MITRE ATT&CK Knowledge Card**: Displays tactic, technique IDs, descriptions, and recommended defensive actions.
5. **Feature Attribution**: Bar chart of top contributing features (escalating vs suppressing risk).
6. **Temporal Attribution**: Timeline indicating which historical time window was most influential.
7. **Network Topology Graph**: NetworkX spring-layout communication graph highlighting suspicious nodes and pivots.
8. **Suspicious Flows Table**: Real-time filtering and inspection of active flows.
9. **One-Click Export**: Download `prediction.json` and `results.csv`.

---

## 10. API Usage

Start the FastAPI backend:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Interactive Swagger API documentation is available at: `http://localhost:8000/docs`.

### Key Endpoints:
- `GET  /health`: Model health, loaded telemetry, and status.
- `POST /upload`: Ingest new CSV or PCAP traffic files.
- `POST /extract-features`: Retrieve time-windowed state vectors $S(t)$.
- `POST /predict`: Execute forward inference (current risk, future timeline, MITRE stage, top features).
- `POST /simulate`: Multi-step forward simulation of state matrices.
- `GET  /risk-timeline`: Fetch the forward rollout risk progression.
- `GET  /attack-stage`: Detailed MITRE report and kill-chain forecast.
- `GET  /explanation`: Feature attribution and historical window influence.

---

## 11. Evaluation

Evaluate the World Model against the Logistic Regression Baseline strictly on an unseen chronological test partition:

```bash
python scripts/evaluate.py
```

Generated Artifacts:
- `reports/model_comparison.csv`: Side-by-side performance table.
- `reports/confusion_matrix.png`: Heatmap on test partition.
- `reports/risk_timeline.png`: True vs. predicted risk calibration over time.
- `reports/results.csv`: Window-by-window ground truth and predicted values.

---

## 12. Explainability

Predictions provide transparent feature and temporal attributions:
- **Integrated Gradients / Path Attribution**: Computes attribution scores along straight-line paths from neutral zero-state baselines to observed states.
- **Directional Impact**: Distinguishes features that escalate risk (`+`) from those that suppress risk (`-`).
- **Temporal Window Attribution**: Identifies which time step in the historical sequence ($t-5, \dots, t$) drove the prediction.
- *Attribution Disclaimer*: All explanations reflect mathematical model sensitivity and feature attribution, not verified physical causality.

---

## 13. MITRE ATT&CK Mapping

The system categorizes enterprise behavior into 6 progression stages:
1. **Stage 0 - Benign / Normal Operations**: Baseline corporate traffic.
2. **Stage 1 - Reconnaissance (`TA0043`)**: Active IP sweeps (`T1595.001`), network service discovery (`T1046`).
3. **Stage 2 - Initial Access (`TA0001`)**: Exploitation of public services (`T1190`), brute force (`T1133`).
4. **Stage 3 - Lateral Movement (`TA0008`)**: Remote SMB/RPC pivots (`T1021.002`), RDP sessions (`T1021.001`).
5. **Stage 4 - Command & Control (`TA0011`)**: Periodic beaconing (`T1071.001`), non-standard protocols (`T1095`).
6. **Stage 5 - Exfiltration (`TA0010`)**: High-volume egress transfer (`T1048.003`), web service theft (`T1567`).

---

## 14. Limitations

1. **Passive Telemetry**: The world model operates on network flow metadata and packet headers. Encrypted payload contents (e.g. encrypted TLS application payloads) are not decrypted.
2. **Dynamic Topologies**: Drastic sudden network topology reconfigurations (e.g., major cloud subnet re-addressing) require retraining the state scaler.
3. **Attribution Scope**: Feature attribution highlights correlation within the model's learned dynamics; it does not constitute deterministic forensic proof of attacker actions.

---

## 15. Future Improvements

1. **Graph Neural Network (GNN) World Model**: Integrating PyTorch Geometric to natively model dynamic graph topology transitions ($G(t) \to G(t+1)$).
2. **Hierarchical Multi-Scale Horizons**: Simultaneous simulation over short-term (10s) and long-term (1 hour) windows.
3. **Active Mitigation Interlock**: Interfacing the prediction head with firewall API hooks (e.g. iptables/pfSense/Palo Alto) for autonomous pre-emptive isolation upon CRITICAL future risk projections.

---

## Running the Complete System in 3 Commands

```bash
# 1. Run unit and integration tests
python -m pytest tests/test_all.py -v

# 2. Run forward prediction on sample traffic
python scripts/predict.py --input data/sample/sample_traffic.csv --horizon 5

# 3. Launch interactive dashboard
streamlit run dashboard/app.py
```
#   A I - C y b e r W M  
 