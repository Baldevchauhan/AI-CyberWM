"""
Predictive Cyber Defence - AI Network World Model Dashboard.
Offline-capable interactive Streamlit command center for network state simulation,
attack risk forecasting, MITRE ATT&CK mapping, and SHAP-aligned explainability.
"""

import sys
import json
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import torch
import joblib

# Add project root to Python path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.utils.config import load_config
from src.data.datasets.generic_csv import GenericCSVAdapter
from src.data.datasets.cicids2018 import CICIDS2018Adapter
from src.data.datasets.ctu13 import CTU13Adapter
from src.data.parsers.pcap_parser import PCAPParser
from src.features.state_builder import NetworkStateBuilder, STATE_FEATURE_NAMES
from src.models.lstm_world_model import LSTMNetworkWorldModel
from src.prediction.risk import RiskEstimator
from src.prediction.attack_stage import AttackStagePredictor
from src.prediction.rollout import ForwardRolloutSimulator
from src.explainability.shap_explainer import WorldModelExplainer
from src.utils.graph import NetworkGraphBuilder


# Page configuration
st.set_page_config(
    page_title="AI World Model - Predictive Cyber Defence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Cyber Threat Intelligence CSS Styling
st.markdown("""
<style>
    /* Dark Cyber Theme */
    .stApp {
        background-color: #0B0F19;
        color: #E2E8F0;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Metrics and Status Cards */
    .cyber-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7), rgba(15, 23, 42, 0.8));
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 10px;
        padding: 18px 22px;
        margin-bottom: 15px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }
    
    .cyber-header {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #94A3B8;
        margin-bottom: 6px;
    }
    
    .cyber-value {
        font-size: 2.1rem;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    
    .status-low { color: #10B981; }
    .status-medium { color: #F59E0B; }
    .status-high { color: #F97316; }
    .status-critical { color: #EF4444; }

    /* Streamlit tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 44px;
        white-space: pre-wrap;
        background-color: #1E293B;
        border-radius: 6px 6px 0 0;
        color: #94A3B8;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0284C7 !important;
        color: #FFFFFF !important;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_system_resources():
    """Caches config, model, scaler, and core components."""
    cfg = load_config()
    device = cfg["system"].get("device", "cpu")
    
    ckpt_path = Path(cfg["training"]["checkpoint_dir"]) / "best_world_model.pt"
    scaler_path = Path(cfg["training"]["checkpoint_dir"]) / "state_scaler.joblib"
    
    model = None
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model = LSTMNetworkWorldModel(
            state_dim=cfg["model"]["state_dim"],
            hidden_dim=cfg["model"]["hidden_dim"],
            latent_dim=cfg["model"]["latent_dim"],
            num_layers=cfg["model"]["num_layers"],
            dropout=cfg["model"]["dropout"],
            num_stages=cfg["model"]["num_stages"]
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        model.eval()

    scaler = joblib.load(scaler_path) if scaler_path.exists() else None

    return cfg, model, scaler, device


cfg, model, scaler, device = load_system_resources()

# Sidebar: Telemetry Ingestion & Hyperparameters
with st.sidebar:
    st.image("https://img.shields.io/badge/Cyber%20World%20Model-v1.0.0-0284C7?style=for-the-badge", use_container_width=True)
    st.markdown("### 📡 Telemetry Ingestion")

    data_source_mode = st.radio(
        "Telemetry Source",
        ["Default Sample Traffic", "Upload CSV File", "Upload PCAP File"]
    )

    dataset_format = st.selectbox(
        "Dataset Adapter / Schema",
        ["generic_csv", "cicids2018", "ctu13"],
        index=0,
        help="Selects column translation adapter for network traffic."
    )

    window_size = st.select_slider(
        "State Window Duration S(t)",
        options=[5, 10, 30, 60],
        value=10,
        help="Fixed time duration to aggregate packet/flow events into state vector S(t)."
    )

    simulation_horizon = st.slider(
        "Rollout Horizon (K-step)",
        min_value=1,
        max_value=10,
        value=5,
        help="Number of forward time windows to project state transition dynamics S(t+1)...S(t+K)."
    )

    enable_graph = st.toggle("Enable Network Graph", value=True)

    st.markdown("---")
    st.caption("🛡️ **AI Predictive Cyber Defence**\nTemporal Dynamics • K-step Rollout • MITRE Mapping • Explainable Alerts")


# Ingest Data based on User Selection
@st.cache_data
def process_telemetry(source_mode, uploaded_file, fmt, w_size):
    if source_mode == "Default Sample Traffic":
        sample_file = ROOT_DIR / "data" / "sample" / "sample_traffic.csv"
        adapter = GenericCSVAdapter()
        df = adapter.load(str(sample_file))
        return df, "data/sample/sample_traffic.csv"
    elif source_mode == "Upload CSV File" and uploaded_file is not None:
        if fmt == "cicids2018":
            adapter = CICIDS2018Adapter()
        elif fmt == "ctu13":
            adapter = CTU13Adapter()
        else:
            adapter = GenericCSVAdapter()
        df = adapter.load(pd.read_csv(uploaded_file))
        return df, uploaded_file.name
    elif source_mode == "Upload PCAP File" and uploaded_file is not None:
        temp_pcap = ROOT_DIR / "data" / "raw" / uploaded_file.name
        temp_pcap.parent.mkdir(parents=True, exist_ok=True)
        with open(temp_pcap, "wb") as f:
            f.write(uploaded_file.getbuffer())
        parser = PCAPParser()
        df = parser.parse(str(temp_pcap))
        return df, uploaded_file.name
    else:
        sample_file = ROOT_DIR / "data" / "sample" / "sample_traffic.csv"
        adapter = GenericCSVAdapter()
        df = adapter.load(str(sample_file))
        return df, "data/sample/sample_traffic.csv"


uploaded_file = None
if data_source_mode in ["Upload CSV File", "Upload PCAP File"]:
    file_type = ["csv"] if data_source_mode == "Upload CSV File" else ["pcap", "pcapng"]
    uploaded_file = st.sidebar.file_uploader(f"Choose a {file_type[0].upper()} file", type=file_type)

flows_df, current_filename = process_telemetry(data_source_mode, uploaded_file, dataset_format, window_size)

# Build States
builder = NetworkStateBuilder(window_size_seconds=window_size)
if scaler is not None:
    builder.scaler = scaler
    builder.is_fitted = True
states_df = builder.build_states(flows_df)

# Check model availability
if model is None:
    st.error("⚠️ World Model checkpoint not found. Please train the model via `python scripts/train.py`.")
    st.stop()

# Build Predictors and Explainer
risk_estimator = RiskEstimator(
    low_threshold=cfg["thresholds"]["low"],
    medium_threshold=cfg["thresholds"]["medium"],
    high_threshold=cfg["thresholds"]["high"]
)
stage_predictor = AttackStagePredictor(feature_names=STATE_FEATURE_NAMES)
simulator = ForwardRolloutSimulator(
    model=model,
    feature_names=STATE_FEATURE_NAMES,
    risk_estimator=risk_estimator,
    stage_predictor=stage_predictor,
    device=device
)
explainer = WorldModelExplainer(model=model, feature_names=STATE_FEATURE_NAMES, device=device)

# Prepare Sliding Sequence
seq_len = cfg["data"]["sequence_length"]
X, _, _, _ = builder.create_sliding_sequences(states_df, seq_length=seq_len, horizon=simulation_horizon)

# Run Inference & Rollout on latest state
current_seq_tensor = torch.tensor(X[-1:], dtype=torch.float32)
current_state_raw = states_df[STATE_FEATURE_NAMES].iloc[-1].values

sim_results = simulator.simulate(
    sequence_tensor=current_seq_tensor,
    current_state_raw=current_state_raw,
    horizon=simulation_horizon,
    window_duration=window_size
)

explanation = explainer.explain_prediction(current_seq_tensor, target="risk")

# Header Section
st.title("🛡️ Predictive Cyber Defence World Model")
st.markdown(f"**Autonomous Temporal Network Dynamics & Kill-Chain Rollout** | Telemetry: `{current_filename}` ({len(flows_df)} flows, {len(states_df)} states of {window_size}s)")

# Top KPI Summary Cards
curr_risk = sim_results["current_attack_probability"]
curr_sev = sim_results["current_severity"]
pred_stage = sim_results["predicted_stage"]
stage_conf = sim_results["stage_confidence"]

col1, col2, col3, col4 = st.columns(4)

with col1:
    sev_class = f"status-{curr_sev.lower()}"
    st.markdown(f"""
    <div class="cyber-card">
        <div class="cyber-header">Current Attack Risk S(t)</div>
        <div class="cyber-value {sev_class}">{curr_risk:.1%}</div>
        <div style="font-size:0.8rem; color:#94A3B8;">Severity Level: <strong class="{sev_class}">{curr_sev}</strong></div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="cyber-card">
        <div class="cyber-header">Predicted Attack Stage</div>
        <div class="cyber-value" style="color: #38BDF8; font-size: 1.6rem;">{pred_stage}</div>
        <div style="font-size:0.8rem; color:#94A3B8;">Model Confidence: <strong>{stage_conf:.1%}</strong></div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    mitre_tactic = sim_results["mitre_report"]["mitre_tactic_name"]
    mitre_id = sim_results["mitre_report"]["mitre_tactic_id"]
    st.markdown(f"""
    <div class="cyber-card">
        <div class="cyber-header">MITRE ATT&CK Tactic</div>
        <div class="cyber-value" style="color: #A855F7; font-size: 1.6rem;">{mitre_id}</div>
        <div style="font-size:0.8rem; color:#94A3B8;">{mitre_tactic}</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    max_future_risk = max(sim_results["future_attack_probabilities"]) if sim_results["future_attack_probabilities"] else curr_risk
    delta_risk = max_future_risk - curr_risk
    delta_color = "#EF4444" if delta_risk > 0.05 else ("#10B981" if delta_risk < -0.05 else "#94A3B8")
    st.markdown(f"""
    <div class="cyber-card">
        <div class="cyber-header">Peak Future Risk (+{simulation_horizon} windows)</div>
        <div class="cyber-value" style="color: {delta_color};">{max_future_risk:.1%}</div>
        <div style="font-size:0.8rem; color:#94A3B8;">Trajectory Drift: <strong>{delta_risk:+.1%}</strong></div>
    </div>
    """, unsafe_allow_html=True)

# Main Multi-Tab Explorer
tab_rollout, tab_mitre, tab_explain, tab_graph, tab_flows, tab_export = st.tabs([
    "📈 Forward Simulation Rollout",
    "🎯 MITRE ATT&CK Intelligence",
    "🔍 Explainability & Attribution",
    "🌐 Network Topology Graph",
    "🔍 Suspicious Flows Inspector",
    "💾 Export Intelligence"
])

with tab_rollout:
    st.subheader(f"Temporal Network Trajectory & K-Step Rollout (Horizon K = {simulation_horizon})")
    st.caption("The World Model projects learned state transitions P(S(t+1) | S(t), history) forward through multi-step autoregressive simulation.")

    timeline_data = sim_results["risk_timeline"]
    df_timeline = pd.DataFrame(timeline_data)

    fig_timeline = go.Figure()

    # Alert threshold bands
    fig_timeline.add_hrect(y0=0.0, y1=0.30, fillcolor="#10B981", opacity=0.08, line_width=0, annotation_text="LOW RISK")
    fig_timeline.add_hrect(y0=0.30, y1=0.60, fillcolor="#F59E0B", opacity=0.08, line_width=0, annotation_text="MEDIUM RISK")
    fig_timeline.add_hrect(y0=0.60, y1=0.80, fillcolor="#F97316", opacity=0.08, line_width=0, annotation_text="HIGH RISK")
    fig_timeline.add_hrect(y0=0.80, y1=1.0, fillcolor="#EF4444", opacity=0.08, line_width=0, annotation_text="CRITICAL RISK")

    # Risk line
    fig_timeline.add_trace(go.Scatter(
        x=[row["label"] for row in timeline_data],
        y=[row["attack_probability"] for row in timeline_data],
        mode="lines+markers",
        name="Projected Attack Risk",
        line=dict(color="#06B6D4", width=3),
        marker=dict(size=10, color="#38BDF8", symbol="diamond"),
        hovertemplate="<b>%{x}</b><br>Attack Probability: %{y:.2%}<extra></extra>"
    ))

    fig_timeline.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0F172A",
        plot_bgcolor="#0F172A",
        height=380,
        margin=dict(l=40, r=40, t=30, b=30),
        yaxis=dict(range=[0.0, 1.05], tickformat=".0%", title="Attack Probability P(Attack)"),
        xaxis=dict(title="Forward Rollout Window Horizon")
    )
    st.plotly_chart(fig_timeline, use_container_width=True)

    # State vector breakdown
    with st.expander("🔬 View Current Network State Vector S(t) (20 Canonical Features)"):
        st.dataframe(pd.DataFrame([current_state_raw], columns=STATE_FEATURE_NAMES), use_container_width=True)


with tab_mitre:
    st.subheader("MITRE ATT&CK Framework Mapping")
    mitre_rep = sim_results["mitre_report"]

    st.markdown(f"### **Tactic:** `{mitre_rep['mitre_tactic_id']} - {mitre_rep['mitre_tactic_name']}`")
    st.info(f"**Description:** {mitre_rep['tactical_description']}")

    st.markdown("#### **Mapped Techniques & Sub-Techniques:**")
    if mitre_rep["techniques"]:
        for tech in mitre_rep["techniques"]:
            st.markdown(f"- **[{tech['id']} - {tech['name']}]({tech['url']})**: {tech['description']}")
    else:
        st.markdown("_No offensive MITRE techniques detected in normal baseline network state._")

    st.markdown("#### **Actionable Defensive Mitigations:**")
    st.success(f"🛡️ **SOC Guidance:** {mitre_rep['recommended_defensive_mitigation']}")

    st.markdown("#### **Observed Behavioral Indicators:**")
    cols = st.columns(len(mitre_rep["key_indicators"]))
    for idx, ind in enumerate(mitre_rep["key_indicators"]):
        with cols[idx % len(cols)]:
            st.markdown(f"🔹 `{ind}`")

    st.caption(f"⚠️ *{mitre_rep['disclaimer']}*")


with tab_explain:
    st.subheader("Model Attribution & Feature Explainability (SHAP / Integrated Gradients)")
    st.caption("Identifies the network traffic properties and historical time windows responsible for escalating model risk scores.")

    top_feats = explanation["top_features"]
    df_explain = pd.DataFrame(top_feats)

    # Bar chart for feature attribution
    colors = ["#EF4444" if row["attribution_score"] > 0 else "#10B981" for _, row in df_explain.iterrows()]

    fig_bar = go.Figure(go.Bar(
        x=df_explain["attribution_score"],
        y=df_explain["feature"],
        orientation="h",
        marker_color=colors,
        text=[f"{s:+.4f}" for s in df_explain["attribution_score"]],
        textposition="outside"
    ))
    fig_bar.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0F172A",
        plot_bgcolor="#0F172A",
        height=340,
        margin=dict(l=10, r=40, t=20, b=20),
        xaxis=dict(title="Attribution Score (Impact on Attack Risk)"),
        yaxis=dict(autorange="reversed")
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # Historical time-window attribution
    st.markdown("#### **Temporal Window Attribution (History Influence)**")
    t_windows = explanation["temporal_window_attribution"]
    df_twin = pd.DataFrame(t_windows)

    fig_twin = px.line(
        df_twin, x="label", y="relative_influence",
        markers=True, template="plotly_dark",
        labels={"label": "Historical Window", "relative_influence": "Relative Influence"}
    )
    fig_twin.update_traces(line_color="#A855F7", marker_size=8)
    fig_twin.update_layout(
        paper_bgcolor="#0F172A", plot_bgcolor="#0F172A",
        height=260, margin=dict(l=10, r=10, t=20, b=20)
    )
    st.plotly_chart(fig_twin, use_container_width=True)

    st.info(f"ℹ️ **Attribution Note:** {explanation['disclaimer']}")


with tab_graph:
    st.subheader("Network Communication Topology & Lateral Pivots")
    if enable_graph:
        graph_builder = NetworkGraphBuilder()
        graph_builder.build_from_flows(flows_df, max_nodes=35)
        fig_graph = graph_builder.to_plotly_figure()
        st.plotly_chart(fig_graph, use_container_width=True)
    else:
        st.info("Network Graph mode is disabled. Toggle 'Enable Network Graph' in the sidebar.")


with tab_flows:
    st.subheader("Suspicious & Active Flow Telemetry")
    col_filter1, col_filter2 = st.columns(2)
    with col_filter1:
        search_ip = st.text_input("Filter by IP Address", "")
    with col_filter2:
        only_malicious = st.checkbox("Show Malicious Flows Only", value=True)

    filtered_df = flows_df.copy()
    if search_ip:
        filtered_df = filtered_df[filtered_df["src_ip"].str.contains(search_ip) | filtered_df["dst_ip"].str.contains(search_ip)]
    if only_malicious:
        filtered_df = filtered_df[filtered_df["label"] == 1]

    st.dataframe(
        filtered_df[["timestamp", "src_ip", "dst_ip", "src_port", "dst_port", "protocol", "packets", "bytes", "label", "attack_stage"]].head(100),
        use_container_width=True
    )


with tab_export:
    st.subheader("Export Threat Intelligence & Simulation Artifacts")
    st.caption("Download structured prediction payloads for SIEM integration or forensic reporting.")

    export_payload = {
        "current_attack_probability": sim_results["current_attack_probability"],
        "current_severity": sim_results["current_severity"],
        "future_attack_probabilities": sim_results["future_attack_probabilities"],
        "risk_timeline": sim_results["risk_timeline"],
        "predicted_attack_stage": sim_results["predicted_stage"],
        "stage_confidence": sim_results["stage_confidence"],
        "mitre_report": sim_results["mitre_report"],
        "top_contributing_features": explanation["top_features"]
    }

    col_exp1, col_exp2 = st.columns(2)
    with col_exp1:
        st.download_button(
            label="📥 Download Prediction JSON (`prediction.json`)",
            data=json.dumps(export_payload, indent=2),
            file_name="prediction.json",
            mime="application/json"
        )
    with col_exp2:
        st.download_button(
            label="📥 Download Network States CSV (`results.csv`)",
            data=states_df.to_csv(index=False),
            file_name="results.csv",
            mime="text/csv"
        )
