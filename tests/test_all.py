"""
Comprehensive Unit and Integration Test Suite for the Cyber World Model System.
"""

import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.data.datasets.base import CANONICAL_COLUMNS
from src.data.datasets.generic_csv import GenericCSVAdapter
from src.data.datasets.cicids2018 import CICIDS2018Adapter
from src.data.datasets.ctu13 import CTU13Adapter
from src.data.parsers.pcap_parser import PCAPParser
from src.features.state_builder import NetworkStateBuilder, STATE_FEATURE_NAMES
from src.models.lstm_world_model import LSTMNetworkWorldModel
from src.models.transformer_world_model import TransformerNetworkWorldModel
from src.models.baseline import LogisticRegressionBaseline
from src.prediction.risk import RiskEstimator
from src.prediction.attack_stage import AttackStagePredictor
from src.prediction.rollout import ForwardRolloutSimulator
from src.mitre.attack_mapping import MITREAttackMapper
from src.explainability.shap_explainer import WorldModelExplainer
from fastapi.testclient import TestClient
from api.main import app


def test_canonical_dataset_adapters():
    """Verify that dataset adapters normalize arbitrary inputs into canonical columns."""
    sample_csv = ROOT_DIR / "data" / "sample" / "sample_traffic.csv"
    adapter = GenericCSVAdapter()
    df = adapter.load(str(sample_csv))

    assert not df.empty
    for col in CANONICAL_COLUMNS:
        assert col in df.columns, f"Missing canonical column: {col}"

    assert (df["label"].isin([0, 1])).all()
    assert (df["attack_stage"].isin(range(6))).all()


def test_network_state_builder():
    """Verify state builder creates valid 20-dim state vectors and sequence sliding windows."""
    sample_csv = ROOT_DIR / "data" / "sample" / "sample_traffic.csv"
    adapter = GenericCSVAdapter()
    df = adapter.load(str(sample_csv))

    builder = NetworkStateBuilder(window_size_seconds=10.0)
    states_df = builder.build_states(df)

    assert not states_df.empty
    assert len(states_df) >= 5

    for feat in STATE_FEATURE_NAMES:
        assert feat in states_df.columns, f"Missing state feature: {feat}"

    builder.fit_scaler(states_df)
    X, Y_next, Y_risk, Y_stage = builder.create_sliding_sequences(states_df, seq_length=6, horizon=5)

    assert X.ndim == 3
    assert X.shape[1] == 6
    assert X.shape[2] == 20
    assert len(X) == len(Y_next) == len(Y_risk) == len(Y_stage)


def test_lstm_world_model_forward_and_rollout():
    """Verify LSTM world model forward pass shapes and K-step autoregressive rollout."""
    state_dim = 20
    seq_len = 6
    batch_size = 4
    horizon = 5

    model = LSTMNetworkWorldModel(
        state_dim=state_dim,
        hidden_dim=32,
        latent_dim=16,
        num_layers=1,
        dropout=0.1,
        num_stages=6
    )

    dummy_x = torch.randn(batch_size, seq_len, state_dim)
    next_state, risk, stage_logits, latent = model(dummy_x)

    assert next_state.shape == (batch_size, state_dim)
    assert risk.shape == (batch_size, 1)
    assert (risk >= 0.0).all() and (risk <= 1.0).all()
    assert stage_logits.shape == (batch_size, 6)
    assert latent.shape == (batch_size, 16)

    # Rollout
    rollout_res = model.rollout(dummy_x[0], steps=horizon)
    assert len(rollout_res["future_states"]) == horizon
    assert len(rollout_res["risk_timeline"]) == horizon
    assert len(rollout_res["stage_timeline"]) == horizon


def test_transformer_world_model_forward_and_rollout():
    """Verify Transformer world model modular alternative."""
    state_dim = 20
    seq_len = 6
    batch_size = 2
    horizon = 3

    model = TransformerNetworkWorldModel(
        state_dim=state_dim,
        hidden_dim=32,
        latent_dim=16,
        num_layers=1,
        nhead=2,
        dropout=0.1,
        num_stages=6
    )

    dummy_x = torch.randn(batch_size, seq_len, state_dim)
    next_state, risk, stage_logits, _ = model(dummy_x)

    assert next_state.shape == (batch_size, state_dim)
    assert risk.shape == (batch_size, 1)
    assert stage_logits.shape == (batch_size, 6)

    rollout_res = model.rollout(dummy_x[0], steps=horizon)
    assert len(rollout_res["future_states"]) == horizon


def test_mitre_mapping():
    """Verify MITRE ATT&CK taxonomy mappings."""
    for stage_id in range(6):
        info = MITREAttackMapper.get_stage_info(stage_id)
        assert "tactic_name" in info
        assert "tactic_id" in info
        assert "defensive_action" in info

    rep = MITREAttackMapper.map_prediction(
        predicted_stage=1,
        confidence=0.88,
        supporting_features=["syn_rate", "unique_dst_ports"]
    )
    assert rep["predicted_stage_name"] == "Reconnaissance"
    assert rep["mitre_tactic_id"] == "TA0043"


def test_explainability_engine():
    """Verify SHAP / Integrated Gradients attribution outputs valid rankings."""
    model = LSTMNetworkWorldModel(state_dim=20, hidden_dim=32, latent_dim=16)
    explainer = WorldModelExplainer(model, feature_names=STATE_FEATURE_NAMES)

    dummy_seq = torch.randn(1, 6, 20)
    attr = explainer.explain_prediction(dummy_seq, target="risk", num_steps=5)

    assert "top_features" in attr
    assert len(attr["top_features"]) > 0
    assert "attribution_score" in attr["top_features"][0]
    assert "temporal_window_attribution" in attr
    assert len(attr["temporal_window_attribution"]) == 6


def test_fastapi_endpoints():
    """Verify FastAPI backend operational endpoints."""
    client = TestClient(app)

    # Health
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "healthy"

    # Predict
    r_pred = client.post("/predict", json={"horizon": 5})
    assert r_pred.status_code == 200
    p_data = r_pred.json()
    assert "current_attack_probability" in p_data
    assert len(p_data["future_attack_probability"]) == 5
    assert "predicted_stage" in p_data

    # Risk Timeline
    r_time = client.get("/risk-timeline")
    assert r_time.status_code == 200
    assert "risk_timeline" in r_time.json()

    # Explanation
    r_exp = client.get("/explanation")
    assert r_exp.status_code == 200
    assert "top_features" in r_exp.json()
