"""
FastAPI Backend Application for AI World Model Predictive Cyber Defence.
Provides RESTful endpoints for network telemetry upload, state extraction,
K-step simulation rollout, risk timelines, MITRE ATT&CK mapping, and SHAP explainability.
"""

import os
import io
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional
import torch
import numpy as np
import pandas as pd
import joblib

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.utils.config import load_config
from src.data.datasets.generic_csv import GenericCSVAdapter
from src.data.parsers.pcap_parser import PCAPParser
from src.features.state_builder import NetworkStateBuilder, STATE_FEATURE_NAMES
from src.models.lstm_world_model import LSTMNetworkWorldModel
from src.prediction.risk import RiskEstimator
from src.prediction.attack_stage import AttackStagePredictor
from src.prediction.rollout import ForwardRolloutSimulator
from src.explainability.shap_explainer import WorldModelExplainer
from src.mitre.attack_mapping import MITREAttackMapper


# Initialize FastAPI app
app = FastAPI(
    title="AI Network World Model API",
    description="Predictive Cyber Defence API powered by temporal world modeling",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global in-memory cache for telemetry and model state
GLOBAL_STATE = {
    "config": None,
    "model": None,
    "scaler": None,
    "builder": None,
    "flows_df": None,
    "states_df": None,
    "last_prediction": None,
    "last_explanation": None,
    "device": "cpu"
}


# Pydantic Schemas
class HealthResponse(BaseModel):
    status: str
    version: str
    model_loaded: bool
    telemetry_loaded: bool
    total_flows: int
    total_states: int


class PredictRequest(BaseModel):
    horizon: int = Field(default=5, ge=1, le=20, description="K-step simulation forward horizon")


class PredictResponse(BaseModel):
    current_attack_probability: float
    current_severity: str
    future_attack_probability: List[float]
    predicted_stage: str
    confidence: float
    top_features: List[str]
    mitre_tactic: str
    mitre_technique: str


class SimulateRequest(BaseModel):
    horizon: int = Field(default=5, ge=1, le=20)
    window_duration: float = Field(default=10.0, ge=1.0)


def get_or_load_resources():
    """Lazily loads config, model, and fitted scaler."""
    if GLOBAL_STATE["config"] is None:
        cfg = load_config()
        GLOBAL_STATE["config"] = cfg
        device = cfg["system"].get("device", "cpu")
        GLOBAL_STATE["device"] = device

        ckpt_path = Path(cfg["training"]["checkpoint_dir"]) / "best_world_model.pt"
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
            GLOBAL_STATE["model"] = model

        scaler_path = Path(cfg["training"]["checkpoint_dir"]) / "state_scaler.joblib"
        if scaler_path.exists():
            GLOBAL_STATE["scaler"] = joblib.load(scaler_path)

        GLOBAL_STATE["builder"] = NetworkStateBuilder(
            window_size_seconds=cfg["data"]["window_size_seconds"]
        )
        if GLOBAL_STATE["scaler"] is not None:
            GLOBAL_STATE["builder"].scaler = GLOBAL_STATE["scaler"]
            GLOBAL_STATE["builder"].is_fitted = True

        # Pre-load sample data so API is immediately operational
        sample_path = Path(cfg["data"]["sample_data_path"])
        if sample_path.exists():
            adapter = GenericCSVAdapter()
            flows = adapter.load(str(sample_path))
            GLOBAL_STATE["flows_df"] = flows
            GLOBAL_STATE["states_df"] = GLOBAL_STATE["builder"].build_states(flows)


@app.on_event("startup")
def startup_event():
    get_or_load_resources()


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health and model status endpoint."""
    get_or_load_resources()
    flows_count = len(GLOBAL_STATE["flows_df"]) if GLOBAL_STATE["flows_df"] is not None else 0
    states_count = len(GLOBAL_STATE["states_df"]) if GLOBAL_STATE["states_df"] is not None else 0

    return HealthResponse(
        status="healthy",
        version="1.0.0",
        model_loaded=GLOBAL_STATE["model"] is not None,
        telemetry_loaded=flows_count > 0,
        total_flows=flows_count,
        total_states=states_count
    )


@app.post("/upload")
async def upload_traffic(
    file: UploadFile = File(...),
    dataset_type: str = Form("generic_csv"),
    window_size: float = Form(10.0)
):
    """
    Ingests network telemetry from CSV or PCAP file.
    Re-aggregates flows into time-windowed states S(t).
    """
    get_or_load_resources()
    filename = file.filename.lower()
    temp_dir = Path("data/raw")
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_file_path = temp_dir / file.filename

    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        if filename.endswith(".pcap") or filename.endswith(".pcapng"):
            parser = PCAPParser()
            flows_df = parser.parse(str(temp_file_path))
        else:
            if dataset_type == "cicids2018":
                from src.data.datasets.cicids2018 import CICIDS2018Adapter
                adapter = CICIDS2018Adapter()
            elif dataset_type == "ctu13":
                from src.data.datasets.ctu13 import CTU13Adapter
                adapter = CTU13Adapter()
            else:
                adapter = GenericCSVAdapter()
            flows_df = adapter.load(str(temp_file_path))

        # Re-build states
        builder = NetworkStateBuilder(window_size_seconds=window_size)
        if GLOBAL_STATE["scaler"] is not None:
            builder.scaler = GLOBAL_STATE["scaler"]
            builder.is_fitted = True

        states_df = builder.build_states(flows_df)

        GLOBAL_STATE["flows_df"] = flows_df
        GLOBAL_STATE["states_df"] = states_df
        GLOBAL_STATE["builder"] = builder

        return {
            "status": "success",
            "filename": file.filename,
            "flows_extracted": len(flows_df),
            "states_constructed": len(states_df),
            "window_size_seconds": window_size
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process telemetry: {str(e)}")


@app.post("/extract-features")
def extract_features():
    """Returns the latest extracted network states S(t) with all 20 canonical features."""
    get_or_load_resources()
    if GLOBAL_STATE["states_df"] is None or GLOBAL_STATE["states_df"].empty:
        raise HTTPException(status_code=404, detail="No telemetry loaded. Upload traffic first.")

    states = GLOBAL_STATE["states_df"]
    return {
        "num_states": len(states),
        "feature_names": STATE_FEATURE_NAMES,
        "latest_state": states.iloc[-1].to_dict(),
        "recent_history": states.tail(10).to_dict(orient="records")
    }


@app.post("/predict", response_model=PredictResponse)
def predict_network_trajectory(req: PredictRequest = PredictRequest()):
    """
    Executes world model inference, predicting current risk, future risk rollout,
    predicted MITRE ATT&CK stage, and top contributing attribution features.
    """
    get_or_load_resources()
    if GLOBAL_STATE["model"] is None:
        raise HTTPException(status_code=500, detail="World Model is not trained or loaded.")
    if GLOBAL_STATE["states_df"] is None or GLOBAL_STATE["states_df"].empty:
        raise HTTPException(status_code=400, detail="No telemetry available. Please upload traffic.")

    cfg = GLOBAL_STATE["config"]
    builder = GLOBAL_STATE["builder"]
    states_df = GLOBAL_STATE["states_df"]
    seq_len = cfg["data"]["sequence_length"]

    X, _, _, _ = builder.create_sliding_sequences(states_df, seq_length=seq_len, horizon=req.horizon)
    current_seq_tensor = torch.tensor(X[-1:], dtype=torch.float32)
    current_state_raw = states_df[STATE_FEATURE_NAMES].iloc[-1].values

    risk_est = RiskEstimator(
        low_threshold=cfg["thresholds"]["low"],
        medium_threshold=cfg["thresholds"]["medium"],
        high_threshold=cfg["thresholds"]["high"]
    )
    stage_pred = AttackStagePredictor(feature_names=STATE_FEATURE_NAMES)
    simulator = ForwardRolloutSimulator(
        model=GLOBAL_STATE["model"],
        feature_names=STATE_FEATURE_NAMES,
        risk_estimator=risk_est,
        stage_predictor=stage_pred,
        device=GLOBAL_STATE["device"]
    )

    sim_results = simulator.simulate(
        sequence_tensor=current_seq_tensor,
        current_state_raw=current_state_raw,
        horizon=req.horizon,
        window_duration=cfg["data"]["window_size_seconds"]
    )

    # Attribution
    explainer = WorldModelExplainer(
        model=GLOBAL_STATE["model"],
        feature_names=STATE_FEATURE_NAMES,
        device=GLOBAL_STATE["device"]
    )
    explanation = explainer.explain_prediction(current_seq_tensor, target="risk")

    top_feat_names = [f["feature"] for f in explanation["top_features"][:4]]

    GLOBAL_STATE["last_prediction"] = sim_results
    GLOBAL_STATE["last_explanation"] = explanation

    tech_name = "N/A"
    techs = sim_results["mitre_report"]["techniques"]
    if techs:
        tech_name = f"{techs[0]['id']} - {techs[0]['name']}"

    return PredictResponse(
        current_attack_probability=sim_results["current_attack_probability"],
        current_severity=sim_results["current_severity"],
        future_attack_probability=sim_results["future_attack_probabilities"],
        predicted_stage=sim_results["predicted_stage"],
        confidence=sim_results["stage_confidence"],
        top_features=top_feat_names,
        mitre_tactic=sim_results["mitre_report"]["mitre_tactic_name"],
        mitre_technique=tech_name
    )


@app.post("/simulate")
def simulate_forward_states(req: SimulateRequest = SimulateRequest()):
    """Full K-step simulation returning state matrices S(t+1)...S(t+K) and risk timeline."""
    predict_res = predict_network_trajectory(PredictRequest(horizon=req.horizon))
    sim_data = GLOBAL_STATE.get("last_prediction", {})
    return {
        "status": "success",
        "horizon": req.horizon,
        "current_attack_probability": sim_data.get("current_attack_probability"),
        "risk_timeline": sim_data.get("risk_timeline", []),
        "future_state_trajectories": sim_data.get("future_state_trajectories", [])
    }


@app.get("/risk-timeline")
def get_risk_timeline():
    """Returns the forward risk timeline generated during the last simulation."""
    if GLOBAL_STATE.get("last_prediction") is None:
        predict_network_trajectory()
    return {
        "risk_timeline": GLOBAL_STATE["last_prediction"]["risk_timeline"]
    }


@app.get("/attack-stage")
def get_attack_stage_info():
    """Returns predicted attack stage and mapped MITRE ATT&CK report."""
    if GLOBAL_STATE.get("last_prediction") is None:
        predict_network_trajectory()
    return GLOBAL_STATE["last_prediction"]["mitre_report"]


@app.get("/explanation")
def get_explanation():
    """Returns top contributing features and temporal window influence."""
    if GLOBAL_STATE.get("last_explanation") is None:
        predict_network_trajectory()
    return GLOBAL_STATE["last_explanation"]
