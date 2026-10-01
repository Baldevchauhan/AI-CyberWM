"""
CLI script to perform predictive forward inference and rollout on network traffic.
Usage:
    python scripts/predict.py --input data/sample/sample_traffic.csv --horizon 5
"""

import argparse
import sys
import json
from pathlib import Path
import torch
import numpy as np
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_config
from src.data.datasets.generic_csv import GenericCSVAdapter
from src.features.state_builder import NetworkStateBuilder, STATE_FEATURE_NAMES
from src.models.lstm_world_model import LSTMNetworkWorldModel
from src.prediction.risk import RiskEstimator
from src.prediction.attack_stage import AttackStagePredictor
from src.prediction.rollout import ForwardRolloutSimulator
from src.explainability.shap_explainer import WorldModelExplainer


def main():
    parser = argparse.ArgumentParser(description="Predictive Cyber Defense Forward Inference")
    parser.add_argument("--input", type=str, required=True, help="Input CSV or PCAP file")
    parser.add_argument("--horizon", type=int, default=5, help="Forward simulation horizon K (steps)")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--output", type=str, default=None, help="Optional output JSON path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    window_sec = cfg["data"]["window_size_seconds"]
    seq_len = cfg["data"]["sequence_length"]
    device = cfg["system"].get("device", "cpu")

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file {input_path} does not exist.")
        sys.exit(1)

    print(f"Loading input traffic from: {input_path}...")
    if input_path.suffix.lower() in [".pcap", ".pcapng"]:
        from src.data.parsers.pcap_parser import PCAPParser
        parser_pcap = PCAPParser()
        flows_df = parser_pcap.parse(str(input_path))
    else:
        adapter = GenericCSVAdapter()
        flows_df = adapter.load(str(input_path))

    builder = NetworkStateBuilder(window_size_seconds=window_sec)
    states_df = builder.build_states(flows_df)

    scaler_path = Path(cfg["training"]["checkpoint_dir"]) / "state_scaler.joblib"
    if scaler_path.exists():
        builder.scaler = joblib.load(scaler_path)
        builder.is_fitted = True
    else:
        builder.fit_scaler(states_df)

    # Prepare sliding sequence
    X, _, _, _ = builder.create_sliding_sequences(states_df, seq_length=seq_len, horizon=args.horizon)

    # Load model
    ckpt_path = Path(cfg["training"]["checkpoint_dir"]) / "best_world_model.pt"
    if not ckpt_path.exists():
        print(f"Model checkpoint not found at {ckpt_path}. Please train the model first with: python scripts/train.py")
        sys.exit(1)

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

    risk_est = RiskEstimator(
        low_threshold=cfg["thresholds"]["low"],
        medium_threshold=cfg["thresholds"]["medium"],
        high_threshold=cfg["thresholds"]["high"]
    )
    stage_pred = AttackStagePredictor(feature_names=STATE_FEATURE_NAMES)
    simulator = ForwardRolloutSimulator(
        model=model,
        feature_names=STATE_FEATURE_NAMES,
        risk_estimator=risk_est,
        stage_predictor=stage_pred,
        device=device
    )

    # Use the most recent observed trajectory
    current_seq_tensor = torch.tensor(X[-1:], dtype=torch.float32)
    current_state_raw = states_df[STATE_FEATURE_NAMES].iloc[-1].values

    print("\nExecuting K-step forward simulation rollout...")
    sim_results = simulator.simulate(
        sequence_tensor=current_seq_tensor,
        current_state_raw=current_state_raw,
        horizon=args.horizon,
        window_duration=window_sec
    )

    print("Computing feature attribution explanations...")
    explainer = WorldModelExplainer(model=model, feature_names=STATE_FEATURE_NAMES, device=device)
    explanation = explainer.explain_prediction(current_seq_tensor, target="risk")

    output_payload = {
        "current_attack_probability": sim_results["current_attack_probability"],
        "current_severity": sim_results["current_severity"],
        "future_attack_probabilities": sim_results["future_attack_probabilities"],
        "risk_timeline": sim_results["risk_timeline"],
        "predicted_attack_stage": sim_results["predicted_stage"],
        "confidence": sim_results["stage_confidence"],
        "mitre_attack_intelligence": sim_results["mitre_report"],
        "top_contributing_features": explanation["top_features"][:5],
        "temporal_window_attribution": explanation["temporal_window_attribution"]
    }

    print("\n" + "="*50)
    print("      CYBER WORLD MODEL PREDICTION REPORT")
    print("="*50)
    print(f"Current Attack Probability: {output_payload['current_attack_probability']:.2%} [{output_payload['current_severity']}]")
    print(f"Predicted Attack Stage:    {output_payload['predicted_attack_stage']} (Confidence: {output_payload['confidence']:.1%})")
    print(f"MITRE Tactic:              {output_payload['mitre_attack_intelligence']['mitre_tactic_name']} ({output_payload['mitre_attack_intelligence']['mitre_tactic_id']})")
    print("\nFuture Risk Timeline (Forward Rollout):")
    for item in output_payload["risk_timeline"]:
        print(f"  {item['label']:<22} -> Risk: {item['attack_probability']:.2%} [{item['severity']}]")

    print("\nTop Contributing Features (Attribution):")
    for idx, feat in enumerate(output_payload["top_contributing_features"], 1):
        print(f"  {idx}. {feat['feature']:<24} {feat['attribution_score']:+.4f} ({feat['direction']})")

    out_file = args.output or "prediction_output.json"
    with open(out_file, "w") as f:
        json.dump(output_payload, f, indent=2)
    print(f"\nSaved complete prediction report to: {out_file}")


if __name__ == "__main__":
    main()
