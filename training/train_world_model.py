"""
World Model Training Pipeline.
Trains the temporal neural world model on historical network states using composite loss:
L_total = lambda_state * MSE(State) + lambda_risk * BCE(Risk) + lambda_stage * CrossEntropy(Stage).
Saves model checkpoint and scaler for inference and simulation rollout.
"""

import os
from pathlib import Path
from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
import joblib

from src.utils.config import load_config
from src.data.datasets.generic_csv import GenericCSVAdapter
from src.features.state_builder import NetworkStateBuilder
from src.data.preprocessing.temporal_split import temporal_sequence_split
from src.models.lstm_world_model import LSTMNetworkWorldModel
from src.models.transformer_world_model import TransformerNetworkWorldModel


def train_world_model(
    config_path: str = "config.yaml",
    dataset_csv_path: str | None = None
) -> Dict[str, Any]:
    """Trains the network world model and saves model checkpoints."""
    cfg = load_config(config_path)
    seed = cfg["system"].get("random_seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    csv_path = dataset_csv_path or cfg["data"]["sample_data_path"]
    window_sec = cfg["data"]["window_size_seconds"]
    seq_len = cfg["data"]["sequence_length"]
    horizon = cfg["data"]["prediction_horizon"]
    device = cfg["system"].get("device", "cpu")

    # Ingest CSV via generic adapter
    adapter = GenericCSVAdapter()
    flows_df = adapter.load(csv_path)

    # Build network states
    builder = NetworkStateBuilder(window_size_seconds=window_sec)
    states_df = builder.build_states(flows_df)

    # Fit scaler on states
    builder.fit_scaler(states_df)

    # Generate sliding sequences
    X, Y_next, Y_risk, Y_stage = builder.create_sliding_sequences(
        states_df, seq_length=seq_len, horizon=horizon
    )

    # Chronological partition
    train_data, val_data, test_data = temporal_sequence_split(
        X, Y_next, Y_stage,
        train_ratio=cfg["data"]["train_split"],
        val_ratio=cfg["data"]["val_split"]
    )
    # Also partition risk labels
    n = len(X)
    train_end = int(n * cfg["data"]["train_split"])
    val_end = int(n * (cfg["data"]["train_split"] + cfg["data"]["val_split"]))

    X_train, Y_next_train, Y_stage_train = train_data
    Y_risk_train = Y_risk[:train_end]

    X_val, Y_next_val, Y_stage_val = val_data
    Y_risk_val = Y_risk[train_end:val_end]

    # DataLoaders
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(Y_next_train, dtype=torch.float32),
        torch.tensor(Y_risk_train, dtype=torch.float32).unsqueeze(1),
        torch.tensor(Y_stage_train, dtype=torch.int64)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(Y_next_val, dtype=torch.float32),
        torch.tensor(Y_risk_val, dtype=torch.float32).unsqueeze(1),
        torch.tensor(Y_stage_val, dtype=torch.int64)
    )

    batch_size = cfg["training"]["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    model_type = cfg["model"].get("type", "lstm")
    state_dim = cfg["model"]["state_dim"]
    hidden_dim = cfg["model"]["hidden_dim"]
    latent_dim = cfg["model"]["latent_dim"]
    num_layers = cfg["model"]["num_layers"]
    dropout = cfg["model"]["dropout"]
    num_stages = cfg["model"]["num_stages"]

    if model_type == "transformer":
        model = TransformerNetworkWorldModel(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=num_layers,
            dropout=dropout,
            num_stages=num_stages
        )
    else:
        model = LSTMNetworkWorldModel(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=num_layers,
            dropout=dropout,
            num_stages=num_stages
        )

    model.to(device)

    # Loss functions & Optimizer
    criterion_state = nn.MSELoss()
    criterion_risk = nn.BCELoss()
    criterion_stage = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"]
    )

    lambda_state = cfg["training"]["lambda_state"]
    lambda_risk = cfg["training"]["lambda_risk"]
    lambda_stage = cfg["training"]["lambda_stage"]
    epochs = cfg["training"]["epochs"]

    best_val_loss = float("inf")
    ckpt_dir = Path(cfg["training"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = ckpt_dir / "best_world_model.pt"
    scaler_path = ckpt_dir / "state_scaler.joblib"

    history = {"train_loss": [], "val_loss": []}

    print(f"Starting training on {device}: {epochs} epochs, {len(train_dataset)} train samples...")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_epoch = 0.0

        for bx, by_next, by_risk, by_stage in train_loader:
            bx, by_next = bx.to(device), by_next.to(device)
            by_risk, by_stage = by_risk.to(device), by_stage.to(device)

            optimizer.zero_grad()
            pred_next, pred_risk, pred_stage, _ = model(bx)

            l_state = criterion_state(pred_next, by_next)
            l_risk = criterion_risk(pred_risk, by_risk)
            l_stage = criterion_stage(pred_stage, by_stage)

            total_loss = (lambda_state * l_state) + (lambda_risk * l_risk) + (lambda_stage * l_stage)
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            train_loss_epoch += total_loss.item() * bx.size(0)

        train_loss_epoch /= max(len(train_dataset), 1)

        # Validation
        model.eval()
        val_loss_epoch = 0.0
        with torch.no_grad():
            for bx, by_next, by_risk, by_stage in val_loader:
                bx, by_next = bx.to(device), by_next.to(device)
                by_risk, by_stage = by_risk.to(device), by_stage.to(device)

                pred_next, pred_risk, pred_stage, _ = model(bx)
                l_state = criterion_state(pred_next, by_next)
                l_risk = criterion_risk(pred_risk, by_risk)
                l_stage = criterion_stage(pred_stage, by_stage)

                total_loss = (lambda_state * l_state) + (lambda_risk * l_risk) + (lambda_stage * l_stage)
                val_loss_epoch += total_loss.item() * bx.size(0)

        val_loss_epoch /= max(len(val_dataset), 1)
        history["train_loss"].append(train_loss_epoch)
        history["val_loss"].append(val_loss_epoch)

        if val_loss_epoch < best_val_loss:
            best_val_loss = val_loss_epoch
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_type": model_type,
                "config": cfg,
                "epoch": epoch,
                "val_loss": val_loss_epoch
            }, best_ckpt_path)

        if epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] Train Loss: {train_loss_epoch:.4f} | Val Loss: {val_loss_epoch:.4f}")

    # Save fitted scaler
    joblib.dump(builder.scaler, scaler_path)
    print(f"Training completed! Best checkpoint saved to {best_ckpt_path}")

    return {
        "best_val_loss": best_val_loss,
        "checkpoint_path": str(best_ckpt_path),
        "scaler_path": str(scaler_path),
        "history": history
    }


if __name__ == "__main__":
    train_world_model()
