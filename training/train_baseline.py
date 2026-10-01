"""
Baseline Training Pipeline.
Trains the static Logistic Regression baseline classifier on the exact same temporal split.
Saves model checkpoint to models/checkpoints/baseline_lr.joblib for fair comparative benchmarking.
"""

from pathlib import Path
from typing import Dict, Any
import numpy as np

from src.utils.config import load_config
from src.data.datasets.generic_csv import GenericCSVAdapter
from src.features.state_builder import NetworkStateBuilder
from src.data.preprocessing.temporal_split import temporal_sequence_split
from src.models.baseline import LogisticRegressionBaseline


def train_baseline(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Fits the static Logistic Regression baseline model."""
    cfg = load_config(config_path)
    csv_path = cfg["data"]["sample_data_path"]
    window_sec = cfg["data"]["window_size_seconds"]
    seq_len = cfg["data"]["sequence_length"]
    horizon = cfg["data"]["prediction_horizon"]

    adapter = GenericCSVAdapter()
    flows_df = adapter.load(csv_path)

    builder = NetworkStateBuilder(window_size_seconds=window_sec)
    states_df = builder.build_states(flows_df)
    builder.fit_scaler(states_df)

    X, Y_next, Y_risk, Y_stage = builder.create_sliding_sequences(
        states_df, seq_length=seq_len, horizon=horizon
    )

    n = len(X)
    train_end = int(n * cfg["data"]["train_split"])

    X_train = X[:train_end]
    Y_risk_train = Y_risk[:train_end]

    baseline = LogisticRegressionBaseline(random_state=cfg["system"]["random_seed"])
    baseline.fit(X_train, Y_risk_train)

    ckpt_dir = Path(cfg["training"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = ckpt_dir / "baseline_lr.joblib"

    baseline.save(str(baseline_path))
    print(f"Logistic Regression baseline saved to: {baseline_path}")

    return {"baseline_path": str(baseline_path)}


if __name__ == "__main__":
    train_baseline()
