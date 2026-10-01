"""
Evaluation and Benchmarking Pipeline.
Compares the Temporal World Model against the Logistic Regression Baseline on unseen test timeline.
Generates genuine metric reports, confusion matrix plot, model comparison table, and risk timelines.
"""

from pathlib import Path
from typing import Dict, Any, Tuple
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    precision_recall_curve,
    auc,
    confusion_matrix,
    mean_squared_error,
    mean_absolute_error,
    accuracy_score
)

from src.utils.config import load_config
from src.data.datasets.generic_csv import GenericCSVAdapter
from src.features.state_builder import NetworkStateBuilder
from src.models.lstm_world_model import LSTMNetworkWorldModel
from src.models.baseline import LogisticRegressionBaseline


def evaluate_models(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Runs rigorous temporal evaluation on the test partition."""
    cfg = load_config(config_path)
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    csv_path = cfg["data"]["sample_data_path"]
    window_sec = cfg["data"]["window_size_seconds"]
    seq_len = cfg["data"]["sequence_length"]
    horizon = cfg["data"]["prediction_horizon"]
    device = cfg["system"].get("device", "cpu")

    # Load data
    adapter = GenericCSVAdapter()
    flows_df = adapter.load(csv_path)

    builder = NetworkStateBuilder(window_size_seconds=window_sec)
    states_df = builder.build_states(flows_df)
    builder.fit_scaler(states_df)

    X, Y_next, Y_risk, Y_stage = builder.create_sliding_sequences(
        states_df, seq_length=seq_len, horizon=horizon
    )

    n = len(X)
    val_end = int(n * (cfg["data"]["train_split"] + cfg["data"]["val_split"]))

    # Test partition (strictly unseen chronological timeline)
    X_test = X[val_end:]
    Y_next_test = Y_next[val_end:]
    Y_risk_test = Y_risk[val_end:]
    Y_stage_test = Y_stage[val_end:]
    Y_binary_test = (Y_risk_test > 0.5).astype(int)

    # 1. Load Trained World Model
    ckpt_path = Path(cfg["training"]["checkpoint_dir"]) / "best_world_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}. Train model first!")

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

    # World Model Inference on Test
    with torch.no_grad():
        x_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        pred_next_t, pred_risk_t, pred_stage_logits_t, _ = model(x_tensor)
        pred_next = pred_next_t.cpu().numpy()
        pred_risk = pred_risk_t.squeeze(1).cpu().numpy()
        pred_stage = torch.argmax(pred_stage_logits_t, dim=-1).cpu().numpy()

    # World model binary predictions
    wm_preds = (pred_risk >= 0.5).astype(int)

    # World Model Metrics
    wm_precision = float(precision_score(Y_binary_test, wm_preds, zero_division=0))
    wm_recall = float(recall_score(Y_binary_test, wm_preds, zero_division=0))
    wm_f1 = float(f1_score(Y_binary_test, wm_preds, zero_division=0))

    try:
        wm_roc_auc = float(roc_auc_score(Y_binary_test, pred_risk))
    except ValueError:
        wm_roc_auc = 0.5

    # PR-AUC
    p_prec, p_rec, _ = precision_recall_curve(Y_binary_test, pred_risk)
    wm_pr_auc = float(auc(p_rec, p_prec))

    # Confusion matrix
    wm_cm = confusion_matrix(Y_binary_test, wm_preds, labels=[0, 1])
    wm_tn, wm_fp, wm_fn, wm_tp = wm_cm.ravel()
    wm_fpr = float(wm_fp / (wm_fp + wm_tn)) if (wm_fp + wm_tn) > 0 else 0.0

    # State prediction error
    state_mse = float(mean_squared_error(Y_next_test, pred_next))
    state_mae = float(mean_absolute_error(Y_next_test, pred_next))

    # Stage accuracy
    stage_acc = float(accuracy_score(Y_stage_test, pred_stage))

    # 2. Load & Evaluate Baseline Model
    baseline_path = Path(cfg["training"]["checkpoint_dir"]) / "baseline_lr.joblib"
    baseline = LogisticRegressionBaseline()
    baseline.load(str(baseline_path))
    bl_metrics = baseline.evaluate(X_test, Y_risk_test)

    # 3. Model Comparison Table
    comparison_df = pd.DataFrame([
        {
            "Model": "Proposed Cyber World Model",
            "Precision": round(wm_precision, 4),
            "Recall": round(wm_recall, 4),
            "F1-Score": round(wm_f1, 4),
            "ROC-AUC": round(wm_roc_auc, 4),
            "PR-AUC": round(wm_pr_auc, 4),
            "FPR": round(wm_fpr, 4),
            "State Prediction MSE": round(state_mse, 4),
            "Stage Accuracy": round(stage_acc, 4),
            "Temporal Horizon Rollout": "Yes (K-step)"
        },
        {
            "Model": "Logistic Regression Baseline",
            "Precision": round(bl_metrics["precision"], 4),
            "Recall": round(bl_metrics["recall"], 4),
            "F1-Score": round(bl_metrics["f1"], 4),
            "ROC-AUC": round(bl_metrics["roc_auc"], 4),
            "PR-AUC": "N/A (Static)",
            "FPR": round(bl_metrics["fpr"], 4),
            "State Prediction MSE": "N/A (No Dynamics)",
            "Stage Accuracy": "N/A (Binary Only)",
            "Temporal Horizon Rollout": "No (Static Window)"
        }
    ])

    comparison_path = reports_dir / "model_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)
    print("\n--- MODEL COMPARISON RESULTS ---")
    print(comparison_df.to_string())

    # 4. Save Confusion Matrix Plot
    plt.figure(figsize=(6, 5))
    sns.heatmap(wm_cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Benign (0)", "Malicious (1)"],
                yticklabels=["Benign (0)", "Malicious (1)"])
    plt.title("World Model Confusion Matrix (Chronological Test Set)")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    cm_path = reports_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=200)
    plt.close()

    # 5. Save Risk Timeline Plot
    plt.figure(figsize=(9, 4))
    plt.plot(Y_risk_test, label="True Network Risk", color="#10B981", linewidth=2.0)
    plt.plot(pred_risk, label="Predicted World Model Risk", color="#EF4444", linestyle="--", linewidth=2.0)
    plt.axhline(0.5, color="gray", linestyle=":", alpha=0.7, label="Alert Threshold (0.5)")
    plt.title("Network Attack Trajectory & Risk Calibration (Test Timeline)")
    plt.xlabel("Test Time Windows (Consecutive 10s states)")
    plt.ylabel("Attack Probability P(Attack)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    risk_plot_path = reports_dir / "risk_timeline.png"
    plt.savefig(risk_plot_path, dpi=200)
    plt.close()

    # 6. Save detailed results CSV
    results_df = pd.DataFrame({
        "time_window_idx": np.arange(len(Y_risk_test)),
        "true_risk": Y_risk_test,
        "predicted_risk": pred_risk,
        "true_stage": Y_stage_test,
        "predicted_stage": pred_stage
    })
    results_path = reports_dir / "results.csv"
    results_df.to_csv(results_path, index=False)

    print(f"\nArtifacts generated:")
    print(f"- {comparison_path}")
    print(f"- {cm_path}")
    print(f"- {risk_plot_path}")
    print(f"- {results_path}")

    return {
        "world_model": {
            "precision": wm_precision,
            "recall": wm_recall,
            "f1": wm_f1,
            "roc_auc": wm_roc_auc,
            "fpr": wm_fpr,
            "state_mse": state_mse
        },
        "baseline": bl_metrics
    }


if __name__ == "__main__":
    evaluate_models()
