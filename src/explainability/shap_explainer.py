"""
Explainability and Model Attribution Engine.
Provides SHAP and Gradient-based feature and temporal window attributions
for the Cyber Network World Model without claiming false causality.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import torch
import torch.nn as nn
from ..models.world_model import BaseWorldModel


class WorldModelExplainer:
    """
    Computes feature-level and time-window attributions for predictions
    using Integrated Gradients and SHAP-aligned baseline perturbations.
    """

    def __init__(
        self,
        model: BaseWorldModel,
        feature_names: List[str],
        device: str = "cpu"
    ):
        self.model = model
        self.feature_names = feature_names
        self.device = device
        self.model.to(self.device)

    def explain_prediction(
        self,
        sequence_tensor: torch.Tensor,
        target: str = "risk",  # "risk" or "stage"
        num_steps: int = 30
    ) -> Dict[str, Any]:
        """
        Computes feature and time-window attributions using path-integrated gradients.
        Args:
            sequence_tensor: (1, seq_len, state_dim)
            target: attribution target ("risk" or "stage")
            num_steps: Riemann approximation steps
        Returns:
            Dict containing sorted feature attributions and time-window importance.
        """
        self.model.eval()
        seq = sequence_tensor.to(self.device).clone()
        seq.requires_grad = True

        # Baseline: neutral quiescent state (zeros)
        baseline = torch.zeros_like(seq)

        # Integrated Gradients
        accumulated_grads = torch.zeros_like(seq)

        for step in range(1, num_steps + 1):
            alpha = float(step / num_steps)
            interpolated = baseline + alpha * (seq - baseline)
            interpolated = interpolated.clone().detach().requires_grad_(True)

            _, risk, stage_logits, _ = self.model.forward(interpolated)

            if target == "risk":
                score = risk.squeeze()
            else:
                score = torch.max(stage_logits.squeeze())

            score.backward()
            if interpolated.grad is not None:
                accumulated_grads += interpolated.grad

        # Average gradient along straight-line path
        avg_grad = accumulated_grads / num_steps
        # Attribution = (x - x') * avg_grad
        attributions = (seq - baseline) * avg_grad
        attributions_np = attributions.squeeze(0).detach().cpu().numpy()  # (seq_len, state_dim)

        # 1. Feature-level overall attribution (aggregate over sequence windows)
        feature_importance_raw = np.mean(attributions_np, axis=0)  # (state_dim,)
        # Also most recent window attribution
        recent_window_attr = attributions_np[-1, :]

        # Combine with weight on recent window
        blended_attr = 0.6 * recent_window_attr + 0.4 * feature_importance_raw

        # Sort features by absolute contribution
        sorted_indices = np.argsort(np.abs(blended_attr))[::-1]

        top_contributors = []
        for idx in sorted_indices:
            feat_name = self.feature_names[idx] if idx < len(self.feature_names) else f"feature_{idx}"
            score_val = float(blended_attr[idx])
            top_contributors.append({
                "feature": feat_name,
                "attribution_score": round(score_val, 4),
                "direction": "positive (escalates risk)" if score_val > 0 else "negative (suppresses risk)"
            })

        # 2. Temporal time-window importance
        window_importance = np.linalg.norm(attributions_np, axis=1)  # (seq_len,)
        norm_sum = np.sum(window_importance) + 1e-8
        norm_window_imp = (window_importance / norm_sum).tolist()

        temporal_windows = []
        seq_len = sequence_tensor.size(1)
        for t_idx, imp in enumerate(norm_window_imp):
            offset = seq_len - 1 - t_idx
            label = "Current window t" if offset == 0 else f"Window t-{offset}"
            temporal_windows.append({
                "window_index": t_idx,
                "label": label,
                "relative_influence": round(float(imp), 4)
            })

        return {
            "top_features": top_contributors[:8],
            "all_features": top_contributors,
            "temporal_window_attribution": temporal_windows,
            "disclaimer": (
                "Feature and temporal attributions reflect model sensitivity and path-integrated gradients. "
                "These scores highlight what signals influenced the neural trajectory, not verified causality."
            )
        }
