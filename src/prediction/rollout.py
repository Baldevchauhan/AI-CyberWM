"""
Multi-step Forward Simulation Rollout Coordinator.
Executes K-step state simulation S(t) -> S(t+1) -> ... -> S(t+K),
computes future risk timelines, and projects attack progression.
"""

from typing import Dict, Any, List
import numpy as np
import torch
from .risk import RiskEstimator
from .attack_stage import AttackStagePredictor
from ..models.world_model import BaseWorldModel


class ForwardRolloutSimulator:
    """Coordinates multi-step network trajectory forecasting."""

    def __init__(
        self,
        model: BaseWorldModel,
        feature_names: List[str],
        risk_estimator: RiskEstimator,
        stage_predictor: AttackStagePredictor,
        device: str = "cpu"
    ):
        self.model = model
        self.feature_names = feature_names
        self.risk_estimator = risk_estimator
        self.stage_predictor = stage_predictor
        self.device = device
        self.model.to(self.device)

    def simulate(
        self,
        sequence_tensor: torch.Tensor,
        current_state_raw: np.ndarray,
        horizon: int = 5,
        window_duration: float = 10.0
    ) -> Dict[str, Any]:
        """
        Executes autoregressive rollout for `horizon` steps.
        Args:
            sequence_tensor: (1, seq_len, state_dim)
            current_state_raw: 1D array of unscaled current state features
            horizon: K forward steps
            window_duration: duration of time window in seconds
        Returns:
            Dictionary with full simulation trajectory, risk timeline, and stage progressions.
        """
        seq = sequence_tensor.to(self.device)
        rollout_results = self.model.rollout(seq, steps=horizon)

        # Immediate current step evaluation
        self.model.eval()
        with torch.no_grad():
            _, current_risk_t, current_stage_logits, _ = self.model.forward(seq)
            current_risk = float(current_risk_t.item())

        future_risks = rollout_results["risk_timeline"]
        future_states = rollout_results["future_states"]

        # Build risk timeline
        risk_timeline = self.risk_estimator.format_risk_timeline(
            current_probability=current_risk,
            future_probabilities=future_risks,
            window_duration=window_duration
        )

        # Stage prediction for immediate next window
        next_state_vec = np.array(future_states[0]) if future_states else current_state_raw
        stage_assessment = self.stage_predictor.predict_from_model(
            stage_logits=current_stage_logits,
            current_state=current_state_raw,
            future_state=next_state_vec
        )

        return {
            "current_attack_probability": round(current_risk, 4),
            "current_severity": self.risk_estimator.classify_severity(current_risk),
            "future_attack_probabilities": [round(p, 4) for p in future_risks],
            "risk_timeline": risk_timeline,
            "predicted_stage": stage_assessment["predicted_stage_name"],
            "stage_confidence": stage_assessment["confidence"],
            "mitre_report": stage_assessment,
            "future_state_trajectories": future_states,
            "horizon": horizon
        }
