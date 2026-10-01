"""
Attack Progression and Stage Predictor.
Combines world model latent state logits with temporal network state dynamics
to forecast attack phase transitions along the cyber kill chain.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from ..mitre.attack_mapping import MITREAttackMapper


class AttackStagePredictor:
    """Predicts attack stage transitions and explains contextual supporting features."""

    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names

    def predict_from_model(
        self,
        stage_logits: torch.Tensor,
        current_state: np.ndarray,
        future_state: np.ndarray
    ) -> Dict[str, Any]:
        """
        Calculates predicted stage, confidence, and supporting feature signatures.
        """
        probs = F.softmax(stage_logits, dim=-1).squeeze(0).detach().cpu().numpy()
        predicted_stage = int(np.argmax(probs))
        confidence = float(probs[predicted_stage])

        # Identify features that are driving this transition from current to future state
        delta_state = future_state - current_state
        top_indices = np.argsort(np.abs(delta_state))[::-1][:4]
        supporting_features = [self.feature_names[i] for i in top_indices if i < len(self.feature_names)]

        # Map to MITRE report
        mitre_report = MITREAttackMapper.map_prediction(
            predicted_stage=predicted_stage,
            confidence=confidence,
            supporting_features=supporting_features
        )
        mitre_report["stage_probabilities"] = {
            i: round(float(p), 4) for i, p in enumerate(probs)
        }

        return mitre_report
