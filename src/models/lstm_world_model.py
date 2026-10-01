"""
LSTM-based Network World Model.
Implements temporal transition dynamics P(S(t+1) | S(t), history),
multi-step autoregressive rollout, attack risk estimation, and MITRE stage forecasting.
"""

from typing import Dict, Any, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from .world_model import BaseWorldModel


class FeatureEncoder(nn.Module):
    """Projects raw or normalized network state features into a dense representation."""

    def __init__(self, state_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FutureStateDecoder(nn.Module):
    """Decodes latent representation and last state into predicted next network state S(t+1)."""

    def __init__(self, latent_dim: int, state_dim: int, hidden_dim: int):
        super().__init__()
        # Residual transition network
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim + state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, state_dim)
        )

    def forward(self, latent: torch.Tensor, current_state: torch.Tensor) -> torch.Tensor:
        concat = torch.cat([latent, current_state], dim=-1)
        residual = self.mlp(concat)
        # Residual prediction: S(t+1) = S(t) + delta S(t)
        return current_state + residual


class AttackRiskHead(nn.Module):
    """Estimates the probability of malicious network activity P(Attack in S(t+1))."""

    def __init__(self, latent_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.net(latent)


class AttackStageHead(nn.Module):
    """Predicts the MITRE ATT&CK progression stage (0 to num_stages-1)."""

    def __init__(self, latent_dim: int, num_stages: int = 6, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_stages)
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.net(latent)


class LSTMNetworkWorldModel(BaseWorldModel):
    """
    Complete Temporal Network World Model based on LSTM.
    Learns state transition dynamics P(S(t+1) | S(t), history).
    """

    def __init__(
        self,
        state_dim: int = 20,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.15,
        num_stages: int = 6
    ):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_stages = num_stages

        # 1. Feature Projection
        self.feature_encoder = FeatureEncoder(state_dim, hidden_dim, dropout)

        # 2. Temporal Sequence Modeling
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # 3. Latent State Projection
        self.latent_proj = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU()
        )

        # 4. Future State Decoder S_hat(t+1)
        self.future_decoder = FutureStateDecoder(latent_dim, state_dim, hidden_dim)

        # 5. Attack Risk Prediction Head
        self.risk_head = AttackRiskHead(latent_dim)

        # 6. Attack Stage Classification Head
        self.stage_head = AttackStageHead(latent_dim, num_stages)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input trajectory (batch_size, seq_len, state_dim)
        Returns:
            next_state: (batch_size, state_dim)
            risk: (batch_size, 1)
            stage_logits: (batch_size, num_stages)
            latent: (batch_size, latent_dim)
        """
        batch_size, seq_len, _ = x.shape
        last_state = x[:, -1, :]  # S(t)

        # Project features
        enc_x = self.feature_encoder(x)  # (batch_size, seq_len, hidden_dim)

        # Temporal LSTM
        lstm_out, _ = self.lstm(enc_x)  # (batch_size, seq_len, hidden_dim)
        final_temporal = lstm_out[:, -1, :]  # Hidden state at t

        # Latent state representation
        latent = self.latent_proj(final_temporal)  # (batch_size, latent_dim)

        # Future state prediction S(t+1)
        next_state = self.future_decoder(latent, last_state)

        # Attack probability and stage prediction
        risk = self.risk_head(latent)
        stage_logits = self.stage_head(latent)

        return next_state, risk, stage_logits, latent

    def rollout(
        self, initial_sequence: torch.Tensor, steps: int = 5
    ) -> Dict[str, Any]:
        """
        K-step forward simulation of network state dynamics.
        Args:
            initial_sequence: (1, seq_len, state_dim) or (seq_len, state_dim)
            steps: K forward rollout steps
        Returns:
            Dictionary with predicted state trajectories, risk timeline, and stage progression.
        """
        self.eval()
        if initial_sequence.ndim == 2:
            current_seq = initial_sequence.unsqueeze(0).clone()
        else:
            current_seq = initial_sequence.clone()

        future_states = []
        future_risks = []
        future_stages = []
        future_stage_probs = []

        with torch.no_grad():
            for step in range(1, steps + 1):
                next_state, risk, stage_logits, _ = self.forward(current_seq)

                stage_probs = F.softmax(stage_logits, dim=-1)
                predicted_stage = torch.argmax(stage_probs, dim=-1).item()
                risk_val = float(risk.item())

                future_states.append(next_state.squeeze(0).cpu().numpy().tolist())
                future_risks.append(risk_val)
                future_stages.append(predicted_stage)
                future_stage_probs.append(stage_probs.squeeze(0).cpu().numpy().tolist())

                # Roll sliding sequence forward: drop oldest state, append predicted next state
                next_state_3d = next_state.unsqueeze(1)  # (1, 1, state_dim)
                current_seq = torch.cat([current_seq[:, 1:, :], next_state_3d], dim=1)

        return {
            "horizon": steps,
            "future_states": future_states,
            "risk_timeline": future_risks,
            "stage_timeline": future_stages,
            "stage_probabilities": future_stage_probs
        }
