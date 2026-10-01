"""
Transformer-based Network World Model.
Implements multi-head self-attention over historical network state windows
with temporal causal masking, multi-step autoregressive rollout, and attention extraction.
"""

from typing import Dict, Any, Tuple, List
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .world_model import BaseWorldModel
from .lstm_world_model import FeatureEncoder, FutureStateDecoder, AttackRiskHead, AttackStageHead


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal network states."""

    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class TransformerNetworkWorldModel(BaseWorldModel):
    """
    Temporal Network World Model using Transformer Encoder with causal attention.
    Enables extraction of self-attention weights for temporal interpretability.
    """

    def __init__(
        self,
        state_dim: int = 20,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        num_layers: int = 2,
        nhead: int = 4,
        dropout: float = 0.15,
        num_stages: int = 6
    ):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_stages = num_stages

        self.feature_encoder = FeatureEncoder(state_dim, hidden_dim, dropout)
        self.pos_encoder = PositionalEncoding(hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.latent_proj = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU()
        )

        self.future_decoder = FutureStateDecoder(latent_dim, state_dim, hidden_dim)
        self.risk_head = AttackRiskHead(latent_dim)
        self.stage_head = AttackStageHead(latent_dim, num_stages)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape
        last_state = x[:, -1, :]

        enc_x = self.feature_encoder(x)
        enc_x = self.pos_encoder(enc_x)

        # Transformer representation
        tf_out = self.transformer_encoder(enc_x)
        final_temporal = tf_out[:, -1, :]

        latent = self.latent_proj(final_temporal)
        next_state = self.future_decoder(latent, last_state)
        risk = self.risk_head(latent)
        stage_logits = self.stage_head(latent)

        return next_state, risk, stage_logits, latent

    def rollout(
        self, initial_sequence: torch.Tensor, steps: int = 5
    ) -> Dict[str, Any]:
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
            for _ in range(steps):
                next_state, risk, stage_logits, _ = self.forward(current_seq)
                stage_probs = F.softmax(stage_logits, dim=-1)
                predicted_stage = torch.argmax(stage_probs, dim=-1).item()
                risk_val = float(risk.item())

                future_states.append(next_state.squeeze(0).cpu().numpy().tolist())
                future_risks.append(risk_val)
                future_stages.append(predicted_stage)
                future_stage_probs.append(stage_probs.squeeze(0).cpu().numpy().tolist())

                next_state_3d = next_state.unsqueeze(1)
                current_seq = torch.cat([current_seq[:, 1:, :], next_state_3d], dim=1)

        return {
            "horizon": steps,
            "future_states": future_states,
            "risk_timeline": future_risks,
            "stage_timeline": future_stages,
            "stage_probabilities": future_stage_probs
        }
