"""
Abstract Base Class for Network World Models.
Defines the interface for state transition modeling, rollout simulation,
risk estimation, and stage progression prediction.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import torch
import torch.nn as nn


class BaseWorldModel(nn.Module, ABC):
    """Abstract interface for all cyber network world models."""

    @abstractmethod
    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Args:
            x: Tensor of shape (batch_size, seq_len, state_dim)
        Returns:
            next_state: Predicted S(t+1) of shape (batch_size, state_dim)
            risk: Predicted attack probability of shape (batch_size, 1)
            stage_logits: Predicted attack stage logits of shape (batch_size, num_stages)
            latent: Latent network representation of shape (batch_size, latent_dim)
        """
        pass

    @abstractmethod
    def rollout(
        self, initial_sequence: torch.Tensor, steps: int = 5
    ) -> Dict[str, Any]:
        """
        Performs K-step forward simulation from the observed sequence.
        Args:
            initial_sequence: Tensor of shape (1, seq_len, state_dim)
            steps: K-step simulation horizon
        Returns:
            Dict containing trajectory of predicted states, risk timeline, and stage progressions.
        """
        pass
