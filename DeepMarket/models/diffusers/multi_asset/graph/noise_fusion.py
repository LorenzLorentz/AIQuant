"""Residual fusion of local and graph messages."""

from __future__ import annotations

import torch
from torch import nn


class NoiseFusion(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int | None = None):
        super().__init__()
        hidden_dim = hidden_dim or feature_dim * 2
        self.gamma = nn.Parameter(torch.tensor(0.0))
        self.net = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(self, eps_local: torch.Tensor, messages: torch.Tensor) -> torch.Tensor:
        residual = self.net(torch.cat([eps_local, messages], dim=-1))
        return eps_local + self.gamma * residual
