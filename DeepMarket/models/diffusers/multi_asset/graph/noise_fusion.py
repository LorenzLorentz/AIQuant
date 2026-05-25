"""Residual fusion of local score estimates and graph messages."""

from __future__ import annotations

import torch
from torch import nn


class NoiseFusion(nn.Module):
    """g(eps, m) = eps + gamma * MLP([eps, m]).

    ``gamma`` is initialized at zero, so the module is an exact no-op at
    construction time. This lets MA_TRADES start from P0 behavior.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int | None = None,
        init_gamma: float = 0.0,
    ):
        super().__init__()
        hidden_dim = hidden_dim or max(32, 2 * feature_dim)
        self.gamma = nn.Parameter(torch.tensor(float(init_gamma), dtype=torch.float32))
        self.mlp = nn.Sequential(
            nn.Linear(2 * feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(self, eps_local, messages):
        delta = self.mlp(torch.cat([eps_local, messages], dim=-1))
        return eps_local + self.gamma.to(dtype=eps_local.dtype, device=eps_local.device) * delta


__all__ = ["NoiseFusion"]
