"""State-dependent directed edge weights for P1 graph coupling."""

from __future__ import annotations

import torch
from torch import nn


class EdgeWeightNet(nn.Module):
    """MLP mapping source/destination stats and relation embedding to w_ji."""

    def __init__(self, stats_dim: int = 4, relation_dim: int = 8, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * stats_dim + relation_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, stats_src, stats_dst, relation_emb):
        """Return weights in (0, 1) with shape matching the input edge axes."""
        features = torch.cat([stats_src, stats_dst, relation_emb], dim=-1)
        return torch.sigmoid(self.net(features)).squeeze(-1)


__all__ = ["EdgeWeightNet"]
