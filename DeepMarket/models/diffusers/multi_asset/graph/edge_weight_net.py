"""State-dependent edge weights for graph coupling."""

from __future__ import annotations

import torch
from torch import nn


class EdgeWeightNet(nn.Module):
    def __init__(self, stats_dim: int, relation_dim: int, hidden_dim: int = 32):
        super().__init__()
        input_dim = stats_dim * 2 + relation_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        stats_src: torch.Tensor,
        stats_dst: torch.Tensor,
        relation: torch.Tensor,
    ) -> torch.Tensor:
        if relation.dim() == 1:
            relation = relation.unsqueeze(0).expand(stats_src.shape[0], -1)
        features = torch.cat([stats_src, stats_dst, relation], dim=-1)
        return torch.sigmoid(self.net(features))
