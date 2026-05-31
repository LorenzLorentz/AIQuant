"""Token-wise directed messages for P1 graph coupling."""

from __future__ import annotations

import torch
from torch import nn


class MessageFunction(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int | None = None):
        super().__init__()
        hidden_dim = hidden_dim or feature_dim * 2
        self.net = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(
        self,
        x_src: torch.Tensor,
        eps_src: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        if edge_weight.dim() == 1:
            edge_weight = edge_weight.unsqueeze(-1)
        weight = edge_weight.view(edge_weight.shape[0], 1, 1)
        token_features = torch.cat([x_src, eps_src], dim=-1)
        return weight * self.net(token_features)
